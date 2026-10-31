"""DocuBharat Streamlit app.

Run with:  streamlit run app.py

Without configuration everything runs locally: PyMuPDF extraction, BM25 retrieval,
rule-based fields and extractive answers. Optional components (see .env.example):
- BharatGen Patram on the user's private ZeroGPU Space reads page images to answer
  questions, summarise and extract fields, including scanned PDFs;
- a hosted LLM (Groq) answers from text passages with quotes checked against them;
- AI4Bharat IndicTrans2 translates answers locally on the CPU.
Uploaded files are held in memory for the session only and are never written to disk.
"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from src import indic_translator, llm_provider, patram_client
from src.config import Settings
from src.field_extractor import ExtractedFields, FieldValue, extract_fields
from src.llm_provider import LLMConfig, LLMError
from src.llm_tasks import GroundedItem, LLMFields, LLMSummary, llm_answer, llm_extract_fields, llm_summarize
from src.pdf_processor import PDFValidationError
from src.pipeline import ProcessedDocument, process_pdf
from src.patram_client import PatramUnavailable, render_page_png
from src.patram_tasks import PatramPagesResult, PatramQA, check_grounding, choose_page, page_text, patram_answer, patram_fields, patram_summary
from src.qa_engine import QAResult, answer_question
from src.retrieval import SearchResult
from src.summarizer import StructuredSummary, summarize
from src.ui_text import highlight, md_escape

try:  # optional .env support; the app works without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("docubharat")
SAMPLE_DIR = Path(__file__).parent / "data" / "sample_documents"
EXAMPLE_QUESTIONS = [
    "What is the last date to apply?",
    "What is the fee?",
    "Who is eligible?",
    "How can I contact the office?",
]
GENERIC_LLM_ERROR = "The AI request failed unexpectedly."
ENGINE_LABELS = {
    "patram": "BharatGen Patram (reads the page image)",
    "groq": "Groq LLM (reads text passages)",
    "extractive": "Extractive only (no AI)",
}
AUTO_PAGE = "Auto (best keyword match)"

st.set_page_config(page_title="DocuBharat", page_icon="📄", layout="wide")
settings = Settings.from_env()


# --------------------------------------------------------------------------- helpers

def page_ref(page: int | None) -> str:
    return f"p. {page}" if page else ""


def reset_document() -> None:
    for key in ("doc", "fields", "summary", "doc_key", "qa_history", "load_error", "llm_cache",
                "pdf_bytes", "patram_cache"):
        st.session_state.pop(key, None)
    st.session_state.pop(f"question_{st.session_state.get('uploader_nonce', 0)}", None)
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1


def load_document(data: bytes, filename: str) -> None:
    """Process a PDF once per unique file and keep the results in session state."""
    try:
        with st.spinner(f"Reading {filename}…"):
            doc = process_pdf(data, filename, settings)
            fields = extract_fields(doc.pages)
            summary = summarize(doc.pages, fields)
    except PDFValidationError as exc:
        st.session_state.load_error = str(exc)
        return
    except Exception:  # never show internals in the browser
        logger.exception("Unexpected error while processing an uploaded PDF")
        st.session_state.load_error = "Something went wrong while reading this PDF. Please try a different file."
        return
    st.session_state.update(
        doc=doc, fields=fields, summary=summary, doc_key=doc.extraction.sha256, qa_history=[], llm_cache={},
        pdf_bytes=data, patram_cache={},  # bytes kept in memory only, to render pages for Patram
    )
    st.session_state.pop("load_error", None)


def get_llm():
    """(provider or None, config, configuration problem). Created once per session."""
    config = LLMConfig.from_env()
    if "llm" not in st.session_state:
        try:
            st.session_state.llm = (llm_provider.create_provider(config), None)
        except LLMError as exc:
            st.session_state.llm = (None, exc.user_message)
    provider, problem = st.session_state.llm
    return provider, config, problem


@st.cache_resource(show_spinner=False)
def get_translator():
    """One IndicTrans2 translator per server, shared by all sessions. The model itself loads on
    the first Translate click. Restart the app after downloading the model."""
    return indic_translator.create_translator()


@st.cache_resource(show_spinner=False)
def get_patram():
    """One Patram client per server (connects to the Space on first use)."""
    return patram_client.create_client()


def run_patram(task: str, key: str, func) -> tuple[object | None, str | None]:
    """Run a Patram task once per (document, task, key); errors are not cached. Returns (result, error)."""
    cache: dict = st.session_state.setdefault("patram_cache", {})
    cache_key = (st.session_state.get("doc_key"), task, key)
    try:
        if cache_key not in cache:
            cache[cache_key] = func()
        return cache[cache_key], None
    except PatramUnavailable as exc:
        return None, str(exc)
    except Exception:
        logger.exception("Unexpected error during Patram task %s", task)
        return None, "The Patram request failed unexpectedly."


def patram_cached(task: str, key: str = "") -> bool:
    return (st.session_state.get("doc_key"), task, key) in st.session_state.get("patram_cache", {})


def cached_llm_call(task: str, key: str, func):
    """Run an LLM task once per (document, task, model, key); errors are not cached."""
    cache: dict = st.session_state.setdefault("llm_cache", {})
    cache_key = (st.session_state.get("doc_key"), task, llm_config.model, key)
    if cache_key not in cache:
        cache[cache_key] = func()
    return cache[cache_key]


def run_llm(task: str, key: str, func) -> tuple[object | None, str | None]:
    """(result, user-safe error message)."""
    try:
        return cached_llm_call(task, key, func), None
    except LLMError as exc:
        return None, exc.user_message
    except Exception:
        logger.exception("Unexpected error during LLM task %s", task)
        return None, GENERIC_LLM_ERROR


def pages_sent_label(last_page: int, truncated: bool) -> str:
    label = "page 1" if last_page == 1 else f"pages 1–{last_page}"
    return label + (" (document was longer; the rest was not sent)" if truncated else "")


def verified_badge(item_verified: bool) -> str:
    return "✅" if item_verified else "⚠️"


# --------------------------------------------------------------------------- sidebar

st.session_state.setdefault("uploader_nonce", 0)
llm, llm_config, llm_problem = get_llm()
translator, translator_reason = get_translator()
patram, patram_reason = get_patram()

with st.sidebar:
    st.header("Document")
    samples = sorted(p.name for p in SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []
    if samples:
        sample = st.selectbox("Try a sample document", samples, index=None, placeholder="Choose a sample…")
        if st.button("Load sample", disabled=sample is None, width="stretch"):
            reset_document()
            load_document((SAMPLE_DIR / sample).read_bytes(), sample)
            st.rerun()
    if st.button("Clear document", width="stretch", disabled="doc" not in st.session_state):
        reset_document()
        st.rerun()

    st.divider()
    st.subheader("Answer engine")
    engines = [e for e, ok in (("patram", patram is not None), ("groq", llm is not None), ("extractive", True)) if ok]
    engine = st.radio("Answer questions with", engines, format_func=ENGINE_LABELS.get, index=0)
    use_patram, use_llm = engine == "patram", engine == "groq"
    if patram is not None:
        st.success("BharatGen Patram: **ready** (private ZeroGPU Space)", icon="🏛️")
        st.caption(
            f"Patram sees one page image per question (the best keyword match unless you pick a page). "
            f"Summary and fields send up to {patram.config.max_pages} page(s), only when you click. "
            f"Space calls since the app started: {patram.calls}. Free quota ≈ 5 GPU-minutes/day."
        )
    else:
        st.caption(f"BharatGen Patram not available: {patram_reason}")
    if llm is not None:
        st.caption(
            f"Groq ({llm_config.display_name}): only the top {llm_config.max_passages} passages "
            f"(≤ {llm_config.max_context_chars:,} characters) are sent per question."
        )
    elif llm_problem:
        st.warning(llm_problem, icon="⚠️")

    st.subheader("Translation (AI4Bharat IndicTrans2)")
    if translator is not None:
        target_language = st.selectbox(
            "Translate answers to",
            [None, *indic_translator.LANGUAGES],
            format_func=lambda code: "Off" if code is None else indic_translator.LANGUAGES[code],
        )
        st.caption(
            "Runs locally on this computer (CPU); nothing is sent online. The model loads on the first "
            "translation. Quotes and source passages are never translated."
        )
    else:
        target_language = None
        st.caption(f"Not available: {translator_reason}")

    st.subheader("How answers are produced")
    answer_line = {
        "patram": "BharatGen Patram reads the page image; numbers in its answer are checked against the page "
        "text. Falls back to the extractive answer if the Space is unavailable.",
        "groq": "Groq LLM from the retrieved passages only; every quote is checked against the document. "
        "Falls back to the extractive answer if the AI is unavailable.",
        "extractive": "extractive — the best matching sentence, verbatim.",
    }[engine]
    st.markdown(
        "\n".join(
            [
                "- **Text extraction:** PyMuPDF · **Retrieval:** BM25 keyword search",
                "- **Rule-based fields & summary:** quoted from the document (always shown)",
                f"- **Answers:** {answer_line}",
            ]
        )
    )
    st.caption(
        f"Limits: {settings.max_upload_mb} MB per file · first {settings.max_pages} pages · "
        f"top {settings.top_k} passages per question. Files stay in memory for this session only."
    )


# --------------------------------------------------------------------------- header & upload

st.title("📄 DocuBharat")
st.markdown(
    "Understand university notices, examination circulars and government documents: "
    "get a structured summary, key dates and fees, and answers backed by the exact source passage."
)

uploaded = st.file_uploader(
    "Upload a PDF", type=["pdf"], key=f"uploader_{st.session_state.uploader_nonce}", help="Digital (text) PDFs work best."
)
if uploaded is not None:
    data = uploaded.getvalue()
    if sha256(data).hexdigest() != st.session_state.get("doc_key"):
        load_document(data, uploaded.name)

if error := st.session_state.get("load_error"):
    st.error(error, icon="🚫")

doc: ProcessedDocument | None = st.session_state.get("doc")
if doc is None:
    st.markdown("#### Get started")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Upload**  \nA PDF notice, circular or announcement (up to 10 MB).")
    c2.markdown("**2. Review**  \nSee the summary, important dates, fees, eligibility and contacts.")
    c3.markdown("**3. Ask**  \nAsk a question and see the answer with the page it came from.")
    if samples:
        st.caption("No PDF handy? Pick a sample document from the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- document overview

extraction = doc.extraction
st.subheader(f"📑 {doc.filename}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pages", extraction.total_pages)
m2.metric("Pages processed", extraction.pages_processed)
m3.metric("Characters extracted", f"{extraction.char_count:,}")
m4.metric("Passages indexed", len(doc.chunks))

for warning in extraction.warnings:
    st.warning(warning, icon="⚠️")

scanned = extraction.needs_ocr
if scanned and patram is None:
    st.markdown(
        "**Why nothing is shown:** this file appears to contain page images rather than text. "
        "The text pipeline cannot read it, and BharatGen Patram (which reads page images) is not configured: "
        f"{patram_reason}"
    )
    st.stop()
if scanned:
    st.info(
        "Scanned PDF: there is no text layer, so only **BharatGen Patram** can read it (from the page images). "
        "Rule-based fields, keyword search and the Groq LLM need text and are skipped. "
        "Patram's numbers cannot be checked against page text for scanned pages.",
        icon="🏛️",
    )
    use_patram, use_llm = True, False
pdf_bytes: bytes = st.session_state.pdf_bytes
page_count = extraction.pages_processed

fields: ExtractedFields = st.session_state.fields
summary: StructuredSummary = st.session_state.summary

tab_summary, tab_fields, tab_qa, tab_text = st.tabs(
    ["📝 Summary", "🔎 Important fields", "💬 Ask a question", "📃 Extracted text"]
)

# --------------------------------------------------------------------------- summary


def show_grounded_list(title: str, items: list[GroundedItem]) -> None:
    st.markdown(f"##### {title}")
    if not items:
        st.caption("Nothing reported.")
    for item in items:
        where = page_ref(item.page) if item.verified else "quote not found in document"
        st.markdown(f"- {verified_badge(item.verified)} {md_escape(item.text)} *({where})*")


def show_llm_summary(ai: LLMSummary) -> None:
    verified = sum(i.verified for i in ai.all_items)
    total = len(ai.all_items)
    with st.container(border=True):
        st.caption(
            f"AI-generated summary · {md_escape(ai.model)} · from {pages_sent_label(ai.pages_sent, ai.truncated)}"
            + f" · {verified}/{total} items matched to a verbatim quote (✅). Check ⚠️ items against the document."
        )
        st.markdown(f"**Purpose:** {md_escape(ai.purpose)}")
        show_grounded_list("Main points", ai.main_points)
        show_grounded_list("Important dates", ai.important_dates)
        show_grounded_list("Required actions", ai.required_actions)
        show_grounded_list("Fees and eligibility", ai.fees_and_eligibility)


def show_patram_pages(result: PatramPagesResult, what: str) -> None:
    pages = [a.page for a in result.answers]
    with st.container(border=True):
        st.caption(
            f"🏛️ AI-generated {what} · {patram_client.MODEL_LABEL} · read from page image(s) "
            f"{', '.join(map(str, pages))} · check against the document"
        )
        for a in result.answers:
            if len(result.answers) > 1:
                st.markdown(f"**Page {a.page}**")
            st.markdown(md_escape(a.text).replace("\n", "  \n"))
            grounding = check_grounding(a.text, page_text(doc.pages, a.page))
            st.caption(f"Numbers: {grounding.label}")


with tab_summary:
    if patram is not None:
        st.markdown("#### 🏛️ BharatGen Patram summary")
        n_calls = min(page_count, patram.config.max_pages)
        if patram_cached("summary") or st.button(
            f"Summarise with Patram ({n_calls} GPU call{'s' if n_calls > 1 else ''})", type="primary"
        ):
            with st.spinner("Patram is reading the page image(s)…"):
                p_summary, err = run_patram("summary", "", lambda: patram_summary(patram, pdf_bytes, page_count))
            if err:
                st.warning(f"Patram summary unavailable: {err}", icon="⚠️")
            else:
                show_patram_pages(p_summary, "summary")
    if scanned:
        st.caption("Rule-based and Groq summaries need a text layer, so they are not available for scanned PDFs.")
    if llm is not None and not scanned:
        st.markdown("#### 🤖 AI summary")
        cache_key = (doc.extraction.sha256, "summary", llm_config.model, "")
        already = cache_key in st.session_state.get("llm_cache", {})
        if already or st.button("Generate AI summary", type="primary"):
            with st.spinner("Asking the AI to summarise…"):
                ai_summary, err = run_llm("summary", "", lambda: llm_summarize(doc.pages, llm))
            if err:
                st.warning(f"AI summary unavailable: {err} The rule-based summary below still works.", icon="⚠️")
            else:
                show_llm_summary(ai_summary)
        st.markdown("#### Rule-based summary")

    if not scanned:
        st.caption("Extractive summary: every line below is quoted from the document. No AI model was used.")
        st.markdown("##### Purpose")
        st.markdown(f"{md_escape(summary.purpose)}  \n*{page_ref(summary.purpose_page)}*")

        st.markdown("##### Main points")
        for point in summary.main_points:
            st.markdown(f"- {md_escape(point.text)} *({page_ref(point.page)})*")

        left, right = st.columns(2)
        with left:
            st.markdown("##### Important dates")
            if summary.important_dates:
                st.dataframe(
                    pd.DataFrame(
                        [{"Date": d.normalized, "As written": d.value, "Type": d.label, "Page": d.page, "Context": d.evidence}
                         for d in summary.important_dates]
                    ),
                    hide_index=True,
                )
            else:
                st.caption("No dates found in the document.")
            st.markdown("##### Fees and amounts")
            if summary.fees:
                for fee in summary.fees:
                    st.markdown(f"- **{md_escape(fee.value)}**: {md_escape(fee.evidence)} *({page_ref(fee.page)})*")
            else:
                st.caption("No fees or amounts found in the document.")
        with right:
            st.markdown("##### Required actions")
            if summary.required_actions:
                for action in summary.required_actions:
                    st.markdown(f"- {md_escape(action.value)} *({page_ref(action.page)})*")
            else:
                st.caption("No required actions found in the document.")
            st.markdown("##### Eligibility")
            if summary.eligibility:
                for item in summary.eligibility:
                    st.markdown(f"- {md_escape(item.value)} *({page_ref(item.page)})*")
            else:
                st.caption("No eligibility conditions found in the document.")

# --------------------------------------------------------------------------- fields


def _field_table(items: list[FieldValue], value_name: str, extra: str | None = None) -> pd.DataFrame:
    rows = []
    for item in items:
        row = {value_name: item.value}
        if extra == "normalized":
            row["Normalised"] = item.normalized
        if extra == "label":
            row["Type"] = item.label
        row.update({"Page": item.page, "Evidence": item.evidence, "Source": "Rule-based"})
        rows.append(row)
    return pd.DataFrame(rows)


def show_llm_fields(ai: LLMFields) -> None:
    verified = sum(i.verified for i in ai.items)
    st.caption(
        f"AI-generated · {ai.model} · from {pages_sent_label(ai.pages_sent, ai.truncated)}"
        + f" · {verified}/{len(ai.items)} values matched to a verbatim quote."
    )
    if not ai.items:
        st.caption("The AI reported no fields.")
        return
    st.dataframe(
        pd.DataFrame(
            [{"Field": i.field.replace("_", " "), "Value": i.text, "Page": i.page,
              "Verified in document": "✅ yes" if i.verified else "⚠️ no", "Quote": i.quote,
              "Source": "AI-generated"} for i in ai.items]
        ),
        hide_index=True,
    )


with tab_fields:
    if not scanned:
        st.caption(
            "Rule-based fields are found with transparent rules (regular expressions and keywords). "
            "Empty means the field was not found; nothing is guessed."
        )
        overview = [
            ("Document title", fields.title),
            ("Issuing organisation", fields.organization),
            ("Reference number", fields.reference_number),
        ]
        st.dataframe(
            pd.DataFrame(
                [{"Field": name, "Value": v.value if v else "Not found", "Page": v.page if v else None,
                  "Source": "Rule-based" if v else ""} for name, v in overview]
            ),
            hide_index=True,
        )
        sections = [
            ("Dates and deadlines", fields.dates, "Date", "label"),
            ("Fees and amounts", fields.fees, "Amount", "normalized"),
            ("Contact details", fields.contacts, "Contact", "label"),
            ("Eligibility requirements", fields.eligibility, "Requirement", None),
            ("Action items", fields.action_items, "Action", None),
        ]
        for title, items, value_name, extra in sections:
            st.markdown(f"##### {title} ({len(items)})")
            if items:
                st.dataframe(_field_table(items, value_name, extra), hide_index=True)
            else:
                st.caption("Not found in the document.")
        st.download_button(
            "Download rule-based fields as JSON",
            data=json.dumps(fields.to_dict(), indent=2, ensure_ascii=False),
            file_name=f"{Path(doc.filename).stem}_fields.json",
            mime="application/json",
        )

    if patram is not None:
        st.markdown("#### 🏛️ BharatGen Patram fields")
        n_calls = min(page_count, patram.config.max_pages)
        if patram_cached("fields") or st.button(f"Extract fields with Patram ({n_calls} GPU call{'s' if n_calls > 1 else ''})"):
            with st.spinner("Patram is reading the page image(s)…"):
                p_fields, err = run_patram("fields", "", lambda: patram_fields(patram, pdf_bytes, doc.pages, page_count))
            if err:
                st.warning(f"Patram field extraction unavailable: {err}", icon="⚠️")
            else:
                st.caption(
                    f"🏛️ AI-generated · {patram_client.MODEL_LABEL} · from page image(s) "
                    f"{', '.join(str(a.page) for a in p_fields.answers)} · numbers checked against the page text"
                )
                if p_fields.fields:
                    st.dataframe(
                        pd.DataFrame(
                            [{"Field": f.field.replace("_", " "), "Patram's label": f.label, "Value": f.value,
                              "Page": f.page, "Numbers": f.grounding.label, "Source": "BharatGen Patram"}
                             for f in p_fields.fields]
                        ),
                        hide_index=True,
                    )
                else:
                    st.caption("Patram's reply had no 'field: value' lines. Its raw reply:")
                    for a in p_fields.answers:
                        st.text(a.text)
    if scanned:
        st.caption("Rule-based and Groq fields need a text layer, so they are not available for scanned PDFs.")

    if llm is not None and not scanned:
        st.markdown("#### 🤖 AI-extracted fields (for comparison)")
        cache_key = (doc.extraction.sha256, "fields", llm_config.model, "")
        already = cache_key in st.session_state.get("llm_cache", {})
        if already or st.button("Extract fields with AI"):
            with st.spinner("Asking the AI to extract fields…"):
                ai_fields, err = run_llm("fields", "", lambda: llm_extract_fields(doc.pages, llm))
            if err:
                st.warning(f"AI field extraction unavailable: {err}", icon="⚠️")
            else:
                show_llm_fields(ai_fields)

# --------------------------------------------------------------------------- question answering


def show_evidence(results: list[SearchResult], heading: str, sent_to_ai: int = 0) -> None:
    if not results:
        return
    st.markdown(f"**{heading}**")
    for i, r in enumerate(results):
        label = f"{r.chunk.chunk_id} · {r.chunk.page_label} · Relevance: {r.relevance_label} · BM25 score {r.score:.2f}"
        if i < sent_to_ai:
            label += " · sent to AI"
        with st.expander(label, expanded=False):
            st.markdown(highlight(r.chunk.text, r.matched_terms).replace("\n", "  \n"))
            st.caption(f"Matched terms: {', '.join(r.matched_terms)} · question coverage {r.coverage:.0%}")


def show_extractive(result: QAResult) -> None:
    if result.found:
        sentence = result.sentences[0]
        with st.container(border=True):
            st.caption(
                f"Extractive answer · quoted verbatim from {page_ref(sentence.page)} "
                f"(passage {sentence.chunk_id}) · not AI-generated"
            )
            st.markdown(f"> {md_escape(result.answer)}")
    else:
        st.warning(result.answer, icon="🔍")


def show_llm(result: QAResult) -> None:
    if not result.found:
        st.warning(f"{result.answer} (The AI read the top passages and found no answer.)", icon="🔍")
        return
    with st.container(border=True):
        st.caption(f"🤖 AI-generated answer · {md_escape(result.model or '')} · based only on the passages below")
        st.markdown(md_escape(result.answer))
        if result.citations:
            st.markdown("**Supporting quotes**")
        for c in result.citations:
            where = f"{page_ref(c.page)}, passage {c.passage_id}" if c.verified else "not found in the document text"
            st.markdown(f"{verified_badge(c.verified)} “{md_escape(c.quote)}” *({where})*")
    if not result.grounded:
        st.warning(
            "None of the AI's supporting quotes could be matched to the document. Treat this answer with "
            "caution and check the extractive answer and source passages.",
            icon="⚠️",
        )


def show_patram(result: PatramQA, index: int) -> None:
    a = result.answer
    with st.container(border=True):
        timing = f"{a.generation_seconds:.1f} s GPU" if a.generation_seconds is not None else ""
        st.caption(
            f"🏛️ AI-generated answer · {patram_client.MODEL_LABEL} · read from the image of {page_ref(a.page)}"
            + (f" · {timing}" if timing else "")
        )
        st.markdown(md_escape(a.text))
        st.caption(f"Numbers: {result.grounding.label}")
    if result.grounding.unverified:
        st.warning(
            f"These numbers in Patram's answer are not in the page text: {', '.join(result.grounding.unverified)}. "
            "Check them against the page image.",
            icon="⚠️",
        )
    with st.expander(f"Page {a.page} as Patram saw it"):
        st.image(render_page_png(pdf_bytes, a.page, patram.config.dpi), caption=f"Page {a.page}")


def show_translation(entry: dict, answer_text: str, index: int) -> None:
    """Translate button and result for one answer; results are kept per language in the entry."""
    if translator is None or target_language is None:
        return
    translations: dict = entry.setdefault("translations", {})
    name = indic_translator.LANGUAGES[target_language]
    if target_language not in translations:
        if not st.button(f"Translate answer to {name}", key=f"translate_{index}_{target_language}"):
            return
        with st.spinner(f"Translating to {name} with IndicTrans2 (first use loads the model)…"):
            try:
                translations[target_language] = translator.translate(answer_text, target_language)
            except indic_translator.TranslationUnavailable as exc:
                st.warning(f"Translation unavailable: {exc} The English answer above is unaffected.", icon="⚠️")
                return
            except Exception:
                logger.exception("Unexpected translation error")
                st.warning("Translation failed unexpectedly. The English answer above is unaffected.", icon="⚠️")
                return
    result = translations[target_language]
    with st.container(border=True):
        st.caption(
            f"🌐 Machine translation ({name}) · {result.model_label} · {result.seconds:.1f} s · "
            "check against the English answer"
        )
        st.markdown(md_escape(result.text))
        if result.missing_numbers:
            st.warning(
                "These numbers from the English answer do not appear in the translation: "
                f"{', '.join(result.missing_numbers)}. Rely on the English answer for dates and amounts.",
                icon="⚠️",
            )


def show_answer(entry: dict, index: int) -> None:
    extractive: QAResult = entry["extractive"]
    ai: QAResult | None = entry.get("llm")
    st.markdown(f"**Q: {md_escape(extractive.question)}**")
    p_result: PatramQA | None = entry.get("patram")
    if entry.get("patram_error"):
        st.info(f"Patram answer unavailable: {entry['patram_error']} Showing the fallback answer instead.", icon="ℹ️")
    if p_result is not None:
        show_patram(p_result, index)
        show_translation(entry, p_result.answer.text, index)
        if not scanned:
            with st.expander("Extractive answer (keyword match, for comparison)"):
                show_extractive(extractive)
            show_evidence(extractive.evidence, "Source passages (keyword search)")
        return
    if scanned:
        return  # no text layer: there is no extractive fallback to show
    if entry.get("llm_error"):
        st.info(f"AI answer unavailable: {entry['llm_error']} Showing the extractive answer instead.", icon="ℹ️")
    if ai is not None:
        show_llm(ai)
        if ai.found:
            show_translation(entry, ai.answer, index)
        with st.expander("Extractive answer (keyword match, for comparison)"):
            show_extractive(extractive)
        show_evidence(extractive.evidence, "Source passages", sent_to_ai=llm_config.max_passages)
    else:
        show_extractive(extractive)
        if extractive.found:
            show_translation(entry, extractive.answer, index)
        heading = "Source passages" if extractive.found else "Closest passages (did not contain enough of the question to answer it)"
        show_evidence(extractive.evidence, heading)


def answer(question_text: str, chosen_page: int | None) -> dict:
    extractive = answer_question(question_text, doc.index, settings.top_k, pages=doc.pages)
    entry = {"extractive": extractive, "llm": None, "llm_error": None, "patram": None, "patram_error": None}
    if use_patram and patram is not None:
        page = chosen_page or choose_page(extractive, page_count)
        with st.spinner(f"Patram is reading page {page}…"):
            entry["patram"], entry["patram_error"] = run_patram(
                "qa",
                f"{page}|{question_text.strip().lower()}",
                lambda: patram_answer(patram, pdf_bytes, doc.pages, page, question_text),
            )
        if entry["patram"] is not None:
            return entry
    if use_llm and llm is not None:
        with st.spinner("Asking the AI…"):
            entry["llm"], entry["llm_error"] = run_llm(
                "qa",
                question_text.strip().lower(),
                lambda: llm_answer(question_text, extractive.evidence, llm, pages=doc.pages),
            )
    return entry


with tab_qa:
    chosen_page = None
    if use_patram:
        st.caption(
            "BharatGen Patram answers from the page image. By default it reads the page that keyword search ranks "
            "highest (page 1 for scanned PDFs); you can pick another page. Each question uses one GPU call."
        )
        page_choice = st.selectbox("Page for Patram", [AUTO_PAGE, *range(1, page_count + 1)])
        chosen_page = None if page_choice == AUTO_PAGE else int(page_choice)
    elif use_llm:
        st.caption(
            "BM25 finds the most relevant passages; the AI answers only from those passages and must quote them. "
            "Each quote is checked against the document (✅ verified, ⚠️ not found)."
        )
    else:
        st.caption(
            "Answers are the single best-matching sentence from the document, found with BM25 keyword search. "
            "If no sentence matches enough of your question, DocuBharat says so instead of guessing."
        )
    st.markdown("Try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    example_clicked = None
    # A new key per loaded document: the browser keeps a widget's old text otherwise.
    question_key = f"question_{st.session_state.uploader_nonce}"
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, width="stretch"):
            example_clicked = example
            st.session_state[question_key] = example  # set before the text box is drawn
    with st.form("qa_form"):
        question = st.text_input("Your question", key=question_key, placeholder="e.g. What is the last date for fee payment?")
        asked = st.form_submit_button("Ask", type="primary")

    history: list[dict] = st.session_state.qa_history
    to_answer = (example_clicked or (question if asked else "")).strip()
    if to_answer:
        history.insert(0, answer(to_answer, chosen_page))
        del history[5:]  # keep the last five answers
    for i, entry in enumerate(history):
        if i == 1:
            st.markdown("---\n**Earlier questions**")
        show_answer(entry, i)

# --------------------------------------------------------------------------- extracted text

with tab_text:
    st.caption("Cleaned text as extracted by PyMuPDF, page by page.")
    for page in doc.pages:
        with st.expander(f"Page {page.page_number} · {len(page.text):,} characters", expanded=page.page_number == 1):
            st.text(page.text or "(no text on this page)")
    with st.expander(f"Retrieval passages ({len(doc.chunks)})"):
        st.dataframe(
            pd.DataFrame(
                [{"Passage": c.chunk_id, "Pages": c.page_label, "Words": c.word_count, "Start of text": c.text[:120]}
                 for c in doc.chunks]
            ),
            hide_index=True,
        )
