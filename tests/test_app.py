"""Smoke tests that run the real Streamlit script headlessly with AppTest (no browser, no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import streamlit as st

from src import indic_translator, llm_provider, patram_client
from src.indic_translator import Translation, TranslationUnavailable
from src.llm_provider import LLMRateLimited
from src.patram_client import PatramAnswer, PatramConfig, PatramUnavailable
from src.translation_checks import missing_numbers
from tests.fakes import FailingProvider, ScriptedProvider

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample_documents"
needs_samples = pytest.mark.skipif(
    not (SAMPLES / "exam_circular.pdf").exists(), reason="sample documents not generated"
)


@pytest.fixture(autouse=True)
def no_real_services(monkeypatch):
    """Tests never use real keys, the real Patram Space (GPU quota!) or the real translation model."""
    for name in ("LLM_API_KEY", "GROQ_API_KEY", "PATRAM_SPACE_ID", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("PATRAM_ENABLED", "false")
    monkeypatch.setenv("TRANSLATION_ENABLED", "false")
    st.cache_resource.clear()  # the translator is cached per server; reset between tests
    yield
    st.cache_resource.clear()


class FakeTranslator:
    def __init__(self, text="परीक्षा शुल्क 1,800 रुपये है।", error=None):
        self.text, self.error, self.calls = text, error, []

    def translate(self, text, target_language):
        self.calls.append((text, target_language))
        if self.error:
            raise self.error
        return Translation(text, self.text, target_language, 0.4, missing_numbers=missing_numbers(text, self.text))


def _with_translator(monkeypatch, fake):
    monkeypatch.setattr(indic_translator, "create_translator", lambda config=None: (fake, "ready"))


def _load_sample(name: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.selectbox[0].set_value(name).run()
    at.sidebar.button[0].click().run()  # "Load sample"
    assert not at.exception
    return at


def _ask(at: AppTest, question: str) -> AppTest:
    next(t for t in at.text_input if t.label == "Your question").set_value(question)
    at.button(key="FormSubmitter:qa_form-Ask").click().run()
    assert not at.exception
    return at


def _markdown(at: AppTest) -> str:
    return "\n".join(md.value for md in at.markdown)


def test_start_screen_renders_without_api_key():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "📄 DocuBharat"
    assert at.sidebar.radio[0].options == ["Extractive only (no AI)"]  # nothing configured
    assert any("Patram not available" in c.value for c in at.sidebar.caption)


@needs_samples
def test_sample_document_summary_and_question():
    at = _load_sample("exam_circular.pdf")
    assert [m.value for m in at.metric][:2] == ["2", "2"]
    assert len(at.tabs) == 4
    _ask(at, "What is the examination fee?")
    assert "1,800" in _markdown(at)
    assert "AI-generated" not in _markdown(at)


@pytest.mark.skipif(not (SAMPLES / "scanned_library_notice.pdf").exists(), reason="sample documents not generated")
def test_scanned_document_explains_ocr():
    at = _load_sample("scanned_library_notice.pdf")
    assert any("OCR" in w.value for w in at.warning)
    assert len(at.tabs) == 0


@needs_samples
def test_llm_answer_is_labelled_and_quotes_are_verified(monkeypatch):
    provider = ScriptedProvider(
        {
            "found": True,
            "answer": "The examination fee is Rs. 1,800 per semester.",
            "citations": [{"passage_id": "C1", "quote": "The examination fee for regular students is Rs. 1,800 per semester."}],
        }
    )
    monkeypatch.setattr(llm_provider, "create_provider", lambda config=None: provider)
    at = _ask(_load_sample("exam_circular.pdf"), "What is the examination fee?")
    text = "\n".join(c.value for c in at.caption) + _markdown(at)
    assert "AI-generated answer" in text
    assert "✅" in text and "p. 1, passage C1" in text
    assert len(provider.messages) == 1

    # The same question again is served from the cache, not a second API call.
    _ask(at, "What is the examination fee?")
    assert len(provider.messages) == 1


@needs_samples
def test_llm_failure_falls_back_to_extractive_answer(monkeypatch):
    monkeypatch.setattr(llm_provider, "create_provider", lambda config=None: FailingProvider(LLMRateLimited(30)))
    at = _ask(_load_sample("exam_circular.pdf"), "What is the examination fee?")
    assert any("AI answer unavailable" in i.value and "rate limit" in i.value for i in at.info)
    assert "1,800" in _markdown(at)
    assert not at.exception


def _translate_button(at: AppTest):
    return next(b for b in at.button if b.label.startswith("Translate answer to"))


def _language_box(at: AppTest):
    return next(s for s in at.sidebar.selectbox if s.label == "Translate answers to")


def test_translation_unavailable_is_explained():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Translation is turned off" in c.value for c in at.sidebar.caption)


@needs_samples
def test_translation_is_on_demand_and_labelled(monkeypatch):
    fake = FakeTranslator()
    _with_translator(monkeypatch, fake)
    at = _load_sample("exam_circular.pdf")
    assert not any(b.label.startswith("Translate answer") for b in at.button)  # off until a language is chosen
    _language_box(at).set_value("hin_Deva").run()
    _ask(at, "What is the examination fee?")
    assert fake.calls == []  # nothing is translated until the user clicks
    _translate_button(at).click().run()
    assert not at.exception
    assert len(fake.calls) == 1 and fake.calls[0][1] == "hin_Deva"
    text = _markdown(at) + "\n".join(c.value for c in at.caption)
    assert "परीक्षा शुल्क" in text and "Machine translation (Hindi)" in text and "IndicTrans2" in text
    assert "1,800" in _markdown(at)  # the English answer stays visible


@needs_samples
def test_translation_warns_when_numbers_are_lost(monkeypatch):
    _with_translator(monkeypatch, FakeTranslator(text="परीक्षा शुल्क है।"))
    at = _load_sample("exam_circular.pdf")
    _language_box(at).set_value("hin_Deva").run()
    _ask(at, "What is the examination fee?")
    _translate_button(at).click().run()
    assert any("do not appear in the translation" in w.value and "1,800" in w.value for w in at.warning)


@needs_samples
def test_translation_failure_keeps_english_answer(monkeypatch):
    _with_translator(monkeypatch, FakeTranslator(error=TranslationUnavailable("The translation model could not run.")))
    at = _load_sample("exam_circular.pdf")
    _language_box(at).set_value("tam_Taml").run()
    _ask(at, "What is the examination fee?")
    _translate_button(at).click().run()
    assert not at.exception
    assert any("Translation unavailable" in w.value for w in at.warning)
    assert "1,800" in _markdown(at)


@needs_samples
def test_question_box_is_empty_after_loading_another_document():
    at = _ask(_load_sample("exam_circular.pdf"), "What is the examination fee?")
    at.sidebar.selectbox[0].set_value("fee_notification.pdf").run()
    at.sidebar.button[0].click().run()  # load another sample
    assert next(t for t in at.text_input if t.label == "Your question").value == ""


@needs_samples
def test_extractive_engine_sends_nothing(monkeypatch):
    provider = ScriptedProvider({"found": False, "answer": "", "citations": []})
    monkeypatch.setattr(llm_provider, "create_provider", lambda config=None: provider)
    fake = FakePatram()
    _with_patram(monkeypatch, fake)
    at = _load_sample("exam_circular.pdf")
    at.sidebar.radio[0].set_value("extractive").run()
    _ask(at, "What is the examination fee?")
    assert provider.messages == [] and fake.calls == []
    assert "1,800" in _markdown(at)


# --------------------------------------------------------------------------- BharatGen Patram (fake Space)


class FakePatram:
    """Stands in for PatramClient: no network, no GPU quota."""

    def __init__(self, text="The examination fee for regular students is Rs. 1,800 per semester.", error=None):
        self.text, self.error, self.calls = text, error, []
        self.config = PatramConfig(space_id="user/space", token="hf_fake", max_pages=2)

    def ask_page(self, pdf_bytes, page, question, max_new_tokens=None):
        self.calls.append((page, question))
        if self.error:
            raise self.error
        return PatramAnswer(question, self.text, page, 1.5, 6.0)


def _with_patram(monkeypatch, fake):
    monkeypatch.setattr(patram_client, "create_client", lambda config=None: (fake, "ready"))


@needs_samples
def test_patram_is_default_engine_and_answers_from_best_page(monkeypatch):
    fake = FakePatram()
    _with_patram(monkeypatch, fake)
    at = _load_sample("exam_circular.pdf")
    assert at.sidebar.radio[0].value == "patram"
    _ask(at, "What is the late fine?")  # best keyword match is on page 1
    assert fake.calls == [(1, "What is the late fine?")]
    captions = "\n".join(c.value for c in at.caption)
    assert "BharatGen Patram" in captions and "read from the image of p. 1" in captions
    assert "all 1 number(s) found in the page text" in captions

    _ask(at, "What is the late fine?")  # cached: no second GPU call
    assert len(fake.calls) == 1


@needs_samples
def test_patram_unverified_number_is_flagged(monkeypatch):
    _with_patram(monkeypatch, FakePatram(text="The examination fee is Rs. 2,500 per semester."))
    at = _ask(_load_sample("exam_circular.pdf"), "What is the examination fee?")
    assert any("not in the page text" in w.value and "2,500" in w.value for w in at.warning)


@needs_samples
def test_patram_failure_falls_back_to_extractive(monkeypatch):
    _with_patram(monkeypatch, FakePatram(error=PatramUnavailable("The free daily GPU quota for Patram is used up.")))
    at = _ask(_load_sample("exam_circular.pdf"), "What is the examination fee?")
    assert any("Patram answer unavailable" in i.value and "quota" in i.value for i in at.info)
    assert "1,800" in _markdown(at)


@needs_samples
def test_patram_reads_scanned_pdf(monkeypatch):
    fake = FakePatram(text="The library will remain open late until 15 December 2026.")
    _with_patram(monkeypatch, fake)
    at = _load_sample("scanned_library_notice.pdf")
    assert any("Scanned PDF" in i.value for i in at.info)
    assert len(at.tabs) == 4
    _ask(at, "Until when is the library open late?")
    assert fake.calls == [(1, "Until when is the library open late?")]
    assert any("not verifiable" in c.value for c in at.caption)


@needs_samples
def test_patram_summary_and_fields_on_click(monkeypatch):
    fake = FakePatram(text="Title: End Semester Examinations\nFee: Rs. 1,800")
    _with_patram(monkeypatch, fake)
    at = _load_sample("exam_circular.pdf")
    assert fake.calls == []  # nothing sent until a button is clicked
    next(b for b in at.button if b.label.startswith("Summarise with Patram")).click().run()
    next(b for b in at.button if b.label.startswith("Extract fields with Patram")).click().run()
    assert not at.exception
    assert len(fake.calls) == 4  # 2 pages x (summary + fields)
    assert any("BharatGen Patram" in str(df.value.get("Source", "")) for df in at.dataframe)
