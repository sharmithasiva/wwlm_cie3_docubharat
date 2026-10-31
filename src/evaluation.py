"""Evaluation of DocuBharat on the 40-case set in data/test_cases/test_cases.json.

Systems compared:
- "extractive": BM25 + best-sentence answers, rule-based fields, extractive summary (no AI)
- "groq":       Groq openai/gpt-oss-20b on retrieved passages (quotes verified)
- "patram":     BharatGen Patram on page images (private ZeroGPU Space)

Every model output is cached in data/test_cases/outputs/<system>_<task>.json as soon as
it arrives, so re-running never repeats a paid-for or quota-limited call. Scoring is
fully deterministic and can be re-run on the cached outputs alone.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.chunker import chunk_pages
from src.config import Settings
from src.field_extractor import (
    DATE_PATTERN,
    EMAIL_PATTERN,
    MONEY_PATTERN,
    PHONE_PATTERN,
    ExtractedFields,
    extract_fields,
    parse_date,
)
from src.pipeline import ProcessedDocument, process_pdf
from src.qa_engine import answer_question
from src.retrieval import BM25Index
from src.summarizer import summarize
from src.translation_checks import missing_numbers

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "test_cases" / "test_cases.json"
SAMPLES_DIR = ROOT / "data" / "sample_documents"
OUTPUT_DIR = ROOT / "data" / "test_cases" / "outputs"

# Phrases with which a free-text model says the document does not contain the answer.
_NOT_FOUND = re.compile(
    # An optional adverb is allowed: Patram answers "Not explicitly stated in document."
    r"not (?:\w+ )?(?:mention|mentioned|specified|stated|provided|available|given|found|included|present|listed|indicated)|"
    r"does not (?:mention|specify|state|provide|contain|include|say|list|indicate)|doesn't (?:mention|specify|state)|"
    r"no (?:information|mention|details?)|cannot be determined|not possible to determine|is not shown",
    re.IGNORECASE,
)


class StopSystem(Exception):
    """Abort the remaining calls for one system (e.g. Patram's daily GPU quota is used up)."""


# --------------------------------------------------------------------------- matching helpers


def norm(text: str) -> str:
    """Lower-case, unify dashes, drop commas and collapse spaces: '1,12,000' == '112000'."""
    text = text.lower().replace(",", "")
    text = re.sub(r"[‐-―−]", "-", text)
    return " ".join(text.split())


def alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def dates_in(text: str) -> set[str]:
    return {iso for m in DATE_PATTERN.finditer(text) if (iso := parse_date(m))}


def as_date(text: str) -> str | None:
    found = dates_in(text)
    return next(iter(found)) if len(found) == 1 else None


_PURE_NUMBER = re.compile(r"^\d[\d,]*(?:\.\d+)?%?$")
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def item_matches(expected: str, answer: str) -> bool:
    """Dates compare as calendar dates; plain numbers as whole numbers ('180' does not match
    '1,800'; '1,12,000' matches '112000'); anything else as a normalised substring."""
    iso = as_date(expected)
    if iso:
        return iso in dates_in(answer)
    if _PURE_NUMBER.match(expected.strip()):
        target = expected.strip().rstrip("%").replace(",", "")
        return target in {n.replace(",", "") for n in _ANY_NUMBER.findall(answer)}
    return norm(expected) in norm(answer)


def must_matched(must: list[list[str]], answer: str) -> bool:
    return all(any(item_matches(alt, answer) for alt in group) for group in must)


def says_not_found(text: str) -> bool:
    return bool(_NOT_FOUND.search(text))


def numbers_in(text: str) -> list[str]:
    return missing_numbers(text, "")


# --------------------------------------------------------------------------- data


def load_cases(path: Path = CASES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class LoadedDoc:
    name: str
    pdf: bytes
    doc: ProcessedDocument
    fields: ExtractedFields


def load_documents(names: set[str], settings: Settings | None = None) -> dict[str, LoadedDoc]:
    docs = {}
    for name in sorted(names):
        pdf = (SAMPLES_DIR / name).read_bytes()
        processed = process_pdf(pdf, name, settings or Settings())
        docs[name] = LoadedDoc(name, pdf, processed, extract_fields(processed.pages))
    return docs


# --------------------------------------------------------------------------- output cache


class OutputCache:
    """Per-system, per-task JSON file; written after every new record."""

    def __init__(self, system: str, task: str, directory: Path = OUTPUT_DIR):
        self.path = directory / f"{system}_{task}.json"
        self.records: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get_or_run(self, key: str, func: Callable[[], dict], offline: bool = False) -> dict | None:
        """Cached record, or run `func`. Records with an "error" (timeouts, rate limits,
        a sleeping Space) are returned but NOT saved, so the next run retries them."""
        if key in self.records:
            return self.records[key]
        if offline:
            return None
        record = func()
        if record.get("error"):
            return record
        self.records[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")
        return record


# --------------------------------------------------------------------------- QA runners


def qa_extractive(case: dict, d: LoadedDoc) -> dict:
    started = time.perf_counter()
    r = answer_question(case["question"], d.doc.index, pages=d.doc.pages)
    return {
        "found": r.found,
        "answer": r.answer,
        "page": r.sentences[0].page if r.sentences else None,
        "grounded": True if r.found else None,  # a verbatim quote from the document
        "seconds": round(time.perf_counter() - started, 3),
        "error": None,
    }


def with_rate_limit_retry(func: Callable[[], Any], attempts: int = 8, sleep=time.sleep) -> Any:
    """Call a Groq task, waiting out free-tier rate limits instead of failing."""
    from src.llm_provider import LLMRateLimited

    for attempt in range(attempts):
        try:
            return func()
        except LLMRateLimited as exc:
            if attempt == attempts - 1:
                raise
            sleep(min(max(exc.retry_after or 20, 5), 90))


def qa_groq(case: dict, d: LoadedDoc, provider, sleep=time.sleep) -> dict:
    from src.llm_provider import LLMError, LLMRateLimited
    from src.llm_tasks import llm_answer

    evidence = d.doc.index.search(case["question"], top_k=4)
    for attempt in range(8):  # the free tier's tokens-per-minute limit needs patience, not failure
        started = time.perf_counter()
        try:
            r = llm_answer(case["question"], evidence, provider, pages=d.doc.pages)
            return {
                "found": r.found,
                "answer": r.answer,
                "page": r.cited_pages[0] if r.cited_pages else None,
                "grounded": r.grounded if r.found else None,
                "citations": [{"passage": c.passage_id, "quote": c.quote, "verified": c.verified} for c in r.citations],
                "seconds": round(time.perf_counter() - started, 2),
                "error": None,
            }
        except LLMRateLimited as exc:
            sleep(min(max(exc.retry_after or 20, 5), 90))
        except LLMError as exc:
            return {"found": False, "answer": "", "page": None, "grounded": None, "seconds": 0, "error": exc.user_message}
    return {"found": False, "answer": "", "page": None, "grounded": None, "seconds": 0, "error": "rate limited repeatedly"}


def qa_patram(case: dict, d: LoadedDoc, client) -> dict:
    from src.patram_client import PatramUnavailable
    from src.patram_tasks import choose_page, patram_answer

    page = choose_page(answer_question(case["question"], d.doc.index, pages=d.doc.pages), d.doc.extraction.pages_processed)
    try:
        r = patram_answer(client, d.pdf, d.doc.pages, page, case["question"])
    except PatramUnavailable as exc:
        if "quota" in str(exc).lower():
            raise StopSystem(str(exc)) from exc
        return {"found": False, "answer": "", "page": page, "grounded": None, "seconds": 0, "error": str(exc)}
    text = r.answer.text
    grounding = r.grounding
    return {
        "found": not says_not_found(text),
        "answer": text,
        "page": page,
        "grounded": (not grounding.unverified) if grounding.checkable else None,
        "numbers_unverified": grounding.unverified,
        "seconds": r.answer.generation_seconds,
        "wall_seconds": round(r.answer.wall_seconds, 2),
        "error": None,
    }


def score_qa(case: dict, record: dict) -> tuple[bool, str]:
    """(correct, failure category)."""
    if record.get("error"):
        return False, "service_error"
    if case["answerable"]:
        if not record["found"]:
            return False, "missed_answer"
        return (True, "correct") if must_matched(case["must"], record["answer"]) else (False, "wrong_or_incomplete")
    return (True, "correct_abstention") if not record["found"] else (False, "hallucinated_answer")


# --------------------------------------------------------------------------- retrieval


def retrieval_hit(case: dict, index: BM25Index, k: int) -> bool | None:
    if not case.get("support_text") or not case["answerable"]:
        return None
    target = norm(case["support_text"])
    return any(target in norm(r.chunk.text) for r in index.search(case["question"], top_k=k))


def small_chunk_index(d: LoadedDoc, words: int = 80, overlap: int = 10) -> BM25Index:
    return BM25Index(chunk_pages(d.doc.pages, words, overlap))


# --------------------------------------------------------------------------- fields


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _amounts(value: str) -> set[int]:
    out = set()
    for n in _NUMBER.findall(value):
        clean = n.replace(",", "")
        try:
            number = float(clean)
        except ValueError:
            continue
        if number >= 10 and not (len(clean) == 4 and clean.startswith("20")):  # skip years
            out.add(int(number))
    return out


def _phones(value: str) -> set[str]:
    return {re.sub(r"\D", "", m.group(0)) for m in PHONE_PATTERN.finditer(DATE_PATTERN.sub(" ", value))
            if 10 <= len(re.sub(r"\D", "", m.group(0))) <= 13}


def empty_field_set() -> dict:
    return {"title": set(), "organization": set(), "reference_number": set(), "dates": set(),
            "amounts": set(), "emails": set(), "phones": set()}


def collect_rule_fields(f: ExtractedFields) -> dict:
    got = empty_field_set()
    for key, value in (("title", f.title), ("organization", f.organization), ("reference_number", f.reference_number)):
        if value:
            got[key].add(value.value)
    got["dates"] = {d.normalized for d in f.dates if d.normalized}
    got["amounts"] = {int(float(m.normalized)) for m in f.fees if m.normalized}
    got["emails"] = {c.value.lower() for c in f.contacts if c.label == "email"}
    got["phones"] = {re.sub(r"\D", "", c.value) for c in f.contacts if c.label == "phone"}
    return got


_FIELD_SLOT = {"title": "title", "issuing_organization": "organization", "reference_number": "reference_number"}


def collect_model_fields(items: list[tuple[str, str]]) -> dict:
    """items: (DocuBharat field name, value) pairs from Groq or Patram."""
    got = empty_field_set()
    for field, value in items:
        if field in _FIELD_SLOT:
            got[_FIELD_SLOT[field]].add(value)
        # Dates, e-mails and phone numbers are collected by type, whatever label the model used.
        if field not in _FIELD_SLOT:
            got["dates"] |= dates_in(value)
            got["emails"] |= {e.lower() for e in EMAIL_PATTERN.findall(value)}
            got["phones"] |= _phones(value)
        # Amounts: values labelled as fees, or any value containing a rupee amount.
        if field == "fee_or_amount":
            got["amounts"] |= _amounts(DATE_PATTERN.sub(" ", value))
        elif field not in _FIELD_SLOT:
            got["amounts"] |= {int(float((m.group("a1") or m.group("a2")).replace(",", "")))
                               for m in MONEY_PATTERN.finditer(value)}
    return got


def score_fields(expected: dict, got: dict, doc_text: str) -> tuple[list[dict], dict]:
    """Slot-level exact matches plus counts of produced values that are not in the document."""
    slots = []
    for key in ("title", "organization", "reference_number"):
        target = alnum(expected[key])
        slots.append({"slot": key, "expected": expected[key],
                      "matched": any(alnum(v) == target for v in got[key]),
                      # relaxed: the value contains the expected text, e.g. "Circular No. KSU/COE/2026/41"
                      "relaxed": any(target and target in alnum(v) for v in got[key])})
    for key, values in (("dates", expected["dates"]), ("amounts", expected["amounts"]),
                        ("emails", expected["emails"]), ("phones", expected["phones"])):
        for value in values:
            hit = value in got[key]
            slots.append({"slot": key[:-1], "expected": value, "matched": hit, "relaxed": hit})
    doc_dates, doc_amounts = dates_in(doc_text), _amounts(DATE_PATTERN.sub(" ", doc_text))
    unsupported = {
        "dates": sorted(d for d in got["dates"] if d not in doc_dates),
        "amounts": sorted(a for a in got["amounts"] if a not in doc_amounts),
    }
    return slots, unsupported


# --------------------------------------------------------------------------- summaries


def rule_summary_text(d: LoadedDoc) -> str:
    s = summarize(d.doc.pages, d.fields)
    parts = [s.purpose, *(p.text for p in s.main_points), *(x.evidence for x in s.important_dates),
             *(x.value for x in s.required_actions), *(x.evidence for x in s.fees), *(x.value for x in s.eligibility)]
    return "\n".join(dict.fromkeys(parts))


def score_summary(case: dict, text: str, doc_text: str) -> dict:
    facts = [{"fact": f["fact"], "covered": any(item_matches(a, text) for a in f["any"])} for f in case["facts"]]
    # Numbers in the summary that never occur in the document (single digits are list markers etc.).
    unsupported = [n for n in missing_numbers(text, doc_text) if len(n.replace(",", "")) > 1]
    return {"facts": facts, "coverage": sum(f["covered"] for f in facts) / len(facts), "unsupported_numbers": unsupported}
