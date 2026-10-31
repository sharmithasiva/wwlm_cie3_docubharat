"""Evaluation set integrity and scoring functions (offline, no model calls)."""

import importlib.util
from pathlib import Path

import pytest

from src import evaluation as ev
from src.field_extractor import ExtractedFields, FieldValue

ROOT = Path(__file__).resolve().parents[1]
CASES = ev.load_cases()


def _sample_source():
    spec = importlib.util.spec_from_file_location("make_samples", ROOT / "scripts" / "make_sample_documents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- dataset integrity


def test_case_counts_and_ids():
    total = len(CASES["qa"]) + len(CASES["fields"]) + len(CASES["summaries"])
    assert total == 40
    ids = [c["id"] for group in ("qa", "fields", "summaries") for c in CASES[group]]
    assert len(ids) == len(set(ids))
    categories = {c["category"] for c in CASES["qa"]}
    assert categories == {"factual", "date", "fee", "absent", "ambiguous", "noisy", "scanned"}


@pytest.mark.skipif(not (ev.SAMPLES_DIR / "exam_circular.pdf").exists(), reason="samples not generated")
def test_support_text_occurs_in_document_on_stated_page():
    names = {c["doc"] for c in CASES["qa"]}
    docs = ev.load_documents(names)
    scanned_source = ev.norm(_sample_source().SCANNED_TEXT)
    for case in CASES["qa"]:
        if not case["support_text"]:
            continue
        d = docs[case["doc"]]
        target = ev.norm(case["support_text"])
        if d.doc.extraction.needs_ocr:
            assert target in scanned_source, case["id"]
            continue
        page = next(p for p in d.doc.pages if p.page_number == case["support_page"])
        assert target in ev.norm(page.text), case["id"]
        assert ev.must_matched(case["must"], page.text) or case["category"] == "ambiguous", case["id"]


@pytest.mark.skipif(not (ev.SAMPLES_DIR / "exam_circular.pdf").exists(), reason="samples not generated")
def test_expected_fields_occur_in_documents():
    docs = ev.load_documents({c["doc"] for c in CASES["fields"]})
    for case in CASES["fields"]:
        text = docs[case["doc"]].doc.full_text
        exp = case["expected"]
        assert ev.alnum(exp["reference_number"]) in ev.alnum(text), case["id"]
        assert set(exp["dates"]) <= ev.dates_in(text), case["id"]
        assert set(exp["amounts"]) <= ev._amounts(text), case["id"]
        assert all(e in text for e in exp["emails"]), case["id"]


# --------------------------------------------------------------------------- matching


@pytest.mark.parametrize(
    "expected, answer, ok",
    [
        ("7 November 2026", "The deadline is November 7, 2026.", True),
        ("7 November 2026", "Pay by 07/11/2026", True),
        ("7 November 2026", "Pay by 14 November 2026", False),
        ("1,12,000", "Rs. 112000 per semester", True),
        ("1,800", "Rs.1,800", True),
        ("10:00 PM", "open till 10:00 pm daily", True),
        ("180", "1,800 seats", False),  # whole-number comparison, not substring
        ("180", "The total intake is 180 seats.", True),
        ("75", "at least 75% marks", True),
        ("600", "Rs. 1,600", False),
    ],
)
def test_item_matches(expected, answer, ok):
    assert ev.item_matches(expected, answer) is ok


def test_must_requires_every_group():
    must = [["20 July 2026"], ["25 July 2026"]]
    assert ev.must_matched(must, "between 20 July 2026 and 25 July 2026")
    assert not ev.must_matched(must, "by 25 July 2026")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The document does not mention the Vice Chancellor.", True),
        ("This information is not specified in the document.", True),
        ("There is no information about bus fees.", True),
        ("Not explicitly stated in document.", True),  # Patram's actual wording
        ("The bus transport fee is Rs. 32,000 per semester, as mentioned in the document.", False),
        ("The fee is Rs. 1,800.", False),
    ],
)
def test_says_not_found(text, expected):
    assert ev.says_not_found(text) is expected


# --------------------------------------------------------------------------- scoring


def _case(answerable=True, must=(("1,800",),)):
    return {"answerable": answerable, "must": [list(g) for g in must]}


@pytest.mark.parametrize(
    "case, record, result",
    [
        (_case(), {"found": True, "answer": "Rs. 1,800", "error": None}, (True, "correct")),
        (_case(), {"found": True, "answer": "Rs. 1,500", "error": None}, (False, "wrong_or_incomplete")),
        (_case(), {"found": False, "answer": "", "error": None}, (False, "missed_answer")),
        (_case(False, ()), {"found": False, "answer": "", "error": None}, (True, "correct_abstention")),
        (_case(False, ()), {"found": True, "answer": "Rs. 500", "error": None}, (False, "hallucinated_answer")),
        (_case(), {"found": False, "answer": "", "error": "timeout"}, (False, "service_error")),
    ],
)
def test_score_qa(case, record, result):
    assert ev.score_qa(case, record) == result


def test_field_scoring_rule_and_model_outputs():
    expected = {"title": "End Semester Exams", "organization": "Kaveri State University", "reference_number": "KSU/1",
                "dates": ["2026-11-07"], "amounts": [1800], "emails": ["a@b.in"], "phones": ["08212419300"]}
    fields = ExtractedFields(
        title=FieldValue("End Semester Exams", 1, ""), organization=FieldValue("KAVERI STATE UNIVERSITY", 1, ""),
        reference_number=FieldValue("KSU/1", 1, ""),
        dates=[FieldValue("7 November 2026", 1, "", "deadline", "2026-11-07")],
        fees=[FieldValue("Rs. 1,800", 1, "", "amount", "1800")],
        contacts=[FieldValue("a@b.in", 1, "", "email"), FieldValue("0821-2419-300", 1, "", "phone")],
    )
    slots, unsupported = ev.score_fields(expected, ev.collect_rule_fields(fields), "7 November 2026 Rs. 1,800")
    assert all(s["matched"] for s in slots) and unsupported == {"dates": [], "amounts": []}

    model = ev.collect_model_fields([
        ("title", "Circular No. KSU/1"), ("issuing_organization", "Kaveri State University"),
        ("date_or_deadline", "Exams: 24 November 2026 to 7 November 2026"), ("fee_or_amount", "Rs. 1,800 per semester"),
        ("fee_or_amount", "Rs. 9,999"), ("contact", "Email a@b.in or call 0821-2419-300"),
    ])
    slots, unsupported = ev.score_fields(expected, model, "7 November 2026 Rs. 1,800")
    by = {s["slot"]: s["matched"] for s in slots}
    assert by == {"title": False, "organization": True, "reference_number": False, "date": True,
                  "amount": True, "email": True, "phone": True}
    assert unsupported == {"dates": ["2026-11-24"], "amounts": [9999]}

    # Unlabelled values are collected by type (dates, e-mails, phones, rupee amounts).
    typed = ev.collect_model_fields([("other", "Helpline 1800-000-3366, mail x@y.in"), ("other", "Fee Rs. 4,500 by 25 June 2026")])
    assert typed["phones"] == {"18000003366"} and typed["emails"] == {"x@y.in"}
    assert typed["amounts"] == {4500} and typed["dates"] == {"2026-06-25"}


def test_summary_scoring():
    case = {"facts": [{"fact": "start", "any": ["24 November 2026"]}, {"fact": "fee", "any": ["1,800"]}]}
    score = ev.score_summary(case, "Exams begin on November 24, 2026. Fee Rs. 2,000.", "24 November 2026 Rs. 1,800")
    assert score["coverage"] == 0.5 and score["unsupported_numbers"] == ["2,000"]


def test_output_cache_runs_once(tmp_path):
    cache = ev.OutputCache("sys", "qa", tmp_path)
    calls = []
    record = cache.get_or_run("F01", lambda: calls.append(1) or {"x": 1})
    again = ev.OutputCache("sys", "qa", tmp_path).get_or_run("F01", lambda: calls.append(1) or {"x": 2})
    assert record == again == {"x": 1} and calls == [1]
    assert ev.OutputCache("sys", "qa", tmp_path).get_or_run("F02", lambda: {"x": 3}, offline=True) is None


def test_output_cache_does_not_save_transient_errors(tmp_path):
    cache = ev.OutputCache("sys", "qa", tmp_path)
    assert cache.get_or_run("D01", lambda: {"error": "timeout"}) == {"error": "timeout"}
    assert ev.OutputCache("sys", "qa", tmp_path).get_or_run("D01", lambda: {"answer": "ok"}) == {"answer": "ok"}


def test_rate_limit_retry_waits_then_succeeds():
    from src.llm_provider import LLMRateLimited

    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMRateLimited(12)
        return "done"

    assert ev.with_rate_limit_retry(flaky, sleep=sleeps.append) == "done"
    assert sleeps == [12, 12]
