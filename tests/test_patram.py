"""Patram client and tasks with a fake Space (offline, no GPU quota used)."""

import concurrent.futures
from pathlib import Path

import pytest

from src.chunker import Chunk
from src.patram_client import (
    PatramClient,
    PatramConfig,
    PatramUnavailable,
    render_page_png,
    status,
)
from src.patram_tasks import (
    FIELDS_QUESTION,
    check_grounding,
    choose_page,
    classify_label,
    parse_fields,
    patram_answer,
    patram_fields,
    patram_summary,
)
from src.pdf_processor import PageText
from src.qa_engine import AnswerSentence, QAResult
from src.retrieval import SearchResult
from tests.conftest import make_pdf

TOKEN = "hf_" + "t" * 30


class FakeJob:
    def __init__(self, outcome):
        self.outcome = outcome

    def result(self, timeout=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeGradioClient:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def submit(self, image, question, max_new_tokens, api_name=None):
        self.calls.append({"image": image, "question": question, "tokens": max_new_tokens, "api": api_name})
        return FakeJob(self.outcomes.pop(0))


def client_with(*outcomes, retries=0, **config):
    fake = FakeGradioClient(*outcomes)
    cfg = PatramConfig(space_id="user/docubharat-patram", token=TOKEN, **config)
    return PatramClient(cfg, client_factory=lambda: fake, sleep=lambda s: None, retries=retries), fake


WriteTimeout = type("WriteTimeout", (Exception,), {})  # stands in for httpx.WriteTimeout


def test_waking_space_timeout_is_retried_once():
    client, fake = client_with(WriteTimeout("upload timed out"), ("Rs. 1,800", 1.2), retries=1)
    assert client.ask_page(PDF, 1, "What is the fee?").text == "Rs. 1,800"
    assert len(fake.calls) == 2


def test_persistent_network_failure_explains_waking_space():
    client, fake = client_with(WriteTimeout("x"), WriteTimeout("x"), retries=1)
    with pytest.raises(PatramUnavailable, match="waking up"):
        client.ask_page(PDF, 1, "What is the fee?")
    assert len(fake.calls) == 2


def test_job_timeout_and_quota_are_not_retried():
    for error in (concurrent.futures.TimeoutError(), RuntimeError("ZeroGPU quota exceeded")):
        client, fake = client_with(error, retries=1)
        with pytest.raises(PatramUnavailable):
            client.ask_page(PDF, 1, "What is the fee?")
        assert len(fake.calls) == 1


PDF = make_pdf(["The examination fee is Rs. 1,800 per semester.", "Contact exams@example.edu."])
PAGES = [PageText(1, "The examination fee is Rs. 1,800 per semester."), PageText(2, "Contact exams@example.edu.")]


# --------------------------------------------------------------------------- configuration


def test_status_requires_space_and_token(monkeypatch):
    monkeypatch.delenv("PATRAM_SPACE_ID", raising=False)
    assert not status(PatramConfig.from_env())[0]
    assert "PATRAM_ENABLED=false" in status(PatramConfig(space_id="a/b", token=TOKEN, enabled=False))[1]
    assert status(PatramConfig(space_id="a/b", token=TOKEN))[0]
    with pytest.raises(PatramUnavailable):
        PatramClient(PatramConfig())


def test_token_not_in_repr():
    assert TOKEN not in repr(PatramConfig(space_id="a/b", token=TOKEN))


def test_render_page_png():
    png = render_page_png(PDF, 2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(PatramUnavailable, match="does not exist"):
        render_page_png(PDF, 5)


# --------------------------------------------------------------------------- client behaviour


def test_ask_page_sends_image_question_and_endpoint():
    client, fake = client_with(("The fee is Rs. 1,800.", 1.7))
    answer = client.ask_page(PDF, 1, "  What is the fee?  ")
    call = fake.calls[0]
    assert call["api"] == "/ask" and call["question"] == "What is the fee?" and call["tokens"] == 256
    assert answer.text == "The fee is Rs. 1,800." and answer.page == 1 and answer.generation_seconds == 1.7
    assert client.calls == 1
    assert not Path(call["image"]["path"]).exists()  # temporary page image is deleted


@pytest.mark.parametrize(
    "error, message",
    [
        (RuntimeError("ZeroGPU quota exceeded for today"), "quota for Patram is used up"),
        (concurrent.futures.TimeoutError(), "took too long"),
        (ConnectionError("dns"), "Could not reach"),
        (type("AuthenticationError", (ValueError,), {})("bad token"), "rejected the token"),
        (type("AppError", (ValueError,), {})("CUDA error"), "reported an error"),
    ],
)
def test_failures_become_user_messages(error, message):
    client, _ = client_with(error)
    with pytest.raises(PatramUnavailable, match=message) as info:
        client.ask_page(PDF, 1, "What is the fee?")
    assert TOKEN not in str(info.value)


@pytest.mark.parametrize("result", [("", 1.0), "not a tuple", (None, None)])
def test_unusable_responses(result):
    client, _ = client_with(result)
    with pytest.raises(PatramUnavailable):
        client.ask_page(PDF, 1, "What is the fee?")


def test_empty_question_makes_no_call():
    client, fake = client_with()
    with pytest.raises(PatramUnavailable):
        client.ask_page(PDF, 1, "   ")
    assert fake.calls == []


# --------------------------------------------------------------------------- tasks


def _chunk(page):
    return Chunk("C1", 0, "text", page, page, 1)


def test_choose_page_prefers_best_sentence_then_passage():
    with_sentence = QAResult("q", True, "a", "extractive", "general",
                             sentences=[AnswerSentence("a", "C1", 2, 1.0, 1.0)])
    assert choose_page(with_sentence, 3) == 2
    passage_only = QAResult("q", False, "n", "extractive", "general",
                            evidence=[SearchResult(_chunk(3), 1.0, 1, ("fee",), 0.5)])
    assert choose_page(passage_only, 3) == 3
    assert choose_page(QAResult("q", False, "n", "extractive", "general"), 2) == 1


def test_grounding_checks_numbers_against_page_text():
    assert check_grounding("Fee is Rs. 1,800.", "The fee is Rs. 1,800 per semester.").label.startswith("all 1")
    wrong = check_grounding("Fee is Rs. 2,000.", "The fee is Rs. 1,800 per semester.")
    assert wrong.unverified == ["2,000"] and "NOT found" in wrong.label
    assert not check_grounding("Open until 15 December 2026.", "").checkable  # scanned page
    assert check_grounding("Apply online.", "Students must apply online via the portal.").label == "no numbers to check"


def test_patram_answer_adds_grounding():
    client, _ = client_with(("The fee is Rs. 1,800 per semester.", 2.0))
    result = patram_answer(client, PDF, PAGES, 1, "What is the fee?")
    assert result.grounding.checkable and result.grounding.unverified == []


def test_parse_fields_maps_labels_and_skips_noise():
    text = (
        "- Title: End Semester Examinations\n"
        "* **Issuing organisation**: Kaveri State University\n"
        "Circular No: KSU/COE/2026/41\n"
        "1. Last date: 7 November 2026\n"
        "Fee: Rs. 1,800\n"
        "Eligibility: 75% attendance\n"
        "Contact: coe@example.in\n"
        "Some sentence without a label\n"
        "Website: not mentioned\n"
    )
    fields = parse_fields(text, 1, "Fee Rs. 1,800 ... 7 November 2026 ... 75% ... KSU/COE/2026/41")
    assert [f.field for f in fields] == [
        "title", "issuing_organization", "reference_number", "date_or_deadline", "fee_or_amount", "eligibility", "contact",
    ]
    assert fields[1].value == "Kaveri State University"
    assert all(f.grounding.unverified == [] for f in fields)


# Patram's actual reply to FIELDS_QUESTION for exam_circular.pdf page 1 (live run, 2026-10-29).
REAL_PATRAM_FIELDS = (
    "Answer: Document title: Circular No. KSU/COE/2026/41, Issuing organisation: Kaveri State University, "
    "Reference number: KSU/COE/2026/41, Important dates and deadlines: Examination period: 24 November 2026 to "
    "12 December 2026, Detailed timetable publication date: 5 November 2026, Examination fee payment deadline: "
    "7 November 2026, Late fine payment deadline: 14 November 2026, Examination Fee: Rs. 1,800 per semester, "
    "Additional fee for supplementary papers: Rs. 350 per paper, Fees acceptance method: Online student portal, "
    "Eligibility: All Heads of Departments and students of the B.Tech, M.Tech, and MBA programmes."
)


def test_parse_real_single_line_patram_reply():
    page = "Circular No. KSU/COE/2026/41 ... 24 November 2026 ... 12 December 2026 ... 5 November 2026 ... " \
           "7 November 2026 ... 14 November 2026 ... Rs. 1,800 ... Rs. 350"
    fields = parse_fields(REAL_PATRAM_FIELDS, 1, page)
    got = [(f.field, f.value) for f in fields]
    assert got[0] == ("title", "Circular No. KSU/COE/2026/41")  # Patram's (wrong) title is kept as reported
    assert ("issuing_organization", "Kaveri State University") in got
    assert ("reference_number", "KSU/COE/2026/41") in got
    assert ("date_or_deadline", "24 November 2026 to 12 December 2026") in got
    assert ("date_or_deadline", "14 November 2026") in got
    assert ("fee_or_amount", "Rs. 1,800 per semester") in got
    # "B.Tech, M.Tech, and MBA" must not be split (no "Label:" follows the commas)
    assert got[-1] == ("eligibility", "All Heads of Departments and students of the B.Tech, M.Tech, and MBA programmes")
    assert len(fields) == 11
    assert all(not f.grounding.unverified for f in fields)


def test_parse_real_semicolon_and_unlabelled_replies():
    # Real Patram replies from the evaluation run (2026-10-31), shortened.
    semicolons = ("Answer: Document title: Fee Structure 2026-27; Issuing organisation: Narmada College of Engineering; "
                  "Reference number: NCE/ACC/2026/18; Fees or amounts: Rs. 48,000 (tuition fee), Rs. 6,500 (development fee)")
    fields = parse_fields(semicolons, 1, "")
    assert [f.field for f in fields] == ["title", "issuing_organization", "reference_number", "fee_or_amount"]
    assert fields[2].value == "NCE/ACC/2026/18"

    sentences = ("Document Title: Admission Notice for M.Tech Programmes. Issuing Organisation: Deccan Institute of "
                 "Technology. Reference Number: DIT/ADM/03. Application deadline: Not explicitly stated. "
                 "Application Fee: Rs. 1,200 for General and OBC candidates, Rs. 600 for SC, ST, and PwD candidates.")
    fields = parse_fields(sentences, 1, "")
    assert [(f.field, f.value) for f in fields] == [
        ("title", "Admission Notice for M.Tech Programmes"),
        ("issuing_organization", "Deccan Institute of Technology"),
        ("reference_number", "DIT/ADM/03"),  # Patram's misread is kept as reported
        ("fee_or_amount", "Rs. 1,200 for General and OBC candidates, Rs. 600 for SC, ST, and PwD candidates"),
    ]

    unlabelled = ("Answer: Directorate of Digital Services, Public Notice No. DDSS/PN/2026/277, 15 November 2026, "
                  "Helpline number 1800-000-3366, Email digitalskills@vindhyain example.gov.in.")
    fields = parse_fields(unlabelled, 1, "")
    assert all(f.field == "other" and f.label == "(no label)" for f in fields)
    assert [f.value for f in fields][:3] == ["Directorate of Digital Services", "Public Notice No. DDSS/PN/2026/277",
                                             "15 November 2026"]


@pytest.mark.parametrize(
    "label, field",
    [("Notice No.", "reference_number"), ("Subject", "title"), ("Application fee", "fee_or_amount"),
     ("Helpline", "contact"), ("Venue", "other")],
)
def test_classify_label(label, field):
    assert classify_label(label) == field


def test_document_tasks_respect_page_limit():
    client, fake = client_with(("Title: A", 1.0), ("Fee: Rs. 5", 1.0), max_pages=2)
    pdf = make_pdf(["Page one", "Page two", "Page three"])
    result = patram_fields(client, pdf, [], 3)
    assert len(fake.calls) == 2 and fake.calls[0]["question"] == FIELDS_QUESTION
    assert [f.field for f in result.fields] == ["title", "fee_or_amount"]

    client, fake = client_with(("- Purpose: exams", 1.0), max_pages=1)
    summary = patram_summary(client, pdf, 3)
    assert len(fake.calls) == 1 and summary.answers[0].page == 1
