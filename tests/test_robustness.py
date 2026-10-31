"""Edge cases, regressions for bugs found in review, and resource-safety checks."""

import time
from pathlib import Path

import pytest

from src.config import Settings
from src.field_extractor import extract_fields
from src.pdf_processor import PageText, PDFValidationError, extract_text
from src.pipeline import process_pdf
from src.qa_engine import answer_question
from src.summarizer import summarize
from src.text_cleaner import remove_repeated_lines, split_sentences
from src.ui_text import highlight, md_escape
from tests.conftest import CIRCULAR_PAGE_1, make_pdf

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample_documents"


# --------------------------------------------------------------------------- configuration


def test_settings_from_env_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("DOCUBHARAT_MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("DOCUBHARAT_MAX_PAGES", "not-a-number")
    monkeypatch.setenv("DOCUBHARAT_TOP_K", "0")
    s = Settings.from_env()
    assert s.max_upload_mb == 5
    assert s.max_upload_bytes == 5 * 1024 * 1024
    assert s.max_pages == 100  # invalid value -> default
    assert s.top_k == 1  # clamped to the minimum


def test_settings_overlap_is_clamped_below_chunk_size(monkeypatch):
    monkeypatch.setenv("DOCUBHARAT_CHUNK_WORDS", "100")
    monkeypatch.setenv("DOCUBHARAT_CHUNK_OVERLAP_WORDS", "90")
    s = Settings.from_env()
    assert s.chunk_overlap_words < s.chunk_words // 2


# --------------------------------------------------------------------------- malformed input


def test_truncated_pdf_does_not_crash(circular_pdf):
    truncated = circular_pdf[: len(circular_pdf) // 2]
    try:
        result = extract_text(truncated, "truncated.pdf")
    except PDFValidationError:
        return  # a clean, user-facing rejection is acceptable
    assert result.pages_processed >= 0  # PyMuPDF may repair it; either way no exception escapes


def test_header_after_leading_garbage_is_accepted():
    data = b"\n\n  " + make_pdf(["Some readable text on this page for the test."])
    assert extract_text(data, "offset.pdf").char_count > 0


# --------------------------------------------------------------------------- regressions from code review


def test_reference_number_must_contain_a_digit():
    f = extract_fields([PageText(1, "Read this notice. No payments are accepted in cash.")])
    assert f.reference_number is None


def test_rs_inside_a_word_is_not_an_amount():
    f = extract_fields([PageText(1, "Classes run from 0900 hrs 30 students per batch. Fee is Rs. 200.")])
    assert [m.normalized for m in f.fees] == ["200"]


def test_body_values_repeated_on_two_pages_are_kept():
    page = "Header line\nIntro {n}\nTable row\nRs. 1,000\nmore body\nand more\nclosing {n}\nFooter text here"
    cleaned = remove_repeated_lines([page.format(n=1), page.format(n=2)])
    assert "Rs. 1,000" in cleaned[0] and "Rs. 1,000" in cleaned[1]
    assert "Footer text here" not in cleaned[0]
    assert "Header line" not in cleaned[1]


def test_sentence_ending_in_single_digit_is_split():
    assert split_sentences("Report to Room 5. Bring your ID.") == ["Report to Room 5.", "Bring your ID."]


# --------------------------------------------------------------------------- safe rendering


def test_markdown_escaping_blocks_links_images_and_html():
    hostile = "![x](http://evil.example/p.png) [click](http://evil.example) <script>alert(1)</script> **bold**"
    escaped = md_escape(hostile)
    for raw in ("![", "](", "<script>", "**"):
        assert raw not in escaped


def test_highlight_bolds_terms_and_skips_single_letters():
    out = highlight("The fees are due. B.E. students pay.", ("fee", "b"))
    assert "**fees**" in out
    assert "**B" not in out and "**b" not in out


# --------------------------------------------------------------------------- Indian-language text


def test_devanagari_text_flows_through_cleaning_retrieval_and_qa():
    from src.chunker import chunk_pages
    from src.retrieval import BM25Index

    pages = [PageText(1, "परीक्षा शुल्क 500 रुपये है।\nअंतिम तिथि 15 नवंबर है।")]
    index = BM25Index(chunk_pages(pages))
    result = answer_question("परीक्षा शुल्क", index, pages=pages)
    assert result.found
    assert result.answer == "परीक्षा शुल्क 500 रुपये है।"


# --------------------------------------------------------------------------- resource limits


@pytest.mark.parametrize("text", ["a." * 10000, "1" * 20000, "1," * 10000, "a.b@" * 5000])
def test_pathological_text_is_processed_quickly(text):
    start = time.perf_counter()
    pages = [PageText(1, text)]
    summarize(pages, extract_fields(pages))
    assert time.perf_counter() - start < 2.0


def test_page_limit_and_speed_on_long_document():
    filler = "Section {i}. General information item {i} about campus facility number {i} and its timings."
    pages = [filler.format(i=i) for i in range(1, 151)]
    pages[36] = CIRCULAR_PAGE_1  # the fee notice is on page 37
    data = make_pdf(pages)
    start = time.perf_counter()
    doc = process_pdf(data, "long.pdf", Settings(max_pages=100))
    elapsed = time.perf_counter() - start
    assert doc.extraction.total_pages == 150
    assert doc.extraction.pages_processed == 100
    assert doc.extraction.truncated
    assert max(c.page_end for c in doc.chunks) == 100
    assert elapsed < 10.0
    result = answer_question("What is the examination fee?", doc.index, pages=doc.pages)
    assert result.found and result.cited_pages == [37]


# --------------------------------------------------------------------------- sample-document regression

GOLDEN = [
    ("exam_circular.pdf", "KSU/COE/2026/41", "What is the examination fee?", "Rs. 1,800"),
    ("admission_notice.pdf", "DIT/ADM/2027/03", "What is the last date to apply?", "15 February 2027"),
    ("scholarship_announcement.pdf", "VSSB/SCH/2026-27/12", "What is the income limit?", "Rs. 2,50,000"),
    ("fee_notification.pdf", "NCE/ACC/2026/18", "What is the tuition fee?", "Rs. 48,000"),
    ("government_notice.pdf", "DDS/PN/2026/77", "Who can apply?", "18 and 60 years"),
    ("noisy_hostel_notice.pdf", "GU/DSW/2026/09", "What is the last date for hostel applications?", "25 June 2026"),
]


@pytest.mark.parametrize("name, reference, question, fragment", GOLDEN)
def test_sample_documents(name, reference, question, fragment):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip("sample documents not generated")
    doc = process_pdf(path.read_bytes(), name)
    fields = extract_fields(doc.pages)
    assert fields.reference_number.value == reference
    assert fields.title is not None and fields.organization is not None
    result = answer_question(question, doc.index, pages=doc.pages)
    assert result.found and fragment in result.answer
    # Running footers ("SYNTHETIC SAMPLE DOCUMENT") must not leak into multi-page summaries.
    if doc.extraction.total_pages > 1:
        assert all("SYNTHETIC" not in p.text for p in summarize(doc.pages, fields).main_points)
