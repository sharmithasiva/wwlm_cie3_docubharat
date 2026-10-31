import pytest

from src.config import Settings
from src.pdf_processor import PDFValidationError, extract_text
from tests.conftest import make_pdf


def test_extracts_text_per_page(circular_pdf):
    result = extract_text(circular_pdf, "circular.pdf")
    assert result.total_pages == 2
    assert result.pages_processed == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "17 November 2026" in result.pages[0].text
    assert "75%" in result.pages[1].text
    assert result.char_count > 500
    assert not result.needs_ocr
    assert len(result.sha256) == 64


def test_blank_pdf_is_flagged_for_ocr(blank_pdf):
    result = extract_text(blank_pdf, "scan.pdf")
    assert result.needs_ocr
    assert result.char_count == 0
    assert any("OCR" in w for w in result.warnings)


def test_single_low_text_page_is_reported_without_ocr_flag():
    data = make_pdf(["A" * 20 + " real content " * 30, "", "More content here " * 20])
    result = extract_text(data, "mixed.pdf")
    assert result.low_text_pages == [2]
    assert not result.needs_ocr


def test_page_limit_truncates():
    data = make_pdf([f"Page body number {i} with enough text to count." * 3 for i in range(5)])
    result = extract_text(data, "long.pdf", Settings(max_pages=3))
    assert result.total_pages == 5
    assert result.pages_processed == 3
    assert result.truncated
    assert any("first 3 of 5" in w for w in result.warnings)


def test_encrypted_pdf_rejected(encrypted_pdf):
    with pytest.raises(PDFValidationError, match="password"):
        extract_text(encrypted_pdf, "locked.pdf")


@pytest.mark.parametrize(
    "data, filename, message",
    [
        (b"", "empty.pdf", "empty"),
        (b"just some text, not a pdf", "fake.pdf", "does not look like a PDF"),
        (b"%PDF-1.7\n this is not a valid body \n%%EOF", "broken.pdf", "could not be opened"),
        (b"%PDF-1.7 whatever", "notes.docx", "Only PDF"),
    ],
)
def test_invalid_inputs_raise_friendly_errors(data, filename, message):
    with pytest.raises(PDFValidationError, match=message):
        extract_text(data, filename)


def test_size_limit(circular_pdf):
    too_big = circular_pdf + b"0" * (1024 * 1024 + 1)
    with pytest.raises(PDFValidationError, match="exceeds the 1 MB limit"):
        extract_text(too_big, "big.pdf", Settings(max_upload_mb=1))
