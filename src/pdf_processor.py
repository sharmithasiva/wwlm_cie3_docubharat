"""PDF validation and per-page text extraction with PyMuPDF.

PyMuPDF does not execute embedded JavaScript; document content is only ever read
as text and is never interpreted as instructions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pymupdf

from src.config import Settings


class PDFValidationError(ValueError):
    """Raised with a user-safe message when a file cannot be processed."""


@dataclass
class PageText:
    page_number: int  # 1-based, as shown to users
    text: str


@dataclass
class ExtractionResult:
    pages: list[PageText]
    total_pages: int
    pages_processed: int
    char_count: int
    sha256: str
    truncated: bool = False
    low_text_pages: list[int] = field(default_factory=list)
    needs_ocr: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def validate_pdf_bytes(data: bytes, filename: str | None, settings: Settings) -> None:
    """Cheap checks that run before the file is handed to the PDF parser."""
    if filename is not None and not filename.lower().endswith(".pdf"):
        raise PDFValidationError("Only PDF files are supported. Please upload a file ending in .pdf.")
    if not data:
        raise PDFValidationError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        size_mb = len(data) / (1024 * 1024)
        raise PDFValidationError(
            f"The file is {size_mb:.1f} MB, which exceeds the {settings.max_upload_mb} MB limit."
        )
    # The PDF spec allows the header anywhere in the first 1024 bytes.
    if b"%PDF-" not in data[:1024]:
        raise PDFValidationError("This file does not look like a PDF (missing PDF header).")


def extract_text(data: bytes, filename: str | None = None, settings: Settings | None = None) -> ExtractionResult:
    settings = settings or Settings()
    validate_pdf_bytes(data, filename, settings)

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for corrupt files
        raise PDFValidationError("The PDF could not be opened. It may be corrupted or malformed.") from exc

    with doc:
        # Some PDFs are encrypted with an empty user password and open fine once authenticated.
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFValidationError(
                "This PDF is password-protected. Please upload an unencrypted copy."
            )
        total_pages = doc.page_count
        if total_pages == 0:
            raise PDFValidationError("The PDF contains no pages.")

        pages_to_read = min(total_pages, settings.max_pages)
        pages: list[PageText] = []
        low_text_pages: list[int] = []
        for index in range(pages_to_read):
            try:
                text = doc[index].get_text("text", sort=True)
            except Exception:
                text = ""
            pages.append(PageText(page_number=index + 1, text=text))
            if len(text.strip()) < settings.min_chars_per_page:
                low_text_pages.append(index + 1)

    char_count = sum(len(p.text.strip()) for p in pages)
    warnings: list[str] = []
    truncated = total_pages > pages_to_read
    if truncated:
        warnings.append(
            f"Only the first {pages_to_read} of {total_pages} pages were processed (page limit)."
        )

    # Mostly-empty documents are probably scanned images; say so instead of returning nothing.
    needs_ocr = char_count == 0 or len(low_text_pages) / pages_to_read >= 0.5
    if needs_ocr:
        warnings.append(
            "Little or no text could be extracted. This PDF is probably scanned or image-based "
            "and needs OCR before it can be analysed."
        )
    elif low_text_pages:
        warnings.append(
            f"Pages {', '.join(map(str, low_text_pages))} have little extractable text "
            "and may be scanned images."
        )

    return ExtractionResult(
        pages=pages,
        total_pages=total_pages,
        pages_processed=pages_to_read,
        char_count=char_count,
        sha256=hashlib.sha256(data).hexdigest(),
        truncated=truncated,
        low_text_pages=low_text_pages,
        needs_ocr=needs_ocr,
        warnings=warnings,
    )
