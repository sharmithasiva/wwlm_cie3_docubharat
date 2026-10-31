"""End-to-end document processing: validate -> extract -> clean -> chunk -> index."""

from __future__ import annotations

from dataclasses import dataclass

from src.chunker import Chunk, chunk_pages
from src.config import Settings
from src.pdf_processor import ExtractionResult, PageText, extract_text
from src.retrieval import BM25Index
from src.text_cleaner import clean_text, remove_repeated_lines


@dataclass
class ProcessedDocument:
    filename: str
    extraction: ExtractionResult
    pages: list[PageText]  # cleaned text, one entry per processed page
    chunks: list[Chunk]
    index: BM25Index

    @property
    def doc_id(self) -> str:
        return self.extraction.sha256[:12]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


def process_pdf(data: bytes, filename: str, settings: Settings | None = None) -> ProcessedDocument:
    """Raises PDFValidationError with a user-safe message for unusable files."""
    settings = settings or Settings()
    extraction = extract_text(data, filename, settings)
    raw_texts = remove_repeated_lines([p.text for p in extraction.pages])
    pages = [PageText(p.page_number, clean_text(text)) for p, text in zip(extraction.pages, raw_texts)]
    chunks = chunk_pages(pages, settings.chunk_words, settings.chunk_overlap_words)
    return ProcessedDocument(
        filename=filename,
        extraction=extraction,
        pages=pages,
        chunks=chunks,
        index=BM25Index(chunks),
    )
