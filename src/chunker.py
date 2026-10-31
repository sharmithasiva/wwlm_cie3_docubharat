"""Split cleaned page text into overlapping word-window chunks that remember their pages."""

from __future__ import annotations

from dataclasses import dataclass

from src.pdf_processor import PageText


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # stable id shown to users, e.g. "C3"
    index: int
    text: str
    page_start: int
    page_end: int
    word_count: int

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"


@dataclass(frozen=True)
class _Word:
    text: str
    page: int
    sep_after: str  # " ", "\n" (line break) or "\n\n" (paragraph/page break)


def _to_words(pages: list[PageText]) -> list[_Word]:
    words: list[_Word] = []
    for page in pages:
        for para in page.text.split("\n\n"):
            for line in para.split("\n"):
                line_words = line.split()
                for i, w in enumerate(line_words):
                    words.append(_Word(w, page.page_number, " " if i < len(line_words) - 1 else "\n"))
            if words and words[-1].page == page.page_number:
                words[-1] = _Word(words[-1].text, words[-1].page, "\n\n")
    return words


def chunk_pages(pages: list[PageText], chunk_words: int = 500, overlap_words: int = 50) -> list[Chunk]:
    """Slide a window of at most `chunk_words` words over the document.

    A window is cut early at a paragraph break if one falls in its second half, so
    chunks rarely split a paragraph. Consecutive chunks share `overlap_words` words
    so that facts on a boundary stay retrievable. Every chunk records the pages its
    words came from. Memory is linear in document size.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    overlap_words = max(0, min(overlap_words, chunk_words // 2))

    words = _to_words(pages)
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        if end < len(words):
            for i in range(end - 1, start + chunk_words // 2 - 1, -1):
                if words[i].sep_after == "\n\n":
                    end = i + 1
                    break
        window = words[start:end]
        text = "".join(w.text + w.sep_after for w in window).strip()
        page_numbers = [w.page for w in window]
        chunks.append(
            Chunk(
                chunk_id=f"C{len(chunks) + 1}",
                index=len(chunks),
                text=text,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                word_count=len(window),
            )
        )
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks
