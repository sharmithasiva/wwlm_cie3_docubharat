"""Normalise raw PDF text: whitespace, hyphenation, broken line endings, page furniture.

`clean_text` is meant to be called once per page so that page-number lines can be
recognised by position (first/last line) without deleting numbers from tables.
"""

from __future__ import annotations

import re
import unicodedata

# Lines that are only page furniture, e.g. "3", "Page 3", "Page 3 of 10", "- 3 -".
_PAGE_NUMBER_LINE = re.compile(r"^(?:page\s*)?[-–]?\s*\d{1,4}\s*(?:of\s*\d{1,4})?\s*[-–]?$", re.IGNORECASE)
# Start of a list item: bullets, "1.", "(a)", "ii)", etc.
_LIST_ITEM = re.compile(r"^(?:[-•*●▪◦]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])\s+")
_SENTENCE_END = (".", ":", ";", "?", "!", "।")


# Words after which a full stop does not end a sentence ("Rs. 500", "Circular No. 14").
_ABBREVIATIONS = frozenset(
    "rs no nos dr prof mr mrs ms sr jr st dept govt univ approx etc viz vs ie eg ref sl sec".split()
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?।](?=\s+)")
# A line ending in one of these words was wrapped mid-sentence.
_CONTINUATION_WORDS = frozenset(
    "a an the of to for and or is are was were be by on at in with from before after than that "
    "which who whose under per as".split()
)
# Header-style label lines ("Subject: ...", "Ref: ...") stand alone.
_LABEL_LINE = re.compile(r"^(?:subject|sub|ref|date|to|from)\s*[:.-]", re.IGNORECASE)


def _normalise_chars(text: str) -> str:
    # NFC rather than NFKC so Indic scripts keep their canonical composed forms.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(chr(0x00AD), "")  # soft hyphen
    text = text.replace(chr(0x00A0), " ").replace(chr(0x200B), "")  # no-break space, zero-width space
    return text.replace("\t", " ")


def _ends_with_abbreviation(line: str) -> bool:
    """True for lines ending in "Rs." or "No." and similar, where the full stop is not a sentence end."""
    words = line.split()
    return bool(words) and words[-1].endswith(".") and words[-1][:-1].lower().strip("(") in _ABBREVIATIONS


def _should_join(previous: str, current: str, previous_was_full_width: bool) -> bool:
    """Join a wrapped line onto the previous one when it is clearly a continuation."""
    if _LIST_ITEM.match(current) or _LABEL_LINE.match(previous):
        return False
    if _ends_with_abbreviation(previous):
        return True  # "Rs." / "No." always continue onto the next line
    if previous.endswith(_SENTENCE_END):
        return False
    # A continuation starts in lower case, follows a trailing comma or a word that cannot
    # end a sentence ("is", "of"), or follows a line that ran to the right margin.
    first = current[0]
    last_word = previous.split()[-1].lower()
    return (
        first.islower()
        or first in "(₹"
        or previous.endswith(",")
        or last_word in _CONTINUATION_WORDS
        or previous_was_full_width
    )


def clean_text(text: str) -> str:
    text = _normalise_chars(text)
    # Re-join words split across lines: "exami-\nnation" -> "examination".
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)

    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    # Drop page numbers only at the top or bottom of the page.
    non_empty = [i for i, line in enumerate(lines) if line]
    for i in (non_empty[:1] + non_empty[-1:]) if non_empty else []:
        if _PAGE_NUMBER_LINE.match(lines[i]):
            lines[i] = ""

    # Lines at least 75% as long as the page's longest lines are treated as running to the margin.
    lengths = sorted(len(line) for line in lines if line)
    full_width = 0.75 * lengths[int(0.9 * (len(lengths) - 1))] if len(lengths) >= 5 else float("inf")

    paragraphs: list[str] = []
    current: list[str] = []
    previous_length = 0
    for line in lines:
        if not line:
            if current:
                paragraphs.append("\n".join(current))
                current = []
            continue
        if current and _should_join(current[-1], line, previous_length >= full_width):
            current[-1] = f"{current[-1]} {line}"
        else:
            current.append(line)
        previous_length = len(line)
    if current:
        paragraphs.append("\n".join(current))

    return "\n\n".join(paragraphs)


def _edge_lines(text: str, edge: int) -> set[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return set(lines[:edge] + lines[-edge:])


def remove_repeated_lines(page_texts: list[str], min_pages: int = 2, edge: int = 2) -> list[str]:
    """Drop running headers/footers: identical lines near the top or bottom of most pages.

    Only the first and last `edge` lines of each page are considered, so a value that
    happens to repeat in the body (e.g. "Rs. 1,000" in two tables) is kept.
    """
    if len(page_texts) < min_pages:
        return page_texts
    counts: dict[str, int] = {}
    edges = [_edge_lines(text, edge) for text in page_texts]
    for lines in edges:
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(min_pages, len(page_texts) // 2 + 1)  # on more than half of the pages
    repeated = {line for line, n in counts.items() if n >= threshold and len(line) > 3}
    cleaned = []
    for text, page_edges in zip(page_texts, edges):
        removable = repeated & page_edges
        cleaned.append("\n".join(ln for ln in text.split("\n") if ln.strip() not in removable))
    return cleaned


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences; each line (list item, heading) is its own unit."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        start = 0
        for match in _SENTENCE_BOUNDARY.finditer(line):
            # The word just before the punctuation; rfind keeps this linear on long lines.
            word_start = max(line.rfind(" ", start, match.start()) + 1, start)
            previous_word = line[word_start : match.start()].lower().strip("(")
            # Keep abbreviations and initials ("B.E. degree", "Rs. 500") inside the sentence.
            last_part = previous_word.split(".")[-1]
            if match.group() == "." and (previous_word in _ABBREVIATIONS or (len(last_part) == 1 and last_part.isalpha())):
                continue
            sentences.append(line[start : match.end()].strip())
            start = match.end()
        if line[start:].strip():
            sentences.append(line[start:].strip())
    return [s for s in sentences if s]
