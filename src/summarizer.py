"""Transparent extractive summary: sentences are selected from the document, never written.

Sentence scores combine word-frequency centrality, position and the presence of
dates, amounts or required actions. The output reuses the rule-based field
extractor for dates, actions, fees and eligibility.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.field_extractor import (
    ACTION_CUES,
    DATE_PATTERN,
    MONEY_PATTERN,
    REFERENCE_PATTERN,
    SUBJECT_PATTERN,
    ExtractedFields,
    FieldValue,
    extract_fields,
)
from src.pdf_processor import PageText
from src.retrieval import tokenize
from src.text_cleaner import split_sentences


@dataclass
class SummarySentence:
    text: str
    page: int


@dataclass
class StructuredSummary:
    purpose: str
    purpose_page: int | None
    main_points: list[SummarySentence]
    important_dates: list[FieldValue]
    required_actions: list[FieldValue]
    fees: list[FieldValue]
    eligibility: list[FieldValue]
    method: str = "extractive (rule-based); sentences are quoted from the document"
    notes: list[str] = field(default_factory=list)


def _is_reference_line(sentence: str) -> bool:
    """Letterhead lines such as "Public Notice No. X Date: Y" carry no summary content."""
    match = REFERENCE_PATTERN.search(sentence)
    return bool(match) and match.start() < 20


def _candidate_sentences(pages: list[PageText]) -> list[tuple[int, int, str]]:
    """(order, page, sentence) for sentences long enough to carry content."""
    out = []
    for page in pages:
        for sentence in split_sentences(page.text):
            if len(sentence.split()) >= 6 and not SUBJECT_PATTERN.match(sentence) and not _is_reference_line(sentence):
                out.append((len(out), page.page_number, sentence))
    return out


def summarize(pages: list[PageText], fields: ExtractedFields | None = None, max_points: int = 5) -> StructuredSummary:
    fields = fields or extract_fields(pages)
    candidates = _candidate_sentences(pages)

    if fields.title:
        purpose, purpose_page = fields.title.value, fields.title.page
    elif candidates:
        purpose, purpose_page = candidates[0][2], candidates[0][1]
    else:
        purpose, purpose_page = "No purpose statement could be identified.", None

    frequencies = Counter(t for _, _, s in candidates for t in tokenize(s))
    scored = []
    for order, page, sentence in candidates:
        tokens = tokenize(sentence)
        if not tokens:
            continue
        centrality = sum(frequencies[t] for t in set(tokens)) / (len(set(tokens)) ** 0.5)
        position = 1.0 / (1 + order * 0.15)  # notices usually lead with the key information
        bonus = (
            0.5 * bool(DATE_PATTERN.search(sentence))
            + 0.5 * bool(MONEY_PATTERN.search(sentence))
            + 0.3 * bool(ACTION_CUES.search(sentence))
        )
        scored.append((centrality / max(frequencies.values()) + position + bonus, order, page, sentence))

    top = sorted(scored, key=lambda s: -s[0])[:max_points]
    # Present selected sentences in document order so the summary reads naturally.
    main_points = [SummarySentence(sentence, page) for _, _, page, sentence in sorted(top, key=lambda s: s[1])
                   if sentence != purpose]

    # One entry per calendar date, preferring the more specific deadline/start label.
    dates_by_day: dict[str | None, FieldValue] = {}
    for item in fields.dates:
        existing = dates_by_day.get(item.normalized)
        if existing is None or (existing.label == "date" and item.label != "date"):
            dates_by_day[item.normalized] = item
    important_dates = sorted(dates_by_day.values(), key=lambda d: d.normalized or "")

    notes = []
    if not candidates:
        notes.append("The document has too little text for a summary.")
    return StructuredSummary(
        purpose=purpose,
        purpose_page=purpose_page,
        main_points=main_points,
        important_dates=important_dates,
        required_actions=fields.action_items[:6],
        fees=fields.fees,
        eligibility=fields.eligibility[:6],
        notes=notes,
    )
