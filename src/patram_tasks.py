"""Document tasks built on BharatGen Patram: question answering, fields and summary.

Patram sees page images, not our extracted text. To keep quota use low:
- questions go to ONE page, chosen by BM25 (or by the user);
- summary and fields send at most `max_pages` pages, only when the user clicks.

Grounding: Patram does not return quotes, so for digital PDFs every number in its
answer (dates, amounts, percentages) is checked against the page's extracted text.
Scanned pages have no text layer, so their answers are marked "not verifiable".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.patram_client import PatramAnswer, PatramClient
from src.pdf_processor import PageText
from src.qa_engine import QAResult
from src.translation_checks import missing_numbers

FIELDS_QUESTION = (
    "List the document title, issuing organisation, reference number, important dates and deadlines, "
    "fees or amounts, eligibility requirements and contact details. "
    "Write one item per line in the form 'field: value'. Only include items that appear in the document."
)
SUMMARY_QUESTION = (
    "Summarise this document page in up to 5 short bullet points: its purpose, key dates, "
    "required actions, and any fees or eligibility conditions."
)

# Map Patram's free-text labels onto DocuBharat's field names.
_FIELD_KEYWORDS = [
    ("reference_number", ("reference", "circular no", "notice no", "notification no", "ref no", "number")),
    ("title", ("title", "subject")),
    ("issuing_organization", ("organisation", "organization", "issued by", "issuer", "institution", "university", "department")),
    ("date_or_deadline", ("date", "deadline", "last day", "schedule", "period")),
    ("fee_or_amount", ("fee", "amount", "cost", "charge", "fine", "scholarship")),
    ("eligibility", ("eligib", "criteria", "requirement", "attendance")),
    ("contact", ("contact", "email", "e-mail", "phone", "helpline", "website")),
]
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_ANSWER_PREFIX = re.compile(r"^answer\s*:\s*", re.IGNORECASE)
_LABEL = r"[A-Z][\w /()&'’-]{1,60}"
# Item separators seen in Patram's replies: ", Label: …", ". Label: …" and "; …".
_NEXT_LABEL = re.compile(rf"(?:[,.]\s+(?={_LABEL}:\s?)|;\s+)")
_EMPTY_VALUE = re.compile(r"^(?:n/?a|none|not (?:\w+ )?(?:mentioned|stated|available|specified|provided))\b", re.IGNORECASE)
_LABEL_VALUE = re.compile(rf"^({_LABEL}):\s+(.+)$")


def choose_page(extractive: QAResult, page_count: int) -> int:
    """Page most likely to hold the answer: the best extractive sentence, else the top BM25 passage."""
    if extractive.sentences:
        return extractive.sentences[0].page
    if extractive.evidence:
        return extractive.evidence[0].chunk.page_start
    return 1 if page_count >= 1 else 0


@dataclass
class Grounding:
    checkable: bool  # False for scanned pages (no text to compare with)
    numbers: list[str] = field(default_factory=list)  # numbers found in Patram's answer
    unverified: list[str] = field(default_factory=list)  # of those, numbers not in the page text

    @property
    def label(self) -> str:
        if not self.checkable:
            return "not verifiable (scanned page has no text layer)"
        if not self.numbers:
            return "no numbers to check"
        if not self.unverified:
            return f"all {len(self.numbers)} number(s) found in the page text"
        return f"{len(self.unverified)} of {len(self.numbers)} number(s) NOT found in the page text"


def check_grounding(answer: str, page_text: str) -> Grounding:
    numbers = missing_numbers(answer, "")  # every number in the answer
    if len(page_text.strip()) < 20:
        return Grounding(False, numbers)
    return Grounding(True, numbers, missing_numbers(answer, page_text))


def page_text(pages: list[PageText], page: int) -> str:
    return next((p.text for p in pages if p.page_number == page), "")


@dataclass
class PatramQA:
    answer: PatramAnswer
    grounding: Grounding


def patram_answer(client: PatramClient, pdf_bytes: bytes, pages: list[PageText], page: int, question: str) -> PatramQA:
    answer = client.ask_page(pdf_bytes, page, question)
    return PatramQA(answer, check_grounding(answer.text, page_text(pages, page)))


@dataclass
class PatramField:
    field: str  # one of the DocuBharat field names, or "other"
    label: str  # Patram's own label
    value: str
    page: int
    grounding: Grounding


def classify_label(label: str) -> str:
    if label == "(no label)":
        return "other"
    lowered = label.lower()
    for name, keywords in _FIELD_KEYWORDS:
        if any(k in lowered for k in keywords):
            return name
    return "other"


def _segments(text: str) -> list[str]:
    """Split Patram's reply into 'label: value' segments.

    Patram does not always follow the one-item-per-line instruction: observed live it
    replied on ONE line, "Answer: Document title: …, Issuing organisation: …, …".
    Segments are therefore split on newlines and on ", " followed by a new "Label: ".
    """
    segments = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip().strip("*").strip()
        line = _ANSWER_PREFIX.sub("", line)
        segments.extend(s.strip() for s in _NEXT_LABEL.split(line) if s.strip())
    return segments


def parse_fields(text: str, page: int, page_content: str) -> list[PatramField]:
    """Parse 'label: value' items; text without a label is skipped (never guessed)."""
    items = []
    segments = _segments(text)
    if not any(_LABEL_VALUE.match(s) for s in segments):
        # Observed live: Patram sometimes lists values with no labels at all
        # ("Directorate of Digital Services, …, 15 November 2026, …"). Keep each comma-separated
        # value as an unlabelled item so dates, e-mails and phone numbers are still usable.
        segments = [f"(no label): {v.strip()}" for s in segments for v in s.split(", ") if v.strip()]
    for segment in segments:
        if ":" not in segment:
            continue
        label, value = (part.strip().strip("*").strip() for part in segment.split(":", 1))
        field_name = classify_label(label)
        # "Important dates and deadlines: Examination period: 24 Nov…" -> use the more specific inner label.
        inner = _LABEL_VALUE.match(value)
        if inner:
            inner_field = classify_label(inner.group(1))
            field_name = inner_field if inner_field != "other" else field_name
            label, value = f"{label} › {inner.group(1)}", inner.group(2).strip()
        if not label or not value or _EMPTY_VALUE.match(value):
            continue
        items.append(PatramField(field_name, label, value.rstrip("."), page, check_grounding(value, page_content)))
    return items


@dataclass
class PatramPagesResult:
    answers: list[PatramAnswer]
    fields: list[PatramField] = field(default_factory=list)


def patram_fields(client: PatramClient, pdf_bytes: bytes, pages: list[PageText], page_count: int) -> PatramPagesResult:
    answers, fields = [], []
    for page in range(1, min(page_count, client.config.max_pages) + 1):
        answer = client.ask_page(pdf_bytes, page, FIELDS_QUESTION, max_new_tokens=384)
        answers.append(answer)
        fields.extend(parse_fields(answer.text, page, page_text(pages, page)))
    return PatramPagesResult(answers, fields)


def patram_summary(client: PatramClient, pdf_bytes: bytes, page_count: int) -> PatramPagesResult:
    answers = [
        client.ask_page(pdf_bytes, page, SUMMARY_QUESTION, max_new_tokens=320)
        for page in range(1, min(page_count, client.config.max_pages) + 1)
    ]
    return PatramPagesResult(answers)
