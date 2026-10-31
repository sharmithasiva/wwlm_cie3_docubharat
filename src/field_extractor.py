"""Rule-based extraction of key fields from notices and circulars.

Everything here is regex and keyword matching. Each value carries `source="rule"`,
the page it came from and the sentence it was found in, so it can be checked
against the document. Fields that are not found stay empty; nothing is guessed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from src.pdf_processor import PageText
from src.text_cleaner import split_sentences

_MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

DATE_PATTERN = re.compile(
    rf"""
    \b(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?[\s-]+(?P<m1>{_MONTH_RE})\.?,?[\s-]+(?P<y1>\d{{4}})\b   # 17 November 2026
    | \b(?P<m2>{_MONTH_RE})\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y2>\d{{4}})\b        # November 17, 2026
    | \b(?P<d3>\d{{1,2}})[/.-](?P<m3>\d{{1,2}})[/.-](?P<y3>\d{{4}})\b                          # 17/11/2026 (day first)
    | \b(?P<y4>\d{{4}})-(?P<m4>\d{{2}})-(?P<d4>\d{{2}})\b                                        # 2026-11-17
    """,
    re.IGNORECASE | re.VERBOSE,
)

MONEY_PATTERN = re.compile(
    r"(?:(?:\bRs\.?|\bINR|₹)\s*(?P<a1>\d[\d,]*(?:\.\d{1,2})?)(?:\s*/-)?"
    r"|\b(?P<a2>\d[\d,]*(?:\.\d{1,2})?)\s*(?:rupees|/-))",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s,;)]+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<![\w/.-])(?:\+91[\s-]?)?\(?\d{2,5}\)?(?:[\s-]?\d{2,5}){1,3}(?![\w/-]|\.\d)")
REFERENCE_PATTERN = re.compile(
    # The reference itself must contain a digit, so "...this notice. No payments..." is not one.
    r"\b(?:Circular|Notification|Notice|Order|Ref(?:erence)?|Memo)\.?\s*(?:No|Number)\.?\s*[:.]?\s*"
    r"(?P<ref>(?=[A-Z0-9/().-]*\d)[A-Z0-9][A-Z0-9/().-]*[A-Z0-9)])",
    re.IGNORECASE,
)
SUBJECT_PATTERN = re.compile(r"^\s*(?:Subject|Sub|Re)\s*[:.-]\s*(?P<subject>.+)$", re.IGNORECASE | re.MULTILINE)

ORG_KEYWORDS = re.compile(
    r"\b(University|Institute|College|Government|Ministry|Department|Directorate|Board|Council|"
    r"Commission|Authority|Corporation|Office|School|Academy|Agency|Vidyalaya|Mission)\b",
    re.IGNORECASE,
)
# Strong cues may appear anywhere before the date; weak ones only directly in front of it,
# so "issued by the Registrar on 3 October" is not read as a deadline.
DEADLINE_CUES = re.compile(
    r"\b(last date|last day|deadline|due date|due on|on or before|not later than|closing date|closes on|latest by)\b",
    re.IGNORECASE,
)
DEADLINE_WEAK_CUES = re.compile(r"\b(by|before|till|until)\s*$|\bfrom\b.*\bto\s*$", re.IGNORECASE)
END_CUES = re.compile(r"\b(conclude|concludes|end|ends|close|closes)\s+(?:on\s+)?$", re.IGNORECASE)
START_CUES = re.compile(r"\b(commence|commences|start|starts|begin|begins|opens?|from)\b", re.IGNORECASE)
ELIGIBILITY_CUES = re.compile(
    r"\b(eligib\w*|criteria|minimum|at least|must have|should have|qualif\w*|age limit|aged|"
    r"income (?:below|less than|up to)|not exceed\w*|percentage|attendance)\b",
    re.IGNORECASE,
)
ACTION_CUES = re.compile(
    r"\b(must|should|are required to|is required to|are requested to|are advised to|are directed to|"
    r"shall|need to|submit|apply|register|pay|download|upload|report to|contact)\b",
    re.IGNORECASE,
)


@dataclass
class FieldValue:
    value: str
    page: int
    evidence: str
    label: str = ""  # e.g. "deadline", "start", "end", "email"
    normalized: str | None = None  # e.g. ISO date or plain number
    source: str = "rule"


@dataclass
class ExtractedFields:
    title: FieldValue | None = None
    organization: FieldValue | None = None
    reference_number: FieldValue | None = None
    dates: list[FieldValue] = field(default_factory=list)
    fees: list[FieldValue] = field(default_factory=list)
    eligibility: list[FieldValue] = field(default_factory=list)
    contacts: list[FieldValue] = field(default_factory=list)
    action_items: list[FieldValue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def deadlines(self) -> list[FieldValue]:
        return [d for d in self.dates if d.label == "deadline"]


def parse_date(match: re.Match) -> str | None:
    """Return an ISO date for a DATE_PATTERN match, or None if it is not a real date."""
    g = match.groupdict()
    try:
        if g["d1"]:
            d, m, y = int(g["d1"]), _MONTHS[g["m1"].lower()], int(g["y1"])
        elif g["m2"]:
            d, m, y = int(g["d2"]), _MONTHS[g["m2"].lower()], int(g["y2"])
        elif g["d3"]:
            d, m, y = int(g["d3"]), int(g["m3"]), int(g["y3"])  # Indian convention: day first
        else:
            d, m, y = int(g["d4"]), int(g["m4"]), int(g["y4"])
        return date(y, m, d).isoformat()
    except (ValueError, KeyError):
        return None


def parse_amount(text: str) -> str:
    return text.replace(",", "")


def _date_label(sentence: str, match_start: int) -> str:
    # Look only at the words before the date, which is where cues like "last date" sit.
    before = sentence[:match_start]
    if END_CUES.search(before):
        return "end"
    if DEADLINE_CUES.search(before) or DEADLINE_WEAK_CUES.search(before):
        return "deadline"
    if START_CUES.search(before):
        return "start"
    return "date"


# Real sentences are far shorter. Garbage text (one huge "word" from a broken PDF) is
# scanned in windows so that regex cost stays linear in document size.
MAX_SENTENCE_CHARS = 1000


def _bounded(sentence: str) -> list[str]:
    if len(sentence) <= MAX_SENTENCE_CHARS:
        return [sentence]
    pieces = []
    while sentence:
        cut = sentence.rfind(" ", 0, MAX_SENTENCE_CHARS) if len(sentence) > MAX_SENTENCE_CHARS else len(sentence)
        cut = cut if cut > 0 else MAX_SENTENCE_CHARS
        pieces.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    return [p for p in pieces if p]


def _iter_sentences(pages: list[PageText]):
    for page in pages:
        for sentence in split_sentences(page.text):
            for piece in _bounded(sentence):
                yield page.page_number, piece


def _extract_title(pages: list[PageText]) -> FieldValue | None:
    for page in pages[:2]:
        match = SUBJECT_PATTERN.search(page.text)
        if match:
            subject = match.group("subject").strip().rstrip(".")
            return FieldValue(subject, page.page_number, match.group(0).strip(), label="subject line")
    if not pages:
        return None
    # Fallback: the first heading-like line that is not the issuing organisation.
    for line in pages[0].text.split("\n")[:12]:
        words = line.split()
        if 3 <= len(words) <= 16 and not ORG_KEYWORDS.search(line) and not REFERENCE_PATTERN.search(line):
            if line.isupper() or line.istitle() or not line.rstrip().endswith("."):
                return FieldValue(line.strip(), pages[0].page_number, line.strip(), label="first heading")
    return None


def _extract_organization(pages: list[PageText]) -> FieldValue | None:
    if not pages:
        return None
    for line in pages[0].text.split("\n")[:8]:
        if ORG_KEYWORDS.search(line) and len(line.split()) <= 14 and not SUBJECT_PATTERN.match(line):
            return FieldValue(line.strip().rstrip(",."), pages[0].page_number, line.strip(), label="letterhead")
    return None


def _is_heading(sentence: str) -> bool:
    """Short line without closing punctuation, e.g. "Eligibility" or "3. Hall Tickets"."""
    return len(sentence.split()) <= 6 and not sentence.rstrip().endswith((".", "?", "!", ";", "।"))


def _valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 10 <= len(digits) <= 13


def extract_fields(pages: list[PageText]) -> ExtractedFields:
    fields = ExtractedFields(
        title=_extract_title(pages),
        organization=_extract_organization(pages),
    )
    seen: set[tuple[str, str]] = set()
    max_sentences_per_list = 12

    def add(bucket_name: str, item: FieldValue, key: str) -> None:
        bucket: list[FieldValue] = getattr(fields, bucket_name)
        if bucket_name in ("eligibility", "action_items") and len(bucket) >= max_sentences_per_list:
            return
        marker = (bucket_name, key.lower())
        if marker not in seen:
            seen.add(marker)
            bucket.append(item)

    for page_number, sentence in _iter_sentences(pages):
        if fields.reference_number is None:
            ref = REFERENCE_PATTERN.search(sentence)
            if ref:
                fields.reference_number = FieldValue(ref.group("ref"), page_number, sentence, label="reference")

        for m in DATE_PATTERN.finditer(sentence):
            iso = parse_date(m)
            if iso:
                label = _date_label(sentence, m.start())
                add("dates", FieldValue(m.group(0), page_number, sentence, label, iso), f"{iso}|{sentence}")

        for m in MONEY_PATTERN.finditer(sentence):
            amount = m.group("a1") or m.group("a2")
            add("fees", FieldValue(m.group(0).strip(), page_number, sentence, "amount", parse_amount(amount)),
                f"{m.group(0)}|{sentence}")

        for m in EMAIL_PATTERN.finditer(sentence):
            add("contacts", FieldValue(m.group(0), page_number, sentence, "email"), m.group(0))
        for m in URL_PATTERN.finditer(sentence):
            url = m.group(0).rstrip(".")
            add("contacts", FieldValue(url, page_number, sentence, "website"), url)
        sentence_without_dates = DATE_PATTERN.sub(" ", sentence)
        for m in PHONE_PATTERN.finditer(sentence_without_dates):
            if _valid_phone(m.group(0)) and not MONEY_PATTERN.search(sentence_without_dates[max(0, m.start() - 5) : m.end()]):
                phone = m.group(0).strip()
                add("contacts", FieldValue(phone, page_number, sentence, "phone"), re.sub(r"\D", "", phone))

        if _is_heading(sentence) or SUBJECT_PATTERN.match(sentence):
            continue  # "3. Eligibility to Appear" names a section; it is not a requirement
        if ELIGIBILITY_CUES.search(sentence) and len(sentence.split()) >= 4:
            add("eligibility", FieldValue(sentence, page_number, sentence, "eligibility"), sentence)
        if ACTION_CUES.search(sentence) and len(sentence.split()) >= 5:
            add("action_items", FieldValue(sentence, page_number, sentence, "action"), sentence)

    return fields
