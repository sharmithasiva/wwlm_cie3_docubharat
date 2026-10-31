"""Retrieval-based question answering.

Phase 1 is purely extractive: BM25 finds candidate chunks, then the single best
sentence is quoted verbatim as the answer. If no sentence
covers enough of the question, the engine says the information was not found
instead of guessing. No text is generated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.field_extractor import DATE_PATTERN, ELIGIBILITY_CUES, EMAIL_PATTERN, MONEY_PATTERN, PHONE_PATTERN
from src.pdf_processor import PageText
from src.retrieval import BM25Index, SearchResult, term_coverage, tokenize
from src.text_cleaner import split_sentences

NOT_FOUND_MESSAGE = "This information was not found in the document."

# A sentence must contain at least this share of the question's content words.
MIN_SENTENCE_COVERAGE = 0.6
# Credit for a question word matched only through a synonym ("start" -> "commence").
SYNONYM_WEIGHT = 0.75

_QUESTION_TYPES = {
    "date": re.compile(r"\b(when|date|deadline|last day|by when|schedule|commence|start|begin|till|until)\b", re.I),
    "amount": re.compile(r"\b(fee|fees|cost|amount|how much|charge|charges|price|fine|rs|rupees?|stipend|allowance|rent|pay)\b", re.I),
    "contact": re.compile(r"\b(contact|email|e-mail|phone|call|helpline|reach|website)\b", re.I),
    "eligibility": re.compile(
        r"\b(eligib\w*|who can|criteria|qualif\w*|minimum|requirements?|required|attendance|marks|percentage)\b", re.I
    ),
}


def question_type(question: str) -> str:
    for name, pattern in _QUESTION_TYPES.items():
        if pattern.search(question):
            return name
    return "general"


def _matches_type(sentence: str, qtype: str) -> bool:
    if qtype == "date":
        return bool(DATE_PATTERN.search(sentence))
    if qtype == "amount":
        return bool(MONEY_PATTERN.search(sentence))
    if qtype == "contact":
        return bool(EMAIL_PATTERN.search(sentence) or PHONE_PATTERN.search(sentence))
    if qtype == "eligibility":
        return bool(ELIGIBILITY_CUES.search(sentence))
    return False


@dataclass
class AnswerSentence:
    text: str
    chunk_id: str
    page: int
    coverage: float
    score: float


@dataclass
class Citation:
    """A quote the LLM says supports its answer, checked against the document."""

    passage_id: str
    quote: str
    page: int | None
    verified: bool  # True only if the quote appears verbatim in that passage


@dataclass
class QAResult:
    question: str
    found: bool
    answer: str
    method: str  # "extractive" or "llm"
    question_type: str
    sentences: list[AnswerSentence] = field(default_factory=list)
    evidence: list[SearchResult] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None
    # Note shown with the answer, e.g. why the app fell back from the LLM to extraction.
    notice: str | None = None

    @property
    def cited_pages(self) -> list[int]:
        pages = {s.page for s in self.sentences} | {c.page for c in self.citations if c.verified and c.page}
        return sorted(pages)

    @property
    def grounded(self) -> bool:
        """Extractive answers are quotes; LLM answers need at least one verified citation."""
        if self.method == "extractive":
            return self.found
        return self.found and any(c.verified for c in self.citations)


def _squash(text: str) -> str:
    return " ".join(text.split())


def locate_page(sentence: str, result: SearchResult, pages: list[PageText] | None) -> int:
    """Exact page of a sentence; chunks can span pages, so check the page texts."""
    if pages and result.chunk.page_start != result.chunk.page_end:
        needle = _squash(sentence)
        for page in pages:
            if result.chunk.page_start <= page.page_number <= result.chunk.page_end and needle in _squash(page.text):
                return page.page_number
    return result.chunk.page_start


def _is_heading(sentence: str) -> bool:
    return (
        len(sentence.split()) <= 6
        and not any(ch.isdigit() for ch in sentence)
        and not sentence.endswith((".", "?", "!", ";"))
    )


def _answer_units(text: str) -> list[tuple[str, str]]:
    """(display text, text to locate on the page) for each sentence of a chunk.

    A section heading such as "Scholarship Amount" is merged into the sentence after
    it, so the heading's words count towards that sentence instead of the heading
    winning on its own.
    """
    sentences = split_sentences(text)
    units = []
    for i, sentence in enumerate(sentences):
        if _is_heading(sentence) and i + 1 < len(sentences) and not _is_heading(sentences[i + 1]):
            continue  # merged into the next sentence
        if i > 0 and _is_heading(sentences[i - 1]) and not _is_heading(sentence):
            units.append((f"{sentences[i - 1].rstrip(':')}: {sentence}", sentence))
        else:
            units.append((sentence, sentence))
    return units


def rank_sentences(question: str, results: list[SearchResult], pages: list[PageText] | None = None) -> list[AnswerSentence]:
    query_terms = list(dict.fromkeys(tokenize(question)))
    qtype = question_type(question)
    ranked: list[AnswerSentence] = []
    seen: set[str] = set()
    full_match_seen = False
    for result in results:
        for display, locate in _answer_units(result.chunk.text):
            if display in seen:  # overlapping chunks repeat sentences
                continue
            seen.add(display)
            tokens = set(tokenize(display))
            coverage = term_coverage(query_terms, tokens, synonym_weight=SYNONYM_WEIGHT)
            if coverage == 0:
                continue
            # Density favours short, specific sentences over long ones that mention everything.
            density = coverage * len(query_terms) / len(tokens)
            # Notices introduce an event before referring back to it, so the first sentence
            # in the top chunk that matches every question word gets a small bonus.
            first_full_match = coverage == 1.0 and not full_match_seen
            full_match_seen = full_match_seen or coverage == 1.0
            score = (
                2.0 * coverage
                + 0.6 * _matches_type(display, qtype)
                + 0.5 * density
                + 0.2 * first_full_match
                + 0.2 / result.rank
            )
            ranked.append(
                AnswerSentence(display, result.chunk.chunk_id, locate_page(locate, result, pages), coverage, score)
            )
    ranked.sort(key=lambda s: -s.score)
    return ranked


def answer_question(
    question: str, index: BM25Index, top_k: int = 4, pages: list[PageText] | None = None
) -> QAResult:
    question = question.strip()
    qtype = question_type(question)
    if not tokenize(question):
        return QAResult(question, False, "Please ask a more specific question about the document.", "extractive", qtype)

    evidence = index.search(question, top_k=top_k)
    ranked = rank_sentences(question, evidence, pages)
    if not ranked or ranked[0].coverage < MIN_SENTENCE_COVERAGE:
        return QAResult(question, False, NOT_FOUND_MESSAGE, "extractive", qtype, evidence=evidence)

    best = ranked[0]
    return QAResult(
        question=question,
        found=True,
        answer=best.text,
        method="extractive",
        question_type=qtype,
        sentences=[best],
        evidence=evidence,
    )
