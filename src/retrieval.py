"""Keyword retrieval over document chunks with BM25 (rank-bm25).

BM25Plus is used instead of BM25Okapi because Okapi's IDF goes negative when a
term appears in most chunks, which happens constantly in short one- or two-chunk
notices. BM25Plus IDF is always positive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Plus

from src.chunker import Chunk

# Latin word characters plus the main Indic Unicode blocks (Devanagari to Malayalam),
# so that vowel signs are not treated as word separators.
_TOKEN = re.compile(r"[\wऀ-ൿ]+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be by can could did do does for from had has have how i if in into is it its
    me my of on or our shall should so than that the their them then there these they this those to
    under was we were what when where which who whom whose why will with would you your about any all
    also been being both each more most other some such only own same very just please tell give much many another number""".split()
)

# Small, transparent query expansion for vocabulary common in Indian notices.
QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "fee": ("fee", "charge", "amount", "rs", "inr", "payment"),
    "cost": ("fee", "charge", "amount"),
    "charge": ("fee", "amount"),
    "deadline": ("last", "date", "due", "deadline"),
    "due": ("last", "date", "deadline"),
    "eligible": ("eligibility", "eligible", "criteria"),
    "eligibility": ("eligibility", "eligible", "criteria"),
    "contact": ("contact", "email", "phone", "helpline"),
    "exam": ("exam", "examination"),
    "examination": ("exam", "examination"),
    "start": ("start", "commence", "begin"),
    "begin": ("start", "commence", "begin"),
    "commence": ("start", "commence", "begin"),
    "apply": ("apply", "application", "register", "registration"),
    "application": ("apply", "application"),
    "register": ("register", "registration", "apply"),
    "registration": ("register", "registration"),
    "document": ("document", "certificate", "proof"),
    "limit": ("limit", "exceed", "maximum", "ceiling"),
    "required": ("required", "must", "minimum", "mandatory"),
    "require": ("required", "must", "minimum", "mandatory"),
    "requirement": ("required", "must", "minimum", "mandatory"),
}


def term_coverage(query_terms: list[str], token_set: set[str], synonym_weight: float = 1.0) -> float:
    """Fraction of query terms present in `token_set`.

    A term matched only through a listed synonym counts `synonym_weight` instead of 1,
    since a synonym is weaker evidence than the user's own word.
    """
    unique = list(dict.fromkeys(query_terms))
    if not unique:
        return 0.0
    hits = 0.0
    for term in unique:
        if term in token_set:
            hits += 1
        elif token_set.intersection(QUERY_EXPANSIONS.get(term, ())):
            hits += synonym_weight
    return hits / len(unique)


def _stem(token: str) -> str:
    # Deliberately tiny plural folding: "fees" -> "fee", "dates" -> "date".
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = (t.lower() for t in _TOKEN.findall(text))
    return [_stem(t) for t in tokens if t not in STOPWORDS]


def expand_query(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        for extra in QUERY_EXPANSIONS.get(token, ()):
            if extra not in expanded:
                expanded.append(extra)
    return expanded


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    matched_terms: tuple[str, ...]
    # Fraction of the user's own query terms found in this chunk (directly or via a synonym).
    coverage: float

    @property
    def relevance_label(self) -> str:
        if self.coverage >= 0.6:
            return "High"
        if self.coverage >= 0.3:
            return "Medium"
        return "Low"


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._token_sets = [set(t) for t in self._tokenized]
        # rank-bm25 divides by corpus size, so an empty corpus has no index.
        self._bm25 = BM25Plus(self._tokenized) if chunks else None

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        """Return up to `top_k` chunks that share at least one term with the query."""
        query_terms = tokenize(query)
        if not query_terms or self._bm25 is None:
            return []
        expanded = expand_query(query_terms)
        # get_scores sums over query tokens, so repeating the user's own terms gives them
        # twice the weight of synonyms added by expansion.
        scores = self._bm25.get_scores(query_terms + expanded)
        candidates = []
        for i, score in enumerate(scores):
            matched = tuple(t for t in dict.fromkeys(expanded) if t in self._token_sets[i])
            if not matched:
                continue  # BM25Plus gives every chunk a small baseline score; ignore non-matches
            candidates.append((float(score), i, matched, term_coverage(query_terms, self._token_sets[i])))

        candidates.sort(key=lambda c: (-c[0], c[1]))
        return [
            SearchResult(chunk=self.chunks[i], score=score, rank=rank + 1, matched_terms=matched, coverage=coverage)
            for rank, (score, i, matched, coverage) in enumerate(candidates[:top_k])
        ]
