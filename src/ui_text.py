"""Safe rendering of untrusted document text as Streamlit Markdown."""

from __future__ import annotations

import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$])")


def md_escape(text: str) -> str:
    """Escape Markdown so document text cannot inject links, images, HTML or formatting."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def highlight(text: str, terms: tuple[str, ...]) -> str:
    """Escaped text with matched query terms in bold.

    Terms are stemmed tokens, so a prefix match also covers plurals ("fee" -> "fees").
    Single-character terms are skipped, otherwise "b" would bold every word starting with b.
    """
    escaped = md_escape(text)
    terms = tuple(t for t in terms if len(t) > 1)
    if not terms:
        return escaped
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    pattern = re.compile(rf"\b(?:{alternatives})\w*", re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", escaped)
