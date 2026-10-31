"""Backend-independent helpers for machine translation of answers.

No translation backend is currently wired into the app (the Bhashini client was
removed at the user's request). These helpers are kept because any future backend,
such as an AI4Bharat IndicTrans2 model, needs the same safeguards:
sentence-sized inputs, and a check that dates and amounts survive translation.
"""

from __future__ import annotations

import re
import unicodedata

from src.text_cleaner import split_sentences

MAX_SEGMENT_CHARS = 400  # IndicTrans2 is trained on sentence-level inputs
MAX_SEGMENTS = 20

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _ascii_digits(text: str) -> str:
    """Map digits from any script (e.g. Devanagari ०-९) to 0-9."""
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() else ch for ch in text)


def missing_numbers(source: str, translated: str) -> list[str]:
    """Numbers (dates, amounts, percentages) in the source that do not survive translation."""
    target_numbers: set[str] = set()
    for n in _NUMBER.findall(_ascii_digits(translated)):
        # "1,800" may stand for 1800, but a model can also write a date as "7,2026",
        # so count both the joined number and each comma-separated part as present.
        target_numbers.add(n.replace(",", ""))
        target_numbers.update(n.split(","))
    missing = []
    for number in _NUMBER.findall(_ascii_digits(source)):
        if number.replace(",", "") not in target_numbers and number not in missing:
            missing.append(number)
    return missing


def segment(text: str) -> list[str]:
    """Split text into sentence-sized pieces; raises ValueError if there are too many."""
    pieces: list[str] = []
    for sentence in split_sentences(text) or [text]:
        while len(sentence) > MAX_SEGMENT_CHARS:
            cut = sentence.rfind(" ", 0, MAX_SEGMENT_CHARS)
            cut = cut if cut > 0 else MAX_SEGMENT_CHARS
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    if len(pieces) > MAX_SEGMENTS:
        raise ValueError("Text is too long to translate in one request.")
    return pieces
