"""Backend-independent translation safeguards (kept after the Bhashini client was removed)."""

import pytest

from src.translation_checks import missing_numbers, segment


@pytest.mark.parametrize(
    "source, translated, missing",
    [
        ("Pay Rs. 1,800 by 7 November 2026.", "7 नवंबर 2026 तक 1,800 रुपये का भुगतान करें।", []),
        ("Pay Rs. 1,800 by 7 November 2026.", "७ नवंबर २०२६ तक १८०० रुपये का भुगतान करें।", []),  # Devanagari digits
        ("Attendance of 75% is required.", "उपस्थिति आवश्यक है।", ["75"]),
        ("Fee Rs. 2,50,000.", "शुल्क 250000 रुपये।", []),  # Indian grouping vs none
        # Real IndicTrans2 Tamil output: the date survives but is written "7,2026".
        ("The last date is 7 November 2026.", "கடைசி தேதி நவம்பர் 7,2026 ஆகும்.", []),
        ("Pay Rs. 1,800.", "ரூ. 1,600 செலுத்தவும்.", ["1,800"]),  # a changed amount is still caught
    ],
)
def test_missing_numbers(source, translated, missing):
    assert missing_numbers(source, translated) == missing


def test_segment_splits_sentences_and_long_text():
    assert segment("First sentence. Second sentence.") == ["First sentence.", "Second sentence."]
    pieces = segment("word " * 200)
    assert all(len(p) <= 400 for p in pieces) and len(pieces) == 3


def test_segment_rejects_overlong_input():
    with pytest.raises(ValueError):
        segment("Sentence. " * 50)
