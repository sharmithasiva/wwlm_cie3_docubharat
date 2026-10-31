import pytest

from src.chunker import chunk_pages
from src.pdf_processor import PageText
from src.text_cleaner import clean_text, remove_repeated_lines, split_sentences


def _numbered_words(start: int, count: int) -> str:
    return " ".join(f"w{i}" for i in range(start, start + count))


def test_clean_text_joins_hyphenation_and_wrapped_lines():
    raw = "The exami-\nnation will be held\nin the main hall.\n\n\n12\n"
    assert clean_text(raw) == "The examination will be held in the main hall."


def test_clean_text_keeps_list_items_and_table_numbers():
    raw = "Fees\n1. Tuition\n2. Hostel\nAmount\n500\nEnd of table."
    cleaned = clean_text(raw)
    assert "1. Tuition\n2. Hostel" in cleaned
    assert "500" in cleaned  # a number mid-page is content, not a page number


def test_clean_text_drops_page_footer():
    assert clean_text("Notice body text.\nPage 2 of 5") == "Notice body text."


def test_clean_text_joins_lines_broken_after_function_words_and_abbreviations():
    raw = "The last date for applications is\n25 June 2026.\nThe total is Rs.\n1,12,000 per semester."
    assert clean_text(raw) == "The last date for applications is 25 June 2026.\nThe total is Rs. 1,12,000 per semester."


def test_subject_line_does_not_absorb_next_line():
    raw = "Subject: Hostel Allotment for the year\nStudents must apply online."
    assert clean_text(raw) == "Subject: Hostel Allotment for the year\nStudents must apply online."


def test_remove_repeated_lines_strips_running_footer():
    pages = ["Body one.\nCONFIDENTIAL FOOTER", "Body two.\nCONFIDENTIAL FOOTER", "Body three."]
    assert remove_repeated_lines(pages) == ["Body one.", "Body two.", "Body three."]


def test_split_sentences_respects_abbreviations():
    text = "Fee is Rs. 500 per paper. Candidates need a B.E. degree. Done!"
    assert split_sentences(text) == ["Fee is Rs. 500 per paper.", "Candidates need a B.E. degree.", "Done!"]


def test_short_document_is_one_chunk_with_page_range():
    pages = [PageText(1, "Alpha paragraph."), PageText(2, "Beta paragraph.")]
    chunks = chunk_pages(pages, chunk_words=100, overlap_words=10)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert chunks[0].page_label == "pp. 1–2"
    assert chunks[0].chunk_id == "C1"


def test_long_text_is_windowed_with_overlap_and_full_coverage():
    pages = [PageText(1, _numbered_words(0, 250))]
    chunks = chunk_pages(pages, chunk_words=100, overlap_words=20)
    assert all(c.word_count <= 100 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == {f"w{i}" for i in range(250)}
    assert [c.chunk_id for c in chunks] == [f"C{i + 1}" for i in range(len(chunks))]


def test_chunks_prefer_paragraph_breaks():
    text = _numbered_words(0, 70) + "\n\n" + _numbered_words(70, 70)
    chunks = chunk_pages([PageText(1, text)], chunk_words=100, overlap_words=0)
    assert chunks[0].text.split()[-1] == "w69"
    assert chunks[1].text.split()[0] == "w70"


def test_page_mapping_across_pages():
    pages = [PageText(p, _numbered_words(p * 100, 60)) for p in (1, 2, 3)]
    chunks = chunk_pages(pages, chunk_words=60, overlap_words=0)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([PageText(1, ""), PageText(2, "   ")]) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "x")], chunk_words=0)
