from src.chunker import Chunk
from src.config import Settings
from src.pipeline import process_pdf
from src.retrieval import BM25Index, tokenize


def _chunk(i: int, text: str, page: int) -> Chunk:
    return Chunk(f"C{i}", i - 1, text, page, page, len(text.split()))


CHUNKS = [
    _chunk(1, "The examination fee is Rs. 1,500 per semester, payable online.", 1),
    _chunk(2, "Students need 75% attendance to be eligible for the examinations.", 2),
    _chunk(3, "Hostel rooms are allotted on a first come first served basis.", 3),
    _chunk(4, "Contact the controller at exams@example.edu for any queries.", 3),
]


def test_tokenize_removes_stopwords_and_folds_plurals():
    assert tokenize("What are the Fees for the Examinations?") == ["fee", "examination"]


def test_tokenize_keeps_devanagari_words_whole():
    assert tokenize("परीक्षा शुल्क") == ["परीक्षा", "शुल्क"]


def test_fee_question_ranks_fee_chunk_first():
    results = BM25Index(CHUNKS).search("How much is the exam fee?", top_k=3)
    assert results[0].chunk.chunk_id == "C1"
    assert results[0].rank == 1
    assert "fee" in results[0].matched_terms
    assert results[0].relevance_label == "High"


def test_query_expansion_finds_synonyms():
    # "charges" is not in the text; expansion maps it to "fee"/"amount".
    results = BM25Index(CHUNKS).search("examination charges", top_k=1)
    assert results[0].chunk.chunk_id == "C1"


def test_eligibility_question():
    results = BM25Index(CHUNKS).search("Who is eligible? What attendance is required?")
    assert results[0].chunk.chunk_id == "C2"


def test_unrelated_question_returns_nothing():
    assert BM25Index(CHUNKS).search("What is the capital of Australia?") == []


def test_top_k_limit_and_scores_descending():
    results = BM25Index(CHUNKS).search("examination fee attendance contact hostel", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_empty_index_and_empty_query():
    assert BM25Index([]).search("fee") == []
    assert BM25Index(CHUNKS).search("the of and") == []


def test_end_to_end_pipeline_retrieves_with_page_numbers(circular_pdf):
    doc = process_pdf(circular_pdf, "circular.pdf", Settings(chunk_words=60, chunk_overlap_words=10))
    assert len(doc.chunks) > 1
    results = doc.index.search("last date for fee payment without fine")
    assert results
    top = results[0]
    assert "5 November 2026" in top.chunk.text
    assert top.chunk.page_end == 2
