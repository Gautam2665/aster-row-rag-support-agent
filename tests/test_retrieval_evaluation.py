from pathlib import Path
import pytest

from src.ingestion import ingest_kb_directory
from src.retrieval import KBVectorStore
from src.models import KBChunk, DocumentMetadata
from src.retrieval_evaluation import (
    RetrievalEvalCase,
    RetrievalEvaluator,
    calculate_precision_at_k,
    calculate_recall_at_k,
    DEFAULT_RETRIEVAL_EVAL_DATASET,
)

KB_DIR = Path("knowledge-base")


@pytest.fixture(scope="module")
def indexed_vector_store():
    """Fixture that ingests knowledge-base/ and indexes chunks into ChromaDB."""
    chunks = ingest_kb_directory(KB_DIR)
    store = KBVectorStore(collection_name="test_eval_kb_store")
    store.clear()
    store.index_chunks(chunks)
    return store


def make_mock_chunk(chunk_id: str, filename: str) -> KBChunk:
    """Helper to construct mock KBChunk for metric testing."""
    meta = DocumentMetadata(
        document_id=filename,
        title="Test Title",
        status="active",
        audience="customer",
        policy_authority="official",
        customer_answering=True,
    )
    return KBChunk(
        chunk_id=chunk_id,
        filename=filename,
        heading="Sample Heading",
        text="Sample text",
        metadata=meta,
    )


def test_perfect_retrieval_metrics():
    """Test 100% precision and 100% recall calculation."""
    retrieved = [
        make_mock_chunk("doc1#0", "doc1.md"),
        make_mock_chunk("doc1#1", "doc1.md"),
    ]
    expected = {"doc1.md"}

    prec = calculate_precision_at_k(retrieved, expected, k=2)
    rec = calculate_recall_at_k(retrieved, expected, k=2)

    assert prec == 1.0
    assert rec == 1.0


def test_partial_and_irrelevant_retrieval_metrics():
    """Test partial precision and recall when retrieved list contains irrelevant chunks."""
    retrieved = [
        make_mock_chunk("doc1#0", "doc1.md"),
        make_mock_chunk("doc2#0", "doc2.md"),
        make_mock_chunk("doc3#0", "doc3.md"),
        make_mock_chunk("doc4#0", "doc4.md"),
    ]
    expected = {"doc1.md"}

    prec = calculate_precision_at_k(retrieved, expected, k=4)
    rec = calculate_recall_at_k(retrieved, expected, k=4)

    assert prec == 0.25  # 1 hit out of 4 retrieved chunks
    assert rec == 1.0   # 1 expected document retrieved out of 1 total expected


def test_zero_retrieval_metrics():
    """Test 0% precision and 0% recall when no matching chunks are retrieved."""
    retrieved = [
        make_mock_chunk("doc2#0", "doc2.md"),
        make_mock_chunk("doc3#0", "doc3.md"),
    ]
    expected = {"doc1.md"}

    prec = calculate_precision_at_k(retrieved, expected, k=2)
    rec = calculate_recall_at_k(retrieved, expected, k=2)

    assert prec == 0.0
    assert rec == 0.0


def test_k_larger_than_retrieved_results():
    """Test calculation when K is larger than the number of available retrieved chunks."""
    retrieved = [
        make_mock_chunk("doc1#0", "doc1.md"),
    ]
    expected = {"doc1.md"}

    prec = calculate_precision_at_k(retrieved, expected, k=10)
    rec = calculate_recall_at_k(retrieved, expected, k=10)

    assert prec == 1.0
    assert rec == 1.0


def test_empty_expected_relevance_set():
    """Test behavior when expected relevant set is empty."""
    retrieved = [
        make_mock_chunk("doc1#0", "doc1.md"),
    ]
    expected = set()

    prec = calculate_precision_at_k(retrieved, expected, k=5)
    rec = calculate_recall_at_k(retrieved, expected, k=5)

    assert prec == 0.0
    assert rec == 0.0


def test_deterministic_repeatability(indexed_vector_store):
    """Verify that running RetrievalEvaluator twice produces identical precision/recall results."""
    evaluator = RetrievalEvaluator(indexed_vector_store)
    
    results_run_1 = evaluator.evaluate_dataset()
    results_run_2 = evaluator.evaluate_dataset()

    assert len(results_run_1) == len(results_run_2)
    for r1, r2 in zip(results_run_1, results_run_2):
        assert r1.case_id == r2.case_id
        assert r1.precision == r2.precision
        assert r1.recall == r2.recall
        assert r1.retrieved_filenames == r2.retrieved_filenames


def test_dataset_execution_no_llm_calls(indexed_vector_store):
    """Verify default retrieval evaluation dataset executes without throwing or requiring LLM/API keys."""
    evaluator = RetrievalEvaluator(indexed_vector_store)
    results = evaluator.evaluate_dataset()

    assert len(results) == len(DEFAULT_RETRIEVAL_EVAL_DATASET)
    
    # Check standard return policy case
    std_res = next(r for r in results if r.case_id == "eval-standard-returns")
    assert std_res.recall == 1.0
    assert "01-returns-policy-current.md" in std_res.retrieved_filenames

    # Check unsupported case shows low/zero recall
    unsupported_res = next(r for r in results if r.case_id == "eval-unsupported-lost-item")
    assert unsupported_res.recall == 0.0
    assert unsupported_res.precision == 0.0
