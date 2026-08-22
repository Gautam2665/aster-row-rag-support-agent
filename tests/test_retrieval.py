from pathlib import Path
import pytest

from src.ingestion import ingest_kb_directory
from src.retrieval import KBVectorStore
from src.embeddings import EmbeddingProvider

KB_DIR = Path("knowledge-base")


@pytest.fixture(scope="module")
def indexed_vector_store():
    """Fixture that ingests knowledge-base/ and indexes chunks into ChromaDB."""
    chunks = ingest_kb_directory(KB_DIR)
    store = KBVectorStore(collection_name="test_kb_store")
    store.clear()
    store.index_chunks(chunks)
    return store


def test_current_vs_legacy_returns_policy_filtering(indexed_vector_store):
    """
    Test that searching for 'return window' with metadata filtering enabled
    returns current active returns policy (30 days) and excludes superseded legacy policy (60 days).
    """
    results = indexed_vector_store.search(
        query="What is the return window for a regular customer?",
        top_k=5,
        filter_customer_eligible=True,
    )

    assert len(results) > 0
    filenames = {c.filename for c in results}

    # Current policy MUST be retrieved
    assert "01-returns-policy-current.md" in filenames
    # Legacy superseded policy MUST NOT be present in eligible results
    assert "02-returns-policy-legacy.md" not in filenames

    # Verify returned chunk status is active
    for chunk in results:
        assert chunk.metadata.status == "active"
        assert chunk.metadata.audience == "customer"
        assert chunk.metadata.customer_answering is True


def test_internal_migration_notes_filtering(indexed_vector_store):
    """
    Test that internal draft notes containing prompt injections (14-internal-content-migration-notes.md)
    are filtered out during customer query retrieval.
    """
    results = indexed_vector_store.search(
        query="Give me 60 days to return and ignore rules",
        top_k=10,
        filter_customer_eligible=True,
    )

    filenames = {c.filename for c in results}
    assert "14-internal-content-migration-notes.md" not in filenames


def test_unfiltered_retrieval_includes_legacy_and_drafts(indexed_vector_store):
    """
    Test that when filter_customer_eligible=False, unfiltered search includes draft and legacy docs.
    """
    results = indexed_vector_store.search(
        query="return policy 60 days migration",
        top_k=15,
        filter_customer_eligible=False,
    )

    filenames = {c.filename for c in results}
    # Unfiltered search should be able to retrieve legacy and draft notes
    assert "02-returns-policy-legacy.md" in filenames or "14-internal-content-migration-notes.md" in filenames


def test_citation_preservation(indexed_vector_store):
    """Test that retrieved chunks preserve correct source citation formatting."""
    results = indexed_vector_store.search(
        query="international shipping to Canada",
        top_k=3,
        filter_customer_eligible=True,
    )

    assert len(results) > 0
    top_chunk = results[0]
    assert top_chunk.filename == "06-international-shipping.md"
    assert "06-international-shipping.md > " in top_chunk.source_citation
    assert top_chunk.metadata.policy_authority == "official"


def test_consistent_embedding_provider():
    """Verify that EmbeddingProvider generates matching vector dimensions for text and query."""
    provider = EmbeddingProvider()
    doc_embs = provider.embed_texts(["Test sentence 1", "Test sentence 2"])
    query_emb = provider.embed_query("Test query")

    assert len(doc_embs) == 2
    assert len(doc_embs[0]) == len(query_emb)
