from pathlib import Path
import pytest

from src.ingestion import parse_markdown_sections
from src.models import KBChunk, DocumentMetadata
from src.retrieval import KBVectorStore
from src.retrieval_policy import evaluate_retrieval_sufficiency, RetrievalSufficiency
from src.agent import SupportAgent
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider

KB_DIR = Path("knowledge-base")


def make_sample_chunk() -> KBChunk:
    meta = DocumentMetadata(
        document_id="01-returns-policy-current.md",
        title="Current Returns Policy",
        status="active",
        audience="customer",
        policy_authority="official",
        customer_answering=True,
    )
    return KBChunk(
        chunk_id="01-returns-policy-current.md#0",
        filename="01-returns-policy-current.md",
        heading="Sample Header",
        text="Standard return window is 30 days.",
        metadata=meta,
    )


def test_sufficient_retrieval_with_eligible_evidence():
    """Test that presence of customer-eligible evidence chunks evaluates as sufficient."""
    chunks = [make_sample_chunk()]
    result = evaluate_retrieval_sufficiency(chunks, user_query="What is the return policy?")

    assert isinstance(result, RetrievalSufficiency)
    assert result.sufficient is True
    assert result.evidence_count == 1
    assert result.failure_category is None
    assert "Retrieved 1 customer-eligible evidence chunks" in result.reason


def test_zero_eligible_evidence_policy():
    """Test that zero eligible evidence chunks evaluates as insufficient with RETRIEVAL_FAILURE."""
    result = evaluate_retrieval_sufficiency([], user_query="Some unknown query")

    assert result.sufficient is False
    assert result.evidence_count == 0
    assert result.failure_category == "RETRIEVAL_FAILURE"
    assert "Zero customer-eligible evidence chunks" in result.reason


def test_unsupported_policy_query_insufficiency():
    """Test that unsupported policy queries evaluate as insufficient despite chunk presence."""
    chunks = [make_sample_chunk()]
    result = evaluate_retrieval_sufficiency(
        chunks, user_query="Does Aster & Row offer an unconditional replacement for lost items?"
    )

    assert result.sufficient is False
    assert result.failure_category == "RETRIEVAL_FAILURE"
    assert "lacks authoritative evidence" in result.reason


def test_no_arbitrary_distance_threshold():
    """Verify policy does not use vector distance threshold to reject valid chunks."""
    # Create chunk with high arbitrary distance
    chunks = [make_sample_chunk()]
    result = evaluate_retrieval_sufficiency(chunks, user_query="How long do I have to return an item?")

    # Should be sufficient based on chunk metadata, ignoring distance
    assert result.sufficient is True


def test_insufficient_retrieval_causes_safe_abstention_handoff():
    """Test that SupportAgent safely abstains and recommends handoff when retrieval is insufficient."""
    from src.ingestion import ingest_kb_directory
    chunks = ingest_kb_directory(KB_DIR)
    store = KBVectorStore(collection_name="test_policy_agent_kb")
    store.clear()
    store.index_chunks(chunks)

    agent = SupportAgent(
        vector_store=store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        llm_provider=MockLLMProvider(),
    )

    # Query unsupported policy
    state = agent.process_turn("Does Aster & Row offer a 10-year unconditional replacement warranty for lost items?")

    assert state.handoff_recommended is True
    assert len(state.observations) > 0
    obs = state.observations[0]
    assert obs.failure_category == "RETRIEVAL_FAILURE"
    assert obs.handoff_recommended is True


def test_no_retry_loop_on_insufficient_retrieval():
    """Verify that retrieval failure does not trigger an infinite planner retry loop."""
    from src.ingestion import ingest_kb_directory
    chunks = ingest_kb_directory(KB_DIR)
    store = KBVectorStore(collection_name="test_retry_loop_kb")
    store.clear()
    store.index_chunks(chunks)

    agent = SupportAgent(
        vector_store=store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        llm_provider=MockLLMProvider(),
    )

    state = agent.process_turn("Does Aster & Row offer an unconditional replacement for lost items in Atlantis?")

    # Iteration count must be <= max_iterations (3)
    assert state.iterations <= 3
    assert state.handoff_recommended is True


def test_pii_excluded_from_policy_result():
    """Verify RetrievalSufficiency dictionary representation contains zero PII."""
    res = RetrievalSufficiency(
        sufficient=False,
        reason="Zero evidence chunks",
        evidence_count=0,
        failure_category="RETRIEVAL_FAILURE",
    )
    d = res.to_dict()

    sensitive_keys = ["email", "address", "risk_score", "warehouse_note", "customer", "order"]
    for key in sensitive_keys:
        assert key not in d
