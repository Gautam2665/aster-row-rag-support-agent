from pathlib import Path
import pytest

from src.ingestion import ingest_kb_directory
from src.retrieval import KBVectorStore
from src.retrieval_trace import RetrievalDiagnostic
from src.agent import SupportAgent
from src.context import ContextBuilder
from src.tools.order_lookup import OrderLookupTool

KB_DIR = Path("knowledge-base")


@pytest.fixture(scope="module")
def indexed_vector_store():
    """Fixture that ingests knowledge-base/ and indexes chunks into ChromaDB."""
    chunks = ingest_kb_directory(KB_DIR)
    store = KBVectorStore(collection_name="test_diag_kb_store")
    store.clear()
    store.index_chunks(chunks)
    return store


@pytest.fixture(scope="module")
def support_agent(indexed_vector_store):
    """Fixture initializing SupportAgent with test vector store."""
    order_tool = OrderLookupTool(data_path=Path("data/orders.json"))
    return SupportAgent(vector_store=indexed_vector_store, order_tool=order_tool)


def test_normal_successful_retrieval_diagnostic(indexed_vector_store):
    """Test that search_with_diagnostics returns valid chunks and populated RetrievalDiagnostic."""
    chunks, diag = indexed_vector_store.search_with_diagnostics(
        query="What is the return window for a regular customer?",
        top_k=5,
        filter_customer_eligible=True,
    )

    assert len(chunks) > 0
    assert isinstance(diag, RetrievalDiagnostic)
    assert diag.retrieval_query == "What is the return window for a regular customer?"
    assert diag.has_usable_evidence is True
    assert diag.num_eligible_chunks == len(chunks)
    assert "01-returns-policy-current.md" in diag.retrieved_filenames
    assert len(diag.distances) == len(chunks)
    assert all(isinstance(d, float) for d in diag.distances)
    assert diag.failure_classification is None
    assert diag.filter_applied is not None


def test_empty_retrieval_diagnostic():
    """Test that querying an empty vector store produces an empty RetrievalDiagnostic."""
    empty_store = KBVectorStore(collection_name="test_empty_diag_kb")
    empty_store.clear()

    chunks, diag = empty_store.search_with_diagnostics(
        query="Random nonexistent term xyz123999",
        top_k=5,
        filter_customer_eligible=True,
    )

    assert len(chunks) == 0
    assert diag.has_usable_evidence is False
    assert diag.num_eligible_chunks == 0
    assert diag.failure_classification == "RETRIEVAL_FAILURE"
    assert diag.retrieved_chunk_ids == []
    assert diag.retrieved_filenames == []


def test_contextualized_multi_turn_retrieval_diagnostic(support_agent):
    """
    Test multi-turn retrieval turn 2 where QueryContextualizer reformulates query.
    Verify diagnostic captures contextualized retrieval_query while state.user_query is preserved.
    """
    session_id = "test_multi_turn_diag_session"
    
    # Turn 1
    support_agent.process_turn("Do you ship internationally?", session_id=session_id)
    
    # Turn 2
    state2 = support_agent.process_turn("What about Canada?", session_id=session_id)

    assert state2.user_query == "What about Canada?"
    assert "Do you ship internationally?" in state2.retrieval_query
    assert "Canada" in state2.retrieval_query

    assert len(state2.retrieval_diagnostics) > 0
    diag = state2.retrieval_diagnostics[-1]
    assert diag.retrieval_query == state2.retrieval_query
    assert "06-international-shipping.md" in diag.retrieved_filenames
    assert diag.has_usable_evidence is True


def test_metadata_filtering_in_diagnostic(indexed_vector_store):
    """Test that customer eligibility filter is recorded in diagnostic and excludes superseded docs."""
    chunks, diag = indexed_vector_store.search_with_diagnostics(
        query="What is the return policy?",
        top_k=10,
        filter_customer_eligible=True,
    )

    assert diag.filter_applied["$and"][0] == {"status": {"$eq": "active"}}
    assert "02-returns-policy-legacy.md" not in diag.retrieved_filenames
    assert "01-returns-policy-current.md" in diag.retrieved_filenames


def test_diagnostic_does_not_contain_pii(support_agent):
    """Verify RetrievalDiagnostic dictionary export contains zero PII or sensitive keys."""
    state = support_agent.process_turn("Check my order ORD-1007", session_id="pii_order_session")
    state_pol = support_agent.process_turn("What is the return window for a regular customer?", session_id="pii_policy_session")
    
    assert len(state_pol.retrieval_diagnostics) > 0

    diag_dict = state_pol.retrieval_diagnostics[0].to_dict()
    sensitive_keys = ["email", "address", "risk_score", "warehouse_note", "customer", "order"]
    
    for key in sensitive_keys:
        assert key not in diag_dict
    
    # Check JSON serializability
    diag_str = str(diag_dict).lower()
    assert "jane.doe@example.com" not in diag_str
    assert "warehouse_note" not in diag_str


def test_diagnostic_does_not_enter_llm_prompt(support_agent):
    """Verify that ContextBuilder output excludes retrieval diagnostic structures."""
    state = support_agent.process_turn("How long is the return policy?", session_id="prompt_test_session")
    
    assert len(state.retrieval_diagnostics) > 0
    diag = state.retrieval_diagnostics[0]

    payload = ContextBuilder.build_prompt_with_context(
        user_question=state.user_query,
        evidence_chunks=state.evidence_chunks,
        history_turns=state.history_turns,
    )

    full_prompt = payload.full_prompt
    assert "RetrievalDiagnostic" not in full_prompt
    assert "num_candidates_returned" not in full_prompt
    assert "num_eligible_chunks" not in full_prompt
    assert str(diag.distances) not in full_prompt


def test_user_query_and_retrieval_query_separation(support_agent):
    """Verify raw user_query remains untouched and retrieval_query remains separate."""
    user_q = "Do you ship to Canada?"
    state = support_agent.process_turn(user_q, session_id="separation_test_session")

    assert state.user_query == user_q
    assert state.retrieval_query == user_q  # Single turn: retrieval_query matches user_q
    assert isinstance(state.retrieval_query, str)
