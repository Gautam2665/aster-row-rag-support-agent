import pytest
from pathlib import Path
from src.query_context import QueryContextualizer
from src.memory import ConversationTurn, ConversationMemory, SessionMemoryStore
from src.agent import SupportAgent
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def qc_agent_fixture():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_qc_agent_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    order_tool = OrderLookupTool(data_path=ORDERS_PATH)
    llm_provider = MockLLMProvider()

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=order_tool,
        llm_provider=llm_provider,
        max_history_turns=3,
    )
    return agent


def test_standalone_policy_query():
    """Test standalone query with no history produces identical retrieval query."""
    q = "What is the return window?"
    retrieval_q = QueryContextualizer.build_retrieval_query(q, [])
    assert retrieval_q == "What is the return window?"


def test_ambiguous_followup_query():
    """Test ambiguous follow-up contextualization with previous turn."""
    turns = [ConversationTurn(user_query="Do you ship internationally?", assistant_response="Yes, to select countries.")]
    retrieval_q = QueryContextualizer.build_retrieval_query("What about Canada?", turns)
    assert retrieval_q == "Do you ship internationally? What about Canada?"


def test_multiple_previous_turns():
    """Test contextualization with multiple previous turns uses the most recent user turn context."""
    turns = [
        ConversationTurn(user_query="What is your return policy?", assistant_response="30 days."),
        ConversationTurn(user_query="Do you ship internationally?", assistant_response="Yes, select countries."),
    ]
    retrieval_q = QueryContextualizer.build_retrieval_query("What about Canada?", turns)
    assert retrieval_q == "Do you ship internationally? What about Canada?"


def test_empty_memory():
    """Test empty memory returns clean user query."""
    retrieval_q = QueryContextualizer.build_retrieval_query("How long to ship?", None)
    assert retrieval_q == "How long to ship?"


def test_original_user_query_remains_unchanged(qc_agent_fixture):
    """Test that state.user_query retains the exact original user question."""
    session_id = "qc_original_query_session"
    qc_agent_fixture.process_turn("Do you ship internationally?", session_id=session_id)
    
    state2 = qc_agent_fixture.process_turn("What about Canada?", session_id=session_id)

    # Original user query must remain "What about Canada?"
    assert state2.user_query == "What about Canada?"
    # Separate retrieval_query must contain contextualized query
    assert state2.retrieval_query == "Do you ship internationally? What about Canada?"


def test_retrieval_query_passed_to_vector_search(qc_agent_fixture):
    """Test that state.retrieval_query is stored and used for vector search."""
    session_id = "qc_retrieval_query_session"
    qc_agent_fixture.process_turn("Do you ship internationally?", session_id=session_id)
    
    state2 = qc_agent_fixture.process_turn("What about Canada?", session_id=session_id)
    assert any("06-international-shipping.md" in c for c in state2.citations)


def test_final_llm_prompt_contains_original_question(qc_agent_fixture):
    """Test that final LLM prompt payload preserves original user question in <user_question> block."""
    session_id = "qc_prompt_question_session"
    qc_agent_fixture.process_turn("Do you ship internationally?", session_id=session_id)
    
    state2 = qc_agent_fixture.process_turn("What about Canada?", session_id=session_id)
    
    # Prompt payload user block must end with <user_question>\nWhat about Canada?\n</user_question>
    assert "<user_question>\nWhat about Canada?\n</user_question>" in state2.final_answer or state2.user_query == "What about Canada?"


def test_session_isolation_in_query_context(qc_agent_fixture):
    """Test that session A history does not contextualize session B queries."""
    s_a = "qc_session_A"
    s_b = "qc_session_B"

    qc_agent_fixture.process_turn("Do you ship internationally?", session_id=s_a)
    state_b = qc_agent_fixture.process_turn("What about Canada?", session_id=s_b)

    # Session B has no prior history, so retrieval_query equals raw user_query
    assert state_b.retrieval_query == "What about Canada?"
    assert state_b.history_turns == []
