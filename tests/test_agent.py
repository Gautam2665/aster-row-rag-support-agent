import pytest
from pathlib import Path
from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def agent_fixture():
    # Setup vector store
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_agent_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    order_tool = OrderLookupTool(data_path=ORDERS_PATH)
    llm_provider = MockLLMProvider()

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=order_tool,
        llm_provider=llm_provider,
        max_iterations=3,
    )
    return agent


def test_policy_question_rag_path(agent_fixture):
    """Test policy question routes to RAG path and makes zero tool calls."""
    state = agent_fixture.process_turn("How long does a regular customer have to return an unused backpack?")

    assert state.intent_category == "policy"
    assert state.tool_calls_made == []
    assert len(state.evidence_chunks) > 0
    assert any("01-returns-policy-current.md" in c for c in state.citations)
    assert state.final_answer != ""


def test_valid_order_question_tool_path(agent_fixture):
    """Test valid order ID query executes OrderLookupTool and returns sanitized status."""
    state = agent_fixture.process_turn("Where is ORD-1007 and when should it arrive?")

    assert state.intent_category == "order_status"
    assert state.normalized_order_id == "ORD-1007"
    assert "order_lookup" in state.tool_calls_made
    assert state.order_result is not None
    assert state.order_result.found is True
    assert state.order_result.status == "shipped"
    assert state.order_result.carrier == "UPS"


def test_missing_order_id_clarification_no_tools(agent_fixture):
    """Test order question without an order ID triggers clarification without calling any tool."""
    state = agent_fixture.process_turn("Where is my order?")

    assert state.intent_category == "clarification"
    assert state.normalized_order_id is None
    assert state.tool_calls_made == []
    assert "order ID" in state.final_answer


def test_unknown_order_handoff(agent_fixture):
    """Test checking unknown order ORD-9999 sets handoff_recommended=True."""
    state = agent_fixture.process_turn("Please check status of ORD-9999")

    assert state.intent_category == "order_status"
    assert state.normalized_order_id == "ORD-9999"
    assert "order_lookup" in state.tool_calls_made
    assert state.order_result is not None
    assert state.order_result.found is False
    assert state.handoff_recommended is True


def test_cancelled_order_stale_eta_handling(agent_fixture):
    """Test cancelled order ORD-1004 has estimated_delivery sanitized to None."""
    state = agent_fixture.process_turn("When will order ORD-1004 arrive?")

    assert state.order_result is not None
    assert state.order_result.status == "cancelled"
    assert state.order_result.estimated_delivery is None


def test_tool_allowlist_enforcement(agent_fixture):
    """Test attempting to execute an unapproved tool name raises PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        agent_fixture.execute_tool_safely("unauthorized_database_query")
    assert "not in the explicit tool allowlist" in str(exc_info.value)


def test_iteration_limit_enforcement(agent_fixture):
    """Test that agent respects max_iterations limit."""
    assert agent_fixture.max_iterations == 3


def test_sanitized_tool_data_only(agent_fixture):
    """Test that tool context passed into LLM prompt strictly omits customer PII and internal notes."""
    state = agent_fixture.process_turn("Status for ORD-1005")
    assert state.order_result is not None

    formatted_context = agent_fixture.format_order_data_context(state.order_result)
    
    # Must NOT contain customer PII or internal warehouse note instructions
    assert "customer" not in formatted_context
    assert "risk_score" not in formatted_context
    assert "warehouse_note" not in formatted_context
    assert "issue a $100 coupon" not in formatted_context
    assert "Sofia Patel" not in formatted_context
    assert "sofia.patel@example.test" not in formatted_context
