import pytest
from pathlib import Path
from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.planner import MockPlanner, LLMPlanner, AgentAction, ActionType, BasePlanner
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def planner_agent_fixture():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_planner_agent_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    order_tool = OrderLookupTool(data_path=ORDERS_PATH)
    llm_provider = MockLLMProvider()
    planner = MockPlanner()

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=order_tool,
        llm_provider=llm_provider,
        planner=planner,
        max_history_turns=3,
    )
    return agent


class FaultyPlanner(BasePlanner):
    """Planner that generates an unapproved action type to test safe fallback handling."""
    def plan_next_action(self, agent_state):
        return AgentAction(action_type="UNAPPROVED_MALICIOUS_ACTION")


def test_policy_retrieve_respond(planner_agent_fixture):
    """Test policy query plans RETRIEVE_KB action and generates grounded response."""
    state = planner_agent_fixture.process_turn("What is the return window?", session_id="planner_policy_s")
    assert state.intent_category == "policy"
    assert len(state.planned_actions) > 0
    assert state.planned_actions[0].action_type == ActionType.RETRIEVE_KB
    assert len(state.evidence_chunks) > 0
    assert state.final_answer != ""


def test_order_lookup_respond(planner_agent_fixture):
    """Test order status query plans LOOKUP_ORDER action and executes order tool."""
    state = planner_agent_fixture.process_turn("Where is ORD-1007?", session_id="planner_order_s")
    assert state.intent_category == "order_status"
    assert len(state.planned_actions) > 0
    assert state.planned_actions[0].action_type == ActionType.LOOKUP_ORDER
    assert state.order_result is not None
    assert state.order_result.status == "shipped"


def test_missing_information_clarify(planner_agent_fixture):
    """Test missing order ID plans CLARIFY action without tool calls."""
    state = planner_agent_fixture.process_turn("Where is my order?", session_id="planner_clarify_s")
    assert state.intent_category == "clarification"
    assert len(state.planned_actions) > 0
    assert state.planned_actions[0].action_type == ActionType.CLARIFY
    assert state.tool_calls_made == []
    assert "order ID" in state.final_answer


def test_unknown_order_handoff(planner_agent_fixture):
    """Test unknown order lookup triggers handoff_recommended = True."""
    state = planner_agent_fixture.process_turn("Check status of ORD-9999", session_id="planner_unknown_s")
    assert state.order_result is not None
    assert state.order_result.found is False
    assert state.handoff_recommended is True


def test_planner_invalid_action_safe_fallback(planner_agent_fixture):
    """Test agent safely falls back when planner produces an unapproved action type."""
    agent_faulty = SupportAgent(
        vector_store=planner_agent_fixture.vector_store,
        order_tool=planner_agent_fixture.order_tool,
        llm_provider=planner_agent_fixture.llm_provider,
        planner=FaultyPlanner()
    )
    state = agent_faulty.process_turn("What is the return window?", session_id="faulty_planner_s")
    assert state.planned_actions[0].action_type == ActionType.HANDOFF
    assert state.handoff_recommended is True


def test_iteration_limit_enforcement(planner_agent_fixture):
    """Test that max_iterations=3 is strictly enforced on every turn."""
    state = planner_agent_fixture.process_turn("What is the warranty policy?", session_id="iter_limit_s")
    assert state.iterations <= 3
    assert state.max_iterations == 3


def test_tool_allowlist_enforcement(planner_agent_fixture):
    """Test that executing an unapproved tool name raises PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        planner_agent_fixture.execute_tool_safely("unapproved_database_tool")
    assert "not in the explicit tool allowlist" in str(exc_info.value)


def test_memory_preservation(planner_agent_fixture):
    """Test that turns are stored and retrieved cleanly across turns."""
    s_id = "planner_memory_preservation_s"
    t1 = planner_agent_fixture.process_turn("Do you ship internationally?", session_id=s_id)
    t2 = planner_agent_fixture.process_turn("What about Canada?", session_id=s_id)

    assert len(t2.history_turns) == 1
    assert t2.history_turns[0].user_query == "Do you ship internationally?"


def test_existing_day1_3_security_behavior(planner_agent_fixture):
    """Verify Day 1-3 security guardrails (PII scrubbing, pre-retrieval filtering, XML tags)."""
    state = planner_agent_fixture.process_turn("Check ORD-1005", session_id="sec_day3_check")
    order_dict = state.order_result.to_dict()
    assert "customer" not in order_dict
    assert "internal" not in order_dict
    assert "warehouse_note" not in order_dict

    state_kb = planner_agent_fixture.process_turn("Check policy updates", session_id="sec_day3_kb")
    retrieved = {c.filename for c in state_kb.evidence_chunks}
    assert "14-internal-content-migration-notes.md" not in retrieved
