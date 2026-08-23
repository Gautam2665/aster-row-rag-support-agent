import pytest
from pathlib import Path
from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.planner import BasePlanner, AgentAction, AgentObservation, ActionType, FailureCategory, MockPlanner
from src.context import ContextBuilder
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def failure_agent_fixture():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_fail_kb")
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
        max_iterations=3,
    )
    return agent


class FailingToolMock(OrderLookupTool):
    """Mock order tool that intentionally raises an exception to test TOOL_ERROR handling."""
    def lookup(self, order_id: str):
        raise RuntimeError("Database connection timed out during lookup")


class ExceptionPlanner(BasePlanner):
    """Planner that intentionally raises an unhandled exception to test PLANNER_FAILURE handling."""
    def plan_next_action(self, agent_state: AgentState) -> AgentAction:
        raise ValueError("LLM provider returned corrupted non-JSON payload")


class UnapprovedActionPlanner(BasePlanner):
    """Planner that attempts to issue an unapproved action type."""
    def plan_next_action(self, agent_state: AgentState) -> AgentAction:
        return AgentAction(action_type="EXECUTE_UNAUTHORIZED_SQL", parameters={"sql": "DROP TABLE"})


def test_successful_retrieval_continues_normally(failure_agent_fixture):
    """Test that a successful retrieval has failure_category set to None."""
    state = failure_agent_fixture.process_turn("What is the return policy?", session_id="fail_succ_s")
    obs = next(o for o in state.observations if o.action_type == ActionType.RETRIEVE_KB)
    assert obs.success is True
    assert obs.failure_category is None


def test_empty_retrieval_classified_as_retrieval_failure(failure_agent_fixture):
    """Test that empty retrieval results in failure_category=RETRIEVAL_FAILURE and handoff_recommended=True."""
    empty_vector_store = KBVectorStore(collection_name="test_empty_kb_for_failure")
    empty_vector_store.clear()
    empty_agent = SupportAgent(
        vector_store=empty_vector_store,
        order_tool=failure_agent_fixture.order_tool,
        llm_provider=failure_agent_fixture.llm_provider,
        planner=failure_agent_fixture.planner,
    )
    state = empty_agent.process_turn("What is the return policy?", session_id="fail_empty_ret_s")
    obs = next(o for o in state.observations if o.action_type == ActionType.RETRIEVE_KB)
    assert obs.failure_category == FailureCategory.RETRIEVAL_FAILURE
    assert state.handoff_recommended is True


def test_unknown_order_classified_as_business_failure(failure_agent_fixture):
    """Test that looking up an unknown order (ORD-9999) is classified as BUSINESS_FAILURE."""
    state = failure_agent_fixture.process_turn("Where is ORD-9999?", session_id="fail_unk_ord_s")
    obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    assert obs.success is False
    assert obs.failure_category == FailureCategory.BUSINESS_FAILURE
    assert state.handoff_recommended is True


def test_tool_exception_classified_as_tool_error(failure_agent_fixture):
    """Test that a tool exception is classified as TOOL_ERROR, triggers HANDOFF, and does NOT retry."""
    failing_agent = SupportAgent(
        vector_store=failure_agent_fixture.vector_store,
        order_tool=FailingToolMock(data_path=ORDERS_PATH),
        llm_provider=failure_agent_fixture.llm_provider,
        planner=failure_agent_fixture.planner,
    )
    state = failing_agent.process_turn("Where is ORD-1007?", session_id="fail_tool_err_s")
    
    assert state.handoff_recommended is True
    obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    assert obs.success is False
    assert obs.failure_category == FailureCategory.TOOL_ERROR
    assert "Database connection timed out" in obs.error_message
    
    # Ensure tool was executed only once (no blind retry loop)
    assert len([o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER]) == 1


def test_planner_failure_triggers_safe_handoff(failure_agent_fixture):
    """Test that planner exception is caught, classified as PLANNER_FAILURE, and falls back safely to HANDOFF."""
    failing_planner_agent = SupportAgent(
        vector_store=failure_agent_fixture.vector_store,
        order_tool=failure_agent_fixture.order_tool,
        llm_provider=failure_agent_fixture.llm_provider,
        planner=ExceptionPlanner(),
    )
    state = failing_planner_agent.process_turn("What is your warranty policy?", session_id="fail_planner_err_s")
    
    assert state.handoff_recommended is True
    assert state.planned_actions[-1].action_type == ActionType.HANDOFF
    obs = state.observations[-1]
    assert obs.failure_category == FailureCategory.PLANNER_FAILURE


def test_invalid_planner_action_never_executed(failure_agent_fixture):
    """Test that unapproved action type is rejected by ActionValidator and converted to PLANNER_FAILURE handoff."""
    unapproved_agent = SupportAgent(
        vector_store=failure_agent_fixture.vector_store,
        order_tool=failure_agent_fixture.order_tool,
        llm_provider=failure_agent_fixture.llm_provider,
        planner=UnapprovedActionPlanner(),
    )
    state = unapproved_agent.process_turn("What is your warranty policy?", session_id="fail_unapproved_s")
    
    assert state.handoff_recommended is True
    # Action type "EXECUTE_UNAUTHORIZED_SQL" must NOT appear in tool_calls_made
    assert "EXECUTE_UNAUTHORIZED_SQL" not in state.tool_calls_made
    assert state.planned_actions[-1].action_type == ActionType.HANDOFF


def test_failure_information_does_not_leak_pii(failure_agent_fixture):
    """Test that failure observations do not expose PII or internal database fields."""
    state = failure_agent_fixture.process_turn("Where is ORD-9999?", session_id="fail_pii_s")
    obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    obs_dict = obs.to_dict()
    
    assert "email" not in obs_dict
    assert "address" not in obs_dict
    assert "risk_score" not in obs_dict


def test_failure_information_does_not_enter_llm_context(failure_agent_fixture):
    """Test that failure_category internal classifications do not leak into prompt payloads."""
    state = failure_agent_fixture.process_turn("Where is ORD-9999?", session_id="fail_prompt_leak_s")
    payload = ContextBuilder.build_prompt_with_context(
        user_question=state.user_query,
        evidence_chunks=state.evidence_chunks,
        history_turns=state.history_turns
    )
    assert "BUSINESS_FAILURE" not in payload.user_prompt
    assert "TOOL_ERROR" not in payload.user_prompt
    assert "PLANNER_FAILURE" not in payload.user_prompt
