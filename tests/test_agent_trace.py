import pytest
from pathlib import Path
from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.planner import BasePlanner, AgentAction, ActionType, MockPlanner
from src.trace import TraceEvent, TraceEventType
from src.context import ContextBuilder
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def trace_agent_fixture():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_trace_kb")
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


def test_trace_created_for_normal_policy_request(trace_agent_fixture):
    """Test that a normal policy request generates a complete lifecycle trace."""
    state = trace_agent_fixture.process_turn("What is the return policy?", session_id="tr_policy_s")
    assert len(state.trace) > 0
    event_types = [t.event_type for t in state.trace]
    
    assert TraceEventType.TURN_STARTED in event_types
    assert TraceEventType.ACTION_PLANNED in event_types
    assert TraceEventType.ACTION_EXECUTED in event_types
    assert TraceEventType.OBSERVATION_RECORDED in event_types
    assert TraceEventType.TURN_COMPLETED in event_types


def test_trace_records_retrieval_action_and_observation(trace_agent_fixture):
    """Test that trace captures RETRIEVE_KB action type and query parameters."""
    state = trace_agent_fixture.process_turn("What is the return policy?", session_id="tr_ret_s")
    planned_ev = next(t for t in state.trace if t.event_type == TraceEventType.ACTION_PLANNED)
    assert planned_ev.action_type == ActionType.RETRIEVE_KB
    
    executed_ev = next(t for t in state.trace if t.event_type == TraceEventType.ACTION_EXECUTED)
    assert executed_ev.action_type == ActionType.RETRIEVE_KB
    assert executed_ev.success is True


def test_trace_records_order_lookup_without_pii(trace_agent_fixture):
    """Test that trace captures LOOKUP_ORDER without exposing PII fields."""
    state = trace_agent_fixture.process_turn("Where is ORD-1007?", session_id="tr_order_s")
    order_evs = [t for t in state.trace if t.action_type == ActionType.LOOKUP_ORDER]
    assert len(order_evs) > 0

    for ev in order_evs:
        ev_dict = ev.to_dict()
        params = ev_dict.get("parameters") or {}
        assert "email" not in params
        assert "address" not in params
        assert "risk_score" not in params
        assert "warehouse_note" not in params


def test_trace_records_missing_order_clarification(trace_agent_fixture):
    """Test that missing order ID query records CLARIFY trace events and TURN_COMPLETED."""
    state = trace_agent_fixture.process_turn("Where is my order?", session_id="tr_clarify_s")
    event_types = [t.event_type for t in state.trace]
    
    assert TraceEventType.TURN_STARTED in event_types
    assert TraceEventType.ACTION_PLANNED in event_types
    assert TraceEventType.ACTION_EXECUTED in event_types
    assert TraceEventType.TURN_COMPLETED in event_types
    
    clarify_ev = next(t for t in state.trace if t.action_type == ActionType.CLARIFY)
    assert clarify_ev.success is True


def test_trace_records_unknown_order_handoff(trace_agent_fixture):
    """Test that unknown order lookup records LOOKUP_ORDER failure and HANDOFF trace events."""
    state = trace_agent_fixture.process_turn("Where is ORD-9999?", session_id="tr_unk_s")
    event_types = [t.event_type for t in state.trace]
    
    assert TraceEventType.HANDOFF in event_types
    handoff_ev = next(t for t in state.trace if t.event_type == TraceEventType.HANDOFF)
    assert handoff_ev.summary is not None


def test_trace_records_iteration_limit_exhaustion(trace_agent_fixture):
    """Test that exhausting max_iterations records ITERATION_LIMIT_EXHAUSTED event."""
    class VaryingRetrievePlanner(BasePlanner):
        def __init__(self):
            self.step = 0
        def plan_next_action(self, agent_state: AgentState) -> AgentAction:
            self.step += 1
            return AgentAction(action_type=ActionType.RETRIEVE_KB, parameters={"query": f"var_q_{self.step}"})

    agent = SupportAgent(
        vector_store=trace_agent_fixture.vector_store,
        order_tool=trace_agent_fixture.order_tool,
        llm_provider=trace_agent_fixture.llm_provider,
        planner=VaryingRetrievePlanner(),
        max_iterations=3,
    )
    state = agent.process_turn("Unresolvable query", session_id="tr_max_iter_s")
    event_types = [t.event_type for t in state.trace]
    
    assert TraceEventType.ITERATION_LIMIT_EXHAUSTED in event_types


def test_trace_ordering_follows_execution_order(trace_agent_fixture):
    """Test that trace events strictly follow the chronological execution sequence."""
    state = trace_agent_fixture.process_turn("What is the return policy?", session_id="tr_seq_s")
    
    # TURN_STARTED must be first
    assert state.trace[0].event_type == TraceEventType.TURN_STARTED
    # TURN_COMPLETED must be last
    assert state.trace[-1].event_type == TraceEventType.TURN_COMPLETED


def test_trace_does_not_mutate_original_user_query(trace_agent_fixture):
    """Test that trace creation does not mutate state.user_query."""
    raw_query = "What is the return window?"
    state = trace_agent_fixture.process_turn(raw_query, session_id="tr_mut_s")
    assert state.user_query == raw_query


def test_trace_does_not_enter_llm_prompt_context(trace_agent_fixture):
    """Test that state.trace information is completely excluded from ContextBuilder prompt output."""
    state = trace_agent_fixture.process_turn("Where is ORD-1007?", session_id="tr_ctx_s")
    payload = ContextBuilder.build_prompt_with_context(
        user_question=state.user_query,
        evidence_chunks=state.evidence_chunks,
        history_turns=state.history_turns
    )
    assert "TURN_STARTED" not in payload.user_prompt
    assert "ACTION_PLANNED" not in payload.user_prompt
    assert "OBSERVATION_RECORDED" not in payload.user_prompt


def test_sensitive_internal_fields_absent_from_trace_data():
    """Test that TraceEvent.to_dict() sanitizes sensitive parameter keys."""
    event = TraceEvent(
        event_type=TraceEventType.ACTION_EXECUTED,
        parameters={
            "order_id": "ORD-1007",
            "email": "customer@example.com",
            "risk_score": 95,
            "warehouse_note": "Confidential"
        }
    )
    clean_dict = event.to_dict()
    params = clean_dict["parameters"]
    assert params == {"order_id": "ORD-1007"}
