import pytest
from pathlib import Path
from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.planner import MockPlanner, AgentAction, AgentObservation, ActionType, BasePlanner
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def obs_agent_fixture():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_obs_agent_kb")
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


class TrackingPlanner(BasePlanner):
    """Planner that records state history passed into plan_next_action for verification."""
    def __init__(self):
        self.state_snapshots = []

    def plan_next_action(self, agent_state: AgentState) -> AgentAction:
        # Snapshot prior observations count
        self.state_snapshots.append(list(agent_state.observations))
        
        if not agent_state.observations:
            if agent_state.normalized_order_id:
                return AgentAction(action_type=ActionType.LOOKUP_ORDER, parameters={"order_id": agent_state.normalized_order_id})
            return AgentAction(action_type=ActionType.RETRIEVE_KB, parameters={"query": agent_state.user_query})
        
        # Second decision: check prior observation
        last_obs = agent_state.observations[-1]
        if last_obs.action_type == ActionType.LOOKUP_ORDER and not last_obs.success:
            return AgentAction(action_type=ActionType.HANDOFF)
        return AgentAction(action_type=ActionType.RESPOND)


def test_observation_recorded_in_state(obs_agent_fixture):
    """Test that executing actions records explicit AgentObservation objects in state.observations."""
    state = obs_agent_fixture.process_turn("What is the return policy?", session_id="obs_rec_s")
    assert len(state.observations) > 0
    first_obs = state.observations[0]
    assert isinstance(first_obs, AgentObservation)
    assert first_obs.action_type == ActionType.RETRIEVE_KB
    assert first_obs.success is True
    assert len(first_obs.result) > 0


def test_second_planner_decision_receives_first_observation(obs_agent_fixture):
    """Test proving that the second planner decision receives the first action's observation in state."""
    tracking_planner = TrackingPlanner()
    agent_tracking = SupportAgent(
        vector_store=obs_agent_fixture.vector_store,
        order_tool=obs_agent_fixture.order_tool,
        llm_provider=obs_agent_fixture.llm_provider,
        planner=tracking_planner
    )

    state = agent_tracking.process_turn("Where is ORD-1007?", session_id="obs_seq_s")
    
    # Verify planner was called twice
    assert len(tracking_planner.state_snapshots) == 2
    # First call: 0 prior observations
    assert len(tracking_planner.state_snapshots[0]) == 0
    # Second call: 1 prior observation (from LOOKUP_ORDER)
    assert len(tracking_planner.state_snapshots[1]) == 1
    assert tracking_planner.state_snapshots[1][0].action_type == ActionType.LOOKUP_ORDER


def test_repeated_actions_avoided_when_evidence_present(obs_agent_fixture):
    """Test proving that repeated actions are avoided once observations contain sufficient evidence."""
    state = obs_agent_fixture.process_turn("Where is ORD-1007?", session_id="obs_no_repeat_s")
    
    action_types = [a.action_type for a in state.planned_actions]
    # Should perform LOOKUP_ORDER then RESPOND (no duplicate LOOKUP_ORDER calls)
    assert action_types == [ActionType.LOOKUP_ORDER, ActionType.RESPOND]
    assert state.tool_calls_made.count("order_lookup") == 1


def test_observations_contain_sanitized_customer_safe_data(obs_agent_fixture):
    """Test that order lookup observation result is sanitized CustomerSafeOrderResult without PII."""
    state = obs_agent_fixture.process_turn("Where is ORD-1007?", session_id="obs_sanitized_s")
    lookup_obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    
    obs_dict = lookup_obs.to_dict()
    summary = obs_dict["result_summary"]
    
    # Must contain customer safe fields and MUST NOT leak customer PII or internal notes
    assert summary["order_id"] == "ORD-1007"
    assert "customer" not in summary
    assert "internal" not in summary
    assert "warehouse_note" not in summary


def test_unknown_order_causes_handoff_observation(obs_agent_fixture):
    """Test that looking up an unknown order produces success=False and handoff_recommended=True in observation."""
    tracking_planner = TrackingPlanner()
    agent_tracking = SupportAgent(
        vector_store=obs_agent_fixture.vector_store,
        order_tool=obs_agent_fixture.order_tool,
        llm_provider=obs_agent_fixture.llm_provider,
        planner=tracking_planner
    )
    state = agent_tracking.process_turn("Where is ORD-9999?", session_id="obs_unknown_s")
    
    lookup_obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    assert lookup_obs.success is False
    assert lookup_obs.handoff_recommended is True
    assert "not found" in lookup_obs.error_message.lower()
