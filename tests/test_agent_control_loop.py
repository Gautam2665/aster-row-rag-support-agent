import pytest
from pathlib import Path
from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.planner import BasePlanner, AgentAction, AgentObservation, ActionType, MockPlanner
from src.ingestion import ingest_kb_directory

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def control_loop_agent():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_ctrl_loop_kb")
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


class InfiniteNonTerminalPlanner(BasePlanner):
    """Planner that continuously proposes the exact same RETRIEVE_KB action without stopping."""
    def plan_next_action(self, agent_state: AgentState) -> AgentAction:
        return AgentAction(
            action_type=ActionType.RETRIEVE_KB,
            parameters={"query": agent_state.user_query}
        )


class ForcedTerminalPlanner(BasePlanner):
    """Planner that returns a specific action for testing."""
    def __init__(self, action: AgentAction):
        self.action = action

    def plan_next_action(self, agent_state: AgentState) -> AgentAction:
        return self.action


def test_clarify_terminates_loop_without_tool_execution(control_loop_agent):
    """Test that CLARIFY immediately terminates turn without order_lookup execution."""
    planner = ForcedTerminalPlanner(AgentAction(action_type=ActionType.CLARIFY))
    agent = SupportAgent(
        vector_store=control_loop_agent.vector_store,
        order_tool=control_loop_agent.order_tool,
        llm_provider=control_loop_agent.llm_provider,
        planner=planner,
    )
    state = agent.process_turn("Where is my order?", session_id="ctrl_clarify")
    assert state.iterations == 1
    assert len(state.tool_calls_made) == 0
    assert "order ID" in state.final_answer
    assert state.observations[-1].action_type == ActionType.CLARIFY


def test_respond_terminates_loop(control_loop_agent):
    """Test that RESPOND terminates the planner loop."""
    planner = ForcedTerminalPlanner(AgentAction(action_type=ActionType.RESPOND))
    agent = SupportAgent(
        vector_store=control_loop_agent.vector_store,
        order_tool=control_loop_agent.order_tool,
        llm_provider=control_loop_agent.llm_provider,
        planner=planner,
    )
    state = agent.process_turn("What is your return window?", session_id="ctrl_respond")
    assert state.iterations == 1
    assert state.observations[-1].action_type == ActionType.RESPOND


def test_handoff_terminates_loop(control_loop_agent):
    """Test that HANDOFF sets handoff_recommended and terminates the loop."""
    planner = ForcedTerminalPlanner(AgentAction(action_type=ActionType.HANDOFF))
    agent = SupportAgent(
        vector_store=control_loop_agent.vector_store,
        order_tool=control_loop_agent.order_tool,
        llm_provider=control_loop_agent.llm_provider,
        planner=planner,
    )
    state = agent.process_turn("I need a human representative", session_id="ctrl_handoff")
    assert state.iterations == 1
    assert state.handoff_recommended is True
    assert state.observations[-1].action_type == ActionType.HANDOFF


def test_retrieve_kb_allows_subsequent_planner_decision(control_loop_agent):
    """Test RETRIEVE_KB produces observation and allows iteration 2 decision."""
    state = control_loop_agent.process_turn("What is the return window?", session_id="ctrl_retrieve")
    assert state.iterations == 2
    assert state.planned_actions[0].action_type == ActionType.RETRIEVE_KB
    assert state.planned_actions[1].action_type == ActionType.RESPOND
    assert state.observations[0].action_type == ActionType.RETRIEVE_KB
    assert len(state.evidence_chunks) > 0


def test_lookup_order_allows_subsequent_planner_decision(control_loop_agent):
    """Test LOOKUP_ORDER produces observation and allows iteration 2 decision."""
    state = control_loop_agent.process_turn("Where is ORD-1007?", session_id="ctrl_lookup")
    assert state.iterations == 2
    assert state.planned_actions[0].action_type == ActionType.LOOKUP_ORDER
    assert state.planned_actions[1].action_type == ActionType.RESPOND
    assert state.observations[0].action_type == ActionType.LOOKUP_ORDER
    assert state.order_result is not None


def test_max_iterations_limit_enforcement(control_loop_agent):
    """Test max_iterations=3 is strictly enforced and terminates safely if limit exhausted."""
    # A planner that always chooses non-terminal actions with slightly different queries
    class VaryingRetrievePlanner(BasePlanner):
        def __init__(self):
            self.step = 0
        def plan_next_action(self, agent_state: AgentState) -> AgentAction:
            self.step += 1
            return AgentAction(action_type=ActionType.RETRIEVE_KB, parameters={"query": f"q_{self.step}"})

    agent = SupportAgent(
        vector_store=control_loop_agent.vector_store,
        order_tool=control_loop_agent.order_tool,
        llm_provider=control_loop_agent.llm_provider,
        planner=VaryingRetrievePlanner(),
        max_iterations=3,
    )
    state = agent.process_turn("Unresolvable query", session_id="ctrl_max_iter")
    assert state.iterations == 3
    assert state.handoff_recommended is True


def test_progress_protection_prevents_repeated_identical_actions(control_loop_agent):
    """Test that progress protection prevents repeated identical non-terminal actions."""
    agent = SupportAgent(
        vector_store=control_loop_agent.vector_store,
        order_tool=control_loop_agent.order_tool,
        llm_provider=control_loop_agent.llm_provider,
        planner=InfiniteNonTerminalPlanner(),
        max_iterations=3,
    )
    state = agent.process_turn("What is your return policy?", session_id="ctrl_prog_protect")
    
    # Iteration 1: RETRIEVE_KB
    # Iteration 2: Duplicate RETRIEVE_KB detected -> Progress protection overrides to RESPOND
    assert state.iterations <= 2
    assert "Progress protection triggered" in state.planned_actions[-1].reasoning


def test_failed_lookup_does_not_loop(control_loop_agent):
    """Test that failed lookup (unknown order) results in handoff_recommended and no uncontrolled retries."""
    state = control_loop_agent.process_turn("Where is ORD-9999?", session_id="ctrl_failed_lookup")
    assert state.handoff_recommended is True
    lookup_obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    assert lookup_obs.success is False


def test_observation_data_sanitized(control_loop_agent):
    """Test that observation data contains sanitized CustomerSafeOrderResult without PII."""
    state = control_loop_agent.process_turn("Where is ORD-1007?", session_id="ctrl_sanitized")
    lookup_obs = next(o for o in state.observations if o.action_type == ActionType.LOOKUP_ORDER)
    obs_summary = lookup_obs.to_dict()["result_summary"]
    
    assert "email" not in obs_summary
    assert "address" not in obs_summary
    assert "risk_score" not in obs_summary
    assert "internal_notes" not in obs_summary
