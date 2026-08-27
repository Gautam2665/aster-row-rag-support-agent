import pytest
from typing import Dict, Any, List

from src.planner import PlannerContext, MockPlanner, LLMPlanner, AgentAction, ActionType, ActionValidator
from src.agent import AgentState, SupportAgent
from src.tools.order_lookup import OrderLookupTool, CustomerSafeOrderResult
from src.llm import MockLLMProvider
from src.models import KBChunk, DocumentMetadata


def create_sample_state() -> AgentState:
    state = AgentState(
        session_id="test_session",
        user_query="Where is order ORD-1007?",
        retrieval_query="Where is order ORD-1007?",
        normalized_order_id="ORD-1007",
        intent_category="order_status",
    )
    return state


def test_planner_context_factory_and_fields():
    """a. Verify PlannerContext contains expected state information and to_dict serializes correctly."""
    state = create_sample_state()
    ctx = PlannerContext.from_agent_state(state)

    assert ctx.user_query == "Where is order ORD-1007?"
    assert ctx.retrieval_query == "Where is order ORD-1007?"
    assert ctx.intent_category == "order_status"
    assert ctx.normalized_order_id == "ORD-1007"
    assert ctx.has_usable_evidence is False
    assert ctx.has_order_result is False
    assert ctx.handoff_recommended is False

    d = ctx.to_dict()
    assert d["user_query"] == state.user_query
    assert d["normalized_order_id"] == "ORD-1007"


def test_planner_context_does_not_expose_raw_state_internals():
    """b. Verify PlannerContext does not copy mutable AgentState internal structures directly."""
    state = create_sample_state()
    ctx = PlannerContext.from_agent_state(state)

    assert not hasattr(ctx, "evidence_chunks")
    assert not hasattr(ctx, "order_result")
    assert not hasattr(ctx, "trace")
    assert not hasattr(ctx, "memory_store")


def test_customer_pii_and_internal_fields_excluded():
    """c. Verify PlannerContext contains zero customer PII or raw database fields."""
    state = create_sample_state()
    raw_order_with_pii = {
        "order_id": "ORD-1007",
        "email": "jane.doe@example.com",
        "address": "123 Main St",
        "risk_score": 95,
        "warehouse_note": "Confidential note",
        "customer": {"name": "Jane Doe"}
    }
    
    # Attach observation with raw PII in summary
    state.observations.append({
        "action_type": "LOOKUP_ORDER",
        "success": True,
        "result_summary": raw_order_with_pii,
    })

    ctx = PlannerContext.from_agent_state(state)
    d = ctx.to_dict()

    # Verify PII keys were stripped during sanitization
    obs_summary = ctx.observations_summary[0]["result_summary"]
    sensitive_keys = ["email", "address", "risk_score", "warehouse_note", "customer"]
    for key in sensitive_keys:
        assert key not in obs_summary

    ctx_str = str(d).lower()
    assert "jane.doe@example.com" not in ctx_str
    assert "123 main st" not in ctx_str


def test_observation_summaries_are_bounded_and_sanitized():
    """d. Verify observation summaries in PlannerContext are structured dicts."""
    state = create_sample_state()
    state.observations.append({
        "action_type": ActionType.LOOKUP_ORDER,
        "success": True,
        "result_summary": {"order_id": "ORD-1007", "status": "shipped"},
        "failure_category": None,
        "handoff_recommended": False,
    })

    ctx = PlannerContext.from_agent_state(state)
    assert len(ctx.observations_summary) == 1
    assert ctx.observations_summary[0]["action_type"] == ActionType.LOOKUP_ORDER
    assert ctx.observations_summary[0]["success"] is True


def test_mock_planner_decisions_with_planner_context():
    """e. Verify MockPlanner produces identical deterministic decisions operating on PlannerContext."""
    planner = MockPlanner()

    # 1. Order status query with order ID -> LOOKUP_ORDER
    state1 = AgentState(session_id="s1", user_query="Check ORD-1007", normalized_order_id="ORD-1007", intent_category="order_status")
    act1 = planner.plan_next_action(state1)
    assert act1.action_type == ActionType.LOOKUP_ORDER
    assert act1.parameters == {"order_id": "ORD-1007"}

    # 2. Clarification -> CLARIFY
    state2 = AgentState(session_id="s2", user_query="Where is my order?", intent_category="clarification")
    act2 = planner.plan_next_action(state2)
    assert act2.action_type == ActionType.CLARIFY

    # 3. Policy question -> RETRIEVE_KB
    state3 = AgentState(session_id="s3", user_query="What is the return policy?", intent_category="policy")
    act3 = planner.plan_next_action(state3)
    assert act3.action_type == ActionType.RETRIEVE_KB

    # 4. Handoff recommended -> HANDOFF
    state4 = AgentState(session_id="s4", user_query="Damaged item", handoff_recommended=True)
    act4 = planner.plan_next_action(state4)
    assert act4.action_type == ActionType.HANDOFF


def test_llm_planner_receives_planner_context():
    """f. Verify LLMPlanner receives PlannerContext and formats structured state."""
    class CapturingLLMProvider(MockLLMProvider):
        def __init__(self):
            super().__init__()
            self.last_user_prompt = None

        def generate(self, system_prompt: str, user_prompt: str) -> str:
            self.last_user_prompt = user_prompt
            return '{"action_type": "RETRIEVE_KB", "parameters": {"query": "returns"}, "reasoning": "policy query"}'

    provider = CapturingLLMProvider()
    planner = LLMPlanner(provider=provider)

    state = create_sample_state()
    action = planner.plan_next_action(state)

    assert action.action_type == ActionType.RETRIEVE_KB
    assert provider.last_user_prompt is not None
    assert "User Query: Where is order ORD-1007?" in provider.last_user_prompt
    assert "Normalized Order ID: ORD-1007" in provider.last_user_prompt
    assert "Has Usable Evidence: False" in provider.last_user_prompt


def test_planner_context_cannot_execute_tools():
    """g. Verify PlannerContext has no execution methods or tool references."""
    ctx = PlannerContext(
        user_query="Check order",
        retrieval_query="Check order",
        intent_category="order_status",
    )

    assert not hasattr(ctx, "execute_tool")
    assert not hasattr(ctx, "run_action")
    assert not hasattr(ctx, "process_turn")


def test_action_validation_boundaries_intact():
    """h. Verify ActionValidator rejects invalid action types planned from PlannerContext."""
    invalid_action = AgentAction(action_type="UNAPPROVED_ACTION", parameters={})
    with pytest.raises(ValueError, match="not allowed"):
        ActionValidator.validate(invalid_action)


from pathlib import Path
from src.retrieval import KBVectorStore
from src.ingestion import ingest_kb_directory

def test_end_to_end_agent_multi_turn_and_handoff():
    """i & j. Verify SupportAgent operates end-to-end with PlannerContext for multi-turn and handoff."""
    chunks = ingest_kb_directory(Path("knowledge-base"))
    vector_store = KBVectorStore(collection_name="test_ctx_e2e_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        llm_provider=MockLLMProvider(),
    )

    # Valid order turn
    state1 = agent.process_turn("Check status for ORD-1007", session_id="ctx_turn_session")
    assert state1.order_result is not None
    assert state1.order_result.found is True

    # Unknown order handoff turn
    state2 = agent.process_turn("Check status for ORD-9999", session_id="ctx_unknown_session")
    assert state2.handoff_recommended is True
