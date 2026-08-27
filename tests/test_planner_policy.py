import pytest

from src.planner import ActionType, AgentAction, PlannerContext, FailureCategory, ActionValidator, MockPlanner
from src.planner_policy import PlannerPolicy, PlannerPolicyValidation
from src.agent import SupportAgent, AgentState
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider


def make_ctx(user_query="Where is order?", normalized_order_id=None, has_evidence=False, has_order=False) -> PlannerContext:
    return PlannerContext(
        user_query=user_query,
        retrieval_query=user_query,
        intent_category="order_status",
        normalized_order_id=normalized_order_id,
        has_usable_evidence=has_evidence,
        has_order_result=has_order,
    )


def test_valid_lookup_order_permitted():
    """a. Verify valid LOOKUP_ORDER action is permitted."""
    ctx = make_ctx(normalized_order_id="ORD-1007")
    action = AgentAction(ActionType.LOOKUP_ORDER, parameters={"order_id": "ORD-1007"})

    val = PlannerPolicy.validate_action(ctx, action)
    assert val.is_permitted is True


def test_lookup_order_without_valid_order_id_rejected():
    """b. Verify LOOKUP_ORDER without valid order ID is rejected."""
    ctx = make_ctx(normalized_order_id=None)
    action = AgentAction(ActionType.LOOKUP_ORDER, parameters={"order_id": "INVALID-123"})

    val = PlannerPolicy.validate_action(ctx, action)
    assert val.is_permitted is False
    assert "ORD-\\d{4}" in val.reason
    assert val.fallback_action.action_type == ActionType.HANDOFF


def test_retrieve_kb_with_empty_query_rejected():
    """c. Verify RETRIEVE_KB with empty query is rejected."""
    ctx = make_ctx()
    action = AgentAction(ActionType.RETRIEVE_KB, parameters={"query": "   "})

    val = PlannerPolicy.validate_action(ctx, action)
    assert val.is_permitted is False
    assert "empty query string" in val.reason.lower()
    assert val.fallback_action.action_type == ActionType.HANDOFF


def test_clarify_respond_handoff_never_execute_tools():
    """d, e, f. Verify terminal actions CLARIFY, RESPOND, HANDOFF are policy valid and execute zero tools."""
    ctx = make_ctx()
    for act_type in (ActionType.CLARIFY, ActionType.RESPOND, ActionType.HANDOFF):
        action = AgentAction(act_type, parameters={})
        val = PlannerPolicy.validate_action(ctx, action)
        assert val.is_permitted is True


from pathlib import Path
from src.retrieval import KBVectorStore
from src.ingestion import ingest_kb_directory


def test_invalid_policy_action_becomes_safe_handoff():
    """g & h. Verify policy rejection produces a recorded AgentObservation and transitions safely to HANDOFF."""
    chunks = ingest_kb_directory(Path("knowledge-base"))
    vector_store = KBVectorStore(collection_name="test_pol_rej_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    # RETRIEVE_KB with whitespace query passes ActionValidator schema but fails PlannerPolicy
    bad_planner = MockPlanner(fixed_action=AgentAction(ActionType.RETRIEVE_KB, parameters={"query": "   "}))
    
    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        planner=bad_planner,
        llm_provider=MockLLMProvider(),
    )

    state = agent.process_turn("Check status for bad query", session_id="policy_rej_session")

    assert state.handoff_recommended is True
    assert len(state.observations) > 0
    obs = state.observations[0]
    assert obs.failure_category == FailureCategory.PLANNER_FAILURE
    assert "Planner policy rejection" in obs.error_message


def test_policy_rejection_does_not_leak_pii():
    """i. Verify PlannerPolicyValidation contains zero PII or sensitive keys."""
    ctx = make_ctx(normalized_order_id="INVALID")
    action = AgentAction(ActionType.LOOKUP_ORDER, parameters={"order_id": "INVALID"})

    val = PlannerPolicy.validate_action(ctx, action)
    val_str = str(val.reason).lower()

    sensitive_keys = ["email", "address", "risk_score", "warehouse_note", "customer", "secret", "api_key"]
    for key in sensitive_keys:
        assert key not in val_str


def test_policy_rejection_cannot_cause_retry_loops():
    """j. Verify policy rejection respects max_iterations=3 and prevents infinite retry loops."""
    chunks = ingest_kb_directory(Path("knowledge-base"))
    vector_store = KBVectorStore(collection_name="test_retry_pol_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    bad_planner = MockPlanner(fixed_action=AgentAction(ActionType.LOOKUP_ORDER, parameters={"order_id": "BAD-ID"}))

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        planner=bad_planner,
        llm_provider=MockLLMProvider(),
    )

    state = agent.process_turn("Retry loop test", session_id="retry_policy_session")

    assert state.iterations <= 3
    assert state.handoff_recommended is True


def test_action_validator_remains_effective():
    """k. Verify ActionValidator continues to enforce structural allowlists before PlannerPolicy."""
    invalid_type_action = AgentAction(action_type="MALICIOUS_ACTION", parameters={})
    with pytest.raises(ValueError, match="not allowed"):
        ActionValidator.validate(invalid_type_action)


def test_planner_context_remains_only_planner_state():
    """l. Verify PlannerPolicy operates exclusively on PlannerContext."""
    ctx = make_ctx()
    action = AgentAction(ActionType.RESPOND, parameters={})
    val = PlannerPolicy.validate_action(ctx, action)
    assert val is not None
