import pytest
from src.planner import (
    AgentAction,
    ActionType,
    ActionValidator,
    BasePlanner,
    MockPlanner,
    LLMPlanner,
)
from src.agent import AgentState
from src.llm import BaseLLMProvider


class CustomMockLLMProvider(BaseLLMProvider):
    """Custom mock provider for testing LLMPlanner responses."""
    def __init__(self, response_text: str = "", raise_error: bool = False):
        self.response_text = response_text
        self.raise_error = raise_error

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.raise_error:
            raise RuntimeError("LLM provider API failure")
        return self.response_text


def test_valid_action_creation():
    """Test creating and validating all valid action types."""
    actions = [
        AgentAction(action_type=ActionType.RETRIEVE_KB, parameters={"query": "return window"}),
        AgentAction(action_type=ActionType.LOOKUP_ORDER, parameters={"order_id": "ORD-1007"}),
        AgentAction(action_type=ActionType.CLARIFY, reasoning="Need order ID"),
        AgentAction(action_type=ActionType.RESPOND, reasoning="Generate grounded response"),
        AgentAction(action_type=ActionType.HANDOFF, reasoning="Escalate to support"),
    ]

    for action in actions:
        assert ActionValidator.validate(action) is True
        assert action.action_type in ActionType.all_allowed()


def test_invalid_action_type_rejection():
    """Test that unapproved action types raise ValueError."""
    invalid_actions = [
        AgentAction(action_type="UNAPPROVED_ACTION"),
        AgentAction(action_type="EXECUTE_SQL"),
        AgentAction(action_type="DELETE_DATABASE"),
    ]

    for action in invalid_actions:
        with pytest.raises(ValueError) as exc_info:
            ActionValidator.validate(action)
        assert "not allowed" in str(exc_info.value)


def test_malformed_order_lookup_arguments():
    """Test that LOOKUP_ORDER with missing, empty, or non-string order_id raises ValueError."""
    malformed_actions = [
        AgentAction(action_type=ActionType.LOOKUP_ORDER, parameters={}),
        AgentAction(action_type=ActionType.LOOKUP_ORDER, parameters={"order_id": ""}),
        AgentAction(action_type=ActionType.LOOKUP_ORDER, parameters={"order_id": "   "}),
        AgentAction(action_type=ActionType.LOOKUP_ORDER, parameters={"order_id": 1007}),
    ]

    for action in malformed_actions:
        with pytest.raises(ValueError) as exc_info:
            ActionValidator.validate(action)
        assert "LOOKUP_ORDER action requires a non-empty string 'order_id'" in str(exc_info.value)


def test_malformed_retrieve_kb_arguments():
    """Test that RETRIEVE_KB with non-string query parameter raises ValueError."""
    action = AgentAction(action_type=ActionType.RETRIEVE_KB, parameters={"query": 12345})
    with pytest.raises(ValueError) as exc_info:
        ActionValidator.validate(action)
    assert "'query' parameter must be a string" in str(exc_info.value)


def test_planner_allowlisting_strictness():
    """Test that ActionType allowlist contains strictly the 5 approved action types."""
    allowed = ActionType.all_allowed()
    assert len(allowed) == 5
    assert "RETRIEVE_KB" in allowed
    assert "LOOKUP_ORDER" in allowed
    assert "CLARIFY" in allowed
    assert "RESPOND" in allowed
    assert "HANDOFF" in allowed


def test_mock_planner_deterministic_decisions():
    """Test MockPlanner produces validated decisions for different state configurations."""
    planner = MockPlanner()

    # Policy state
    state_policy = AgentState(session_id="s1", user_query="What is the return window?", intent_category="policy")
    action1 = planner.plan_next_action(state_policy)
    assert action1.action_type == ActionType.RETRIEVE_KB

    # Order state with ID
    state_order = AgentState(session_id="s2", user_query="Where is ORD-1007?", normalized_order_id="ORD-1007", intent_category="order_status")
    action2 = planner.plan_next_action(state_order)
    assert action2.action_type == ActionType.LOOKUP_ORDER
    assert action2.parameters["order_id"] == "ORD-1007"

    # Clarification state
    state_clarify = AgentState(session_id="s3", user_query="Where is my order?", intent_category="clarification")
    action3 = planner.plan_next_action(state_clarify)
    assert action3.action_type == ActionType.CLARIFY

    # Handoff state
    state_handoff = AgentState(session_id="s4", user_query="Check ORD-9999", handoff_recommended=True)
    action4 = planner.plan_next_action(state_handoff)
    assert action4.action_type == ActionType.HANDOFF


# --- LLMPlanner Tests ---

def test_llm_planner_valid_structured_action():
    """Test LLMPlanner successfully parses and validates a structured JSON action from LLM output."""
    json_resp = """```json
    {
        "action_type": "LOOKUP_ORDER",
        "parameters": {"order_id": "ORD-1007"},
        "reasoning": "User provided order ID ORD-1007"
    }
    ```"""
    provider = CustomMockLLMProvider(response_text=json_resp)
    planner = LLMPlanner(provider=provider)

    state = AgentState(session_id="s_llm_1", user_query="Where is ORD-1007?", normalized_order_id="ORD-1007")
    action = planner.plan_next_action(state)

    assert action.action_type == ActionType.LOOKUP_ORDER
    assert action.parameters == {"order_id": "ORD-1007"}
    assert action.reasoning == "User provided order ID ORD-1007"


def test_llm_planner_invalid_action_type():
    """Test LLMPlanner safely falls back to HANDOFF when LLM returns an unapproved action type."""
    json_resp = '{"action_type": "UNAPPROVED_ACTION", "parameters": {}}'
    provider = CustomMockLLMProvider(response_text=json_resp)
    planner = LLMPlanner(provider=provider)

    state = AgentState(session_id="s_llm_2", user_query="Do something forbidden")
    action = planner.plan_next_action(state)

    assert action.action_type == ActionType.HANDOFF
    assert "fallback" in action.reasoning.lower()


def test_llm_planner_missing_parameters():
    """Test LLMPlanner safely falls back to HANDOFF when LOOKUP_ORDER is missing order_id."""
    json_resp = '{"action_type": "LOOKUP_ORDER", "parameters": {}}'
    provider = CustomMockLLMProvider(response_text=json_resp)
    planner = LLMPlanner(provider=provider)

    state = AgentState(session_id="s_llm_3", user_query="Where is my order?")
    action = planner.plan_next_action(state)

    assert action.action_type == ActionType.HANDOFF
    assert "fallback" in action.reasoning.lower()


def test_llm_planner_malformed_json_output():
    """Test LLMPlanner safely falls back to HANDOFF when LLM returns plain text non-JSON."""
    text_resp = "I think we should check the order for the customer."
    provider = CustomMockLLMProvider(response_text=text_resp)
    planner = LLMPlanner(provider=provider)

    state = AgentState(session_id="s_llm_4", user_query="Check my package")
    action = planner.plan_next_action(state)

    assert action.action_type == ActionType.HANDOFF
    assert "fallback" in action.reasoning.lower()


def test_llm_planner_provider_failure():
    """Test LLMPlanner safely falls back to HANDOFF when LLM provider raises an API exception."""
    provider = CustomMockLLMProvider(raise_error=True)
    planner = LLMPlanner(provider=provider)

    state = AgentState(session_id="s_llm_5", user_query="What is the return window?")
    action = planner.plan_next_action(state)

    assert action.action_type == ActionType.HANDOFF
    assert "fallback" in action.reasoning.lower()
