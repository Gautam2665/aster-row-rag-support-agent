import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.planner import ActionType, AgentAction, PlannerContext, FailureCategory


@dataclass
class PlannerPolicyValidation:
    """
    Result of evaluating an AgentAction against application-level state policies.
    """
    is_permitted: bool
    reason: str
    fallback_action: Optional[AgentAction] = None


class PlannerPolicy:
    """
    Deterministic, state-aware planner policy enforcement layer.
    Ensures LLM planner recommendations comply with state rules before tool execution.

    Flow:
        LLM Output -> ActionValidator (Schema/Allowlist) -> PlannerPolicy (State Rules) -> SupportAgent Execution
    """

    ORDER_ID_REGEX = re.compile(r"^ORD-\d{4}$", re.IGNORECASE)

    @classmethod
    def validate_action(cls, context: PlannerContext, action: AgentAction) -> PlannerPolicyValidation:
        """
        Validates whether the proposed AgentAction is permitted under current PlannerContext state.
        Does NOT execute tools, make network requests, or call LLMs.
        """
        if not action or not isinstance(action, AgentAction):
            return PlannerPolicyValidation(
                is_permitted=False,
                reason="Invalid action object provided to policy.",
                fallback_action=AgentAction(ActionType.HANDOFF, reasoning="Invalid action object fallback"),
            )

        act_type = action.action_type
        params = action.parameters or {}

        # 1. Policy Rule for LOOKUP_ORDER
        if act_type == ActionType.LOOKUP_ORDER:
            order_id = params.get("order_id") or context.normalized_order_id
            if not order_id or not isinstance(order_id, str) or not cls.ORDER_ID_REGEX.match(order_id.strip()):
                return PlannerPolicyValidation(
                    is_permitted=False,
                    reason=f"LOOKUP_ORDER action proposed without a valid order ID matching 'ORD-\\d{{4}}' (got '{order_id}').",
                    fallback_action=AgentAction(
                        ActionType.HANDOFF,
                        reasoning="Policy rejected LOOKUP_ORDER due to missing/invalid order_id parameter."
                    ),
                )

        # 2. Policy Rule for RETRIEVE_KB
        elif act_type == ActionType.RETRIEVE_KB:
            query = params.get("query")
            if query is not None and isinstance(query, str) and not query.strip():
                return PlannerPolicyValidation(
                    is_permitted=False,
                    reason="RETRIEVE_KB action proposed with an empty query string.",
                    fallback_action=AgentAction(
                        ActionType.HANDOFF,
                        reasoning="Policy rejected RETRIEVE_KB due to empty query parameter."
                    ),
                )

        # 3. Policy Rule for CLARIFY, RESPOND, HANDOFF (Zero-tool terminal actions)
        elif act_type in ActionType.TERMINAL_ACTIONS:
            pass

        return PlannerPolicyValidation(
            is_permitted=True,
            reason="Action is policy permitted.",
            fallback_action=None,
        )
