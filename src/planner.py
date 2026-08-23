import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set

from src.llm import BaseLLMProvider, get_default_provider


class ActionType:
    """Allowed action types for the agent planner."""
    RETRIEVE_KB = "RETRIEVE_KB"
    LOOKUP_ORDER = "LOOKUP_ORDER"
    CLARIFY = "CLARIFY"
    RESPOND = "RESPOND"
    HANDOFF = "HANDOFF"

    @classmethod
    def all_allowed(cls) -> Set[str]:
        return {
            cls.RETRIEVE_KB,
            cls.LOOKUP_ORDER,
            cls.CLARIFY,
            cls.RESPOND,
            cls.HANDOFF,
        }


@dataclass
class AgentAction:
    """
    Represents a structured action planned by the agent planner.
    """
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }


@dataclass
class AgentObservation:
    """
    Represents the execution observation of a single AgentAction.
    Contains only sanitized evidence or customer-safe order results.
    """
    action_type: str
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    handoff_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        summary = None
        if self.result is not None:
            if isinstance(self.result, list):
                summary = f"Retrieved {len(self.result)} items"
            elif hasattr(self.result, "to_dict"):
                summary = self.result.to_dict()
            else:
                summary = str(self.result)

        return {
            "action_type": self.action_type,
            "success": self.success,
            "result_summary": summary,
            "error_message": self.error_message,
            "handoff_recommended": self.handoff_recommended,
        }


class ActionValidator:
    """
    Strict action validator ensuring planned actions comply with allowlist rules
    and exact parameter schemas.
    """

    ALLOWED_ACTIONS = ActionType.all_allowed()
    ORDER_ID_REGEX = re.compile(r"^ORD-\d{4}$", re.IGNORECASE)

    @classmethod
    def validate(cls, action: AgentAction) -> bool:
        if not action or not isinstance(action, AgentAction):
            raise ValueError("Invalid action object: Must be an instance of AgentAction.")

        if action.action_type not in cls.ALLOWED_ACTIONS:
            raise ValueError(
                f"Action type '{action.action_type}' is not allowed. "
                f"Allowed action types: {sorted(list(cls.ALLOWED_ACTIONS))}"
            )

        params = action.parameters if isinstance(action.parameters, dict) else {}

        # 1. Parameter schema validation for LOOKUP_ORDER
        if action.action_type == ActionType.LOOKUP_ORDER:
            allowed_keys = {"order_id"}
            unexpected_keys = set(params.keys()) - allowed_keys
            if unexpected_keys:
                raise ValueError(f"LOOKUP_ORDER action received unexpected parameters: {unexpected_keys}")

            order_id = params.get("order_id")
            if not order_id or not isinstance(order_id, str) or not order_id.strip():
                raise ValueError("LOOKUP_ORDER action requires a non-empty string 'order_id' parameter.")
            
            cleaned_oid = order_id.strip()
            if not cls.ORDER_ID_REGEX.match(cleaned_oid):
                raise ValueError(f"LOOKUP_ORDER order_id '{order_id}' does not match required format 'ORD-\\d{{4}}'.")

        # 2. Parameter schema validation for RETRIEVE_KB
        elif action.action_type == ActionType.RETRIEVE_KB:
            allowed_keys = {"query"}
            unexpected_keys = set(params.keys()) - allowed_keys
            if unexpected_keys:
                raise ValueError(f"RETRIEVE_KB action received unexpected parameters: {unexpected_keys}")

            query = params.get("query")
            if query is not None and not isinstance(query, str):
                raise ValueError("RETRIEVE_KB action 'query' parameter must be a string if provided.")

        # 3. Parameter schema validation for zero-parameter actions (CLARIFY, RESPOND, HANDOFF)
        elif action.action_type in (ActionType.CLARIFY, ActionType.RESPOND, ActionType.HANDOFF):
            if params and len(params) > 0:
                raise ValueError(f"{action.action_type} action rejects unexpected parameters: {set(params.keys())}")

        return True


class BasePlanner(ABC):
    """Abstract base interface for agent planners."""

    @abstractmethod
    def plan_next_action(self, agent_state: Any) -> AgentAction:
        """Given current AgentState, plan the next validated AgentAction."""
        pass


class MockPlanner(BasePlanner):
    """
    Deterministic mock planner for unit testing without live LLM calls.
    Decides the next action based on structured state, intent, and prior observations.
    """

    def __init__(self, fixed_action: Optional[AgentAction] = None):
        self.fixed_action = fixed_action

    def plan_next_action(self, agent_state: Any) -> AgentAction:
        if self.fixed_action:
            ActionValidator.validate(self.fixed_action)
            return self.fixed_action

        # Inspect state and prior observations
        intent = getattr(agent_state, "intent_category", "policy")
        order_id = getattr(agent_state, "normalized_order_id", None)
        handoff = getattr(agent_state, "handoff_recommended", False)
        observations = getattr(agent_state, "observations", [])
        evidence_chunks = getattr(agent_state, "evidence_chunks", [])
        order_result = getattr(agent_state, "order_result", None)

        if handoff:
            action = AgentAction(
                action_type=ActionType.HANDOFF,
                reasoning="Handoff recommended due to policy conflict, unknown order, or privacy request."
            )
        elif intent == "clarification":
            action = AgentAction(
                action_type=ActionType.CLARIFY,
                reasoning="Missing order ID for order status question; requesting clarification."
            )
        elif observations:
            # Check prior observations to avoid repeating actions
            last_obs = observations[-1]
            if last_obs.action_type == ActionType.LOOKUP_ORDER:
                if order_result and not order_result.found:
                    action = AgentAction(action_type=ActionType.HANDOFF, reasoning="Order not found; triggering handoff.")
                else:
                    action = AgentAction(action_type=ActionType.RESPOND, reasoning="Order status retrieved; generating response.")
            elif last_obs.action_type == ActionType.RETRIEVE_KB:
                if not evidence_chunks:
                    action = AgentAction(action_type=ActionType.HANDOFF, reasoning="No evidence retrieved; triggering handoff.")
                else:
                    action = AgentAction(action_type=ActionType.RESPOND, reasoning="KB evidence retrieved; generating response.")
            else:
                action = AgentAction(action_type=ActionType.RESPOND, reasoning="Observation recorded; responding.")
        elif intent == "order_status" and order_id:
            action = AgentAction(
                action_type=ActionType.LOOKUP_ORDER,
                parameters={"order_id": order_id},
                reasoning=f"Valid order ID '{order_id}' present; looking up order status."
            )
        else:
            action = AgentAction(
                action_type=ActionType.RETRIEVE_KB,
                parameters={"query": getattr(agent_state, "retrieval_query", getattr(agent_state, "user_query", ""))},
                reasoning="Policy question; retrieving KB evidence."
            )

        ActionValidator.validate(action)
        return action


PLANNER_SYSTEM_PROMPT = """You are the action planner for the Aster & Row customer support AI agent.
Analyze the user query, session context, and prior action observations, then output a JSON object deciding the single next action.

CRITICAL RULES:
1. You MUST respond with ONLY a valid JSON object matching this schema:
   {
     "action_type": "<ACTION_TYPE>",
     "parameters": { ... },
     "reasoning": "<brief technical rationale>"
   }
2. The "action_type" MUST be strictly one of:
   - "RETRIEVE_KB": To search knowledge-base policies. Parameters: {"query": "search query string"}
   - "LOOKUP_ORDER": To check order status. Parameters: {"order_id": "ORD-XXXX"}
   - "CLARIFY": When an order status is requested but the order ID is missing. Parameters: {}
   - "RESPOND": To generate a grounded response when evidence is already available. Parameters: {}
   - "HANDOFF": When human escalation is required due to policy conflict, privacy, unknown order, or missing info. Parameters: {}
3. If prior observations show that KB retrieval or order lookup has ALREADY succeeded, plan "RESPOND".
4. If prior observations show an unknown order or empty evidence, plan "HANDOFF".
5. Do NOT repeat an action if evidence is already present in state.
6. Do NOT output markdown outside the JSON block. Do NOT include any unapproved action types.
"""


class LLMPlanner(BasePlanner):
    """
    LLM-driven action planner that uses the BaseLLMProvider abstraction
    to generate structured JSON actions, validated by ActionValidator.
    Incorporate prior action observations into planning context.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or get_default_provider()

    @staticmethod
    def _clean_json_response(raw_text: str) -> str:
        """Strip markdown code blocks or surrounding text to extract raw JSON string."""
        text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_obj:
            return match_obj.group(1).strip()
        return text

    def plan_next_action(self, agent_state: Any) -> AgentAction:
        """
        Calls the LLMProvider to generate a JSON action plan, parses it,
        and validates it through ActionValidator.
        If malformed or invalid, safely falls back to a HANDOFF action.
        """
        user_query = getattr(agent_state, "user_query", "")
        retrieval_q = getattr(agent_state, "retrieval_query", user_query)
        order_id = getattr(agent_state, "normalized_order_id", None)
        intent = getattr(agent_state, "intent_category", "policy")
        observations = getattr(agent_state, "observations", [])

        obs_summaries = []
        for i, obs in enumerate(observations, 1):
            obs_summaries.append(f"Observation {i}: Action={obs.action_type}, Success={obs.success}, Summary={obs.to_dict()['result_summary']}")
        obs_text = "\n".join(obs_summaries) if obs_summaries else "None"

        user_prompt = (
            f"User Query: {user_query}\n"
            f"Retrieval Query Context: {retrieval_q}\n"
            f"Detected Intent: {intent}\n"
            f"Normalized Order ID: {order_id or 'None'}\n"
            f"Prior Observations:\n{obs_text}\n"
            f"Plan the single next ActionType JSON object."
        )

        try:
            raw_response = self.provider.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            json_text = self._clean_json_response(raw_response)
            data = json.loads(json_text)

            if not isinstance(data, dict):
                raise ValueError("Parsed LLM planner output is not a JSON dictionary.")

            action = AgentAction(
                action_type=data.get("action_type", ""),
                parameters=data.get("parameters", {}),
                reasoning=data.get("reasoning", "LLM planned action"),
            )

            # Strict allowlist & parameter schema validation
            ActionValidator.validate(action)
            return action

        except Exception as e:
            # Safe fallback for malformed JSON, invalid action type, or provider error
            return AgentAction(
                action_type=ActionType.HANDOFF,
                parameters={},
                reasoning=f"LLM planner error / invalid output fallback: {str(e)}"
            )
