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

    TERMINAL_ACTIONS = {CLARIFY, RESPOND, HANDOFF}
    NON_TERMINAL_ACTIONS = {RETRIEVE_KB, LOOKUP_ORDER}

    @classmethod
    def all_allowed(cls) -> Set[str]:
        return {
            cls.RETRIEVE_KB,
            cls.LOOKUP_ORDER,
            cls.CLARIFY,
            cls.RESPOND,
            cls.HANDOFF,
        }

    @classmethod
    def is_terminal(cls, action_type: str) -> bool:
        return action_type in cls.TERMINAL_ACTIONS


@dataclass
class AgentAction:
    """
    Represents a structured action planned by the agent planner.
    """
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return ActionType.is_terminal(self.action_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }


class FailureCategory:
    """Explicit failure categories for agent execution and recovery policy."""
    TOOL_ERROR = "TOOL_ERROR"               # Tool execution raised an exception or failed
    BUSINESS_FAILURE = "BUSINESS_FAILURE"   # Operation executed successfully but domain rules could not fulfill (e.g. unknown order)
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE" # Knowledge base retrieval returned no usable evidence
    PLANNER_FAILURE = "PLANNER_FAILURE"     # Planner provider failed or produced unapproved/invalid action


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
    failure_category: Optional[str] = None
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
            "failure_category": self.failure_category,
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


@dataclass
class PlannerContext:
    """
    Explicit, bounded context payload supplied to the agent planner.
    Isolates planner decision-making from the full, mutable AgentState object.
    Guarantees no customer PII or raw database objects enter the planner.
    """
    user_query: str
    retrieval_query: str
    intent_category: str
    normalized_order_id: Optional[str] = None
    has_usable_evidence: bool = False
    evidence_count: int = 0
    has_order_result: bool = False
    order_found: Optional[bool] = None
    observations_summary: List[Dict[str, Any]] = field(default_factory=list)
    failure_category: Optional[str] = None
    handoff_recommended: bool = False
    iterations: int = 0

    @classmethod
    def from_agent_state(cls, state: Any) -> "PlannerContext":
        """
        Deterministic factory constructing a bounded PlannerContext from an AgentState object.
        """
        if isinstance(state, cls):
            return state

        user_query = getattr(state, "user_query", "")
        retrieval_query = getattr(state, "retrieval_query", user_query)
        intent_category = getattr(state, "intent_category", "policy")
        normalized_order_id = getattr(state, "normalized_order_id", None)
        handoff_recommended = getattr(state, "handoff_recommended", False)
        iterations = getattr(state, "iterations", 0)

        evidence_chunks = getattr(state, "evidence_chunks", [])
        has_usable_evidence = bool(evidence_chunks and len(evidence_chunks) > 0)
        evidence_count = len(evidence_chunks) if evidence_chunks else 0

        order_result = getattr(state, "order_result", None)
        has_order_result = order_result is not None
        order_found = getattr(order_result, "found", None) if order_result else None

        raw_observations = getattr(state, "observations", [])
        obs_summaries = []
        last_failure_cat = None

        for obs in raw_observations:
            if hasattr(obs, "to_dict"):
                obs_dict = obs.to_dict()
            elif isinstance(obs, dict):
                obs_dict = obs.copy()
            else:
                obs_dict = {
                    "action_type": str(getattr(obs, "action_type", "")),
                    "success": bool(getattr(obs, "success", False)),
                    "result_summary": str(getattr(obs, "result", "")),
                    "failure_category": getattr(obs, "failure_category", None),
                    "handoff_recommended": bool(getattr(obs, "handoff_recommended", False)),
                }

            # Security sanitization: strip any sensitive customer keys
            res_summary = obs_dict.get("result_summary")
            if isinstance(res_summary, dict):
                clean_res = {
                    k: v for k, v in res_summary.items()
                    if k not in ("email", "address", "risk_score", "warehouse_note", "customer")
                }
                obs_dict["result_summary"] = clean_res

            obs_summaries.append(obs_dict)
            if obs_dict.get("failure_category"):
                last_failure_cat = obs_dict.get("failure_category")

        return cls(
            user_query=user_query,
            retrieval_query=retrieval_query,
            intent_category=intent_category,
            normalized_order_id=normalized_order_id,
            has_usable_evidence=has_usable_evidence,
            evidence_count=evidence_count,
            has_order_result=has_order_result,
            order_found=order_found,
            observations_summary=obs_summaries,
            failure_category=last_failure_cat,
            handoff_recommended=handoff_recommended,
            iterations=iterations,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert PlannerContext to dictionary, ensuring zero PII or raw DB objects exist."""
        return {
            "user_query": self.user_query,
            "retrieval_query": self.retrieval_query,
            "intent_category": self.intent_category,
            "normalized_order_id": self.normalized_order_id,
            "has_usable_evidence": self.has_usable_evidence,
            "evidence_count": self.evidence_count,
            "has_order_result": self.has_order_result,
            "order_found": self.order_found,
            "observations_summary": self.observations_summary,
            "failure_category": self.failure_category,
            "handoff_recommended": self.handoff_recommended,
            "iterations": self.iterations,
        }


class BasePlanner(ABC):
    """Abstract base interface for agent planners operating on PlannerContext."""

    @abstractmethod
    def plan_next_action(self, context: Any) -> AgentAction:
        """Given a PlannerContext (or AgentState), plan the next validated AgentAction."""
        pass


class MockPlanner(BasePlanner):
    """
    Deterministic mock planner for unit testing without live LLM calls.
    Decides the next action based on structured PlannerContext.
    """

    def __init__(self, fixed_action: Optional[AgentAction] = None):
        self.fixed_action = fixed_action

    def plan_next_action(self, context: Any) -> AgentAction:
        if self.fixed_action:
            ActionValidator.validate(self.fixed_action)
            return self.fixed_action

        planner_ctx = PlannerContext.from_agent_state(context)

        if planner_ctx.handoff_recommended:
            action = AgentAction(
                action_type=ActionType.HANDOFF,
                reasoning="Handoff recommended due to policy conflict, unknown order, or privacy request."
            )
        elif planner_ctx.intent_category == "clarification":
            action = AgentAction(
                action_type=ActionType.CLARIFY,
                reasoning="Missing order ID for order status question; requesting clarification."
            )
        elif planner_ctx.observations_summary:
            last_obs = planner_ctx.observations_summary[-1]
            last_action_type = last_obs.get("action_type")
            if last_action_type == ActionType.LOOKUP_ORDER:
                if planner_ctx.has_order_result and planner_ctx.order_found is False:
                    action = AgentAction(action_type=ActionType.HANDOFF, reasoning="Order not found; triggering handoff.")
                else:
                    action = AgentAction(action_type=ActionType.RESPOND, reasoning="Order status retrieved; generating response.")
            elif last_action_type == ActionType.RETRIEVE_KB:
                if not planner_ctx.has_usable_evidence:
                    action = AgentAction(action_type=ActionType.HANDOFF, reasoning="No evidence retrieved; triggering handoff.")
                else:
                    action = AgentAction(action_type=ActionType.RESPOND, reasoning="KB evidence retrieved; generating response.")
            else:
                action = AgentAction(action_type=ActionType.RESPOND, reasoning="Observation recorded; responding.")
        elif planner_ctx.intent_category == "order_status" and planner_ctx.normalized_order_id:
            action = AgentAction(
                action_type=ActionType.LOOKUP_ORDER,
                parameters={"order_id": planner_ctx.normalized_order_id},
                reasoning=f"Valid order ID '{planner_ctx.normalized_order_id}' present; looking up order status."
            )
        else:
            action = AgentAction(
                action_type=ActionType.RETRIEVE_KB,
                parameters={"query": planner_ctx.retrieval_query or planner_ctx.user_query},
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
    to generate structured JSON actions from PlannerContext, validated by ActionValidator.
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

    def plan_next_action(self, context: Any) -> AgentAction:
        """
        Calls the LLMProvider to generate a JSON action plan from PlannerContext, parses it,
        and validates it through ActionValidator.
        If malformed or invalid, safely falls back to a HANDOFF action.
        """
        planner_ctx = PlannerContext.from_agent_state(context)

        obs_summaries = []
        for i, obs in enumerate(planner_ctx.observations_summary, 1):
            action_type = obs.get("action_type", "")
            success = obs.get("success", False)
            summary = obs.get("result_summary", "")
            obs_summaries.append(f"Observation {i}: Action={action_type}, Success={success}, Summary={summary}")
        obs_text = "\n".join(obs_summaries) if obs_summaries else "None"

        user_prompt = (
            f"User Query: {planner_ctx.user_query}\n"
            f"Retrieval Query Context: {planner_ctx.retrieval_query}\n"
            f"Detected Intent: {planner_ctx.intent_category}\n"
            f"Normalized Order ID: {planner_ctx.normalized_order_id or 'None'}\n"
            f"Has Usable Evidence: {planner_ctx.has_usable_evidence} (count: {planner_ctx.evidence_count})\n"
            f"Has Order Result: {planner_ctx.has_order_result} (found: {planner_ctx.order_found})\n"
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
