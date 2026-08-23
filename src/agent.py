import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.models import KBChunk, DocumentMetadata
from src.ingestion import parse_markdown_file, parse_markdown_sections
from src.embeddings import EmbeddingProvider
from src.retrieval import KBVectorStore
from src.prompt_builder import build_grounded_prompt
from src.llm import BaseLLMProvider, GroundedResponse, get_default_provider
from src.tools.order_lookup import OrderLookupTool, CustomerSafeOrderResult
from src.memory import SessionMemoryStore, ConversationTurn
from src.context import ContextBuilder
from src.query_context import QueryContextualizer
from src.planner import BasePlanner, MockPlanner, AgentAction, AgentObservation, ActionType, ActionValidator, FailureCategory
from src.trace import TraceEvent, TraceEventType


@dataclass
class AgentState:
    """
    State object tracking a single interaction turn through the agent state machine.
    """
    session_id: str
    user_query: str
    retrieval_query: Optional[str] = None
    normalized_order_id: Optional[str] = None
    intent_category: str = "general_policy"  # "policy", "order_status", "clarification"

    # Context & Execution State
    evidence_chunks: List[KBChunk] = field(default_factory=list)
    order_result: Optional[CustomerSafeOrderResult] = None
    tool_calls_made: List[str] = field(default_factory=list)
    history_turns: List[ConversationTurn] = field(default_factory=list)
    planned_actions: List[AgentAction] = field(default_factory=list)
    observations: List[AgentObservation] = field(default_factory=list)
    trace: List[TraceEvent] = field(default_factory=list)
    
    # Bounded Loop Control
    iterations: int = 0
    max_iterations: int = 3
    handoff_recommended: bool = False

    # Outputs
    final_answer: Optional[str] = None
    citations: List[str] = field(default_factory=list)


class SupportAgent:
    """
    Bounded Support Agent state machine orchestrating RAG retrieval,
    query contextualization, secure order lookups, prompt building,
    bounded conversation memory, action planning, observation tracking, and LLM generation.
    """

    ORDER_ID_REGEX = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
    ORDER_STATUS_PHRASES = [
        "where is my order", "order status", "track my order", "where is my package",
        "status of my order", "when will my order arrive", "where is order", "status of order",
        "where is my", "when will it arrive", "where is"
    ]
    PRIVACY_KEYWORDS = {"email", "address", "risk score", "internal note", "fraud review", "risk_score"}

    def __init__(
        self,
        vector_store: Optional[KBVectorStore] = None,
        order_tool: Optional[OrderLookupTool] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        planner: Optional[BasePlanner] = None,
        memory_store: Optional[SessionMemoryStore] = None,
        max_iterations: int = 3,
        max_history_turns: int = 5,
    ):
        self.vector_store = vector_store or KBVectorStore()
        self.order_tool = order_tool or OrderLookupTool()
        self.llm_provider = llm_provider or get_default_provider()
        self.planner = planner or MockPlanner()
        self.memory_store = memory_store or SessionMemoryStore(max_turns_per_session=max_history_turns)
        self.max_iterations = max_iterations

        # Explicit tool allowlist
        self.allowed_tools = {
            "order_lookup": self.order_tool
        }

    def extract_order_id(self, text: str) -> Optional[str]:
        """Extract and normalize order ID (e.g. 'ORD-1007') from text."""
        match = self.ORDER_ID_REGEX.search(text)
        if match:
            return match.group(0).upper()
        return None

    def detect_intent(self, user_query: str, order_id: Optional[str]) -> str:
        """Classify user query into policy, order_status, or clarification intent."""
        query_lower = user_query.lower()

        if order_id:
            return "order_status"

        # Only trigger clarification if the user is explicitly asking for order status / tracking
        is_order_tracking_query = any(phrase in query_lower for phrase in self.ORDER_STATUS_PHRASES) or query_lower.strip() in ("where is my order", "where is my order?", "where's my order")
        
        if is_order_tracking_query:
            return "clarification"
        
        return "policy"

    def execute_tool_safely(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool only if present in the explicit allowlist."""
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"Tool '{tool_name}' is not in the explicit tool allowlist.")
        
        tool = self.allowed_tools[tool_name]
        if tool_name == "order_lookup":
            return tool.lookup(kwargs.get("order_id", ""))
        raise ValueError(f"Unknown tool execution handler for '{tool_name}'.")

    def format_order_data_context(self, order_res: CustomerSafeOrderResult) -> str:
        """Format sanitized CustomerSafeOrderResult into safe XML data context."""
        if not order_res.found:
            return (
                f"<order_lookup_data>\n"
                f"Status: ORDER_NOT_FOUND\n"
                f"Order ID: {order_res.order_id}\n"
                f"Message: {order_res.error_message}\n"
                f"Note to Assistant: Explain that order '{order_res.order_id}' was not found. Ask customer to verify their order ID or offer human support escalation.\n"
                f"</order_lookup_data>"
            )

        items_str = ", ".join(f"{it.quantity}x {it.name} (Final Sale: {it.final_sale})" for it in order_res.items)
        eta_str = order_res.estimated_delivery if order_res.estimated_delivery else "Unavailable / Not Applicable"

        return (
            f"<order_lookup_data>\n"
            f"Order ID: {order_res.order_id}\n"
            f"Status: {order_res.status}\n"
            f"Membership Tier: {order_res.membership_tier}\n"
            f"Placed At: {order_res.placed_at}\n"
            f"Carrier: {order_res.carrier or 'None'}\n"
            f"Tracking Number: {order_res.tracking_number or 'None'}\n"
            f"Estimated Delivery: {eta_str}\n"
            f"Customer Safe Message: {order_res.customer_safe_message}\n"
            f"Items: {items_str}\n"
            f"</order_lookup_data>"
        )

    def process_turn(self, user_query: str, session_id: str = "default_session") -> AgentState:
        """
        Execute a single turn through the bounded state machine with planner-driven action execution,
        explicit observation recording, and structured lifecycle trace logging.
        """
        normalized_order_id = self.extract_order_id(user_query)
        intent = self.detect_intent(user_query, normalized_order_id)

        # Retrieve bounded conversation history for this session
        recent_history = self.memory_store.get_recent_turns(session_id)

        # Construct separate retrieval-oriented query via QueryContextualizer
        retrieval_query = QueryContextualizer.build_retrieval_query(user_query, recent_history)

        state = AgentState(
            session_id=session_id,
            user_query=user_query,
            retrieval_query=retrieval_query,
            normalized_order_id=normalized_order_id,
            intent_category=intent,
            history_turns=recent_history,
            max_iterations=self.max_iterations,
        )

        state.trace.append(
            TraceEvent(
                event_type=TraceEventType.TURN_STARTED,
                iteration=0,
                summary=f"Session: {session_id} | Intent: {intent}",
                parameters={"session_id": session_id}
            )
        )

        # Check Privacy Keywords Handoff Rule
        query_lower = user_query.lower()
        if any(kw in query_lower for kw in self.PRIVACY_KEYWORDS):
            state.handoff_recommended = True
            state.trace.append(
                TraceEvent(
                    event_type=TraceEventType.HANDOFF,
                    iteration=0,
                    summary="Handoff recommended due to privacy keywords in user query"
                )
            )

        # Bounded Planning & Observation Loop
        while state.iterations < state.max_iterations:
            state.iterations += 1

            # Plan next action based on current state & prior observations
            try:
                action = self.planner.plan_next_action(state)
                ActionValidator.validate(action)
            except Exception as e:
                # Safe fallback: PLANNER_FAILURE
                action = AgentAction(
                    action_type=ActionType.HANDOFF,
                    reasoning=f"Planner validation/execution error fallback: {str(e)}"
                )
                obs = AgentObservation(
                    action_type=ActionType.HANDOFF,
                    success=False,
                    result="Planner validation or execution failure fallback",
                    error_message=f"Planner failure: {str(e)}",
                    failure_category=FailureCategory.PLANNER_FAILURE,
                    handoff_recommended=True,
                )
                state.observations.append(obs)

            # Progress Protection: prevent repeated identical non-terminal actions
            is_duplicate_non_terminal = False
            for prior_action in state.planned_actions:
                if (
                    prior_action.action_type == action.action_type
                    and prior_action.parameters == action.parameters
                    and not ActionType.is_terminal(action.action_type)
                ):
                    is_duplicate_non_terminal = True
                    break

            if is_duplicate_non_terminal:
                fallback_type = ActionType.RESPOND if (state.evidence_chunks or state.order_result) else ActionType.HANDOFF
                action = AgentAction(
                    action_type=fallback_type,
                    reasoning="Progress protection triggered: Repeated identical non-terminal action proposed."
                )

            state.planned_actions.append(action)

            state.trace.append(
                TraceEvent(
                    event_type=TraceEventType.ACTION_PLANNED,
                    iteration=state.iterations,
                    action_type=action.action_type,
                    parameters=action.parameters,
                    summary=f"Action planned: {action.action_type}"
                )
            )

            # ACTION EXECUTION 1: Clarification Action (Terminal)
            if action.action_type == ActionType.CLARIFY:
                state.final_answer = (
                    "Could you please provide your order ID (for example, ORD-1007) so I can check your order status?"
                )
                state.tool_calls_made = []
                obs = AgentObservation(
                    action_type=ActionType.CLARIFY,
                    success=True,
                    result="Requested order ID clarification from user"
                )
                state.observations.append(obs)
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.ACTION_EXECUTED,
                        iteration=state.iterations,
                        action_type=ActionType.CLARIFY,
                        success=True,
                        summary="Clarification requested from user for missing order ID"
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.OBSERVATION_RECORDED,
                        iteration=state.iterations,
                        action_type=ActionType.CLARIFY,
                        success=True,
                        summary="Recorded clarification observation"
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.TURN_COMPLETED,
                        iteration=state.iterations,
                        success=True,
                        summary="Turn completed with clarification response"
                    )
                )
                self.memory_store.add_turn(session_id, user_query, state.final_answer)
                return state

            # ACTION EXECUTION 2: Lookup Order Action (Non-terminal)
            if action.action_type == ActionType.LOOKUP_ORDER:
                raw_oid = action.parameters.get("order_id", "") or state.normalized_order_id or ""
                target_oid = self.extract_order_id(raw_oid) or raw_oid.strip()
                
                try:
                    order_res = self.execute_tool_safely("order_lookup", order_id=target_oid)
                except Exception as tool_err:
                    state.handoff_recommended = True
                    obs = AgentObservation(
                        action_type=ActionType.LOOKUP_ORDER,
                        success=False,
                        result=None,
                        error_message=f"Tool error: {str(tool_err)}",
                        failure_category=FailureCategory.TOOL_ERROR,
                        handoff_recommended=True,
                    )
                    state.observations.append(obs)
                    state.trace.append(
                        TraceEvent(
                            event_type=TraceEventType.ACTION_EXECUTED,
                            iteration=state.iterations,
                            action_type=ActionType.LOOKUP_ORDER,
                            success=False,
                            error_message=f"Tool execution failed: {str(tool_err)}",
                            summary="Order lookup tool raised an exception"
                        )
                    )
                    state.trace.append(
                        TraceEvent(
                            event_type=TraceEventType.HANDOFF,
                            iteration=state.iterations,
                            action_type=ActionType.LOOKUP_ORDER,
                            success=False,
                            summary="Handoff recommended due to tool error"
                        )
                    )
                    break

                state.order_result = order_res
                state.tool_calls_made.append("order_lookup")

                failure_cat = None
                if not order_res.found:
                    state.handoff_recommended = True
                    failure_cat = FailureCategory.BUSINESS_FAILURE

                # Retrieve policy context for order item details
                state.evidence_chunks = self.vector_store.search(state.retrieval_query, top_k=2, filter_customer_eligible=True)

                obs = AgentObservation(
                    action_type=ActionType.LOOKUP_ORDER,
                    success=order_res.found,
                    result=order_res,
                    error_message=order_res.error_message if not order_res.found else None,
                    failure_category=failure_cat,
                    handoff_recommended=state.handoff_recommended,
                )
                state.observations.append(obs)

                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.ACTION_EXECUTED,
                        iteration=state.iterations,
                        action_type=ActionType.LOOKUP_ORDER,
                        parameters={"order_id": target_oid},
                        success=order_res.found,
                        error_message=order_res.error_message if not order_res.found else None,
                        summary=f"Order lookup executed for {target_oid}. Found={order_res.found}"
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.OBSERVATION_RECORDED,
                        iteration=state.iterations,
                        action_type=ActionType.LOOKUP_ORDER,
                        success=order_res.found,
                        error_message=order_res.error_message if not order_res.found else None,
                        summary=f"Recorded order observation for {target_oid}"
                    )
                )
                if not order_res.found:
                    state.trace.append(
                        TraceEvent(
                            event_type=TraceEventType.HANDOFF,
                            iteration=state.iterations,
                            action_type=ActionType.LOOKUP_ORDER,
                            success=False,
                            summary="Handoff recommended due to unknown order ID"
                        )
                    )
                continue

            # ACTION EXECUTION 3: Retrieve KB Action (Non-terminal)
            if action.action_type == ActionType.RETRIEVE_KB:
                search_q = action.parameters.get("query") or state.retrieval_query
                state.evidence_chunks = self.vector_store.search(search_q, top_k=12, filter_customer_eligible=True)
                
                retrieved_filenames = {c.filename for c in state.evidence_chunks}
                
                failure_cat = None
                # Handoff Rule A: Source conflict between active policies 11 and 12
                if "11-product-care.md" in retrieved_filenames and "12-breeze-tumbler-product-card.md" in retrieved_filenames:
                    state.handoff_recommended = True
                    failure_cat = FailureCategory.BUSINESS_FAILURE
                
                # Handoff Rule B: Insufficient info or no chunks returned
                if not state.evidence_chunks:
                    state.handoff_recommended = True
                    failure_cat = FailureCategory.RETRIEVAL_FAILURE
                elif "vegan" in user_query.lower():
                    state.handoff_recommended = True
                    failure_cat = FailureCategory.BUSINESS_FAILURE
                
                # Handoff Rule C: Damaged / defective item exception requiring human review
                if any(kw in query_lower for kw in ("damaged", "defective", "broken")):
                    state.handoff_recommended = True
                    failure_cat = FailureCategory.BUSINESS_FAILURE

                obs = AgentObservation(
                    action_type=ActionType.RETRIEVE_KB,
                    success=len(state.evidence_chunks) > 0,
                    result=state.evidence_chunks,
                    error_message="No evidence retrieved" if not state.evidence_chunks else None,
                    failure_category=failure_cat,
                    handoff_recommended=state.handoff_recommended,
                )
                state.observations.append(obs)

                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.ACTION_EXECUTED,
                        iteration=state.iterations,
                        action_type=ActionType.RETRIEVE_KB,
                        parameters={"query": search_q},
                        success=len(state.evidence_chunks) > 0,
                        summary=f"KB retrieval executed. Retrieved {len(state.evidence_chunks)} evidence chunks."
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.OBSERVATION_RECORDED,
                        iteration=state.iterations,
                        action_type=ActionType.RETRIEVE_KB,
                        success=len(state.evidence_chunks) > 0,
                        summary=f"Recorded retrieval observation with {len(state.evidence_chunks)} evidence chunks"
                    )
                )
                if state.handoff_recommended:
                    state.trace.append(
                        TraceEvent(
                            event_type=TraceEventType.HANDOFF,
                            iteration=state.iterations,
                            action_type=ActionType.RETRIEVE_KB,
                            success=False,
                            summary="Handoff recommended due to policy constraints or source conflict"
                        )
                    )
                continue

            # ACTION EXECUTION 4: Handoff Action (Terminal)
            if action.action_type == ActionType.HANDOFF:
                state.handoff_recommended = True
                # Preserve existing failure observation if already recorded in exception handler
                existing_obs = state.observations[-1] if state.observations else None
                if not (existing_obs and existing_obs.action_type == ActionType.HANDOFF and existing_obs.failure_category == FailureCategory.PLANNER_FAILURE):
                    obs = AgentObservation(
                        action_type=ActionType.HANDOFF,
                        success=True,
                        result="Handoff recommended to human support",
                        handoff_recommended=True,
                    )
                    state.observations.append(obs)
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.ACTION_EXECUTED,
                        iteration=state.iterations,
                        action_type=ActionType.HANDOFF,
                        success=True,
                        summary="Handoff action executed"
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.HANDOFF,
                        iteration=state.iterations,
                        action_type=ActionType.HANDOFF,
                        success=True,
                        summary="Handoff recommended to human support"
                    )
                )
                break

            # ACTION EXECUTION 5: Respond Action (Terminal)
            if action.action_type == ActionType.RESPOND:
                obs = AgentObservation(
                    action_type=ActionType.RESPOND,
                    success=True,
                    result="Proceeding to grounded response generation",
                    handoff_recommended=state.handoff_recommended,
                )
                state.observations.append(obs)
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.ACTION_EXECUTED,
                        iteration=state.iterations,
                        action_type=ActionType.RESPOND,
                        success=True,
                        summary="Proceeding to grounded response generation"
                    )
                )
                break

        # Safety Fallback: if loop ended via max_iterations without a terminal answer / observation
        if not state.final_answer and state.iterations >= state.max_iterations:
            has_terminal_obs = any(o.action_type in ActionType.TERMINAL_ACTIONS for o in state.observations)
            if not has_terminal_obs:
                state.handoff_recommended = True
                state.observations.append(
                    AgentObservation(
                        action_type=ActionType.HANDOFF,
                        success=False,
                        result="Iteration limit exhausted without terminal action",
                        error_message="Max iterations reached",
                        handoff_recommended=True,
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.ITERATION_LIMIT_EXHAUSTED,
                        iteration=state.iterations,
                        success=False,
                        error_message="Max iterations reached without terminal action",
                        summary=f"Exhausted max_iterations={state.max_iterations}"
                    )
                )
                state.trace.append(
                    TraceEvent(
                        event_type=TraceEventType.HANDOFF,
                        iteration=state.iterations,
                        success=False,
                        summary="Handoff triggered due to iteration limit exhaustion"
                    )
                )

        # Format order data context if present
        order_data_context = None
        if state.order_result:
            order_data_context = self.format_order_data_context(state.order_result)

        # Build grounded prompt with raw user_query and multi-turn conversation context
        prompt_payload = ContextBuilder.build_prompt_with_context(
            user_question=user_query,
            evidence_chunks=state.evidence_chunks,
            history_turns=recent_history,
            order_data_context=order_data_context,
        )

        # Call LLM Provider
        answer = self.llm_provider.generate(
            system_prompt=prompt_payload["system_prompt"],
            user_prompt=prompt_payload["user_prompt"],
        )

        state.final_answer = answer.strip()
        state.citations = [c.source_citation for c in state.evidence_chunks if c.source_citation]

        # Save turn to session memory store with raw user_query
        self.memory_store.add_turn(session_id, user_query, state.final_answer)

        state.trace.append(
            TraceEvent(
                event_type=TraceEventType.TURN_COMPLETED,
                iteration=state.iterations,
                success=True,
                summary=f"Turn completed. Final answer generated. Handoff={state.handoff_recommended}"
            )
        )

        return state

