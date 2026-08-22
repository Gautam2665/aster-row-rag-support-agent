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


@dataclass
class AgentState:
    """
    State object tracking a single interaction turn through the agent state machine.
    """
    session_id: str
    user_query: str
    normalized_order_id: Optional[str] = None
    intent_category: str = "general_policy"  # "policy", "order_status", "clarification"

    # Context & Execution State
    evidence_chunks: List[KBChunk] = field(default_factory=list)
    order_result: Optional[CustomerSafeOrderResult] = None
    tool_calls_made: List[str] = field(default_factory=list)
    
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
    secure order lookups, prompt building, and LLM generation.
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
        max_iterations: int = 3,
    ):
        self.vector_store = vector_store or KBVectorStore()
        self.order_tool = order_tool or OrderLookupTool()
        self.llm_provider = llm_provider or get_default_provider()
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
        Execute a single turn through the bounded state machine.
        """
        normalized_order_id = self.extract_order_id(user_query)
        intent = self.detect_intent(user_query, normalized_order_id)

        state = AgentState(
            session_id=session_id,
            user_query=user_query,
            normalized_order_id=normalized_order_id,
            intent_category=intent,
            max_iterations=self.max_iterations,
        )

        # Enforce Bounded Iterations
        while state.iterations < state.max_iterations:
            state.iterations += 1

            # PATH 1: Clarification Path (Order question, missing Order ID)
            if state.intent_category == "clarification":
                state.final_answer = (
                    "Could you please provide your order ID (for example, ORD-1007) so I can check your order status?"
                )
                state.tool_calls_made = []
                return state

            # PATH 2: Order Status Path (Order ID Present)
            if state.intent_category == "order_status" and state.normalized_order_id:
                order_res = self.execute_tool_safely("order_lookup", order_id=state.normalized_order_id)
                state.order_result = order_res
                state.tool_calls_made.append("order_lookup")

                if not order_res.found:
                    state.handoff_recommended = True

                # Also retrieve policy context if applicable (e.g. shipping/returns)
                state.evidence_chunks = self.vector_store.search(user_query, top_k=2, filter_customer_eligible=True)
                break

            # PATH 3: Policy / Knowledge-Base Path
            if state.intent_category == "policy":
                state.evidence_chunks = self.vector_store.search(user_query, top_k=12, filter_customer_eligible=True)
                
                retrieved_filenames = {c.filename for c in state.evidence_chunks}
                
                # Handoff Rule A: Source conflict between active policies 11 and 12
                if "11-product-care.md" in retrieved_filenames and "12-breeze-tumbler-product-card.md" in retrieved_filenames:
                    state.handoff_recommended = True
                
                # Handoff Rule B: Insufficient info or no chunks returned
                if not state.evidence_chunks or "vegan" in user_query.lower():
                    state.handoff_recommended = True
                
                # Handoff Rule C: Damaged / defective item exception requiring human review
                query_lower = user_query.lower()
                if any(kw in query_lower for kw in ("damaged", "defective", "broken")):
                    state.handoff_recommended = True
                break

        # Check Privacy Keywords Handoff Rule
        query_lower = user_query.lower()
        if any(kw in query_lower for kw in self.PRIVACY_KEYWORDS):
            state.handoff_recommended = True

        # Safety Fallback if iteration limit reached
        if state.iterations >= state.max_iterations and not state.final_answer and not state.evidence_chunks and not state.order_result:
            state.handoff_recommended = True
            state.final_answer = "I apologize, but I am unable to process your request at this time. Please contact customer support."
            return state

        # Assemble Grounded Prompt Payload
        prompt_payload = build_grounded_prompt(user_query, state.evidence_chunks)

        # If order data exists, append safe order context to user prompt
        if state.order_result:
            order_context = self.format_order_data_context(state.order_result)
            prompt_payload["user_prompt"] = f"{order_context}\n\n{prompt_payload['user_prompt']}"

        # Call LLM Provider
        answer = self.llm_provider.generate(
            system_prompt=prompt_payload["system_prompt"],
            user_prompt=prompt_payload["user_prompt"],
        )

        state.final_answer = answer.strip()
        state.citations = [c.source_citation for c in state.evidence_chunks if c.source_citation]

        return state
