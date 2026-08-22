# 13. Bounded State Machine & State Management

## 1. Agent State Model ([`AgentState`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py#L14-L29))

The agent tracks execution state within a structured dataclass across each turn:

```python
@dataclass
class AgentState:
    session_id: str
    user_query: str
    normalized_order_id: Optional[str] = None
    intent_category: str = "general_policy"

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
```

---

## 2. Bounded Iteration Control

Autonomous agent loops can easily fall into infinite retries if tool execution fails or LLM outputs become repetitive. We enforce a strict iteration bound (`max_iterations = 3`):

```python
while state.iterations < state.max_iterations:
    state.iterations += 1
    ...
```

If `state.iterations >= state.max_iterations` without reaching a valid answer state, the machine safely triggers human escalation:
```python
state.handoff_recommended = True
state.final_answer = "I apologize, but I am unable to process your request at this time. Please contact customer support."
```

---

## 3. Special Handoff Recommendation Triggers

In addition to unknown orders (`ORD-9999`), the state machine sets `handoff_recommended = True` when:
1. **Source Policy Conflicts**: Retrieved chunks contain active conflicting policies (`11-product-care.md` vs `12-breeze-tumbler-product-card.md`).
2. **Missing Knowledge / Insufficient Evidence**: Queries regarding unsupported topics (e.g. vegan adhesives).
3. **Damaged Item Exceptions**: Reports of damaged/defective final-sale items requiring human approval.
4. **Privacy Requests**: Customer attempts to query PII (`email`, `address`, `risk_score`).
