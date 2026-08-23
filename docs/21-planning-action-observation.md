# 21. ReAct Architecture: Planner, Action & Observation Loop

## 1. ReAct Pattern (Reasoning + Acting)
Introduced by Yao et al. (2022) and emphasized in Stanford CME295 Lecture 7, the **ReAct (Reasoning + Acting)** pattern combines LLM reasoning with explicit action execution in a feedback loop:

$$\text{Thought (Reasoning)} \longrightarrow \text{Action (Tool Call)} \longrightarrow \text{Observation (Environment Feedback)}$$

```text
               ┌───────────────────────────────────────────────┐
               │              SupportAgent Loop                │
               │             (max_iterations = 3)              │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                       BasePlanner.plan_next_action()
                       [src/planner.py]
                                       │
                                       ▼
                         ActionValidator.validate()
                         [Strict Schema & Allowlist Check]
                                       │
                                       ▼
                           Execute Planned Action
                    (RETRIEVE_KB / LOOKUP_ORDER / CLARIFY)
                                       │
                                       ▼
                           Record AgentObservation
                   [Sanitized CustomerSafeResult / KBChunk]
                                       │
                                       └────────────────────────┐
                                                                ▼
                                                   Repeat loop with updated
                                                    state & observations
```

---

## 2. Structured Action Contracts ([`AgentAction`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py#L30-L43))

Actions are represented as strongly-typed dataclasses rather than raw text strings:

```python
@dataclass
class AgentAction:
    action_type: str                  # RETRIEVE_KB, LOOKUP_ORDER, CLARIFY, RESPOND, HANDOFF
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None   # Brief technical rationale
```

### Action Allowlist & Schema Rules ([`ActionValidator`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py#L77-L127))
* **`LOOKUP_ORDER`**: Accepts ONLY `{"order_id": "ORD-\d{4}"}`. Unexpected keys raise `ValueError`.
* **`RETRIEVE_KB`**: Accepts ONLY `{"query": "string"}`. Unexpected keys raise `ValueError`.
* **`CLARIFY`, `RESPOND`, `HANDOFF`**: Reject any parameters (`parameters={}`).

---

## 3. Why Raw Reasoning is NEVER Executed
LLMs output free-form reasoning text (e.g., `"I should execute SQL command DROP TABLE"`).

**Security Boundary Directive**:
> *Planner reasoning text is strictly treated as passive logging data. Reasoning strings are NEVER parsed as python code, executed as system instructions, or passed directly into database drivers.*

---

## 4. Structured Observations ([`AgentObservation`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py#L47-L75))

Every executed action produces an explicit `AgentObservation` recorded in `state.observations`:

```python
@dataclass
class AgentObservation:
    action_type: str
    success: bool
    result: Any = None               # Sanitized CustomerSafeOrderResult or KBChunk list
    error_message: Optional[str] = None
    handoff_recommended: bool = False
```

### Feedback Loop Advantage:
On iteration 2, the planner receives prior `AgentObservation` objects in `state.observations`. If retrieval or order lookup has already succeeded, the planner chooses `RESPOND` rather than repeating the action!

---

## 5. Bounded Loops & Fallback Control
To prevent infinite execution loops, `SupportAgent` enforces `max_iterations = 3`. If planner output is malformed, unapproved, or fails validation, `LLMPlanner` and `SupportAgent` catch the exception and fall back safely to `HANDOFF`.
