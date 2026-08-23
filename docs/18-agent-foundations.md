# 18. AI Agent Foundations & State Machines

## 1. What an Agent Actually Is
An **AI Agent** is an autonomous software system that uses a Large Language Model (LLM) as its central reasoning engine to iteratively perceive inputs, maintain internal state, plan actions, execute tools, observe environment feedback, and achieve a specified goal within explicit safety boundaries.

Unlike a simple function or linear script, an agent operates in a continuous control loop:
$$\text{Perceive} \longrightarrow \text{Plan} \longrightarrow \text{Execute Action} \longrightarrow \text{Observe Outcome} \longrightarrow \text{Decide Next Step}$$

---

## 2. LLM Pipelines vs. Workflows vs. Autonomous Agents

It is critical to distinguish three levels of system complexity:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. SIMPLE LLM PIPELINE                                                   │
│    User Input ──► Static Prompt ──► LLM ──► Text Output                  │
│    (Single call, no state, no tools, no dynamic decisions)              │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. DETERMINISTIC WORKFLOW (Chain)                                        │
│    Input ──► Retain Data ──► Vector Search ──► Format Prompt ──► LLM     │
│    (Hardcoded sequence of steps, fixed control flow, no dynamic routing)│
├──────────────────────────────────────────────────────────────────────────┤
│ 3. AUTONOMOUS AGENT STATE MACHINE                                        │
│    Input ──► State Machine Loop:                                         │
│              ├── Planner decides Action (RETRIEVE_KB / LOOKUP_ORDER)    │
│              ├── Tool execution firewall returns Observation             │
│              └── Loop repeats until RESPOND or max_iterations reached     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. State in an Agent System ([`AgentState`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py#L18-L42))

State is the single source of truth tracking an interaction turn. Without explicit state management, an agent cannot maintain multi-step context or bound its execution loop.

In our production codebase ([`src/agent.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py#L18-L42)), state is explicitly tracked via `AgentState`:

```python
@dataclass
class AgentState:
    session_id: str
    user_query: str
    retrieval_query: Optional[str] = None
    normalized_order_id: Optional[str] = None
    intent_category: str = "general_policy"

    # Context & Execution State
    evidence_chunks: List[KBChunk] = field(default_factory=list)
    order_result: Optional[CustomerSafeOrderResult] = None
    tool_calls_made: List[str] = field(default_factory=list)
    history_turns: List[ConversationTurn] = field(default_factory=list)
    planned_actions: List[AgentAction] = field(default_factory=list)
    observations: List[AgentObservation] = field(default_factory=list)
    
    # Bounded Loop Control
    iterations: int = 0
    max_iterations: int = 3
    handoff_recommended: bool = False

    # Outputs
    final_answer: Optional[str] = None
    citations: List[str] = field(default_factory=list)
```

### Stanford CME295 Connection:
As taught in Stanford CME295 Lecture 7, compound AI systems move beyond single-shot prompt engineering by maintaining explicit, inspectable state variables rather than relying on LLM hidden state.
