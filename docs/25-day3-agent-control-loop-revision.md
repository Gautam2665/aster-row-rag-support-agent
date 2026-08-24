# 25. Day 3 Agent Control Loop & Stanford CME295 Mental Model

This module provides the core mental model for Day 3. It synthesizes the engineering decisions, architectural guardrails, and interview concepts without relying on code line details.

---

## 1. Memory $\neq$ Context $\neq$ Retrieval Query

This is the most critical distinction in Day 3 agent architecture.

```text
Conversation Memory
        │
        │ stores past turns (FIFO)
        ▼
  Recent History
        │
        ├────────────────────────┐
        ▼                        ▼
  ContextBuilder            QueryContextualizer
        │                        │
        ▼                        ▼
  LLM Prompt Context       Vector Retrieval Query
```

### Core Definitions
* **Memory stores what happened**: Retains short-term conversation turns in a bounded FIFO queue (`SessionMemoryStore`). Memory is **untrusted customer context** and must never override authoritative knowledge-base documents.
* **Context determines what the LLM sees**: `ContextBuilder` formats recent turns inside `<conversation_history>` XML tags within the prompt payload without mutating internal state.
* **Query Contextualization determines vector search**: `QueryContextualizer` reformulates ambiguous follow-up turns into a standalone retrieval query for ChromaDB.

### Critical Engineering Separation: `user_query != retrieval_query`
For a multi-turn user conversation:
* Turn 1: *"Do you ship internationally?"*
* Turn 2: *"What about Canada?"*

If we pass Turn 2 verbatim to ChromaDB (`"What about Canada?"`), vector search fails because the text lacks product or shipping keywords.

**Separation Rule**:
* `user_query` (**Preserved Verbatim**): `"What about Canada?"` $\rightarrow$ Placed inside `<user_question>` tag in the final prompt to preserve customer intent and tone.
* `retrieval_query` (**Contextualized Vector Search**): `"Do you ship internationally? What about Canada?"` $\rightarrow$ Passed **only** to ChromaDB vector search.

---

## 2. Agent $\neq$ LLM

The LLM is **not** the agent. The agent is the surrounding deterministic application container.

```text
                 LLM
                  │
                  ▼
             AgentAction (Action Proposal)
                  │
                  ▼
           ActionValidator
                  │
            valid? │
             ┌────┴────┐
            YES        NO
             │          │
             ▼          ▼
       SupportAgent    HANDOFF (Safe Fallback)
             │
             ▼
      Execute Tool / RAG
```

### Interview Takeaway
> *"The LLM is a decision-making model operating inside the agent state machine, not the authority over application tools or infrastructure. The LLM only proposes an `AgentAction`; the application validates parameters via `ActionValidator`, checks security policies, executes tools, and records sanitized observations."*

---

## 3. Planner $\rightarrow$ Executor $\rightarrow$ Observation Loop

The central ReAct loop dynamics:

$$\text{Plan} \longrightarrow \text{Validate} \longrightarrow \text{Execute} \longrightarrow \text{Observe} \longrightarrow \text{Plan Again}$$

### Walkthrough Example (`ORD-1007`)
1. **User**: *"Where is ORD-1007?"*
2. **Planner**: Proposes `AgentAction(action_type=LOOKUP_ORDER, order_id="ORD-1007")`.
3. **Executor**: `SupportAgent` passes `ORD-1007` to `OrderLookupTool.lookup_order()`.
4. **Observation**: `AgentObservation` recorded (`success=True`, `status="shipped"`, `carrier="UPS"`).
5. **Planner**: Inspects observation, recognizes order is found, proposes `AgentAction(action_type=RESPOND)`.
6. **Executor**: Invokes `ContextBuilder` and generates final grounded response.

### Why Observations Are Crucial
Without state-aware observations, an agent enters an ungrounded loop (`LOOKUP` $\rightarrow$ `LOOKUP` $\rightarrow$ `LOOKUP`). With observations, prior results inform the next planner iteration.

---

## 4. Why Bounded Loops Matter

### Interview Question
> *"Why don't you let the LLM execute actions continuously until it decides to stop?"*

### Engineering Answer
> *"Because LLMs are probabilistic models that can hallucinate, get stuck in repeated tool calls, or output invalid schemas. Our application enforces strict safety bounds: `max_iterations = 3`, progress protection (blocking duplicate consecutive tool calls), and terminal action classification (`RESPOND`, `CLARIFY`, `HANDOFF`)."*

### Control Architecture
$$\text{LLM Proposes} \longrightarrow \text{Application Validates} \longrightarrow \text{Application Executes} \longrightarrow \text{Application Bounds}$$

---

## 5. Failure Taxonomy & Recovery Policy

Failures in a production agent are **state transitions**, not infrastructure crashes or invitations to retry blindly.

| Failure Category | Meaning | System Recovery Response |
| :--- | :--- | :--- |
| `TOOL_ERROR` | Infrastructure exception or tool crash | Record failed observation, set `handoff_recommended=True`, break execution loop. |
| `BUSINESS_FAILURE` | Tool ran successfully but domain rules were unsatisfied (e.g. unknown `ORD-9999`) | Record observation, advise customer safely or recommend human handoff. |
| `RETRIEVAL_FAILURE` | Vector search returned 0 eligible evidence chunks | Safely abstain from fabricating policy answers; recommend human handoff. |
| `PLANNER_FAILURE` | LLM output invalid JSON or unapproved action | Intercept with `ActionValidator`, fallback immediately to safe `HANDOFF`. |

---

## 6. Trace vs Memory

| Dimension | Conversation Memory | Agent Execution Trace |
| :--- | :--- | :--- |
| **Purpose** | Retains customer dialogue history | Inspectable developer log of agent operations |
| **Audience** | Formatted into LLM context for future turns | Developers, debugging CLI (`--debug`), auditing |
| **LLM Exposure** | Included inside `<conversation_history>` | **STRICTLY EXCLUDED** from LLM prompt payloads |
| **PII Rules** | Customer queries / sanitized answers | PII fields (`email`, `address`, `risk_score`) scrubbed |

$$\text{Conversation History} \longrightarrow \text{ContextBuilder} \longrightarrow \text{LLM Prompt}$$
$$\text{Execution Trace} \longrightarrow \text{Developer Debugging Log / CLI Output}$$

---

## 7. Deterministic State vs Semantic LLM Evaluation

Evaluating an agent requires separating deterministic control assertions from semantic text generation checks.

```text
                       Evaluation Case Execution
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
Deterministic State Assertions                       Semantic Assertions
• Tool execution check                              • Phrase inclusion (must_include)
• Tool allowlist validation                         • Phrase exclusion (must_not_include)
• PII sanitization check                            • Tone & helpfulness
• Citation lineage check                            
• Handoff recommendation flag                        
         │                                                   │
         ▼                                                   ▼
   PASS / FAIL                                       PASS / FAIL / UNAVAILABLE
```

### Why `UNVERIFIED_REQUIRES_LLM` is Rigorous Engineering
Under offline test fixtures (`MockLLMProvider`), state assertions are 100% verified (`PASS`). Semantic text assertions require a live LLM (`UNAVAILABLE`). Marking tests as `UNVERIFIED_REQUIRES_LLM` rather than fake-passing guarantees honest evaluation reporting.

---

## 8. The One-Page Day 3 Mental Model

```text
                                USER
                                 │
                                 ▼
                          Session Memory
                                 │
                                 ▼
                         Query Contextualizer
                                 │
                                 ▼
                           ┌───────────┐
                           │  Planner  │
                           └─────┬─────┘
                                 │
                                 ▼
                          Action Proposal
                                 │
                                 ▼
                          ActionValidator
                                 │
                                 ▼
                            SupportAgent
                         ┌───────┴───────┐
                         │               │
                     Retrieval         Tool
                         │               │
                         └───────┬───────┘
                                 ▼
                             Observation
                                 │
                                 ▼
                            Bounded Loop
                           (max_iter = 3)
                                 │
                         ┌───────┴───────┐
                         │               │
                     Continue        Terminal
                         │               │
                         └───────┐   ┌───┘
                                 ▼   ▼
                           ContextBuilder
                                 │
                                 ▼
                            LLM Provider
                                 │
                                 ▼
                              Response
                                 │
                                 ▼
                           AgentTrace Log
```
