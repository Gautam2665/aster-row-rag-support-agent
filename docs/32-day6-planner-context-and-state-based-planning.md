# 32. Day 6 Planner Context & State-Based Planning Guide

This module covers the design and architecture of explicit `PlannerContext`, isolating agent state from planner decision-making, observation-driven feedback loops, and keeping execution authority strictly inside the application layer.

---

## 1. Why Agents Need State-Aware Planning

In simple single-turn RAG systems, a model takes a user prompt, retrieves documents, and responds. In multi-turn **Agentic Systems**, an agent must perform multi-step reasoning, execute tools, observe outcomes, and adapt its next action dynamically.

```text
                     AgentState (Mutable Runtime Memory)
                                     │
                                     ▼
                PlannerContext.from_agent_state(state)
                        (Bounded, Sanitized Context)
                                     │
                                     ▼
                        BasePlanner.plan_next_action()
                        (MockPlanner / LLMPlanner)
                                     │
                                     ▼
                        ActionValidator.validate()
                        (Strict Schema & Allowlist)
                                     │
                                     ▼
                       SupportAgent Execution Core
                    (Sole Application Tool Executor)
```

---

## 2. AgentState vs. PlannerContext

| Architectural Dimension | `AgentState` (Application Core) | `PlannerContext` (Planner Context Contract) |
| :--- | :--- | :--- |
| **Purpose** | Complete, mutable runtime state tracking session, trace, memory, tools, and vector chunks | Bounded, read-only context snapshot tailored strictly for next-action planning |
| **PII / Database Exposure** | Holds full trace events, order objects, and raw query context | **Strictly sanitized**: zero emails, addresses, risk scores, warehouse notes, or DB handles |
| **Execution Authority** | Executes tools (`OrderLookupTool`), modifies state, and updates memory | **Data payload only**: zero execution methods (`execute_tool`, `run`) |
| **Lifecycle Scope** | Persists across iteration steps during turn execution | Immutably constructed per planning step (`PlannerContext.from_agent_state`) |

---

## 3. Observation $\rightarrow$ Next-Action Feedback Loop

A core principle of the ReAct (Reasoning + Acting) loop is incorporating prior action outcomes (`AgentObservation`) into the next decision step:

1. **Iteration 1**: User asks *"Where is ORD-1007?"* $\rightarrow$ Planner outputs `LOOKUP_ORDER` (`order_id="ORD-1007"`).
2. **Execution**: `SupportAgent` calls `OrderLookupTool` $\rightarrow$ Records `AgentObservation(action_type="LOOKUP_ORDER", success=True)`.
3. **Iteration 2**: `PlannerContext` captures `has_order_result=True` and `order_found=True`.
4. **Next Plan**: Planner sees prior success $\rightarrow$ Outputs terminal `RESPOND` action instead of repeating order lookup.

---

## 4. Why the Application Must Remain the Execution Authority

* **Separation of Planning and Execution**: The planner's sole role is recommending an action (`AgentAction`). It possesses zero capabilities to execute database queries or call external APIs directly.
* **Schema Validation Boundary**: `ActionValidator` sits strictly between the planner and the execution engine, ensuring malformed or injection-crafted action JSON cannot reach execution.
* **Deterministic Fallback**: If an LLM planner produces invalid JSON or unknown action types, `SupportAgent` catches the validation error and falls back to safe `HANDOFF`.

---

## 5. Compact Interview Revision Sheet

### Q1: Why should an LLM planner receive a bounded `PlannerContext` rather than raw application state?
> **Answer**: To enforce security and token efficiency. Passing raw application objects risks leaking customer PII (emails, addresses, internal risk scores) into LLM prompts. A bounded `PlannerContext` exposes only high-level status flags (`has_usable_evidence`, `order_found`) needed for action planning.

### Q2: What is the difference between planning authority and execution authority?
> **Answer**: Planning authority (`Planner`) recommends what to do next (`AgentAction`). Execution authority (`SupportAgent`) validates the action against security rules (`ActionValidator`) and executes the tools. Decoupling them prevents LLM hallucinated actions from executing arbitrary code.

### Q3: How does your agent prevent infinite tool execution loops?
> **Answer**: Via two mechanisms: (1) `PlannerContext` tracks prior `AgentObservation` summaries so the planner sees when a tool has already succeeded, and (2) Progress Protection in `SupportAgent` detects repeated identical non-terminal actions and forces a terminal `RESPOND` or `HANDOFF`.

### Q4: What happens if an LLM planner produces invalid JSON or an unapproved action type?
> **Answer**: `ActionValidator` raises a `ValueError`. The agent's control loop catches the exception, classifies it as a `PLANNER_FAILURE`, records the observation, and executes a safe human handoff escalation without crashing.
