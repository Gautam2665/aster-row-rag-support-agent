# 33. Day 6 Planner Policy & Execution Boundary Guide

This module covers the explicit architectural separation between **LLM Action Planning** and **Application Policy Authorization**, explaining why non-deterministic models must never act as security boundaries.

---

## 1. Planning vs. Execution Architecture

```text
                                Customer Query
                                      │
                                      ▼
                               PlannerContext
                        (Sanitized State Snapshot)
                                      │
                                      ▼
                                 LLM Planner
                        (Generates Proposed Action)
                                      │
                                      ▼
                                 AgentAction
                                      │
                                      ▼
                        ActionValidator (Layer 1)
                     (Structural Allowlist & Schema)
                                      │
                                      ▼
                         PlannerPolicy (Layer 2)
                     (Deterministic State Policy Rules)
                                      │
                                      ▼
                            SupportAgent Core
                     (Sole Execution Authority)
                                      │
                                      ▼
                          Tool / Vector Search
                                      │
                                      ▼
                            AgentObservation
```

---

## 2. Structural Validation vs. State Policy Validation

To achieve robust defense-in-depth, validation is split into two complementary layers:

| Layer | Component | Scope & Purpose | Example Check |
| :--- | :--- | :--- | :--- |
| **Layer 1** | `ActionValidator` | **Structural & Schema Validation**: Ensures action type exists in allowlist and parameter keys match strict JSON schemas. | `ActionType` is `"LOOKUP_ORDER"` and parameters contain string key `"order_id"`. |
| **Layer 2** | `PlannerPolicy` | **State & Policy Validation**: Ensures the action is permitted under the current application state before tool execution. | Rejects `LOOKUP_ORDER` if `order_id` parameter is missing or invalid (`ORD-\d{4}`). |

---

## 3. Why an LLM Must Never Be the Authorization Layer

1. **Non-Determinism**: Large Language Models are probabilistic text generators. They can suffer from hallucination, jailbreaking, or prompt injection, leading them to recommend invalid or unapproved actions.
2. **LLM Proposes, Application Disposes**: In professional enterprise architecture, the LLM is treated as an **untrusted proposal engine**. The application container retains sole execution authority and enforces deterministic business policies.

---

## 4. Safe Rejection & Handoff Escalation

When `PlannerPolicy.validate_action()` evaluates an action as `is_permitted = False`:
1. Tool execution is **completely blocked**.
2. An `AgentObservation` is recorded with `failure_category = PLANNER_FAILURE`.
3. `state.handoff_recommended = True` is set.
4. The action is safely substituted with `fallback_action = AgentAction(ActionType.HANDOFF)`.
5. Bounded iteration counter (`max_iterations = 3`) prevents infinite retry loops.

---

## 5. Compact Interview Revision Sheet

### Q1: Why do you need both `ActionValidator` and `PlannerPolicy`?
> **Answer**: `ActionValidator` provides structural schema validation (checking allowlists and JSON types). `PlannerPolicy` provides state-aware policy validation (checking whether state conditions permit executing that tool right now). Decoupling them keeps structural parsing clean while enforcing state rules.

### Q2: Can an LLM planner execute tools directly in your system?
> **Answer**: No. The planner outputs only a JSON `AgentAction` recommendation. `SupportAgent` is the sole execution authority, validating the action through `ActionValidator` and `PlannerPolicy` before calling any tool code.

### Q3: How does `PlannerPolicy` handle an invalid `LOOKUP_ORDER` recommendation?
> **Answer**: If the planner proposes `LOOKUP_ORDER` without a valid order ID parameter matching `ORD-\d{4}`, `PlannerPolicy` marks `is_permitted = False`, blocks tool execution, records a `PLANNER_FAILURE` observation, and safely escalates to human handoff.

### Q4: Why is `PlannerPolicy` strictly deterministic?
> **Answer**: Security authorization rules must be 100% predictable, testable, and immune to prompt injection or model hallucination. Running deterministic Python functions ensures 0 LLM latency, 0 token costs, and 100% security guarantees.
