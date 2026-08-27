# 35. Day 6 Final Integration & Interview Revision Guide

This document presents the complete end-to-end architecture audit, component boundaries, failure recovery taxonomy, and interview master revision guide for the Aster & Row Bounded Customer Support AI Agent.

---

## 1. Complete End-to-End System Architecture

```text
                                Customer Query
                                      │
                                      ▼
                      Query Contextualization & Memory
                       (SessionMemoryStore & Intent)
                                      │
                                      ▼
                            AgentState Core Model
                                      │
                                      ▼
                         PlannerContext.from_agent_state()
                        (Sanitized, Read-Only Payload)
                                      │
                                      ▼
                        BasePlanner (Mock / LLM Planner)
                         (Proposes AgentAction Recommendation)
                                      │
                                      ▼
                         ActionValidator (Layer 1)
                      (Structural Allowlist & JSON Schema)
                                      │
                                      ▼
                          PlannerPolicy (Layer 2)
                      (Deterministic State Policy Rules)
                                      │
                                      ▼
                         SupportAgent Execution Core
                      (Sole Application Execution Authority)
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
       Order Lookup Tool                          KB Vector Search
      (PII-Scrubbed Status)                     search_with_diagnostics()
                 │                                         │
                 ▼                                         ▼
         AgentObservation                         evaluate_retrieval_sufficiency()
                 │                                         │
                 │                                         ▼
                 │                                 assess_evidence()
                 │                     ┌───────────────────┼───────────────────┐
                 │                     │                   │                   │
                 │                  USABLE            INSUFFICIENT          CONFLICT
                 │                     │                   │                   │
                 │                     ▼                   ▼                   ▼
                 │               ContextBuilder         HANDOFF             HANDOFF
                 │             (Data-Instruction)       No LLM              No LLM
                 │                     │              Generation          Generation
                 │                     ▼
                 │               LLM Generation
                 │            (Grounded Response)
                 │                     │
                 └─────────────────────┼───────────────────┘
                                       │
                                       ▼
                             Trace & Memory Update
                                       │
                                       ▼
                                 Final Response
```

---

## 2. Core Subsystem Boundaries

| Subsystem Component | Role & Scope | Security & Authorization Boundary |
| :--- | :--- | :--- |
| `AgentState` | Mutable turn state tracking session, trace, memory, order results, and evidence. | Internal application state container. |
| `PlannerContext` | Bounded, sanitized snapshot supplied to `BasePlanner`. | **Strips all customer PII** (emails, addresses, risk scores, warehouse notes, DB handles). No execution methods. |
| `ActionValidator` | **Layer 1 Validator**: Checks structural JSON syntax, parameter schema types, and action allowlists (`RETRIEVE_KB`, `LOOKUP_ORDER`, `CLARIFY`, `RESPOND`, `HANDOFF`). | Rejects malformed JSON or unapproved action names. |
| `PlannerPolicy` | **Layer 2 Policy**: Evaluates state-aware prerequisites (e.g. valid `ORD-\d{4}` for order lookup, non-empty search query). | Rejects unpermitted actions, logs `PLANNER_FAILURE`, and forces safe `HANDOFF`. |
| `SupportAgent` | Sole execution authority for running tools, updating state, and coordinating turn flow. | Retains sole authority to call external tools or database APIs. |
| `retrieval_policy.py` | Evaluates vector search chunk count and distance thresholds (`evaluate_retrieval_sufficiency`). | Rejects empty or low-relevance vector results (`RETRIEVAL_FAILURE`). |
| `evidence_policy.py` | Assesses retrieved evidence for active authoritative status vs. active-source policy conflicts (`assess_evidence`). | Detects genuine active-source conflicts (`CONFLICT`), skipping LLM generation (`BUSINESS_FAILURE`). |
| `generation_policy.py` | Enforces prompt framing (`<retrieved_evidence>`, `<order_lookup_data>`, `<user_question>`) and citation rules. | Isolates untrusted retrieved text from system instructions. |

---

## 3. Failure Taxonomy & Safe Handoff Rules

```text
                                Failure Event Detected
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
           TOOL_ERROR             RETRIEVAL_FAILURE        BUSINESS_FAILURE
      (Tool Exception /        (Empty Search / Low       (Unknown Order / Active
       Database Error)            Relevance Evidence)      Policy Source Conflict)
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          │
                                          ▼
                         SupportAgent Escalation Protocol
                         1. Mark handoff_recommended = True
                         2. Record AgentObservation with Failure Category
                         3. Record TraceEvent HANDOFF
                         4. Skip LLM Generation (or return safe handoff text)
                         5. Terminate turn (no infinite retry loop)
```

---

## 4. How to Explain This Project in 2 Minutes (Elevator Pitch)

> "I built a production-grade, enterprise-ready customer support AI agent designed around **bounded agentic execution** and **defense-in-depth security**.
>
> Unlike naive RAG pipelines that pass raw user queries directly to an LLM with full tool access, my system enforces strict separation between **planning authority** and **execution authority**.
>
> When a customer sends a query, the agent constructs a sanitized `PlannerContext` that strips all PII (emails, addresses, internal risk scores). The planner—whether deterministic or LLM-driven—recommends an `AgentAction`. That recommendation passes through a two-tier validation barrier: `ActionValidator` checks structural schemas, and `PlannerPolicy` checks state-aware business rules.
>
> Only if both pass does the application core (`SupportAgent`) execute tools like `OrderLookupTool` or vector search. Retrieved evidence undergoes deterministic sufficiency and conflict evaluation (`assess_evidence()`); if active official documents contradict each other, the system abstains from generating a hallucinated response and escalates to human handoff.
>
> The entire architecture is bounded to a maximum of 3 iterations with progress protection, verified by 237 pytest tests and 20 evaluation cases."

---

## 5. Master Interview Question & Answer Bank

### Q1: Why did you separate `PlannerContext` from `AgentState`?
> **Answer**: `AgentState` contains full, mutable application objects, including order lookup results and execution traces. If an LLM planner receives `AgentState` directly, it risks leaking customer PII or raw database handles into the prompt context. `PlannerContext` acts as a read-only, sanitized view that exposes only the high-level status flags (`has_usable_evidence`, `order_found`) needed for action planning.

### Q2: Why is the LLM not allowed to execute tools directly?
> **Answer**: Because LLMs are probabilistic models prone to hallucination, jailbreaking, and prompt injection. In enterprise agentic systems, the LLM is an untrusted decision recommender. Application code must retain sole execution authority, validating every proposed action against schema allowlists (`ActionValidator`) and state policy rules (`PlannerPolicy`) before touching database tools or external APIs.

### Q3: What is the difference between `ActionValidator` and `PlannerPolicy`?
> **Answer**: `ActionValidator` is Layer 1 structural validation: it verifies that the action name is in the allowlist and parameters conform to JSON schemas (e.g. `LOOKUP_ORDER` requires a string `order_id`). `PlannerPolicy` is Layer 2 state-aware validation: it checks if current application state permits executing that action right now (e.g., verifying `order_id` matches `ORD-\d{4}`).

### Q4: How does the agent handle genuine policy conflicts between official documents?
> **Answer**: Through `assess_evidence()`. If vector search retrieves active official chunks from distinct documents that contain contradictory directives (such as care instructions requiring hand-washing while a product card claims all components are dishwasher safe), `assess_evidence()` sets status to `CONFLICT`. The agent skips LLM generation, marks `handoff_recommended = True`, records a `BUSINESS_FAILURE` observation, and safely escalates to human support.

### Q5: How do you prevent infinite execution loops in your agent control loop?
> **Answer**: Through two safeguards: (1) Hard iteration bounding via `max_iterations = 3`, and (2) Progress Protection, which inspects prior planned actions and observations to detect repeated identical non-terminal actions and forces a terminal `RESPOND` or `HANDOFF`.

### Q6: How does the system prevent PII leakage to LLM providers or customer-facing answers?
> **Answer**: `OrderLookupTool` scrubs internal fields (`email`, `address`, `risk_score`, `warehouse_note`) into `CustomerSafeOrderResult`. `PlannerContext` sanitizes observation summaries before prompt formatting. `generation_evaluation.py` deterministically scans all outputs for forbidden internal field names.

### Q7: Why use `search_with_diagnostics()` instead of raw vector search?
> **Answer**: Diagnostics capture non-sensitive RAG metadata (`retrieval_query`, candidate counts, chunk IDs, similarity distances, filter flags) on `AgentState.retrieval_diagnostics` for evaluation and tracing without polluting the prompt payload.

### Q8: What happens if the LLM provider fails or returns malformed JSON?
> **Answer**: `LLMPlanner` catches JSON parsing and validation exceptions, logs a `PLANNER_FAILURE` observation, and safely falls back to an `AgentAction(ActionType.HANDOFF)`. The agent never crashes or exposes stack traces to the user.

### Q9: What is the difference between `RETRIEVAL_FAILURE` and `BUSINESS_FAILURE`?
> **Answer**: `RETRIEVAL_FAILURE` means the vector store returned no eligible or sufficient evidence chunks. `BUSINESS_FAILURE` means retrieval or tool execution succeeded, but domain rules could not fulfill the request (e.g., order ID not found in database, or active policy documents in conflict).

### Q10: Why build custom evaluation frameworks instead of using Ragas or DeepEval?
> **Answer**: To maintain 100% deterministic, zero-API-key reproducibility for local testing and CI/CD. External frameworks introduce heavy dependencies, network latency, non-deterministic scoring, and API key requirements. Our evaluation runner cleanly separates deterministic state assertions (`PASS` offline) from semantic judge assertions (`UNVERIFIED_REQUIRES_LLM` when API keys are absent).
