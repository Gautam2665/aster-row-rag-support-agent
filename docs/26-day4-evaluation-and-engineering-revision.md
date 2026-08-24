# 26. Day 4 Evaluation & Engineering Revision Guide

This module covers the core principles of evaluation-driven development, deterministic versus semantic testing, regression prevention, failure taxonomies, and system architecture for production AI agent applications.

---

## 1. Evaluation-Driven Development (EDD) for LLM Systems

Traditional software engineering relies on deterministic unit testing ($\text{Input} \rightarrow \text{Fixed Output}$). In LLM-based Compound AI Systems, output prose is probabilistic and non-deterministic.

Evaluation-Driven Development (EDD) structures test suites around **state assertions** and **semantic expectations**:
* **State Assertions (Deterministic)**: Did the agent execute the correct tool? Were PII fields scrubbed? Was the superseded policy excluded? Was handoff recommended?
* **Semantic Assertions (Probabilistic)**: Does the generated response convey the required facts without hallucinating forbidden claims?

---

## 2. Deterministic vs. Semantic Evaluation

```text
                                Evaluation Suite
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  Deterministic State Assertions                         Semantic Text Assertions
  • Tool allowlist validation                           • Phrase inclusion (must_include)
  • Parameter schema validation                         • Concept presence (must_include_concepts)
  • PII sanitization (CustomerSafeOrderResult)           • Exclusion rules (must_not_include)
  • Source lineage check                                • Tone & refusal verification
  • Handoff recommendation flag                         
            │                                                     │
            ▼                                                     ▼
     PASS / FAIL                                           PASS / FAIL / UNAVAILABLE
```

### Why `UNVERIFIED_REQUIRES_LLM` is Rigorous Engineering
Running evaluation suites against mock LLMs (`MockLLMProvider`) cannot evaluate live prose generation. Marking state assertions as `PASS` and semantic assertions as `UNVERIFIED_REQUIRES_LLM` prevents false test confidence while maintaining 100% deterministic test coverage offline.

---

## 3. Scenario-Based Evaluation

Scenario-based evaluation tests multi-turn customer journeys, edge cases, and security attacks rather than isolated single-turn prompts.

### Visible vs. Custom Evaluation Cases
* **Visible Cases (`evaluation/visible-cases.json`)**: 15 candidate evaluation cases covering return windows, TrailPlus membership, damaged item exceptions, Canadian shipping, missing order IDs, stale ETAs, PII privacy, prompt injections, and source conflicts.
* **Custom Cases (`evaluation/custom-cases.json`)**: 5 original evaluation cases testing session isolation, internal data refusal, retrieval abstention, multi-turn order follow-ups, and planner failure recovery.

---

## 4. Regression Testing in Compound AI Systems

When modifying prompts, context builders, or tool handlers, regression tests ensure existing security and state boundaries remain unbroken.

| Regression Risk | Prevention Mechanism | Regression Test |
| :--- | :--- | :--- |
| **Session Cross-Contamination** | Session-isolated memory queues (`SessionMemoryStore`) | `test_custom_eval_cases.py::test_session_isolation_case_behavior` |
| **PII / Internal Data Disclosures** | Security firewall (`CustomerSafeOrderResult`) | `test_order_lookup.py` & `test_custom_eval_cases.py::test_internal_data_refusal` |
| **Unbounded Tool Execution Loops** | Progress protection & `max_iterations = 3` | `test_agent_control_loop.py` |
| **Unauthorized Action Execution** | Schema validation (`ActionValidator`) | `test_planner_contract.py` & `test_custom_eval_cases.py::test_planner_failure_recovery` |

---

## 5. Failure Taxonomy & State Recovery

A production agent treats failures as **typed state transitions**, not infrastructure crashes.

```text
                                Execution Failure
                                        │
        ┌───────────────────┬───────────┴───────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
    TOOL_ERROR      BUSINESS_FAILURE        RETRIEVAL_FAILURE   PLANNER_FAILURE
  (Exception in     (Unknown order /       (0 evidence chunks  (Malformed JSON /
   tool execution)   source conflict)        from vector search) invalid action)
        │                   │                       │                   │
        └───────────────────┴───────────┬───────────┴───────────────────┘
                                        ▼
                             Record AgentObservation
                               (failure_category)
                                        │
                                        ▼
                             Safe Handoff Transition
                             (handoff_recommended=True)
```

---

## 6. Observability (Trace) vs. Conversation Memory

```text
Conversation History ────► ContextBuilder ────► LLM Prompt Context
                                                      
Execution Trace      ────► Developer Log  ────► CLI Debug Output (--debug)
```

### Critical Separation Rules
* **Memory** stores customer dialogue turns to inform future turns. Memory is **untrusted customer context** and enters prompt payloads wrapped in XML tags.
* **Trace** (`AgentTrace`) logs internal lifecycle events (`TraceEvent`). Trace logs are **strictly excluded** from LLM prompt context to prevent context pollution and prompt injection vectors.

---

## 7. How the Day 1–4 Architecture Fits Together

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │ DAY 1: Vector RAG Core                                                 │
  │ Markdown Ingestion ──► Heading Chunking ──► ChromaDB ──► Metadata Filter│
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ DAY 2: Grounded Generation & Tool Firewall                             │
  │ BaseLLMProvider ──► OrderLookupTool (PII Scrubbing) ──► Prompt Context │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ DAY 3: Bounded ReAct Planning & Session Memory                         │
  │ SessionMemoryStore ──► QueryContextualizer ──► LLMPlanner ──► Validator│
  │                     ──► Bounded Loop (max_iter=3) ──► AgentTrace       │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ DAY 4: Evaluation & CLI Runtime                                        │
  │ Visible + Custom Cases ──► Dual-Layer Eval ──► Interactive CLI Adapter │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Compact Interview Revision Sheet

* **Q: How does the system prevent hallucinating order details?**
  * *A*: Order lookups are restricted to `OrderLookupTool`. Raw order files never enter prompt context. Order results are sanitized into `CustomerSafeOrderResult` objects before reaching the LLM.
* **Q: How are superseded policies handled?**
  * *A*: Pre-retrieval metadata filtering (`status == "active"`, `policy_authority == "official"`) drops legacy documents at the ChromaDB vector index query layer.
* **Q: Why separate `user_query` from `retrieval_query`?**
  * *A*: `retrieval_query` reformulates ambiguous follow-ups (*"What about Canada?"*) with past session turns for vector search, while `user_query` is preserved verbatim in prompt context to retain customer tone.
* **Q: Why validate planner actions with an application validator?**
  * *A*: The LLM proposes actions; `ActionValidator` enforces parameter schemas and tool allowlists. The LLM cannot execute arbitrary code or unauthorized tools.
* **Q: How are execution loops bounded?**
  * *A*: The state machine enforces `max_iterations = 3`, progress protection against duplicate non-terminal actions, and explicit terminal action transitions (`RESPOND`, `CLARIFY`, `HANDOFF`).
* **Q: What is the difference between deterministic and semantic evaluation?**
  * *A*: Deterministic evaluation verifies tool calls, security, PII scrubbing, and lineage state directly from `AgentState`. Semantic evaluation verifies live LLM text generation against phrase/concept assertions.
