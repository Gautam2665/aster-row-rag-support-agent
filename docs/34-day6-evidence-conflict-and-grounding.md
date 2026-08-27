# 34. Day 6 Evidence Conflict & Grounding Policy Guide

This module covers the deterministic **Evidence Policy Assessment Layer**, explaining how agentic support architectures handle retrieval candidates vs. usable evidence, non-authoritative filtering, and active-source policy conflicts.

---

## 1. Evidence Assessment Architecture

```text
                           RETRIEVE_KB Action
                                   │
                                   ▼
                   KBVectorStore.search_with_diagnostics()
                         (Retrieved Chunks + Distances)
                                   │
                                   ▼
                      evaluate_retrieval_sufficiency()
                     (Checks empty / low-relevance threshold)
                                   │
                                   ▼
                          assess_evidence()
                  (Deterministic Evidence Policy Assessment)
                                   │
                     ┌─────────────┼─────────────┐
                     │             │             │
                  USABLE     INSUFFICIENT    CONFLICT
                     │             │             │
                     ▼             ▼             ▼
             ContextBuilder     HANDOFF       HANDOFF
                   │            No LLM        No LLM
                   ▼           Generation    Generation
             LLM Generation
```

---

## 2. Evidence Classification Taxonomy

| Status | Trigger Condition | Agent Behavior | Failure Category |
| :--- | :--- | :--- | :--- |
| `USABLE` | Active, customer-eligible authoritative chunks retrieved without contradictory policy directives. | Continues to `ContextBuilder` $\rightarrow$ LLM grounded generation. | None |
| `INSUFFICIENT` | Empty retrieval or only non-authoritative (superseded, internal draft) chunks returned. | **Skips LLM generation** $\rightarrow$ Triggers human handoff. | `RETRIEVAL_FAILURE` |
| `CONFLICT` | Multiple active official documents provide contradictory policy directives for the same item/topic. | **Skips LLM generation** $\rightarrow$ Triggers human handoff. Never chooses one source arbitrarily. | `BUSINESS_FAILURE` |

---

## 3. Why an LLM Must Not Arbitrarily Resolve Active Policy Conflicts

1. **Hallucination Risk**: When official documents contradict each other (e.g., care guide vs. product card), an LLM prompted to give a definitive answer will arbitrarily choose one or blend them into misleading advice.
2. **Customer Trust & Product Damage**: If the agent wrongly tells a customer that a hand-wash-only tumbler body is "100% dishwasher safe", the customer suffers product damage and breach of warranty terms.
3. **Application Control**: The application must deterministically identify active policy conflicts, abstain from ungrounded generation, and escalate to a human support supervisor for policy clarification.

---

## 4. Compact Interview Revision Sheet

### Q1: What is the difference between `INSUFFICIENT` evidence and `CONFLICT` evidence?
> **Answer**: `INSUFFICIENT` evidence occurs when retrieval returns no usable authoritative chunks (e.g., empty search or deprecated/internal drafts). `CONFLICT` evidence occurs when multiple active official documents return contradictory policy directives for the same topic. Both skip LLM generation and trigger handoff, but `INSUFFICIENT` is classified as `RETRIEVAL_FAILURE` whereas `CONFLICT` is classified as `BUSINESS_FAILURE`.

### Q2: Why shouldn't you use an LLM to resolve contradictory policy documents?
> **Answer**: Because LLMs are probabilistic models, not business authority systems. Asking an LLM to pick between conflicting official sources results in arbitrary guesses or hallucinated compromises. Enterprise RAG architectures require deterministic abstention and human escalation when active policy sources conflict.

### Q3: How does `assess_evidence()` evaluate source authority?
> **Answer**: It inspects `KBChunk` metadata fields (`status`, `audience`, `policy_authority`). Only chunks with `status == "active"`, `audience in ("customer", "all")`, and `policy_authority == "official"` qualify as usable authoritative evidence. Internal drafts or deprecated policies are filtered out.

### Q4: How does your agent prevent infinite loops when evidence conflict is detected?
> **Answer**: Upon detecting `EvidenceStatus.CONFLICT`, `assess_evidence()` marks `state.handoff_recommended = True`, records a `BUSINESS_FAILURE` observation, and breaks the execution loop immediately. The agent never retries retrieval or repeats the turn.
