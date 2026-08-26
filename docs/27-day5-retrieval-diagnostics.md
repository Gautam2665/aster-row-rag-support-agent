# 27. Day 5 Retrieval Quality & Retrieval Diagnostics Guide

This module covers RAG retrieval observability, vector search diagnostics, precision/recall trade-offs, top-k selection, and failure classification in production AI agents.

---

## 1. What Are Retrieval Diagnostics?

Retrieval diagnostics are **non-sensitive execution metadata** captured during vector search operations to inspect what the retrieval layer actually fetched before context assembly and LLM generation.

```text
                  User / QueryContextualizer
                              │
                              ▼
                     KBVectorStore.search_with_diagnostics()
                              │
       ┌──────────────────────┴──────────────────────┐
       ▼                                             ▼
  retrieved_chunks                              RetrievalDiagnostic
  (KBChunk evidence)                           (Observability metadata)
       │                                             │
       ▼                                             ▼
ContextBuilder (Prompt Payload)               AgentState.retrieval_diagnostics
       │                                       (Excluded from LLM Prompt)
       ▼                                             │
  LLM Generation                                     ▼
                                              Developer Log / Evaluation
```

---

## 2. Retrieval Failure vs. Generation Failure

In RAG systems, quality breakdowns occur in two distinct stages:

| Failure Type | Stage | Root Cause | Example Symptom | Diagnostic Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval Failure** | Vector Search | 0 chunks returned, relevant policy missing, or vector distance too high | Agent says "I don't know" or retrieves legacy document | `num_eligible_chunks == 0` or `failure_classification == "RETRIEVAL_FAILURE"` |
| **Generation Failure** | LLM Synthesis | Retrieved chunks contain answer, but LLM ignores evidence or hallucinates claims | Chunks state "30 days", but LLM outputs "45 days" | Grounding assertion failure (`must_include`) despite `has_usable_evidence == True` |

### Why Distinguishing Them Is Critical
If an agent outputs an incorrect return window:
* If `retrieved_filenames` contains `02-returns-policy-legacy.md`, it is a **retrieval filtering failure**.
* If `retrieved_filenames` contains `01-returns-policy-current.md`, but LLM generated 60 days, it is a **generation hallucination failure**.

Without retrieval diagnostics, developers waste time prompt-engineering when the vector search index returned the wrong context.

---

## 3. Precision vs. Recall Intuition in RAG

* **Recall**: What fraction of relevant policy evidence was successfully retrieved?
  * *High Recall*: Retrieve 12 chunks so we don't miss niche exception rules.
  * *Risk*: Lowers precision, fills prompt context with irrelevant noise, increases cost, and risks "lost-in-the-middle" LLM attention degradation.
* **Precision**: What fraction of retrieved chunks are actually relevant to the user query?
  * *High Precision*: Retrieve top 2 chunks with strict similarity thresholds.
  * *Risk*: Lowers recall, risks missing crucial policy nuances or multi-document exceptions.

---

## 4. Top-K Trade-Offs

| Top-K Choice | Pros | Cons | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| **Small ($K = 2 - 4$)** | Lower latency, lower token cost, high prompt signal-to-noise ratio | Misses multi-source policy exceptions (e.g. final sale + damaged item) | Simple single-document Q&A |
| **Balanced ($K = 10 - 12$)** *(Aster & Row Default)* | Captures primary policy + exception clauses + product care rules | Slightly higher prompt tokens | Multi-source policy resolution & conflict detection |
| **Large ($K = 25+$)** | Maximum recall | Distracts LLM, exhausts context window, increases hallucination rate | Initial candidate retrieval before re-ranking |

---

## 5. Why Retrieval Metadata Must Remain Separate from LLM Context

Including retrieval diagnostics (`distances`, `num_candidates_returned`, `chunk_ids`) in the LLM prompt is an anti-pattern:
1. **Context Window Waste**: Scores and chunk IDs consume prompt tokens without aiding natural language understanding.
2. **Prompt Injection Risk**: Diagnostic data from untrusted files could contain malformed strings.
3. **Model Confusion**: LLMs may quote distance numbers or internal chunk IDs to customers (e.g., *"According to chunk_9912 with distance 0.231..."*).

`SupportAgent` attaches `RetrievalDiagnostic` to `AgentState.retrieval_diagnostics` for evaluation and developer debugging, while `ContextBuilder` excludes it completely from `ContextPayload`.

---

## 6. How Retrieval Diagnostics Map to Our RAG Architecture

```python
# KBVectorStore returns both evidence chunks and safe diagnostic metadata
chunks, diagnostic = vector_store.search_with_diagnostics(
    query=retrieval_query,
    top_k=12,
    filter_customer_eligible=True
)

# Diagnostic captures non-sensitive search metrics:
diagnostic.retrieval_query          # "Do you ship internationally? What about Canada?"
diagnostic.num_candidates_returned  # Candidate chunks evaluated
diagnostic.num_eligible_chunks      # Eligible chunks after metadata filter
diagnostic.retrieved_filenames      # ["06-international-shipping.md"]
diagnostic.distances                # [0.1824, 0.2911]
diagnostic.has_usable_evidence      # True
diagnostic.filter_applied           # {"status": "active", "policy_authority": "official", ...}
```

---

## 7. Compact Interview Revision Sheet

* **Q: What is a retrieval diagnostic in a RAG agent?**
  * *A*: A structured, non-sensitive metadata record (`RetrievalDiagnostic`) capturing vector query parameters, candidate counts, eligible counts, filenames, and distance scores for observability.
* **Q: How do you differentiate a retrieval failure from a generation failure?**
  * *A*: Inspection of `RetrievalDiagnostic`. If `has_usable_evidence` is `False` or required sources are missing from `retrieved_filenames`, it is a retrieval failure. If authoritative chunks are present but the LLM output is wrong, it is a generation failure.
* **Q: Why keep retrieval diagnostics out of the LLM prompt?**
  * *A*: Diagnostics are operational metrics for evaluation and debugging. Passing them to the LLM wastes context tokens and risks leaking internal metadata into customer responses.
* **Q: How does `KBVectorStore` preserve backward compatibility?**
  * *A*: `search()` calls `search_with_diagnostics()` internally and returns `List[KBChunk]`. Existing callers continue receiving identical evidence chunks without code breakage.
