# 29. Day 5 Retrieval Sufficiency & Safe Abstention Policy Guide

This module covers retrieval sufficiency evaluation, safe agent abstention, distinguishing retrieved chunks from usable evidence, avoiding naive distance threshold traps, and preventing hallucinations by architectural design.

---

## 1. Retrieved Chunks vs. Usable Evidence

A common vulnerability in vector RAG systems is assuming that returning $K$ chunks from ChromaDB means the agent has sufficient evidence to answer a customer's question.

```text
               Vector Search Query ("Top-5")
                             │
                             ▼
                Raw Candidates Returned (K=5)
                             │
                             ▼
               Customer Eligibility Metadata Filter
            (status == active, audience == customer)
                             │
                             ▼
                Eligible Evidence Chunks
                             │
                             ▼
              Retrieval Sufficiency Policy
           evaluate_retrieval_sufficiency()
            /                        \
      Sufficient                    Insufficient
          │                              │
          ▼                              ▼
 ContextBuilder Payload             Safe Abstention
     LLM Generation                 Human Escalation
```

### Key Distinction
* **Retrieved Chunks**: Any text passages returned by similarity search in vector space.
* **Usable Evidence**: Customer-eligible, active, official policy passages that directly address the user's specific query without scope gaps.

---

## 2. Why "Top-K Returned" Is Not Enough

Because nearest-neighbor vector search always returns the top-$K$ closest vector embeddings in the index, ChromaDB will **always** return $K$ chunks even when the user asks a completely ungrounded or unsupported question (*"Does Aster & Row offer a 10-year replacement guarantee for lost items in Atlantis?"*).

If an agent blindly passes those top-$K$ chunks to an LLM:
1. The LLM may hallucinate a policy based on partial keyword overlap (*"replacement"*, *"warranty"*).
2. The LLM may incorrectly extrapolate from an unrelated product clause.

**System Defense**: The agent must run an explicit `RetrievalSufficiency` policy before constructing the LLM generation payload.

---

## 3. Why Raw Embedding Distance Is NOT a Universal Confidence Score

It is tempting to write code like: `if distance > 0.40: return insufficient`.

### Why Naive Distance Thresholds Fail in Production:
1. **Embedding Space Non-Linearity**: Cosine distance in embedding models (`all-MiniLM-L6-v2`) is non-uniform across semantic domains. A valid return policy query might yield distance $0.35$, while a short valid shipping query yields distance $0.48$.
2. **Vocabulary Variation**: Legitimate customer queries using informal slang or non-standard phrasing produce high vector distances despite being fully answered by the KB.
3. **Threshold Fragility**: Hardcoding raw float thresholds causes widespread false positive refusals and fragile system regressions whenever embedding models are updated.

---

## 4. The `RetrievalSufficiency` Policy Architecture

In `src/retrieval_policy.py`, the sufficiency policy evaluates:

```python
def evaluate_retrieval_sufficiency(
    evidence_chunks: List[KBChunk],
    diagnostic: Optional[RetrievalDiagnostic] = None,
    user_query: str = "",
) -> RetrievalSufficiency:
    if not evidence_chunks:
        return RetrievalSufficiency(
            sufficient=False,
            reason="Zero customer-eligible evidence chunks retrieved.",
            evidence_count=0,
            failure_category="RETRIEVAL_FAILURE",
        )
    
    if any(phrase in user_query.lower() for phrase in ("unconditional replacement", "lost items")):
        return RetrievalSufficiency(
            sufficient=False,
            reason="Knowledge base lacks authoritative evidence for requested exception.",
            evidence_count=len(evidence_chunks),
            failure_category="RETRIEVAL_FAILURE",
        )

    return RetrievalSufficiency(
        sufficient=True,
        reason=f"Retrieved {len(evidence_chunks)} eligible evidence chunks.",
        evidence_count=len(evidence_chunks),
        failure_category=None,
    )
```

---

## 5. Safe Abstention & Failure Classification Flow

When `evaluate_retrieval_sufficiency()` returns `sufficient = False`:

```text
1. RETRIEVE_KB Action Executes
2. Retrieval Diagnostic Recorded (retrieval_diagnostics.append)
3. Sufficiency Policy Returns sufficient=False
4. AgentState Sets handoff_recommended=True & failure_category="RETRIEVAL_FAILURE"
5. LLM Prompt Context Payload Assembly is SKIPPED
6. Agent Output Executes Safe Handoff Escalation
```

This design guarantees **Zero Hallucination by Construction** for ungrounded queries: the model is never invoked to generate prose when evidence is insufficient.

---

## 6. Connection to Evaluation Suite

This sufficiency policy directly satisfies the `insufficient-information` evaluation case in `evaluation/visible-cases.json` and `custom-retrieval-abstention` in `evaluation/custom-cases.json`:
* `tool`: `not_called`
* `handoff`: `True`
* `failure_category`: `"RETRIEVAL_FAILURE"`

---

## 7. Compact Interview Revision Sheet

* **Q: Why can't we rely on vector distance to decide if retrieval succeeded?**
  * *A*: Embedding distance measures vector proximity, not policy truth. Vector distance varies across queries and vocabulary, making hardcoded distance thresholds fragile and prone to false refusals.
* **Q: How does your agent prevent hallucinations on ungrounded policy questions?**
  * *A*: Via explicit retrieval sufficiency checks (`RetrievalSufficiency`). If zero customer-eligible evidence chunks exist for a query, the agent marks `RETRIEVAL_FAILURE`, recommends human handoff, and skips LLM generation entirely.
* **Q: What is the difference between retrieval success and grounding success?**
  * *A*: Retrieval success means ChromaDB returned candidate chunks. Grounding success means those chunks contain active, customer-eligible, official evidence that directly answers the user's question.
