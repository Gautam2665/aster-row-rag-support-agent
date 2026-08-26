# 28. Day 5 Deterministic Retrieval Evaluation (Precision@K & Recall@K) Guide

This module covers standalone retrieval evaluation, Precision@K, Recall@K, ground-truth relevance matching, and why vector similarity distances are not evaluation metrics.

---

## 1. Why Separate Retrieval Evaluation from Generation Evaluation?

In traditional software systems, a single end-to-end integration test passes or fails. In RAG systems, end-to-end text generation output is non-deterministic and conflates two failure modes:
1. **Retrieval Failure**: The vector index failed to return the correct document.
2. **Generation Failure**: The LLM received the correct document but hallucinated or ignored the evidence.

Evaluating vector search independently using **Precision@K** and **Recall@K** allows engineers to measure, debug, and optimize index performance (chunk sizes, embedding models, metadata filters, top-K selection) **without making expensive LLM API calls** or relying on non-deterministic model outputs.

---

## 2. Precision@K and Recall@K Definitions

```text
               Retrieved Top-K Chunks (R_k)           Expected Relevant IDs (E)
          ┌───────────────────────────────────┐     ┌───────────────────────────┐
          │  Chunk 1: 07-warranty.md          │     │ 01-returns-policy-current │
          │  Chunk 2: 01-returns-policy... [✓]│ ◄───┤                           │
          │  Chunk 3: 09-trailplus...         │     └───────────────────────────┘
          │  Chunk 4: 03-final-sale...        │
          │  Chunk 5: 01-returns-policy... [✓]│
          └───────────────────────────────────┘
```

### Precision@K
What fraction of the $K$ retrieved chunks are actually relevant?
$$\text{Precision}@K = \frac{\text{Relevant Chunks Retrieved in Top-}K}{K}$$

* *Example Above*: 2 out of 5 retrieved chunks belong to `01-returns-policy-current.md`.
* $\text{Precision}@5 = \frac{2}{5} = 0.40$ (40%).

### Recall@K
What fraction of the ground-truth expected relevant documents were successfully retrieved in the top-$K$?
$$\text{Recall}@K = \frac{\text{Unique Expected Documents Hit in Top-}K}{\text{Total Expected Relevant Documents}}$$

* *Example Above*: 1 expected document (`01-returns-policy-current.md`), which was hit.
* $\text{Recall}@5 = \frac{1}{1} = 1.00$ (100%).

---

## 3. Confusion Matrix in Vector Search

| Metric Category | Definition in Vector Retrieval | RAG Impact |
| :--- | :--- | :--- |
| **True Positive (TP)** | Retrieved chunk matches expected relevant ground-truth document | High signal evidence for LLM prompt |
| **False Positive (FP)** | Retrieved chunk is irrelevant to the query | Wastes prompt tokens, risks hallucination/distraction |
| **False Negative (FN)** | Expected relevant document was NOT retrieved | Leads to "insufficient info" handoffs or ungrounded answers |

---

## 4. Why Vector Similarity Distances Are NOT Evaluation Metrics

A common mistake is treating ChromaDB's cosine distance output (e.g. `distance = 0.1824`) as proof that retrieval was correct.

### Why Distance $\neq$ Relevance:
1. **Semantic Vector Spacing**: Cosine distance measures vector proximity in embedding space, not semantic correctness. A retrieved passage can have a low distance ($0.15$) to a query because of shared vocabulary (*"returns"*, *"window"*) while actually being a **superseded 60-day policy** or an **internal migration note**.
2. **Ground-Truth Necessity**: True evaluation requires comparing retrieved chunk IDs against human-annotated ground-truth expected documents (`expected_relevant_ids`).

---

## 5. Limitations of Manually Created Evaluation Datasets

* **Coverage Limits**: Small datasets (e.g. 6–10 cases) test critical policy paths but do not represent millions of long-tail user queries.
* **Granularity Choice**: Matching by filename vs. exact chunk ID:
  * *Filename Matching*: Flexible across chunking strategy changes.
  * *Chunk ID Matching*: Precise, but brittle if document headers or chunk splits change.

---

## 6. How Retrieval Evaluation Connects to Visible and Custom Evaluation Cases

```text
Visible & Custom Evaluation Cases ────► State Assertions & LLM Prose Verification
(evaluation/*.json)                      (End-to-End System Evaluation)

Retrieval Evaluation Dataset      ────► Precision@K & Recall@K Metrics
(src/retrieval_evaluation.py)            (Standalone Vector Search Quality)
```

The standalone `RetrievalEvaluator` verifies vector search quality independently, ensuring the RAG pipeline feeds high-quality evidence into `ContextBuilder` before LLM generation begins.

---

## 7. Compact Interview Revision Sheet

* **Q: How do you measure RAG retrieval performance without an LLM?**
  * *A*: Using deterministic metrics **Precision@K** and **Recall@K** calculated by comparing retrieved chunk IDs against ground-truth expected document IDs (`RetrievalEvaluator`).
* **Q: What does Precision@5 = 0.20 mean?**
  * *A*: Out of the 5 top-ranked chunks returned by ChromaDB, exactly 1 chunk was relevant to the query.
* **Q: What does Recall@5 = 1.00 mean?**
  * *A*: 100% of the expected ground-truth relevant documents were retrieved within the top 5 results.
* **Q: Why can't vector distance scores replace evaluation metrics?**
  * *A*: Vector distance measures embedding proximity, not policy truth. An irrelevant or superseded document can have a low distance score if its keywords match the query.
