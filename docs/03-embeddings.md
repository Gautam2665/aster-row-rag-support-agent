# 03. Embeddings, Vector Spaces & Cosine Similarity

## 1. What is an Embedding?
An **embedding** is a dense numerical vector representation of text in a continuous multi-dimensional mathematical space $\mathbb{R}^d$.

## 2. Text → Vector
An embedding model converts natural language strings into floating-point numbers:

```text
"Return within 30 days" ──► [0.024, -0.115, 0.892, ..., 0.041] (384 dimensions)
```

## 3. Why Numerical Representation?
Computers cannot compute mathematical distance or similarity directly on raw text strings. Converting text into dense vectors enables linear algebra operations (dot products, vector norms) to measure semantic relationships instantaneously across millions of documents.

## 4. Semantic Similarity
Semantic similarity means two pieces of text share conceptual meaning even if they use entirely different vocabulary:
* Text A: *"How long do I have to send my product back?"*
* Text B: *"Customers may return eligible products within 30 days of delivery."*

Though A and B share virtually no matching keywords, their vector embeddings point in nearly the same mathematical direction.

---

## 5. Query Embedding & 6. Document Embedding
* **Document Embedding**: Calculated offline during ingestion for each [`KBChunk`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py#L47-L86) and stored in ChromaDB.
* **Query Embedding**: Calculated online at runtime for incoming user questions.

---

## 7. Same Vector Space Requirement (CRITICAL RULE)

> **MANDATORY ARCHITECTURAL RULE**: Document embeddings and Query embeddings **must** be generated using the exact same embedding model instance and vector dimension.

### Why You Cannot Mix Models:
* `all-MiniLM-L6-v2`: Generates **384-dimensional** vectors.
* `text-embedding-3-small`: Generates **1536-dimensional** vectors.

Vectors from different models inhabit incompatible mathematical coordinate systems. Mixing vectors across models or dimensions produces invalid dot products and runtime exceptions. Our [`EmbeddingProvider`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/embeddings.py) strictly enforces model uniformity across indexing and retrieval.

---

## 8. Cosine Similarity Math

Cosine similarity evaluates the directional alignment ($\theta$) between two vectors $u$ and $v$:

$$\text{cosine similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

## 9. Interpreting Cosine Similarity Scores
* **1.0**: Perfect semantic alignment (pointing in the exact same direction).
* **0.0**: Orthogonal (completely independent / semantically unrelated).
* **-1.0**: Diametrically opposite semantic meaning.

---

## 10. Similarity ≠ Correctness / Distance Distinction

> **Crucial Distinction**: Cosine similarity measures **directional alignment**, NOT distance or factual truth.

* **Cosine Similarity**: Ranges from $-1.0$ to $1.0$. Higher score = greater directional alignment.
* **Cosine Distance**: Derived as $\text{distance} = 1 - \text{similarity}$. Lower score = closer proximity.

A high similarity score means two texts discuss the same topic—it does **not** mean the document is authoritative, current, or correct!

---

## 11. Our Educational Experiment 1 ([`experiments/embedding_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/embedding_experiment.py))

We conducted an experiment comparing a user query against two document sentences:

### Inputs:
* **Query (#2)**: *"How long do I have to send my product back?"*
* **Doc #1**: *"Customers may return eligible products within 30 days of delivery."*
* **Doc #3**: *"Customers can track their shipment using the carrier tracking number."*

### Observed Results (`all-MiniLM-L6-v2`, 384 dimensions):
* **Query vs. Doc #1 (Return Policy)**: **`0.6970`**
* **Query vs. Doc #3 (Carrier Tracking)**: **`0.1706`**

## 12. What We Learned From Experiment 1
Vector embeddings capture semantic relationships beyond keyword matching. Even though Query #2 used *"send my product back"* and Doc #1 used *"return eligible products"*, the embedding model mapped them to the same neighborhood (similarity `0.6970` vs `0.1706` for tracking).

---

## 13. Embedding Model Choice & 14. Dimensions

| Model | Provider | Dimensions | Primary Use Case |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Sentence-Transformers (Local) | 384 | Fast, zero-cost, privacy-preserving offline CPU inference. |
| `text-embedding-3-small` | OpenAI API | 1536 | High-accuracy commercial API embedding. |

---

## 15. Interview Questions & Answers

### Q1: Why must query embeddings use the same model as document embeddings?
> **Answer**: Because vector similarity depends on comparing vectors within the exact same coordinate system. Mixing a 384-dim vector with a 1536-dim vector makes vector space math impossible.

### Q2: What is the mathematical difference between Cosine Similarity and Cosine Distance?
> **Answer**: Cosine similarity measures directional alignment (ranging from $-1$ to $1$, where $1$ is identical alignment). Cosine distance measures separation, defined as $\text{distance} = 1 - \text{similarity}$.
