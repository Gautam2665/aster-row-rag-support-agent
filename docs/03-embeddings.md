# 03. Embeddings, Vector Spaces & Cosine Similarity

## What Are Embeddings?

An embedding model maps text into a high-dimensional vector space $\mathbb{R}^d$ where semantically similar phrases are positioned close to one another, regardless of exact keyword matching.

For example:
* `"How long do I have to send my product back?"`
* `"Customers may return eligible products within 30 days of delivery."`

These sentences share zero matching vocabulary keywords, yet their vector representations align closely because they share the conceptual meaning of **return time limits**.

---

## Cosine Similarity Math

Cosine similarity measures the **directional alignment** (similarity) between two vectors $u$ and $v$:

$$\text{cosine similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

* **Score = 1.0**: Identical semantic direction (perfect alignment).
* **Score = 0.0**: Orthogonal (unrelated topics).
* **Score = -1.0**: Diametrically opposite vectors.

> **Note**: Cosine similarity measures similarity/alignment, not distance. Higher scores indicate greater semantic alignment.

---

## Vector Space Consistency Rule

> **CRITICAL RULE**: Document chunk embeddings and user query embeddings **must** be generated using the exact same embedding model instance and dimension space.

### Why You Cannot Mix Models:
* `all-MiniLM-L6-v2`: Generates **384-dimensional** dense vectors.
* `text-embedding-3-small`: Generates **1536-dimensional** dense vectors.

Vectors from different models live in completely distinct coordinate systems. Computing dot products or distances across different dimensions or models produces mathematical errors and invalid retrieval results. Our `EmbeddingProvider` strictly enforces a single unified model instance across document indexing and query vectorization.
