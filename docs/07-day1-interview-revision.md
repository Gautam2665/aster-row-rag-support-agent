# 07. Day 1 Rapid Interview Revision Guide

## 30-Second RAG Elevator Pitch
> *"RAG (Retrieval-Augmented Generation) is an architecture that dynamically fetches relevant, authoritative knowledge chunks from an external database and provides them to an LLM context window at query time. In our Day 1 system, we parse Markdown documents into section chunks preserving YAML frontmatter metadata, embed chunks into a vector space using Sentence-Transformers/OpenAI, and store them in ChromaDB. Crucially, because vector similarity measures topical relevance rather than authority, we enforce metadata eligibility filtering (status=='active', customer_answering==true) to prevent superseded policies or prompt injection notes from reaching the LLM."*

---

## Rapid-Fire Q&A Cheat Sheet

### 1. Explain Embeddings
Text strings are converted into dense mathematical floating-point vectors in a high-dimensional space where semantically related concepts sit close together, allowing similarity calculations without exact keyword matches.

### 2. Explain Section Chunking
Instead of blindly splitting text every $N$ characters (which breaks sentences and headings), we split along Markdown section headers (`##`), keeping headings, paragraphs, and tables together as single retrievable units.

### 3. Explain Cosine Similarity Formula
Cosine similarity measures vector directional alignment:
$$\text{cosine similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
Score ranges from $-1.0$ (opposite) to $1.0$ (identical direction). Cosine similarity measures alignment, not distance.

### 4. Explain Vector Database & Why ChromaDB?
Vector databases store embeddings alongside text and metadata, performing fast HNSW graph nearest-neighbor search. We chose ChromaDB because it is embedded, local, persistent, lightweight, and supports metadata queries (`$and`).

### 5. Why Metadata Filtering?
Vector similarity search retrieves *relevance candidates*, not *eligible evidence*. Superseded policies (e.g. 60-day return window) score high similarity for return questions. Metadata filtering on `status == "active"` excludes outdated or draft policies.

### 6. Why Not Top-1 Retrieval?
Single-vector retrieval is fragile. A single chunk may miss complementary details (e.g. damaged item return exceptions). Top-K candidate retrieval followed by metadata filtering ensures complete evidence selection.

### 7. Why RAG Doesn't Eliminate Hallucination?
If retrieval fetches wrong/insufficient context, or if the system prompt fails to enforce strict grounding, the LLM can still hallucinate. System prompts must enforce safe abstention when evidence is missing.

### 8. Why Not LangChain or LlamaIndex?
Frameworks like LangChain introduce heavy abstraction layers, opaque default chunkers, and version churn. Writing custom modular Python components (~150 lines) provided 100% control over metadata preservation and deterministic unit testing (`pytest`).

### 9. Pre-Filtering vs. Post-Filtering vs. The Naive Top-K Bug
* **Native Pre-filtering (Best Practice)**: Vector DB restricts search space using metadata (`where={"status": "active"}`) *before/during* vector search. Guarantees all $K$ returned results are active/valid.
* **Broad Candidate Post-filtering**: Retrieve broad pool ($N=25$), filter in Python, take top $K$. Compute-heavy fallback if DB lacks native filtering.
* **Naive Post-filtering (Dangerous Bug)**: Retrieve small Top-3 *first*, then filter in Python. If the top 3 nearest vectors are superseded/draft items, post-filtering drops all 3, leaving the LLM with **zero context**!
