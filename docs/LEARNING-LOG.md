# Day 1 Personal Learning Journal & Technical Evolution

## 1. What I Knew Initially
* Basic concept of RAG ("send documents to LLM to answer questions").
* Basic idea of vector embeddings and ChromaDB.

---

## 2. What I Learned Today

1. **RAG Architecture**: Why full-corpus prompts fail (context limits, cost, TTFT latency, lost-in-the-middle attention degradation).
2. **Document Segmentation**: Why fixed-character chunking breaks structure, and how Markdown heading-aware chunking preserves section titles for citations (`01-returns-policy-current.md > Standard return window`).
3. **Vector Mathematics**: Embeddings represent text in vector space. Cosine similarity $\frac{u \cdot v}{\|u\| \|v\|}$ measures directional alignment.
4. **Vector Space Consistency**: Document chunks and user query embeddings **must** use the exact same embedding model instance and vector dimensionality.
5. **Candidates vs Evidence**: Semantic vector search finds *relevance candidates*. Metadata filtering (`status == "active"`, `customer_answering == true`) determines *evidence eligibility*.
6. **Prompt Security**: Pre-retrieval metadata filtering drops internal draft scratchpads (`14-internal-content-migration-notes.md`) containing prompt injection payloads before they reach the LLM.

---

## 3. Key Mistakes & Corrections Made During Day 1

1. **Misconception**: Initially confused Cosine Similarity with Cosine Distance.
   * **Correction**: Cosine similarity measures directional alignment (higher = more aligned). Cosine distance is derived as $\text{distance} = 1 - \text{similarity}$ (lower = closer).
2. **Misconception**: Thought LLMs perform vector search.
   * **Correction**: The vector database retriever performs nearest-neighbor vector search. The LLM receives plain text context.
3. **Misconception**: Confused a chunk with an embedding.
   * **Correction**: A chunk is a text string + metadata object ([`KBChunk`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py#L47-L86)). An embedding is its numerical float vector.
4. **Misconception**: Assumed high semantic similarity implies correctness.
   * **Correction**: Proved via experiment ([`legacy_policy_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/legacy_policy_experiment.py)) that superseded legacy policies (60 days) produce high similarity scores (~0.65) for return questions. Metadata filtering is mandatory.
5. **Misconception**: Added fallback logic between OpenAI and SentenceTransformer embeddings in an educational script.
   * **Correction**: Recognized that fallback logic must **never** enter production, because vector dimensions (1536 vs 384) cannot be mixed in a single vector index.

---

## 4. What I Implemented Today

* [`src/models.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py): `DocumentMetadata` and `KBChunk` domain models.
* [`src/ingestion.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/ingestion.py): Markdown YAML frontmatter parser, section chunker, section subdivider, and batch directory ingester.
* [`src/embeddings.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/embeddings.py): `EmbeddingProvider` enforcing unified vector spaces.
* [`src/retrieval.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/retrieval.py): `KBVectorStore` ChromaDB vector index with metadata eligibility filtering.
* [`experiments/embedding_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/embedding_experiment.py): Keyword independence semantic experiment.
* [`experiments/legacy_policy_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/legacy_policy_experiment.py): Superseded legacy policy experiment.
* **14 Unit Tests** ([`tests/`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/tests/)): Passing test suite covering models, ingestion, and vector retrieval.

---

## 5. What I Can Now Explain in an Interview

* Why RAG is superior to fine-tuning or full-corpus prompt stuffing.
* How Markdown section chunking works and why heading citations are critical.
* Mathematical formula and interpretation of Cosine Similarity.
* Why vector databases (ChromaDB) use HNSW graph indices for $O(\log N)$ retrieval.
* Why semantic vector search alone fails on corporate policy updates without metadata filtering.
* How pre-retrieval metadata filtering blocks prompt injections.

---

## 6. Questions & Topics to Revisit on Day 2

* How to integrate the LLM (Gemini / OpenAI) with retrieved context.
* System prompt design for strict grounding, citations, and safe abstention.
* Function calling / tool use for mock order lookups (`orders.json`).
* Multi-source conflict handling (e.g. tumbler dishwasher care conflict).
