# ADR 001: RAG Foundation & Vector Retrieval Architecture

## Status
Accepted (Day 1)

---

## Context & Problem Statement
Aster & Row requires a RAG support agent operating over corporate Markdown documentation and order snapshots. The system must handle realistic corporate data quality issues, such as superseded policies (30 vs 60 days return window), internal draft notes, prompt injection payloads inside migration scratchpads, and strict citation requirements.

We needed to establish the initial data models, document chunking strategy, embedding pipeline, and vector storage retrieval layer.

---

## Decision

We chose to implement a **lightweight, modular Python architecture** using standard Python data structures and **ChromaDB** as the local vector store, without using heavy orchestration frameworks like LangChain or LangGraph.

### Architecture Breakdown:
1. **Domain Data Models** ([`src/models.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py)): `DocumentMetadata` (parsing frontmatter) and `KBChunk` (representing section chunks with metadata and source citations).
2. **Ingestion & Semantic Section Chunking** ([`src/ingestion.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/ingestion.py)): Markdown header-based parsing (`#`, `##`) that preserves document heading context and subdivides large sections at paragraph breaks while keeping parent metadata.
3. **Unified Embedding Provider** ([`src/embeddings.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/embeddings.py)): Enforces a single consistent model (`all-MiniLM-L6-v2` or `text-embedding-3-small`) for both document chunks and user queries.
4. **Vector Store & Metadata Filtering** ([`src/retrieval.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/retrieval.py)): Local ChromaDB vector index with strict metadata eligibility filtering (`status == "active"`, `policy_authority == "official"`, `customer_answering == True`).

---

## Why This Architecture?

* **Full Transparency & Debuggability**: Every step of parsing, chunking, embedding, and filtering is explicitly defined and easily testable without opaque framework magic.
* **Deterministic Unit Testing**: Clean module separation allows unit testing each component individually (`pytest`) from models to full vector retrieval.
* **Metadata Authority Control**: Native frontmatter metadata filtering guarantees that superseded policies and prompt-injection scratchpads never reach the LLM context.

---

## Alternatives Considered & Why Rejected

### 1. LangChain / LlamaIndex
* **Why Rejected**: LangChain introduces deep abstraction hierarchies, hidden prompt wrappers, unpredictable default chunkers, and version instability. Writing a clean ~150-line custom ingestion and retrieval layer gave us 100% control over metadata preservation and vector filtering without framework bloat.

### 2. Pure In-Memory Numpy Cosine Search
* **Why Rejected**: While simple, an in-memory matrix multiplication script lacks persistent storage, efficient HNSW indexing, and structured metadata querying operators (`$and`, `$eq`). ChromaDB provides a lightweight local vector store with built-in metadata querying.

### 3. Model Fallback / Mixing Embeddings in Production
* **Why Rejected**: Generating document embeddings with one model (e.g. 384-dim) and query embeddings with another (e.g. 1536-dim) violates vector space consistency and causes mathematical calculation errors. We mandated a single `EmbeddingProvider` instance across indexing and retrieval.

---

## Tradeoffs & Consequences

* **Pros**: Zero heavy framework dependencies, instant test suite execution (<1s for models/ingestion, ~10s for vector search), clear auditability, robust metadata filtering.
* **Cons**: We manually maintain the Markdown parser and section splitter code instead of importing a pre-built library class.
