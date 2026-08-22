# 06. Day 1 Architecture & Implementation Mapping

## 1. System Architecture Overview

```text
                                KNOWLEDGE BASE
                               (14 .md files)
                                      │
                                      ▼
                             ┌─────────────────┐
                             │    Ingestion    │
                             │ (src/ingestion) │
                             └────────┬────────┘
                                      │
                                      ▼
                                   KBChunk
                             (src/models.py)
                                      │
                                      ▼
                              EmbeddingProvider
                            (src/embeddings.py)
                                      │
                                      ▼
                                Vector Store
                             (ChromaDB Index)
                                      ▲
                                      │
                                Query Vector
                                      ▲
                                      │
                             EmbeddingProvider
                                      ▲
                                      │
                                 USER QUERY

                              VECTOR SEARCH
                                    │
                                    ▼
                             Candidate Chunks
                                    │
                                    ▼
                          Metadata Eligibility
                           (status, audience)
                                    │
                                    ▼
                            Relevant Evidence
```

---

## 2. Codebase Implementation Mapping (`src/`)

### A. [`src/models.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py) — Domain Data Models
* **`DocumentMetadata`**: Dataclass representing YAML frontmatter:
  * `document_id`, `title`, `status`, `audience`, `policy_authority`
  * `effective_date`, `last_reviewed`, `supersedes`, `superseded_by`, `customer_answering`
  * `is_active_official` property evaluating customer-facing eligibility.
* **`KBChunk`**: Dataclass representing indexed chunks:
  * `chunk_id`, `filename`, `heading`, `text`, `metadata`
  * `source_citation` property formatting `"filename > heading"`.

### B. [`src/ingestion.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/ingestion.py) — Frontmatter Parser & Section Chunker
* **`parse_markdown_file(file_path)`**: Reads `.md` file, parses PyYAML frontmatter into `DocumentMetadata`, returns body string.
* **`parse_markdown_sections(filename, body_text, metadata)`**: Splits Markdown body along header boundaries (`#`, `##`), creates `KBChunk` objects.
* **`split_section_text(text, max_chars=1000)`**: Subdivides sections larger than 1000 characters at paragraph breaks (`\n\n`) while maintaining parent heading context and document metadata.
* **`ingest_kb_directory(directory_path)`**: Scans `knowledge-base/` directory and yields indexed `KBChunk` list.

### C. [`src/embeddings.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/embeddings.py) — Unified Embedding Provider
* **`EmbeddingProvider`**: Enforces a single embedding model instance across document chunks and user queries (`all-MiniLM-L6-v2` locally or `text-embedding-3-small` via API key).

### D. [`src/retrieval.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/retrieval.py) — Vector Storage & Filtered Retrieval
* **`KBVectorStore`**: Wraps ChromaDB vector collection (`hnsw:space: cosine`).
* **`index_chunks(chunks)`**: Generates embeddings and stores chunk text, IDs, vectors, and metadata in ChromaDB.
* **`search(query, top_k=10, filter_customer_eligible=True)`**: Converts query to vector, queries ChromaDB, applies metadata `$and` filter, and reconstructs `KBChunk` objects with source citations.

---

## 3. Data Flow Execution Sequence

```text
1. User message arrives: "What is the return window?"
2. KBVectorStore.search() calls EmbeddingProvider.embed_query("What is the return window?")
3. ChromaDB executes HNSW vector search against stored KBChunk vectors.
4. ChromaDB applies metadata filter:
   {
     "$and": [
       {"status": {"$eq": "active"}},
       {"policy_authority": {"$eq": "official"}},
       {"audience": {"$eq": "customer"}},
       {"customer_answering": {"$eq": true}}
     ]
   }
5. Candidate chunks from 02-returns-policy-legacy.md (superseded) and 14-internal-migration-notes.md (draft) are dropped.
6. KBChunk from 01-returns-policy-current.md (active, 30 days) is returned with citation "01-returns-policy-current.md > Returns Policy > Standard return window".
```
