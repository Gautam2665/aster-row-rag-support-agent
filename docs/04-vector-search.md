# 04. Vector Databases & Similarity Search

## 1. What a Vector Database Stores
Unlike traditional relational databases (SQL) that index scalar values or key-value pairs, a **vector database** stores multi-dimensional floating-point vectors alongside text documents and metadata payloads:

```json
{
  "id": "01-returns-policy-current.md#standard-return-window",
  "vector": [0.024, -0.115, ..., 0.041],
  "document": "Customers on the standard plan may request a return within 30 calendar days...",
  "metadata": {
    "filename": "01-returns-policy-current.md",
    "heading": "Returns Policy > Standard return window",
    "status": "active",
    "audience": "customer",
    "policy_authority": "official",
    "customer_answering": true
  }
}
```

---

## 2. Why ChromaDB?
For our Aster & Row RAG implementation, we selected **ChromaDB** as the local vector store because:
1. **Lightweight & Embedded**: Runs locally inside Python without external daemon setups or cloud dependencies.
2. **Metadata Filtering**: Supports rich `$and` / `$eq` boolean queries directly over document frontmatter attributes.
3. **Persistence**: Supports both in-memory client mode for fast unit tests and `PersistentClient` for disk persistence.
4. **Zero-Framework Bloat**: Operates independently of LangChain or LlamaIndex wrappers.

---

## 3. Cosine Distance in ChromaDB
ChromaDB configures space metric via `metadata={"hnsw:space": "cosine"}`. 
In ChromaDB, distances are returned as cosine distances:

$$\text{distance} = 1 - \text{cosine similarity}$$

* **Distance near 0.0**: Extremely close semantic alignment.
* **Distance near 1.0**: Unrelated vectors.

---

## 4. HNSW (Hierarchical Navigable Small World) at a High Level
Performing exact brute-force Nearest Neighbor search across millions of vectors requires $O(N \cdot d)$ floating-point operations per query, which is too slow.

ChromaDB uses **HNSW (Hierarchical Navigable Small World)** graphs:
* HNSW constructs a multi-layer graph index (similar to a skip-list for vectors).
* The top layers contain sparse long-range connections for fast multi-dimensional traversal.
* The bottom layers contain dense local connections for accurate nearest-neighbor refinement.
* HNSW delivers $O(\log N)$ approximate nearest neighbor (ANN) search latency.

---

## 5. Offline Indexing vs. Online Querying

```text
OFFLINE INDEXING (Run once or on doc update):
Directory Markdown Files ──► Parse Frontmatter ──► Chunk Sections ──► Embed ──► Add to ChromaDB

ONLINE RETRIEVAL (Run per user message):
User Question ──► Embed Query ──► HNSW Graph Query ──► Top-K Candidates ──► Metadata Filter ──► Evidence
```

---

## 6. Top-K Retrieval & Why Top-K Isn't Final Evidence
* **Top-K Retrieval**: Vector search fetches a broad candidate pool of nearest neighbor vectors (e.g. $K = 10$).
* **Why Top-K Isn't Final Evidence**: Nearest vector neighbors may include superseded 60-day policies, internal draft notes, or prompt injection scratchpads. Top-K provides *relevance candidates*, NOT *eligible evidence*.

---

## 7. Metadata Stored Alongside Vectors
ChromaDB indexes scalar metadata values alongside vector IDs. Our implementation ([`src/retrieval.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/retrieval.py#L40-L58)) stores:
* `filename`, `heading`, `source_citation`
* `document_id`, `title`
* `status`, `audience`, `policy_authority`, `customer_answering`
* `supersedes`, `superseded_by`

---

## 8. Our ChromaDB Implementation ([`src/retrieval.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/retrieval.py))

```python
class KBVectorStore:
    def __init__(self, collection_name="aster_row_kb"):
        self.embedding_provider = EmbeddingProvider()
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def search(self, query: str, top_k=10, filter_customer_eligible=True):
        query_emb = self.embedding_provider.embed_query(query)
        where_filter = None
        if filter_customer_eligible:
            where_filter = {
                "$and": [
                    {"status": {"$eq": "active"}},
                    {"policy_authority": {"$eq": "official"}},
                    {"audience": {"$eq": "customer"}},
                    {"customer_answering": {"$eq": True}}
                ]
            }

        return self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter
        )
```

---

## 9. Interview Questions & Answers

### Q1: What algorithm does ChromaDB use for fast vector search?
> **Answer**: ChromaDB uses HNSW (Hierarchical Navigable Small World) graph indexing, which builds a multi-layer graph to achieve logarithmic $O(\log N)$ approximate nearest neighbor search time.

### Q2: Why store metadata inside the vector database instead of a separate database?
> **Answer**: Storing metadata alongside vectors enables single-pass vector filtering (`where` clause). The vector database filters out ineligible items (e.g. `status != active`) during nearest-neighbor traversal, avoiding unnecessary overhead.
