# 01. RAG Foundations

## Why RAG Instead of Passing the Entire Knowledge Base?

When building AI support agents over corporate knowledge bases, passing the entire raw corpus directly into the LLM prompt is problematic for several reasons:

1. **Context Window & Latency Overhead**: As the document corpus grows, sending hundreds of thousands of words per user request increases API latency and token cost exponentially.
2. **"Lost in the Middle" Attention Degradation**: LLMs exhibit degraded recall when critical facts are buried in massive prompt context windows.
3. **Data Quality & Poisoning Vulnerabilities**: Raw corpora often contain legacy policies, internal notes, or prompt injection payloads. Including the entire raw corpus exposes the model to prompt poisoning and hallucinated legacy claims.

Retrieval-Augmented Generation (RAG) solves these issues by extracting, indexing, and selecting only the top relevant, authoritative passages before calling the model.

---

## Documents vs. Chunks

* **Document**: A complete file (e.g. `01-returns-policy-current.md`) containing multiple sub-topics, frontmatter metadata, headers, and paragraphs.
* **Chunk**: A smaller, semantically self-contained unit of text extracted from a document. Chunks are the fundamental unit of embedding, indexing, vector similarity search, and LLM context injection.

### The RAG Lifecycle
```text
Raw Markdown File 
   └── YAML Frontmatter Parsing -> DocumentMetadata
   └── Section Header Splitting  -> KBChunk[]
   └── Vector Embedding          -> Float Vector (e.g. 384-dim)
   └── Chroma Indexing          -> Local Persistent Vector Store
   └── Query Vector Search       -> Candidate Top-K Chunks
   └── Metadata Eligibility Filter -> Authoritative Evidence Chunks
```
