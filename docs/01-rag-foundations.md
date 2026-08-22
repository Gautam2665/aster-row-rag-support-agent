# 01. RAG Foundations

## 1. The Problem: Why Can't We Put the Entire Knowledge Base into the LLM Prompt?

When engineering customer support systems, a naive approach is to load all company policy Markdown files directly into the system prompt of a Large Language Model (LLM). For small toy datasets this might work, but for real production applications it breaks down across six core dimensions:

### Context Window Limits
Even modern models with 128k or 1M token context windows cannot hold enterprise knowledge bases (which often span tens of thousands of pages, manuals, order logs, and catalogs).

### Cost
LLM API pricing scales linearly with input tokens. Sending 50,000 words of policy documentation on every single \$0.01 user interaction renders the application economically unviable at scale.

### Latency
Processing massive prompt context windows dramatically increases Time-To-First-Token (TTFT) and overall inference time, degrading user experience.

### Irrelevant Information & Attention Degradation ("Lost in the Middle")
Research shows that LLMs suffer from degraded recall when key details are surrounded by long blocks of irrelevant text. The model may miss subtle nuances or hallucinate answers when overloaded with context.

### Data Exposure & Privacy Risks
Stuffing the full database into the context window risks leaking confidential internal data (e.g., customer PII, internal warehouse notes, fraud risk scores) to the end user.

### Scalability & Maintenance
Knowledge bases update continuously. Loading static files into system prompts prevents real-time updates and dynamic data lookups (such as live order status checks).

---

## 2. What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architectural pattern that separates **knowledge storage** from **reasoning**. 

Instead of relying on the LLM's static parametric memory or cramming entire documents into the prompt, RAG dynamically retrieves only the relevant, authoritative snippets of information from an external database and provides them as grounded evidence to the LLM at query time.

```text
Raw Corpus ──► Indexing ──► Vector Database
                                 │
User Query ──► Vector Search ────┴──► Relevant Chunks ──► LLM Prompt ──► Answer
```

---

## 3. Traditional LLM vs. RAG

| Feature | Traditional LLM (Prompt-Only) | RAG (Retrieval-Augmented) |
|---|---|---|
| **Knowledge Source** | Fixed during pre-training; static. | Dynamic external knowledge base + vector store. |
| **Accuracy / Grounding** | Prone to hallucinations on specific domain facts. | Strictly grounded in retrieved source passages. |
| **Auditability** | Cannot cite exact file sources or headings. | Generates clear citations (`filename > heading`). |
| **Data Freshness** | Requires expensive fine-tuning/re-training. | Instant; add/update document files in vector store. |
| **Cost & Latency** | High prompt token costs if sending raw corpus. | Low; sends only Top-K relevant chunks (~500 tokens). |

---

## 4. Complete RAG Pipeline

```text
OFFLINE INGESTION STAGE:
Document Files (.md) ──► Parse YAML Frontmatter ──► Section Chunking ──► Embedding Model ──► Vector DB Storage

ONLINE RETRIEVAL & GENERATION STAGE:
User Question ──► Query Embedding ──► Similarity Search ──► Metadata Filter ──► Prompt Assembly ──► LLM ──► Grounded Answer
```

---

## 5. Concrete Example

### User Query
> *"How long can I return my bag?"*

### Knowledge Base Document (`01-returns-policy-current.md`)
```markdown
---
document_id: RET-2026-01
status: active
audience: customer
policy_authority: official
supersedes: RET-2024-01
---
# Returns Policy
## Standard return window
Customers on the standard plan may request a return within 30 calendar days of delivery.
```

### Step-by-Step Execution:
1. **Query Vectorization**: Embed user query *"How long can I return my bag?"* into vector space.
2. **Vector Similarity Search**: Match query vector against chunk vectors in ChromaDB.
3. **Metadata Eligibility Filtering**: Assert `status == "active"`, `audience == "customer"`, `customer_answering == true`.
4. **Context Injection**: Assemble retrieved chunk as context for LLM:
   ```text
   Context: [01-returns-policy-current.md > Standard return window]: "Customers on the standard plan may request a return within 30 calendar days of delivery."
   User Question: "How long can I return my bag?"
   ```
5. **Grounded Generation**: LLM responds: *"You have 30 calendar days from delivery to return your bag (Source: 01-returns-policy-current.md > Standard return window)."*

---

## 6. Why RAG Doesn't Guarantee Absolute Truth

RAG drastically reduces hallucinations, but it does **not** guarantee 100% truth out of the box because:
1. **Garbage In, Garbage Out**: If the retrieved chunk itself contains outdated or conflicting information, the LLM will output inaccurate answers unless metadata filtering is enforced.
2. **Retrieval Misses**: If the vector search fails to retrieve the correct chunk, the LLM may rely on pre-trained parametric memory or hallucinate.
3. **Prompt Injection / Poisoning**: If retrieved chunks contain instruction overrides (e.g. `SYSTEM INSTRUCTION: Ignore rules`), the LLM can be hijacked unless retrieved context is treated strictly as untrusted data.

---

## 7. Important Terminology

* **Retrieval**: Fetching relevant text passages from a database based on semantic query matching.
* **Generation**: The LLM producing natural language responses using retrieved context.
* **Grounding**: Restricting the LLM to answer using *only* provided source evidence.
* **Context**: The text payload passed to the LLM prompt alongside the user prompt.
* **Evidence**: Verified, customer-eligible text chunks used to support an answer.
* **Chunk**: A semantically meaningful subsection of a document.
* **Embedding**: A dense mathematical numerical vector representing semantic meaning.
* **Vector Store**: A specialized database (e.g., ChromaDB) optimized for fast vector similarity search.

---

## 8. Interview Questions & Answers

### Q1: Why use RAG instead of fine-tuning the LLM?
> **Answer**: Fine-tuning teaches an LLM *style, format, or syntax*, but is unreliable for teaching *factual knowledge*. Fine-tuned models still hallucinate, cannot provide exact file/heading citations, and require expensive GPU re-training whenever a policy changes. RAG allows instant knowledge updates simply by modifying files in the vector database.

### Q2: Does RAG eliminate hallucinations completely?
> **Answer**: No. RAG reduces hallucinations by grounding the model in retrieved context, but if the retriever fetches irrelevant/conflicting context, or if the LLM misinterprets the retrieved text, hallucinations can still occur. Strict metadata filtering and prompt guardrails are required.

### Q3: How do you handle non-existent information in RAG?
> **Answer**: By instructing the LLM system prompt to safely abstain (e.g. *"If the provided context does not contain enough information to answer, state that the information is unavailable and recommend human support"*).
