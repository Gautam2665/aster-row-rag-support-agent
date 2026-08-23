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
6. **Misconception**: Post-filtering small Top-K results in Python (Naive Post-filtering Anti-Pattern).
   * **Correction**: Filtering *after* retrieving a small Top-3 causes a severe failure bug (if top 3 nearest vectors are superseded/draft items, post-filtering drops all 3, leaving zero context). Native Pre-filtering inside ChromaDB (`where` clause) ensures vector search operates *only* over active documents, guaranteeing all returned items are valid evidence.

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

## 7. Day 2 Personal Learning Journal & Technical Evolution

### Key Day 2 Concepts Learned
1. **Data-Instruction Separation**: Encapsulating retrieved knowledge chunks in `<retrieved_evidence>` XML tags and instructing the model to treat XML text purely as passive data neutralizes prompt injection attacks.
2. **Dependency Inversion Principle**: Coupling application logic to `BaseLLMProvider` interface rather than direct `OpenAI()` calls enables zero-cost offline unit testing via `MockLLMProvider`.
3. **Tool Boundary Security Firewall**: Tool outputs are injected into LLM context. `OrderLookupTool` must sanitize data into `CustomerSafeOrderResult`, purging PII (`email`, `address`) and malicious warehouse notes (`ORD-1005`) before data reaches the model.
4. **Stale Field Scrubbing**: Cancelled orders (`ORD-1004`) in raw databases often retain stale carrier ETAs. The tool firewall scrubs `estimated_delivery` to `None`.
5. **Bounded Agent State Machine**: Bounding agent loop iterations (`max_iterations=3`) prevents infinite execution loops. Intent classification bypasses tool execution when order IDs are missing.
6. **Honest Evaluation Tagging**: 15/15 visible evaluation cases pass 100% of deterministic state assertions (tools, privacy, lineage, handoff). Semantic text generation assertions are tagged `UNVERIFIED_REQUIRES_LLM` under MockLLMProvider rather than fake-passing.

---

## 8. What I Implemented in Day 2
* [`src/prompt_builder.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/prompt_builder.py): Grounded response prompt builder with XML delimitation.
* [`src/llm.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/llm.py): `BaseLLMProvider`, `MockLLMProvider`, `OpenAILLMProvider`, and `GroundedResponse` pipeline.
* [`src/tools/order_lookup.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/tools/order_lookup.py): `OrderLookupTool` security firewall & `CustomerSafeOrderResult`.
* [`src/agent.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py): `SupportAgent` bounded state machine & intent classifier.
* [`src/evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/evaluation.py): `EvaluationRunner` & visible evaluation cases analyzer.
* **43 Unit Tests** across 8 test suites passing cleanly in `pytest`.

---

## 9. Day 3 Personal Learning Journal & Technical Evolution

### Key Day 3 Concepts Learned
1. **Bounded Conversation Memory**: Memory stores short-term dialogue in a FIFO queue (`max_turns=5`), scoped strictly per session (`SessionMemoryStore`). Memory is UNTRUSTED DATA and NEVER overrides authoritative KB evidence.
2. **Query Contextualization**: Separating `retrieval_query` (used for ChromaDB search) from `user_query` (raw verbatim input preserved in `<user_question>`) ensures high-precision vector search for ambiguous follow-ups without altering the user's voice.
3. **ReAct Planner-Action-Observation Loop**: Decoupling action planning (`LLMPlanner`), action validation (`ActionValidator`), tool authorization (`SupportAgent.execute_tool_safely`), and observation feedback (`AgentObservation`) prevents arbitrary code execution and eliminates duplicate action loops.
4. **Stanford CME295 Alignment**: Mapped codebase components to Stanford CME295 Lecture 7 (Transformers & Large Language Models), mastering Compound AI Systems, Contextual Retrieval, Function Calling, ReAct Agent Loops, Safety, and Evaluation.

---

## 10. What I Implemented in Day 3
* [`src/memory.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/memory.py): `ConversationTurn`, `ConversationMemory`, and `SessionMemoryStore`.
* [`src/context.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/context.py): `ContextBuilder` assembling grounded prompts with `<conversation_history>` XML tags.
* [`src/query_context.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/query_context.py): `QueryContextualizer` constructing search queries without modifying raw `user_query`.
* [`src/planner.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py): ReAct architecture (`ActionType`, `AgentAction`, `AgentObservation`, `ActionValidator`, `FailureCategory`, `LLMPlanner`).
* [`src/trace.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/trace.py): `AgentTrace` & `TraceEvent` with automatic PII sanitization.
* [`src/agent.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py): `SupportAgent` state machine orchestrating the bounded ReAct planning loop (`max_iterations=3`).
* [`src/evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/evaluation.py): `EvaluationRunner` with offline state assertion verification and optional live LLM semantic validation.
* [`src/cli.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/cli.py): Minimal CLI application runtime supporting offline, `--live`, and `--debug` modes.
* **154 Unit Tests** across 21 test files passing 100% cleanly in `pytest`.
