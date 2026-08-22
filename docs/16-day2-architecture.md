# 16. Day 2 System Architecture & Codebase Mapping

## 1. Day 2 End-to-End System Architecture

```text
                                  USER QUERY
                                      │
                                      ▼
                           SupportAgent.process_turn()
                           [src/agent.py]
                                      │
                         extract_order_id()
                         detect_intent()
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
      "clarification"           "order_status"              "policy"
    (Order Query w/o ID)       (Order Query w/ ID)     (KB Policy Query)
            │                         │                         │
            ▼                         ▼                         ▼
   Ask for Order ID          OrderLookupTool.lookup()   KBVectorStore.search()
   (Zero Tool Calls)         [src/tools/order_lookup]  [src/retrieval.py]
            │                         │                         │
            │            [found=False]├──► Handoff              │
            │            [found=True] └──► Safe Context        │
            │                         │                         │
            └────────────────────┬────┴─────────────────────────┘
                                 ▼
                    build_grounded_prompt()
                    [src/prompt_builder.py]
                                 │
                                 ▼
                     BaseLLMProvider.generate()
                     [src/llm.py]
                                 │
                                 ▼
                      GroundedResponse Output
```

---

## 2. Codebase Directory Mapping

```text
ai-agent-intern-test/
├── src/
│   ├── models.py             # DocumentMetadata and KBChunk domain models
│   ├── ingestion.py          # Markdown YAML parser & heading-aware section chunker
│   ├── embeddings.py         # EmbeddingProvider (Vector Space Consistency)
│   ├── retrieval.py          # KBVectorStore (ChromaDB + Native Pre-filtering)
│   ├── prompt_builder.py     # Grounded Prompt Builder & Data-Instruction Separation
│   ├── llm.py                # LLM Provider Abstraction (Mock & OpenAI)
│   ├── agent.py              # SupportAgent Bounded State Machine & Intent Classifier
│   ├── evaluation.py         # EvaluationRunner & Visible Case Evaluator
│   └── tools/
│       ├── __init__.py
│       └── order_lookup.py   # OrderLookupTool Security Firewall & Data Projection
├── tests/
│   ├── test_models.py        # 4 unit tests
│   ├── test_ingestion.py     # 5 unit tests
│   ├── test_retrieval.py     # 5 unit tests
│   ├── test_prompt_builder.py# 5 unit tests
│   ├── test_llm.py           # 5 unit tests
│   ├── test_order_lookup.py  # 7 unit tests
│   ├── test_agent.py         # 8 unit tests
│   └── test_evaluation.py    # 4 unit tests (Total: 43 unit tests)
├── experiments/
│   ├── embedding_experiment.py     # Semantic similarity experiment
│   ├── legacy_policy_experiment.py # Superseded legacy policy experiment
│   ├── e2e_rag_demo.py             # End-to-end RAG demonstration
│   └── real_llm_experiment.py     # Live OpenAI inference experiment
├── evaluation/
│   ├── visible-cases.json     # 15 assignment visible evaluation cases
│   └── evaluation_results.json# Generated machine-readable evaluation report
└── docs/                      # 18 Study & Architecture Revision Modules
```
