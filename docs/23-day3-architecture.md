# 23. Complete Architecture Evolution: Day 1 $\rightarrow$ Day 3

## 1. Complete Day 3 System Architecture

```text
                                  USER QUERY
                                      │
                                      ▼
                         SupportAgent.process_turn()
                         [src/agent.py]
                                      │
                         SessionMemoryStore.get_turns()
                         [src/memory.py]
                                      │
                        QueryContextualizer.build()
                        [src/query_context.py]
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
     state.user_query                                state.retrieval_query
   (Original Verbatim Text)                        (Contextualized Vector Search)
             │                                                 │
             └────────────────────────┬────────────────────────┘
                                      ▼
                           [ Bounded Planning Loop ]
                            (max_iterations = 3)
                                      │
                           LLMPlanner.plan_next_action()
                           ActionValidator.validate()
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
     RETRIEVE_KB                LOOKUP_ORDER                  CLARIFY /
   KBVectorStore.search()      OrderLookupTool               RESPOND /
   (ChromaDB + Pre-filter)     (PII & Note Scrubbing)        HANDOFF
           │                          │                          │
           └─────────────────────┬────┴──────────────────────────┘
                                 ▼
                     Record AgentObservation
                     (state.observations.append)
                                 │
                                 ▼
                    ContextBuilder.build_prompt()
                    [src/context.py]
                    (XML Data-Instruction Tags)
                                 │
                                 ▼
                     BaseLLMProvider.generate()
                     [src/llm.py]
                                 │
                                 ▼
                      GroundedResponse Output
```

---

## 2. Day 1 $\rightarrow$ Day 3 Architecture Evolution

| Phase | Core Capabilities Implemented | Key Architecture & Security Guardrails |
| :--- | :--- | :--- |
| **DAY 1** | Markdown Ingestion, Heading Chunking, SentenceTransformers, ChromaDB HNSW Vector Store | Cosine similarity, pre-retrieval metadata filtering (`status=="active"`), vector space consistency. |
| **DAY 2** | Grounded Prompt Builder, LLM Abstraction (`BaseLLMProvider`), Safe Order Lookup Tool, Support Agent | XML evidence delimitation, PII scrubbing, stale ETA removal (`ORD-1004`), intent detection, tool allowlist. |
| **DAY 3** | Bounded Conversation Memory (`SessionMemoryStore`), Query Contextualization (`QueryContextualizer`), Planner-Action-Observation Loop (`LLMPlanner`, `AgentObservation`, `ActionValidator`) | Short-term memory FIFO queue (`max_turns=5`), raw query preservation (`user_query` vs `retrieval_query`), ReAct loop (`max_iterations=3`), observation feedback, strict action parameter contracts. |

---

## 3. Complete Codebase Map

```text
ai-agent-intern-test/
├── src/
│   ├── models.py             # KBChunk & DocumentMetadata domain models
│   ├── ingestion.py          # Heading-aware markdown chunking & directory ingester
│   ├── embeddings.py         # EmbeddingProvider enforcing single embedding model
│   ├── retrieval.py          # KBVectorStore (ChromaDB + Native Pre-filtering)
│   ├── prompt_builder.py     # Grounded Prompt Builder & System Directives
│   ├── llm.py                # LLM Provider Abstraction (BaseLLMProvider, Mock, OpenAI)
│   ├── memory.py             # Bounded Conversation Memory (ConversationTurn, SessionMemoryStore)
│   ├── context.py            # Structured ContextBuilder (XML conversation_history tags)
│   ├── query_context.py      # QueryContextualizer (separate retrieval_query building)
│   ├── planner.py            # ReAct Planner (AgentAction, AgentObservation, ActionValidator, LLMPlanner)
│   ├── agent.py              # SupportAgent Bounded State Machine Orchestrator
│   ├── evaluation.py         # EvaluationRunner & Visible Case Evaluator
│   └── tools/
│       ├── __init__.py
│       └── order_lookup.py   # OrderLookupTool Security Firewall & CustomerSafeOrderResult
├── tests/                    # 100 Unit Tests Across 15 Suites (100% Passing)
│   ├── test_models.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_prompt_builder.py
│   ├── test_llm.py
│   ├── test_order_lookup.py
│   ├── test_agent.py
│   ├── test_evaluation.py
│   ├── test_memory.py
│   ├── test_memory_security.py
│   ├── test_query_context.py
│   ├── test_planner.py
│   ├── test_planner_contract.py
│   ├── test_agent_planner.py
│   └── test_observations.py
└── docs/                      # 25 Comprehensive Study & Revision Modules
```
