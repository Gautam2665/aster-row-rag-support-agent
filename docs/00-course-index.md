# Aster & Row RAG Learning & Interview Revision Notes — Master Index

This repository serves as both a production implementation and a self-contained external memory/study curriculum for building reliable Retrieval-Augmented Generation (RAG) AI systems.

---

## Day 1 Study Curriculum

1. **[01-rag-foundations.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/01-rag-foundations.md)**: Core RAG principles, why context windows fail, document vs chunk lifecycle, complete pipeline flow, and fundamental terminology.
2. **[02-chunking.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/02-chunking.md)**: Document segmentation, fixed-size vs semantic section chunking, heading hierarchy retention, chunk subdivision, parent-child context, and code mapping.
3. **[03-embeddings.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/03-embeddings.md)**: Mathematical vector embeddings, cosine similarity formula, query vectorization, vector space consistency, and educational experiment 1.
4. **[04-vector-search.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/04-vector-search.md)**: Vector stores, ChromaDB architecture, HNSW index overview, cosine distance vs similarity, candidate retrieval, and indexing vs querying.
5. **[05-reliable-rag.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/05-reliable-rag.md)**: Why semantic similarity alone fails (30-day current vs 60-day legacy experiment), metadata eligibility rules, prompt injection defense, and evidence selection.
6. **[06-day1-architecture.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/06-day1-architecture.md)**: Full ASCII system architecture diagram, component-by-component breakdown, and codebase implementation map (`src/`).
7. **[07-day1-interview-revision.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/07-day1-interview-revision.md)**: 30-second elevator pitch and rapid-fire interview revision Q&A.
8. **[LEARNING-LOG.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/LEARNING-LOG.md)**: Personal learning journal, initial knowledge, misconceptions corrected, and implementation milestones.

---

## Day 2 Study Curriculum

8. **[08-grounded-generation.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/08-grounded-generation.md)**: Grounded response prompt building, system directives, Data-Instruction separation (`<retrieved_evidence>`), safe abstention, conflict handling, and citation mandates.
9. **[09-llm-provider-abstraction.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/09-llm-provider-abstraction.md)**: Dependency Inversion Principle, `BaseLLMProvider`, `MockLLMProvider` vs `OpenAILLMProvider`, and decoupling application code from specific LLM SDKs.
10. **[10-tool-security-boundary.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/10-tool-security-boundary.md)**: Why the tool boundary is a security firewall, preventing PII leaks (`customer.email`, `customer.address`) and data-driven prompt injections (`ORD-1005` warehouse note injection).
11. **[11-safe-order-tool.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/11-safe-order-tool.md)**: Implementation of `OrderLookupTool` and `CustomerSafeOrderResult`, input normalization, status precedence rules, scrubbing stale carrier ETAs for cancelled orders (`ORD-1004`), null ETA preservation (`ORD-1011`).
12. **[12-agent-architecture.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/12-agent-architecture.md)**: High-level agent orchestration, state machine concept, intent classification (`policy`, `order_status`, `clarification`), and tool allowlist.
13. **[13-bounded-state-machine.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/13-bounded-state-machine.md)**: `AgentState` dataclass, finite state machine transitions, bounded loop iteration limits (`max_iterations=3`), clarification without tool execution, unknown order handoff.
14. **[14-evaluation-and-testing.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/14-evaluation-and-testing.md)**: Design of `EvaluationRunner`, deterministic state assertions vs LLM semantic validation tagging (`UNVERIFIED_REQUIRES_LLM`), machine-readable JSON reports, test suite.
15. **[15-visible-case-analysis.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/15-visible-case-analysis.md)**: Detailed breakdown of all 15 visible evaluation cases: case ID, category, what it tests, architectural mechanism handling it, and what remains live-LLM dependent.
16. **[16-day2-architecture.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/16-day2-architecture.md)**: Full Day 2 system architecture ASCII diagram and complete codebase mapping (`src/prompt_builder.py`, `src/llm.py`, `src/tools/order_lookup.py`, `src/agent.py`, `src/evaluation.py`).
17. **[17-day2-interview-revision.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/17-day2-interview-revision.md)**: 30-second elevator pitch and rapid Q&A cheat sheet for Day 2 concepts.

---

## Day 3 Study Curriculum

18. **[18-agent-foundations.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/18-agent-foundations.md)**: AI Agent definition, LLM vs Workflow vs Autonomous Agent, state tracking in compound AI systems (`AgentState`).
19. **[19-conversation-memory.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/19-conversation-memory.md)**: Bounded short-term conversation memory (`SessionMemoryStore`), FIFO queue eviction (`max_turns=5`), session isolation, why memory $\neq$ RAG.
20. **[20-query-contextualization.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/20-query-contextualization.md)**: Context construction (`ContextBuilder`), query contextualization (`QueryContextualizer`), separating `retrieval_query` from `user_query`, prompt grounding.
21. **[21-planning-action-observation.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/21-planning-action-observation.md)**: ReAct architecture (Thought-Action-Observation loop), structured actions (`AgentAction`), strict schema validation (`ActionValidator`), tool execution authorization, observation feedback (`AgentObservation`), bounded loops (`max_iterations=3`).
22. **[22-agent-security-and-control.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/22-agent-security-and-control.md)**: Defense-in-depth security matrix, mapping theoretical agent concepts to `src/` codebase implementation, Data-Instruction separation, PII sanitization, non-executable reasoning.
23. **[23-day3-architecture.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/23-day3-architecture.md)**: Complete Day 3 ASCII system architecture diagram, Day 1 $\rightarrow$ Day 3 architecture evolution, complete codebase file map (`src/` and `tests/`).
24. **[24-day3-interview-revision.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/24-day3-interview-revision.md)**: 30-second Day 3 elevator pitch, Stanford CME295 (Lecture 7) concept mapping matrix, rapid Q&A cheat sheet.

---

## Architectural Decision Records (ADR)

* **[decisions/001-rag-foundation.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/decisions/001-rag-foundation.md)**: RAG Foundation Architecture — Lightweight Python components + local ChromaDB vs LangChain / LangGraph frameworks.

---

## Experiments Directory

* **[`experiments/embedding_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/embedding_experiment.py)**: Educational experiment demonstrating semantic similarity over exact keyword matching.
* **[`experiments/legacy_policy_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/legacy_policy_experiment.py)**: Educational experiment demonstrating why vector similarity alone retrieves superseded policies.
* **[`experiments/e2e_rag_demo.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/e2e_rag_demo.py)**: End-to-end RAG educational demonstration pipeline.
* **[`experiments/real_llm_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/real_llm_experiment.py)**: Educational experiment for live LLM inference with OpenAI (`gpt-4o-mini`).
