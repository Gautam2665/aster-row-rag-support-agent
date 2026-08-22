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

## Architectural Decision Records (ADR)

* **[decisions/001-rag-foundation.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/decisions/001-rag-foundation.md)**: RAG Foundation Architecture — Lightweight Python components + local ChromaDB vs LangChain / LangGraph frameworks.

---

## Experiments Directory

* **[`experiments/embedding_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/embedding_experiment.py)**: Educational experiment demonstrating semantic similarity over exact keyword matching.
* **[`experiments/legacy_policy_experiment.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/experiments/legacy_policy_experiment.py)**: Educational experiment demonstrating why vector similarity alone retrieves superseded policies.
