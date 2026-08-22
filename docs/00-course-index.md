# Aster & Row RAG Learning & Interview Revision Notes

This directory contains learning notes, architectural concepts, and design decisions recorded during the development of the Aster & Row RAG Support Agent.

---

## Course Index

* **[01-rag-foundations.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/01-rag-foundations.md)**: Core RAG principles, why full-corpus context fails, and document vs chunk lifecycle.
* **[02-chunking.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/02-chunking.md)**: Markdown section chunking, heading hierarchy preservation, citation tracking, and handling large sections.
* **[03-embeddings.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/03-embeddings.md)**: Vector embeddings, cosine similarity math, query vectorization, and vector space consistency rules.
* **[04-reliable-retrieval.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/04-reliable-retrieval.md)**: Candidate retrieval vs evidence selection, why vector similarity fails on legacy policies, metadata filtering, prompt injection defense, and citation formatting.

---

## Architecture Decision Logs (ADR)

* **[decisions/001-rag-foundation.md](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/docs/decisions/001-rag-foundation.md)**: RAG Foundation Architecture — Lightweight Python components + local ChromaDB vs LangChain / LangGraph frameworks.
