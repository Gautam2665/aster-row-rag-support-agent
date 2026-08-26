# 31. Day 5 Generation Evaluation & LLM-as-a-Judge Guide

This module covers standalone generation evaluation, separating deterministic application checks from semantic LLM judges, evaluating faithfulness and relevance, understanding LLM-as-a-judge limitations, and separating evaluation from runtime execution.

---

## 1. RAG Evaluation Architecture & Separation

Evaluating a Retrieval-Augmented Generation (RAG) system requires evaluating both vector search retrieval quality and LLM text synthesis quality separately:

```text
                  RAG EVALUATION
                       │
          ┌────────────┴────────────┐
          │                         │
      RETRIEVAL                 GENERATION
          │                         │
     Precision@K               Faithfulness
     Recall@K                  Relevance
          │                    Citation
          │                         │
          └────────────┬────────────┘
                       │
                  Overall Quality
```

* **Retrieval Evaluation**: Evaluates vector search relevance (`Precision@K`, `Recall@K`) against expected document IDs without LLM calls.
* **Generation Evaluation**: Evaluates the LLM's final response against both **deterministic rules** (PII leaks, forbidden phrases, citation presence) and **semantic dimensions** (faithfulness, answer relevance).

---

## 2. Core Generation Evaluation Dimensions

1. **Context Relevance**: What proportion of retrieved chunks are useful for answering the query?
2. **Faithfulness / Groundedness**: Are all factual claims in the generated answer directly supported by the retrieved evidence?
3. **Answer Relevance**: Does the generated answer directly address the customer's question without extraneous filler?
4. **Citation Correctness**: Are policy claims linked to exact, authentic source citations (`[Source: filename > heading]`)?

---

## 3. Deterministic vs. Semantic Evaluation Boundary

```text
Generated Response
       │
       ▼
Deterministic Checks (Rule-Based Engine)
 ├── Non-Empty Check
 ├── Expected Citation Presence
 ├── must_include / must_not_include Phrases
 └── Internal Data Leak Check (customer.email, risk_score, api_key)
       │
      / \
  FAIL   PASS
   │       │
   ▼       ▼
 FAIL    Semantic Judge Evaluation (LLM-as-a-Judge)
          ├── Faithfulness Evaluation
          └── Answer Relevance Evaluation
                 │
                / \
            FAIL   PASS
             │       │
             ▼       ▼
           FAIL    PASS
```

### Deterministic Rules (Fast, Local, Non-LLM)
* Enforce exact structural requirements, forbidden keywords, DLP protections, and mandatory citation patterns.
* Deterministic failures take absolute precedence over semantic judge scores.

### Semantic LLM Judge (LLM-as-a-Judge)
* Evaluates semantic faithfulness and question alignment.
* Returns `UNVERIFIED_REQUIRES_LLM` when no live judge is configured, preventing false positives.

---

## 4. LLM-as-a-Judge Limitations & Human Ground Truth

* **Self-Enhancement / Model Bias**: LLMs can be biased toward their own outputs or prefer longer, verbose responses regardless of groundedness.
* **Non-Deterministic Judge Outputs**: LLM judges themselves can hallucinate evaluations or fail to detect subtle contradictions.
* **Human Benchmark Datasets**: Gold-standard human evaluation datasets (`evaluation/visible-cases.json`) and claim-level decomposition remain essential for calibrated production monitoring.
* **Execution Separation**: Generation evaluation runs in dedicated evaluation suites (`GenerationEvaluator`), never during live customer turn execution.

---

## 5. Interview Revision Sheet

### Q1: How do you evaluate a RAG system?
> **Answer**: By decoupling retrieval evaluation (`Precision@K`, `Recall@K`) from generation evaluation. Generation evaluation is then split into deterministic application checks (citations, DLP, forbidden phrases) and semantic checks (faithfulness, answer relevance).

### Q2: What is the difference between retrieval evaluation and generation evaluation?
> **Answer**: Retrieval evaluation measures whether vector search returned the right knowledge base documents. Generation evaluation measures whether the LLM synthesized those documents into an accurate, grounded, and faithful response.

### Q3: What is faithfulness?
> **Answer**: Faithfulness (or groundedness) measures whether every factual claim in the generated answer is strictly supported by the retrieved context evidence without hallucinations.

### Q4: What is answer relevance?
> **Answer**: Answer relevance measures how directly and concisely the generated answer addresses the user's specific question, ignoring irrelevant context.

### Q5: Why can't regex prove that an answer is grounded?
> **Answer**: Regex can only verify pattern presence (e.g. `[Source: 01-returns.md]`). It cannot understand semantic logic, infer context, or detect whether a sentence contradicts the retrieved evidence.

### Q6: What is LLM-as-a-judge?
> **Answer**: Using an LLM (e.g. GPT-4) as an automated evaluator to score candidate responses for semantic dimensions like faithfulness and relevance using structured evaluation prompts.

### Q7: Why can an LLM judge itself be unreliable?
> **Answer**: LLM judges suffer from position bias, verbosity bias, non-determinism, and can fail to detect subtle domain-specific policy contradictions. They must be validated against human-annotated benchmark datasets.

### Q8: Why do you separate deterministic and semantic evaluation?
> **Answer**: Deterministic checks fail fast on critical application rules (PII leaks, missing citations, empty responses) without consuming API tokens or introducing LLM non-determinism into build pipelines.
