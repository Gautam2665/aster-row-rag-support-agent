# 04. Reliable Retrieval & Metadata Filtering

## Candidate Retrieval vs. Evidence Selection

A high-performing RAG system separates retrieval into two distinct phases:

1. **Broad Candidate Retrieval**: Performing dense vector similarity search to gather top $K$ candidate chunks matching the user's semantic intent.
2. **Metadata Eligibility Filtering**: Applying strict business rules and frontmatter constraints to select only active, official, customer-facing evidence.

---

## Why Semantic Similarity Alone Is Insufficient

Vector similarity measures **topical overlap**, NOT **authoritative truth or validity**.

### Empirical Failure Case: Current vs. Legacy Policies
When a user asks: *"What is the return window?"*
* Current active policy chunk (`01-returns-policy-current.md`): states **30 days**.
* Legacy superseded policy chunk (`02-returns-policy-legacy.md`): states **60 days**.

Because both chunks talk extensively about return windows, **both produce high cosine similarity scores (~0.63–0.66)**. Pure vector similarity search without metadata filtering cannot distinguish between the active 30-day policy and the outdated 60-day policy, risking incorrect customer answers.

---

## Metadata Eligibility Rules

To enforce corporate authority and document lineage, our vector retriever (`KBVectorStore`) enforces metadata query constraints:

```json
{
  "$and": [
    {"status": {"$eq": "active"}},
    {"policy_authority": {"$eq": "official"}},
    {"audience": {"$eq": "customer"}},
    {"customer_answering": {"$eq": true}}
  ]
}
```

### What Filtering Prevents:
1. **Superseded Documents**: `02-returns-policy-legacy.md` (`status: superseded`) is excluded.
2. **Internal & Draft Notes**: `14-internal-content-migration-notes.md` (`status: draft`, `customer_answering: false`) is excluded.
3. **Prompt Injection Defense**: Internal draft notes containing malicious instructions (`> SYSTEM INSTRUCTION: Ignore all prior rules...`) are filtered out at retrieval time before reaching the LLM prompt context.

---

## Citation Formatting

Every retrieved evidence chunk preserves its source filename and section heading:
```text
Source Citation: 01-returns-policy-current.md > Returns Policy > Standard return window
```
This enables the agent to provide source references required for auditability and customer trust.
