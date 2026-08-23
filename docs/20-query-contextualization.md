# 20. Query Contextualization & Context Construction

## 1. The Multi-Turn Follow-Up Problem
When a user asks an ambiguous follow-up question like `"What about Canada?"` after previously asking `"Do you ship internationally?"`, performing vector search on raw `"What about Canada?"` yields poor embeddings. The vector database returns generic Canada references rather than international shipping policies.

---

## 2. Separation of Concerns: `retrieval_query` vs. `user_query`

We introduce a dedicated `QueryContextualizer` ([`src/query_context.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/query_context.py)) that separates query representation into two distinct properties:

```text
User Input: "What about Canada?" (Turn 2)
  │
  ├──► state.user_query = "What about Canada?"
  │    (Preserved RAW & UNCHANGED for final LLM prompt payload <user_question>)
  │
  └──► state.retrieval_query = "Do you ship internationally? What about Canada?"
       (Contextualized query string used ONLY for ChromaDB vector search)
```

---

## 3. Why the Original User Query Must Remain Unchanged

1. **Prompt Grounding Integrity**: The customer asked `"What about Canada?"`. Injected user text in the final prompt must reflect their exact words inside `<user_question>`. Rewriting user input inside the final prompt alters the user's voice and risks hallucinated intent.
2. **Auditability & Logging**: Debugging agent trajectories requires knowing the exact verbatim string submitted by the user.
3. **Deterministic Retrieval Query Building**: By generating `retrieval_query` deterministically (combining recent user turn topic with current query), we achieve rich vector search context **without adding an extra expensive LLM API call** for query rewriting.

---

## 4. Context Construction Pipeline ([`src/context.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/context.py))

`ContextBuilder` constructs the final prompt payload by assembling four distinct context blocks:

```xml
=== SYSTEM INSTRUCTIONS ===
[Grounded directives & Data-Instruction Separation rules]

=== USER CONTEXT & QUERY ===
<conversation_history>
[TURN 1]
User: Do you ship internationally?
Assistant: Yes, Aster & Row ships to select international destinations...
</conversation_history>

<order_lookup_data>
[Sanitized CustomerSafeOrderResult XML if applicable]
</order_lookup_data>

<retrieved_evidence>
[Active policy KBChunk items with filename & heading headers]
</retrieved_evidence>

<user_question>
What about Canada?
</user_question>
```
