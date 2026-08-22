# 14. Evaluation Runner & Automated Testing Architecture

## 1. Evaluation Architecture Overview ([`src/evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/evaluation.py))

To validate the support agent against the assignment's 15 visible evaluation cases (`evaluation/visible-cases.json`), we built a custom evaluation engine.

```text
evaluation/visible-cases.json ──► [ EvaluationRunner ] ──► SupportAgent.process_turn()
                                           │
                                           ├──► Deterministic Assertions (PASS / FAIL)
                                           └──► Semantic Generation Tagging (UNVERIFIED_REQUIRES_LLM)
                                           │
                                           ▼
                             evaluation/evaluation_results.json + Terminal Summary
```

---

## 2. Deterministic vs. Semantic Evaluation Strategy

A crucial engineering design choice in our harness is **refusing to fake-pass LLM semantic generation behavior**.

### A. Deterministic State Assertions (Evaluated Instantly)
Evaluated directly from structured `AgentState` outputs:
* **Tool Invocation**: Asserts tool called (`order_lookup`) or bypassed (`not_called`).
* **Source Lineage**: Asserts required documents (e.g. `01-returns-policy-current.md`) are present and forbidden documents (`02-returns-policy-legacy.md`, `14-internal-content-migration-notes.md`) are excluded.
* **Privacy Sanitization**: Asserts `CustomerSafeOrderResult` excludes PII keys (`email`, `address`, `risk_score`).
* **Handoff Expectations**: Asserts `handoff_recommended` matches expected boolean state.
* **Clarification Intent**: Asserts missing order IDs trigger clarification intent with zero tool calls.

### B. Honest LLM Status Tagging (`UNVERIFIED_REQUIRES_LLM`)
When running under `MockLLMProvider` (offline CI mode), assertions testing final natural language output concepts (`must_include`, `must_not_invent`) are tagged **`UNVERIFIED_REQUIRES_LLM`** rather than fake-passing.

---

## 3. Reporting Outputs
* **Machine-Readable JSON**: Saved to [`evaluation/evaluation_results.json`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/evaluation/evaluation_results.json).
* **Terminal Summary Table**:
  ```text
  Summary: 0 PASSED | 15 UNVERIFIED (Requires Live LLM) | 0 FAILED
  ```
