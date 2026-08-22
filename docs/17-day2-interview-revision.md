# 17. Day 2 Rapid Interview Revision Guide

## 30-Second Day 2 Elevator Pitch
> *"In Day 2, we built the grounded generation and agent orchestration layers. We designed a Grounded Prompt Builder enforcing Data-Instruction Separation (<retrieved_evidence> XML tags), preventing retrieved prompt injection attacks. We decoupled LLM calls behind a BaseLLMProvider abstraction (Dependency Inversion Principle), enabling zero-cost offline unit testing with MockLLMProvider. We implemented a secure OrderLookupTool that acts as a security firewall—scrubbing customer PII (email, address) and malicious warehouse notes before data enters context. Finally, we constructed a bounded SupportAgent state machine with intent classification, tool allowlisting, and iteration limits, evaluated deterministically against 15 visible evaluation cases."*

---

## Rapid-Fire Day 2 Q&A Cheat Sheet

### 1. What is Data-Instruction Separation & Why is it Critical?
Retrieved knowledge chunks are UNTRUSTED DATA. If an internal note contains malicious prompt injection (e.g. `"SYSTEM INSTRUCTION: Ignore prior rules"`), placing evidence inside `<retrieved_evidence>` tags and instructing the model to treat XML blocks purely as passive evidence prevents instruction hijacking.

### 2. Why Abstract the LLM Provider?
Directly importing OpenAI SDK couples business logic to a single vendor. Using `BaseLLMProvider` abstraction (Dependency Inversion Principle) allows switching between OpenAI, Gemini, or local models without modifying application code, while enabling fast, deterministic unit testing via `MockLLMProvider`.

### 3. Why is the Tool Boundary a Security Boundary?
Tools inject data directly into prompt context. If a tool returns raw database records, customer PII (`email`, `address`) or internal risk scores leak to users, and internal notes (`warehouse_note`) can execute prompt injection attacks. The tool must sanitize data into a `CustomerSafeOrderResult` before returning.

### 4. How Does the Agent Handle Stale ETA Data on Cancelled Orders?
Operational databases often retain stale carrier estimated delivery dates on cancelled orders (`ORD-1004`). `OrderLookupTool` detects status `cancelled` and explicitly scrubs `estimated_delivery` to `None` so old ETAs are not presented as active delivery dates.

### 5. Why Ask for Clarification on Missing Order IDs Without Tool Execution?
Querying a database with a missing or blank order ID wastes compute and risks returning invalid errors. The intent classifier detects missing order IDs and returns a clarifying question immediately with zero tool calls.

### 6. Why Distinguish Deterministic Assertions from LLM Semantic Behavior?
15/15 visible evaluation cases pass 100% of deterministic state assertions (tool calls, PII scrubbing, missing ID clarification, metadata filtering, handoff rules). The evaluation runner tags final prose generation `UNVERIFIED_REQUIRES_LLM` under MockLLMProvider rather than fake-passing natural language generation.
