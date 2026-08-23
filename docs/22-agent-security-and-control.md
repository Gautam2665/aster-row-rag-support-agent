# 22. Agent Security Boundaries & Threat Mitigation

## 1. Defense-in-Depth Security Matrix

Our agent architecture enforces five distinct security boundaries across the ingestion, retrieval, planning, tool execution, and prompt building layers:

```text
┌─────────────────────────┬──────────────────────────────────┬────────────────────────────────────────┐
│ Security Layer          │ Threat / Vulnerability           │ Defense Mechanism in Code              │
├─────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┤
│ 1. Pre-Retrieval Filter │ Draft Prompt Injection Documents │ Native ChromaDB Pre-filtering          │
│                         │ (e.g. 14-migration-notes.md)     │ where={"status": "active"}             │
├─────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┤
│ 2. Tool Firewall        │ Customer PII & Malicious DB      │ CustomerSafeOrderResult projection     │
│                         │ Warehouse Notes (ORD-1005)       │ Strips email, address, risk_score      │
├─────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┤
│ 3. Action Allowlist     │ Arbitrary Code / Tool Injection  │ ActionValidator & allowed_tools dict   │
│                         │ (e.g. EXECUTE_SQL)               │ Rejects unapproved action types        │
├─────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┤
│ 4. Prompt Delimitation  │ Indirect Prompt Injection via    │ <retrieved_evidence> &                 │
│                         │ Evidence or Chat History         │ <conversation_history> XML Delimiters  │
├─────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┤
│ 5. Loop Bounding        │ Infinite Retry & Resource Denial │ Bounded State Loop                     │
│                         │ of Service (DoS)                 │ max_iterations = 3 limit               │
└─────────────────────────┴──────────────────────────────────┴────────────────────────────────────────┘
```

---

## 2. Mapping Theoretical Agent Concepts to `src/` Codebase

| Theoretical Concept | Codebase File & Class / Function | Purpose |
| :--- | :--- | :--- |
| **Agent State** | [`src/agent.py` $\rightarrow$ `AgentState`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py#L18-L42) | Single source of truth tracking turns, intent, evidence, tools, and observations. |
| **State Machine Loop** | [`src/agent.py` $\rightarrow$ `SupportAgent.process_turn()`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/agent.py#L140-L246) | Orchestrates bounded iteration loop (`max_iterations=3`). |
| **Action Planner** | [`src/planner.py` $\rightarrow$ `LLMPlanner` / `MockPlanner`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py#L88-L222) | Decides structured next action using `BaseLLMProvider`. |
| **Action Validator** | [`src/planner.py` $\rightarrow$ `ActionValidator`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py#L77-L127) | Validates action type allowlist and parameter schemas. |
| **Tool Security Firewall** | [`src/tools/order_lookup.py` $\rightarrow$ `OrderLookupTool`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/tools/order_lookup.py) | Strips customer PII and internal warehouse notes before data enters context. |
| **Observation Storage** | [`src/planner.py` $\rightarrow$ `AgentObservation`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/planner.py#L47-L75) | Records sanitized execution feedback in `state.observations`. |
| **Conversation Memory** | [`src/memory.py` $\rightarrow$ `SessionMemoryStore`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/memory.py) | Bounded FIFO queue per session (`max_turns=5`). |
| **Query Contextualization**| [`src/query_context.py` $\rightarrow$ `QueryContextualizer`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/query_context.py) | Builds separate `retrieval_query` for vector search without modifying raw `user_query`. |
| **Grounded Generation** | [`src/context.py` $\rightarrow$ `ContextBuilder`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/context.py) | Assembles grounded prompt payload with XML untrusted data tags. |
