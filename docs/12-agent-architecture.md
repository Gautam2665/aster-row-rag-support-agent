# 12. Support Agent Architecture & Intent Classification

## 1. High-Level System Architecture

The Support Agent orchestrates user queries across three primary capability paths:
1. **Knowledge-Base Retrieval (RAG)**: Policy, shipping, return window, or product care questions.
2. **Order Lookup Tool Execution**: Order status or delivery tracking queries with an explicit order ID (`ORD-1007`).
3. **Clarification Handling**: Order queries missing an order ID (asks customer for ID without calling tools).

```text
                                USER QUERY
                                    │
                                    ▼
                         extract_order_id()
                         detect_intent()
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
     "clarification"          "order_status"              "policy"
   (Order Query w/o ID)      (Order Query w/ ID)     (KB Policy Query)
           │                        │                        │
           ▼                        ▼                        ▼
  Ask for Order ID         execute_tool_safely()      KBVectorStore.search()
  (Zero Tool Calls)        OrderLookupTool.lookup()   (Native Pre-filtering)
           │                        │                        │
           └───────────────────┬────┴────────────────────────┘
                               ▼
                   build_grounded_prompt()
                   llm_provider.generate()
                               │
                               ▼
                        [FINAL ANSWER]
```

---

## 2. Intent Classification Engine

Intent classification deterministically categorizes user inputs to prevent unnecessary tool execution:

```python
def detect_intent(self, user_query: str, order_id: Optional[str]) -> str:
    query_lower = user_query.lower()

    if order_id:
        return "order_status"

    is_order_tracking_query = any(phrase in query_lower for phrase in self.ORDER_STATUS_PHRASES)
    
    if is_order_tracking_query:
        return "clarification"
    
    return "policy"
```

---

## 3. Explicit Tool Allowlist Security

To prevent arbitrary code execution or unapproved database queries, tool execution is gated behind an explicit registry dictionary:

```python
self.allowed_tools = {
    "order_lookup": self.order_tool
}

def execute_tool_safely(self, tool_name: str, **kwargs) -> Any:
    if tool_name not in self.allowed_tools:
        raise PermissionError(f"Tool '{tool_name}' is not in the explicit tool allowlist.")
    return self.allowed_tools[tool_name].lookup(kwargs.get("order_id", ""))
```
