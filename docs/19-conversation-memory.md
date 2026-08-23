# 19. Bounded Conversation Memory & Multi-Turn Context

## 1. Short-Term / Bounded Conversation Memory
In enterprise customer support, conversations span multiple turns (e.g., Turn 1: *"Do you ship internationally?"* $\rightarrow$ Turn 2: *"What about Canada?"*). 

To support multi-turn reasoning without context window degradation or unbounded token costs, we implement a **Bounded Conversation Memory** using a First-In-First-Out (FIFO) queue ([`src/memory.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/memory.py)).

```text
Turn 1 ──► [Turn 1]
Turn 2 ──► [Turn 1, Turn 2]
Turn 3 ──► [Turn 1, Turn 2, Turn 3]
...
Turn 6 ──► [Turn 2, Turn 3, Turn 4, Turn 5, Turn 6]  <-- (Turn 1 evicted FIFO)
```

---

## 2. Why Conversation Memory $\neq$ RAG

A common misconception in junior AI engineering is treating conversation memory and RAG as interchangeable. They solve entirely different system challenges:

| Dimension | Conversation Memory ([`src/memory.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/memory.py)) | Knowledge-Base RAG ([`src/retrieval.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/retrieval.py)) |
| :--- | :--- | :--- |
| **Data Source** | Recent user & assistant chat messages | Authoritative corporate policy docs & product cards |
| **Scope** | Session-specific, short-term transient dialogue | Global, persistent organizational knowledge base |
| **Trust Level** | **UNTRUSTED DATA** (User/assistant dialog context) | **AUTHORITATIVE EVIDENCE** (Vetted active policies) |
| **Retrieval Mechanism** | Bounded FIFO memory queue (`max_turns=5`) | Vector semantic search + ChromaDB pre-filtering |
| **Conflict Rule** | **Never** overrides RAG evidence | Strictly overrides chat memory claims |

---

## 3. Session Isolation Architecture ([`SessionMemoryStore`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/memory.py#L42-L72))

To prevent cross-customer data leaks, memory is strictly scoped to a unique `session_id`:

```python
class SessionMemoryStore:
    def __init__(self, max_turns_per_session: int = 5):
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(max_turns=self.max_turns_per_session)
        return self._sessions[session_id]
```

`Session A` history is physically isolated in memory from `Session B`, guaranteeing privacy boundary enforcement across concurrent users.
