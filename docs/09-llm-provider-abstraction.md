# 09. LLM Provider Abstraction & Dependency Inversion

## 1. The Engineering Problem: SDK Lock-in
In many naive AI applications, code search reveals `import openai` or `client = OpenAI()` calls sprinkled across retrieval logic, agent state handlers, and API routes.

### Why Direct Coupling Fails:
* **Vendor Lock-in**: Switching from OpenAI (`gpt-4o-mini`) to Google (`Gemini`) or Anthropic requires touching dozens of files.
* **Testing Friction**: Unit tests end up making live network calls or requiring complex `unittest.mock` patching of external SDK clients.
* **Cost & Flakiness**: Running CI test suites incurs API costs and risks failure due to third-party rate limits.

---

## 2. Dependency Inversion Principle (DIP)

We apply SOLID's **Dependency Inversion Principle**:
> *High-level application modules should not depend on low-level SDK implementations; both should depend on abstractions.*

```text
Application Orchestrator (generate_grounded_response)
                         │
                         ▼
             BaseLLMProvider (Abstract Class)
            ┌────────────┴────────────┐
            ▼                         ▼
    MockLLMProvider          OpenAILLMProvider
  (Deterministic Offline)     (Live API Inference)
```

---

## 3. Code Implementation Breakdown ([`src/llm.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/llm.py))

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Abstract method implemented by all LLM providers."""
        pass

class MockLLMProvider(BaseLLMProvider):
    """Offline provider used for zero-cost, deterministic unit testing."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "NO ELIGIBLE EVIDENCE RETRIEVED" in user_prompt:
            return "I am sorry, but the supplied information is insufficient..."
        return "Based on Aster & Row policy, customers have 30 calendar days..."

class OpenAILLMProvider(BaseLLMProvider):
    """Production provider encapsulating OpenAI ChatCompletions API."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        return response.choices[0].message.content or ""
```

---

## 4. Structured Response Output ([`GroundedResponse`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/llm.py#L10-L17))

The generation function `generate_grounded_response()` returns a clean structured object rather than a raw string:

```python
@dataclass
class GroundedResponse:
    answer: str                     # Final generated answer text
    source_citations: List[str]      # Inline citations list
    used_evidence_count: int         # Count of chunks passed into context
    prompt_payload: Dict[str, str]   # Formatted system and user prompts
    provider_name: str               # Provider name ("MockLLMProvider", "OpenAILLMProvider")
```
