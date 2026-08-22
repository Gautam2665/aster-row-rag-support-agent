from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import List, Dict, Any, Optional

from src.models import KBChunk
from src.prompt_builder import build_grounded_prompt


@dataclass
class GroundedResponse:
    """Structured response object containing LLM answer and evidence provenance."""
    answer: str
    source_citations: List[str]
    used_evidence_count: int
    prompt_payload: Dict[str, str]
    provider_name: str


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers (Dependency Inversion Principle)."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text given system instructions and user prompt."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Fake LLM provider for deterministic unit testing without API calls."""

    def __init__(self, fixed_response: Optional[str] = None):
        self.fixed_response = fixed_response
        self.last_system_prompt: Optional[str] = None
        self.last_user_prompt: Optional[str] = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt

        if self.fixed_response:
            return self.fixed_response

        if "NO ELIGIBLE EVIDENCE RETRIEVED" in user_prompt:
            return "I am sorry, but the supplied information is insufficient to answer your request. Please contact Aster & Row customer support for assistance."

        return (
            "Based on Aster & Row policy, customers on the standard plan may request a return "
            "within 30 calendar days of delivery. "
            "[Source: 01-returns-policy-current.md > Returns Policy > Standard return window]"
        )


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI implementation of LLM provider abstraction."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY environment variable is missing or invalid.")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


def get_default_provider() -> BaseLLMProvider:
    """Factory function to return configured LLM provider or MockLLMProvider if API key is absent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        try:
            return OpenAILLMProvider()
        except Exception:
            pass
    return MockLLMProvider()


def generate_grounded_response(
    user_question: str,
    evidence_chunks: List[KBChunk],
    provider: Optional[BaseLLMProvider] = None,
) -> GroundedResponse:
    """
    Main grounded response generation pipeline.
    1. Uses build_grounded_prompt() to construct system and user prompts.
    2. Calls the provided (or default) LLMProvider.
    3. Returns a structured GroundedResponse.
    """
    active_provider = provider or get_default_provider()
    prompt_payload = build_grounded_prompt(user_question, evidence_chunks)

    answer_text = active_provider.generate(
        system_prompt=prompt_payload["system_prompt"],
        user_prompt=prompt_payload["user_prompt"],
    )

    citations = [c.source_citation for c in evidence_chunks if c.source_citation]

    provider_class_name = active_provider.__class__.__name__

    return GroundedResponse(
        answer=answer_text.strip(),
        source_citations=citations,
        used_evidence_count=len(evidence_chunks),
        prompt_payload=prompt_payload,
        provider_name=provider_class_name,
    )
