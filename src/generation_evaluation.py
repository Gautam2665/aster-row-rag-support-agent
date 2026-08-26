from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re

from src.models import KBChunk

FORBIDDEN_INTERNAL_FIELDS = [
    "customer.email",
    "customer.address",
    "risk_score",
    "warehouse_note",
    "api_key",
    "secret",
    "credentials",
]


@dataclass
class GenerationEvalCase:
    """
    Representation of an evaluation case for LLM generation quality.
    """
    case_id: str
    user_query: str
    generated_answer: str
    retrieved_evidence: List[KBChunk] = field(default_factory=list)
    expected_source_ids: List[str] = field(default_factory=list)
    must_include: List[str] = field(default_factory=list)
    must_not_include: List[str] = field(default_factory=list)


@dataclass
class GenerationEvalResult:
    """
    Structured result of evaluating a GenerationEvalCase combining
    deterministic application checks with optional semantic LLM judge results.
    """
    case_id: str
    deterministic_status: str  # "PASS" or "FAIL"
    semantic_status: str  # "PASS", "FAIL", "UNAVAILABLE", or "UNVERIFIED_REQUIRES_LLM"
    overall_status: str  # "PASS", "FAIL", or "UNVERIFIED_REQUIRES_LLM"
    deterministic_failures: List[str] = field(default_factory=list)
    semantic_failures: List[str] = field(default_factory=list)
    citations_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "deterministic_status": self.deterministic_status,
            "semantic_status": self.semantic_status,
            "overall_status": self.overall_status,
            "deterministic_failures": self.deterministic_failures,
            "semantic_failures": self.semantic_failures,
            "citations_found": self.citations_found,
        }


class BaseGenerationJudge(ABC):
    """
    Abstract interface for semantic LLM generation judges.
    """
    @abstractmethod
    def evaluate(
        self,
        user_query: str,
        retrieved_evidence: List[KBChunk],
        generated_answer: str,
    ) -> Dict[str, Any]:
        pass


class MockGenerationJudge(BaseGenerationJudge):
    """
    Mock semantic judge for testing generation evaluation pipelines without API calls.
    """
    def __init__(
        self,
        default_faithfulness: bool = True,
        default_relevance: bool = True,
        reason: str = "Mock evaluation result",
    ):
        self.default_faithfulness = default_faithfulness
        self.default_relevance = default_relevance
        self.reason = reason
        self.last_evaluated_query: Optional[str] = None
        self.last_evaluated_evidence: Optional[List[KBChunk]] = None

    def evaluate(
        self,
        user_query: str,
        retrieved_evidence: List[KBChunk],
        generated_answer: str,
    ) -> Dict[str, Any]:
        self.last_evaluated_query = user_query
        self.last_evaluated_evidence = retrieved_evidence
        return {
            "faithfulness": self.default_faithfulness,
            "answer_relevance": self.default_relevance,
            "reason": self.reason,
        }


class GenerationEvaluator:
    """
    Evaluator performing deterministic application checks and optional
    semantic LLM evaluation on generated answers.
    """
    def __init__(self, judge: Optional[BaseGenerationJudge] = None):
        self.judge = judge

    def evaluate_case(self, eval_case: GenerationEvalCase) -> GenerationEvalResult:
        deterministic_failures: List[str] = []
        citations_found: List[str] = []

        answer = eval_case.generated_answer or ""
        answer_lower = answer.lower()

        # Check A: Non-empty answer
        if not answer.strip():
            deterministic_failures.append("Generated answer is empty.")

        # Check B: Citation / source presence
        for expected_src in eval_case.expected_source_ids:
            if expected_src in answer or expected_src.lower() in answer_lower:
                citations_found.append(expected_src)
            else:
                deterministic_failures.append(f"Missing expected source citation: '{expected_src}'")

        # Extract citations present in text [Source: filename > heading]
        found_matches = re.findall(r"\[Source:\s*([^\]]+)\]", answer)
        for match in found_matches:
            if match not in citations_found:
                citations_found.append(match)

        # Check C: must_include
        for phrase in eval_case.must_include:
            if phrase.lower() not in answer_lower:
                deterministic_failures.append(f"Answer missing required phrase: '{phrase}'")

        # Check D: must_not_include
        for phrase in eval_case.must_not_include:
            if phrase.lower() in answer_lower:
                deterministic_failures.append(f"Answer contains forbidden phrase: '{phrase}'")

        # Check E: Internal-data protection
        for forbidden in FORBIDDEN_INTERNAL_FIELDS:
            if forbidden.lower() in answer_lower:
                deterministic_failures.append(f"Answer exposes forbidden internal data field: '{forbidden}'")

        deterministic_status = "FAIL" if deterministic_failures else "PASS"

        semantic_failures: List[str] = []
        semantic_status = "UNAVAILABLE"

        # Execute semantic judge if configured
        if self.judge is not None:
            judge_res = self.judge.evaluate(
                user_query=eval_case.user_query,
                retrieved_evidence=eval_case.retrieved_evidence,
                generated_answer=eval_case.generated_answer,
            )
            faithfulness = judge_res.get("faithfulness", False)
            relevance = judge_res.get("answer_relevance", False)

            if faithfulness and relevance:
                semantic_status = "PASS"
            else:
                semantic_status = "FAIL"
                if not faithfulness:
                    semantic_failures.append("Semantic failure: Generated answer is unfaithful to evidence.")
                if not relevance:
                    semantic_failures.append("Semantic failure: Generated answer is irrelevant to user query.")

        # Determine overall status
        if deterministic_status == "FAIL":
            overall_status = "FAIL"
        elif semantic_status == "PASS":
            overall_status = "PASS"
        elif semantic_status == "FAIL":
            overall_status = "FAIL"
        else:
            overall_status = "UNVERIFIED_REQUIRES_LLM"

        return GenerationEvalResult(
            case_id=eval_case.case_id,
            deterministic_status=deterministic_status,
            semantic_status=semantic_status,
            overall_status=overall_status,
            deterministic_failures=deterministic_failures,
            semantic_failures=semantic_failures,
            citations_found=citations_found,
        )
