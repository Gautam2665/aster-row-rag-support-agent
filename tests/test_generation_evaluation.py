import pytest

from src.models import KBChunk, DocumentMetadata
from src.generation_evaluation import (
    GenerationEvalCase,
    GenerationEvalResult,
    GenerationEvaluator,
    MockGenerationJudge,
)


def sample_chunk() -> KBChunk:
    return KBChunk(
        chunk_id="01-returns-policy-current.md#0",
        filename="01-returns-policy-current.md",
        heading="Returns",
        text="Standard return window is 30 days.",
        metadata=DocumentMetadata(
            document_id="01-returns-policy-current.md",
            title="Returns",
            status="active",
            audience="customer",
            policy_authority="official",
            customer_answering=True,
        ),
    )


def test_empty_answer_fails_deterministic_eval():
    """1. Verify empty generated answer fails deterministic checks."""
    case = GenerationEvalCase(
        case_id="empty-ans",
        user_query="What is the policy?",
        generated_answer="   ",
    )
    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "FAIL"
    assert res.overall_status == "FAIL"
    assert any("empty" in f.lower() for f in res.deterministic_failures)


def test_correct_citation_passes():
    """2. Verify presence of expected citation passes deterministic checks."""
    case = GenerationEvalCase(
        case_id="cite-pass",
        user_query="What is the return window?",
        generated_answer="The return window is 30 days. [Source: 01-returns-policy-current.md > Returns]",
        expected_source_ids=["01-returns-policy-current.md"],
    )
    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "PASS"
    assert "01-returns-policy-current.md" in res.citations_found


def test_missing_expected_citation_fails():
    """3. Verify missing expected citation fails deterministic checks."""
    case = GenerationEvalCase(
        case_id="cite-fail",
        user_query="What is the return window?",
        generated_answer="The return window is 30 days.",
        expected_source_ids=["01-returns-policy-current.md"],
    )
    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "FAIL"
    assert any("Missing expected source citation" in f for f in res.deterministic_failures)


def test_must_include_passes_and_fails():
    """4 & 5. Verify must_include phrase check."""
    case_pass = GenerationEvalCase(
        case_id="inc-pass",
        user_query="What is the window?",
        generated_answer="Items must be returned within 30 days.",
        must_include=["30 days"],
    )
    case_fail = GenerationEvalCase(
        case_id="inc-fail",
        user_query="What is the window?",
        generated_answer="Items can be returned anytime.",
        must_include=["30 days"],
    )
    evaluator = GenerationEvaluator()

    res_pass = evaluator.evaluate_case(case_pass)
    res_fail = evaluator.evaluate_case(case_fail)

    assert res_pass.deterministic_status == "PASS"
    assert res_fail.deterministic_status == "FAIL"
    assert any("missing required phrase" in f.lower() for f in res_fail.deterministic_failures)


def test_must_not_include_failure():
    """6. Verify must_not_include detects forbidden phrases."""
    case = GenerationEvalCase(
        case_id="not-inc-fail",
        user_query="Is shipping free?",
        generated_answer="We offer free lifetime returns.",
        must_not_include=["lifetime returns"],
    )
    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "FAIL"
    assert any("forbidden phrase" in f.lower() for f in res.deterministic_failures)


def test_internal_pii_secret_content_detected():
    """7. Verify internal PII and secret field leaks are detected."""
    case = GenerationEvalCase(
        case_id="pii-leak",
        user_query="Order info",
        generated_answer="Your order details: customer.email = jane@example.com, warehouse_note = fragile.",
    )
    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "FAIL"
    assert any("forbidden internal data field" in f.lower() for f in res.deterministic_failures)


def test_deterministic_pass_without_judge_equals_unverified():
    """8. Verify deterministic PASS with no judge yields UNVERIFIED_REQUIRES_LLM."""
    case = GenerationEvalCase(
        case_id="unverified-case",
        user_query="Return window?",
        generated_answer="Return window is 30 days.",
    )
    evaluator = GenerationEvaluator(judge=None)
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "PASS"
    assert res.semantic_status == "UNAVAILABLE"
    assert res.overall_status == "UNVERIFIED_REQUIRES_LLM"


def test_mock_judge_pass_and_fail():
    """9 & 10. Verify mock judge PASS and FAIL integration."""
    case = GenerationEvalCase(
        case_id="judge-case",
        user_query="Return policy?",
        generated_answer="30 days policy.",
    )
    
    judge_pass = MockGenerationJudge(default_faithfulness=True, default_relevance=True)
    judge_fail = MockGenerationJudge(default_faithfulness=False, default_relevance=True)

    eval_pass = GenerationEvaluator(judge=judge_pass)
    eval_fail = GenerationEvaluator(judge=judge_fail)

    res_pass = eval_pass.evaluate_case(case)
    res_fail = eval_fail.evaluate_case(case)

    assert res_pass.overall_status == "PASS"
    assert res_fail.overall_status == "FAIL"
    assert res_fail.semantic_status == "FAIL"


def test_deterministic_failure_precedes_semantic_pass():
    """11. Verify deterministic failure takes precedence over a passing semantic judge."""
    case = GenerationEvalCase(
        case_id="override-case",
        user_query="Return policy?",
        generated_answer="   ",  # Fails empty check
    )
    judge_pass = MockGenerationJudge(default_faithfulness=True, default_relevance=True)
    evaluator = GenerationEvaluator(judge=judge_pass)
    res = evaluator.evaluate_case(case)

    assert res.deterministic_status == "FAIL"
    assert res.overall_status == "FAIL"


def test_no_api_calls_required():
    """12. Verify default evaluator runs completely offline."""
    case = GenerationEvalCase(
        case_id="offline-case",
        user_query="Check return time",
        generated_answer="30 days.",
    )
    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_case(case)
    assert res is not None


def test_evidence_and_query_supplied_to_judge():
    """13 & 14. Verify retrieved evidence and user_query are passed unchanged to judge."""
    chunks = [sample_chunk()]
    query = "Original user query string"
    case = GenerationEvalCase(
        case_id="input-pass-case",
        user_query=query,
        generated_answer="30 days.",
        retrieved_evidence=chunks,
    )
    judge = MockGenerationJudge()
    evaluator = GenerationEvaluator(judge=judge)
    evaluator.evaluate_case(case)

    assert judge.last_evaluated_query == query
    assert judge.last_evaluated_evidence == chunks
