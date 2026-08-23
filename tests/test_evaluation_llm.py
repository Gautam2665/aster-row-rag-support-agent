import os
import json
import pytest
from pathlib import Path
from src.evaluation import EvaluationRunner, CaseAssertionResult, CaseEvalReport
from src.llm import BaseLLMProvider, MockLLMProvider
from src.agent import SupportAgent


class CustomAnswerMockLLMProvider(BaseLLMProvider):
    """Mock LLM provider returning custom text for testing evaluation semantic checks."""
    def __init__(self, custom_answer: str):
        self.custom_answer = custom_answer

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.custom_answer


def test_semantic_must_include_passes(monkeypatch):
    """Test that live semantic check PASSES when must_include phrases are present."""
    custom_llm = CustomAnswerMockLLMProvider("You have 30 calendar days from delivery to return unused items.")
    runner = EvaluationRunner()
    runner.agent.llm_provider = custom_llm

    case_dict = {
        "id": "test-semantic-pass",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "How long do I have to return an item?"}],
        "expect": {
            "must_include": ["30 calendar days", "delivery"],
            "required_sources": ["01-returns-policy-current.md"]
        }
    }
    report = runner.evaluate_case(case_dict)
    assert report.deterministic_status == "PASS"
    assert report.live_llm_status == "PASS"
    assert report.overall_status == "PASS"


def test_semantic_must_include_fails():
    """Test that live semantic check FAILS when a must_include phrase is missing."""
    custom_llm = CustomAnswerMockLLMProvider("You have 60 days to return your order.")
    runner = EvaluationRunner()
    runner.agent.llm_provider = custom_llm

    case_dict = {
        "id": "test-semantic-fail",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "How long do I have to return an item?"}],
        "expect": {
            "must_include": ["30 calendar days"],
        }
    }
    report = runner.evaluate_case(case_dict)
    assert report.live_llm_status == "FAIL"
    assert report.overall_status == "FAIL"


def test_semantic_must_not_include_passes():
    """Test that live semantic check PASSES when forbidden phrases are absent."""
    custom_llm = CustomAnswerMockLLMProvider("Returns are valid for 30 calendar days.")
    runner = EvaluationRunner()
    runner.agent.llm_provider = custom_llm

    case_dict = {
        "id": "test-forbidden-pass",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "Return window?"}],
        "expect": {
            "must_not_include": ["60 days", "free label"]
        }
    }
    report = runner.evaluate_case(case_dict)
    assert report.live_llm_status == "PASS"


def test_semantic_must_not_include_fails():
    """Test that live semantic check FAILS when a forbidden phrase is present."""
    custom_llm = CustomAnswerMockLLMProvider("We offer 60 days return window and a free label.")
    runner = EvaluationRunner()
    runner.agent.llm_provider = custom_llm

    case_dict = {
        "id": "test-forbidden-fail",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "Return window?"}],
        "expect": {
            "must_not_include": ["60 days"]
        }
    }
    report = runner.evaluate_case(case_dict)
    assert report.live_llm_status == "FAIL"
    assert report.overall_status == "FAIL"


def test_deterministic_pass_plus_no_live_llm_equals_unverified():
    """Test that MockLLMProvider with passing state assertions produces UNVERIFIED_REQUIRES_LLM."""
    runner = EvaluationRunner()  # uses MockLLMProvider by default
    reports = runner.run_all()
    assert all(r.deterministic_status == "PASS" for r in reports)
    assert all(r.live_llm_status == "UNAVAILABLE" for r in reports)
    assert all(r.overall_status == "UNVERIFIED_REQUIRES_LLM" for r in reports)


def test_deterministic_failure_equals_fail_regardless_of_llm():
    """Test that deterministic assertion failure forces overall_status=FAIL regardless of LLM output."""
    custom_llm = CustomAnswerMockLLMProvider("30 calendar days delivery")
    runner = EvaluationRunner()
    runner.agent.llm_provider = custom_llm

    case_dict = {
        "id": "test-det-fail",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "Return policy?"}],
        "expect": {
            "must_include": ["30 calendar days"],
            "required_sources": ["NON_EXISTENT_FILE.md"]  # Deterministic failure
        }
    }
    report = runner.evaluate_case(case_dict)
    assert report.deterministic_status == "FAIL"
    assert report.overall_status == "FAIL"


def test_api_key_missing_does_not_crash_evaluation(monkeypatch):
    """Test that missing OPENAI_API_KEY gracefully falls back without crashing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = EvaluationRunner(use_live_llm=True)
    assert isinstance(runner.agent.llm_provider, MockLLMProvider)
    reports = runner.run_all()
    assert len(reports) == 15


def test_api_key_never_included_in_reports_or_trace_data(monkeypatch):
    """Test that API keys do not leak into evaluation reports or trace outputs."""
    fake_key = "sk-proj-FAKEKEY1234567890ABCDEF"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    
    runner = EvaluationRunner()
    reports = runner.run_all()
    report_json = json.dumps([r.to_dict() for r in reports])
    
    assert fake_key not in report_json
