import json
from pathlib import Path
import pytest

from src.evaluation import EvaluationRunner, CaseEvalReport
from src.agent import SupportAgent
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.ingestion import ingest_kb_directory

VISIBLE_CASES_PATH = Path("evaluation/visible-cases.json")
KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


@pytest.fixture(scope="module")
def eval_runner():
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="test_eval_runner_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    order_tool = OrderLookupTool(data_path=ORDERS_PATH)
    llm_provider = MockLLMProvider()

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=order_tool,
        llm_provider=llm_provider,
    )
    return EvaluationRunner(cases_json_path=VISIBLE_CASES_PATH, agent=agent)


def test_runner_loads_cases(eval_runner):
    """Test loading visible-cases.json loads 15 test cases."""
    cases = eval_runner.load_cases()
    assert len(cases) == 15
    case_ids = [c["id"] for c in cases]
    assert "standard-return-window" in case_ids
    assert "valid-order-lookup" in case_ids
    assert "retrieved-prompt-injection" in case_ids


def test_runner_executes_all_cases(eval_runner, tmp_path):
    """Test executing all cases produces structured reports and writes evaluation_results.json."""
    output_json = tmp_path / "evaluation_results.json"
    reports = eval_runner.run_all(output_report_path=output_json)

    assert len(reports) == 15
    assert output_json.exists()

    saved_data = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved_data["total_cases"] == 15
    assert saved_data["failed_cases"] == 0


def test_runner_reports_unverified_under_mock_llm(eval_runner):
    """Test that cases requiring semantic LLM judgment are tagged UNVERIFIED_REQUIRES_LLM under MockLLMProvider."""
    cases = eval_runner.load_cases()
    std_case = next(c for c in cases if c["id"] == "standard-return-window")
    
    report = eval_runner.evaluate_case(std_case)
    assert report.overall_status == "UNVERIFIED_REQUIRES_LLM"
    
    semantic_assertion = next(a for a in report.assertions if a.name == "semantic_llm_generation_check")
    assert semantic_assertion.status == "UNVERIFIED_REQUIRES_LLM"


def test_runner_deterministic_assertions_pass(eval_runner):
    """Test that deterministic tool, privacy, and handoff assertions pass cleanly."""
    cases = eval_runner.load_cases()
    
    # Valid order lookup case
    valid_order_case = next(c for c in cases if c["id"] == "valid-order-lookup")
    report = eval_runner.evaluate_case(valid_order_case)
    tool_assertion = next(a for a in report.assertions if a.name == "tool_execution_check")
    assert tool_assertion.status == "PASS"

    # Missing order ID case
    missing_id_case = next(c for c in cases if c["id"] == "missing-order-id")
    report_missing = eval_runner.evaluate_case(missing_id_case)
    tool_assertion_missing = next(a for a in report_missing.assertions if a.name == "tool_execution_check")
    assert tool_assertion_missing.status == "PASS"

    # Privacy case
    privacy_case = next(c for c in cases if c["id"] == "order-data-privacy")
    report_privacy = eval_runner.evaluate_case(privacy_case)
    privacy_assertion = next(a for a in report_privacy.assertions if a.name == "privacy_pii_sanitization_check")
    assert privacy_assertion.status == "PASS"
