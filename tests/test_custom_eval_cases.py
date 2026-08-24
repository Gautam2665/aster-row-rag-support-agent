import pytest
from pathlib import Path
from src.evaluation import EvaluationRunner


def test_custom_cases_loading():
    """Test loading evaluation/custom-cases.json via EvaluationRunner."""
    custom_path = Path("evaluation/custom-cases.json")
    runner = EvaluationRunner(cases_json_path=custom_path)
    cases = runner.load_cases()
    assert len(cases) == 5
    case_ids = [c["id"] for c in cases]
    assert "custom-session-isolation" in case_ids
    assert "custom-internal-data-refusal" in case_ids
    assert "custom-retrieval-abstention" in case_ids
    assert "custom-multi-turn-order-followup" in case_ids
    assert "custom-planner-failure-recovery" in case_ids


def test_custom_cases_execution():
    """Test running all 5 custom evaluation cases through EvaluationRunner."""
    custom_path = Path("evaluation/custom-cases.json")
    runner = EvaluationRunner(cases_json_path=custom_path)
    reports = runner.run_all()

    assert len(reports) == 5
    assert all(r.deterministic_status == "PASS" for r in reports)
    assert all(r.overall_status == "UNVERIFIED_REQUIRES_LLM" for r in reports)


def test_session_isolation_case_behavior():
    """Test custom-session-isolation case specifically."""
    runner = EvaluationRunner(cases_json_path=Path("evaluation/custom-cases.json"))
    cases = runner.load_cases()
    case_dict = next(c for c in cases if c["id"] == "custom-session-isolation")
    report = runner.evaluate_case(case_dict)

    assert report.deterministic_status == "PASS"
    assert len(report.tool_calls_made) == 0
    assert report.handoff_recommended is False


def test_internal_data_refusal_case_behavior():
    """Test custom-internal-data-refusal case specifically."""
    runner = EvaluationRunner(cases_json_path=Path("evaluation/custom-cases.json"))
    cases = runner.load_cases()
    case_dict = next(c for c in cases if c["id"] == "custom-internal-data-refusal")
    report = runner.evaluate_case(case_dict)

    assert report.deterministic_status == "PASS"
    assert report.handoff_recommended is True


def test_retrieval_abstention_case_behavior():
    """Test custom-retrieval-abstention case specifically."""
    runner = EvaluationRunner(cases_json_path=Path("evaluation/custom-cases.json"))
    cases = runner.load_cases()
    case_dict = next(c for c in cases if c["id"] == "custom-retrieval-abstention")
    report = runner.evaluate_case(case_dict)

    assert report.deterministic_status == "PASS"
    assert report.handoff_recommended is True


def test_multi_turn_order_followup_case_behavior():
    """Test custom-multi-turn-order-followup case specifically."""
    runner = EvaluationRunner(cases_json_path=Path("evaluation/custom-cases.json"))
    cases = runner.load_cases()
    case_dict = next(c for c in cases if c["id"] == "custom-multi-turn-order-followup")
    report = runner.evaluate_case(case_dict)

    assert report.deterministic_status == "PASS"
    assert "order_lookup" in report.tool_calls_made


def test_planner_failure_recovery_case_behavior():
    """Test custom-planner-failure-recovery case specifically."""
    runner = EvaluationRunner(cases_json_path=Path("evaluation/custom-cases.json"))
    cases = runner.load_cases()
    case_dict = next(c for c in cases if c["id"] == "custom-planner-failure-recovery")
    report = runner.evaluate_case(case_dict)

    assert report.deterministic_status == "PASS"
    assert report.handoff_recommended is True
    assert len(report.tool_calls_made) == 0


def test_combined_cases_execution():
    """Test evaluating both visible-cases.json and custom-cases.json together."""
    runner = EvaluationRunner(cases_json_path=[
        Path("evaluation/visible-cases.json"),
        Path("evaluation/custom-cases.json")
    ])
    reports = runner.run_all()

    assert len(reports) == 20  # 15 visible + 5 custom
    assert all(r.deterministic_status == "PASS" for r in reports)
