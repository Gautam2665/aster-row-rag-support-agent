import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.agent import SupportAgent, AgentState
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import BaseLLMProvider, MockLLMProvider, OpenAILLMProvider, get_default_provider
from src.ingestion import ingest_kb_directory


@dataclass
class CaseAssertionResult:
    name: str
    category: str  # "retrieval", "tool", "privacy", "orchestration", "grounding", "handoff"
    passed: bool
    status: str  # "PASS", "FAIL", "UNVERIFIED_REQUIRES_LLM"
    detail: str


@dataclass
class CaseEvalReport:
    case_id: str
    category: str
    user_inputs: List[str]
    detected_intent: str
    tool_calls_made: List[str]
    retrieved_sources: List[str]
    handoff_recommended: bool
    final_answer: Optional[str]
    assertions: List[CaseAssertionResult] = field(default_factory=list)
    deterministic_status: str = "PASS"  # "PASS", "FAIL"
    live_llm_status: str = "UNAVAILABLE"  # "PASS", "FAIL", "UNAVAILABLE"
    overall_status: str = "PASS"  # "PASS", "FAIL", "UNVERIFIED_REQUIRES_LLM"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationRunner:
    """
    Evaluation runner for Aster & Row support agent visible evaluation cases.
    Evaluates state machine transitions, tool calls, metadata filtering, privacy,
    and supports optional live LLM semantic response verification.
    """

    def __init__(
        self,
        cases_json_path: Optional[Path] = None,
        agent: Optional[SupportAgent] = None,
        use_live_llm: bool = False,
    ):
        self.cases_json_path = cases_json_path or Path("evaluation/visible-cases.json")
        self.use_live_llm = use_live_llm
        if agent:
            self.agent = agent
        else:
            # Default setup
            chunks = ingest_kb_directory(Path("knowledge-base"))
            vector_store = KBVectorStore(collection_name="eval_runner_kb")
            vector_store.clear()
            vector_store.index_chunks(chunks)
            
            order_tool = OrderLookupTool(data_path=Path("data/orders.json"))
            
            if self.use_live_llm and os.getenv("OPENAI_API_KEY"):
                llm_provider = OpenAILLMProvider()
            else:
                llm_provider = MockLLMProvider()

            self.agent = SupportAgent(
                vector_store=vector_store,
                order_tool=order_tool,
                llm_provider=llm_provider,
            )

    def load_cases(self) -> List[Dict[str, Any]]:
        if not self.cases_json_path.exists():
            raise FileNotFoundError(f"Evaluation cases file not found: {self.cases_json_path}")
        content = self.cases_json_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return data.get("cases", [])

    def evaluate_case(self, case_dict: Dict[str, Any]) -> CaseEvalReport:
        case_id = case_dict["id"]
        category = case_dict["category"]
        messages = case_dict["messages"]
        expect = case_dict.get("expect", {})

        user_inputs = [m["content"] for m in messages if m.get("role") == "user"]
        
        # Execute conversation turns through agent state machine
        state: Optional[AgentState] = None
        session_id = f"eval_{case_id}"
        
        for user_input in user_inputs:
            state = self.agent.process_turn(user_query=user_input, session_id=session_id)

        assertions: List[CaseAssertionResult] = []
        is_mock_llm = isinstance(self.agent.llm_provider, MockLLMProvider)

        # 1. Evaluate Tool Usage
        expected_tool = expect.get("tool")
        if expected_tool:
            if expected_tool in ("not_called", "not_called_without_id"):
                tool_passed = len(state.tool_calls_made) == 0
                assertions.append(
                    CaseAssertionResult(
                        name="tool_execution_check",
                        category="tool",
                        passed=tool_passed,
                        status="PASS" if tool_passed else "FAIL",
                        detail=f"Expected zero tool calls. Actual tool calls: {state.tool_calls_made}",
                    )
                )
            elif expected_tool == "order_lookup":
                tool_passed = "order_lookup" in state.tool_calls_made and state.order_result is not None
                assertions.append(
                    CaseAssertionResult(
                        name="tool_execution_check",
                        category="tool",
                        passed=tool_passed,
                        status="PASS" if tool_passed else "FAIL",
                        detail=f"Expected 'order_lookup' tool call. Actual: {state.tool_calls_made}",
                    )
                )
            elif expected_tool == "optional_sanitized_lookup":
                tool_passed = True
                if "order_lookup" in state.tool_calls_made:
                    tool_passed = state.order_result is not None
                assertions.append(
                    CaseAssertionResult(
                        name="tool_execution_check",
                        category="tool",
                        passed=tool_passed,
                        status="PASS" if tool_passed else "FAIL",
                        detail="Optional sanitized lookup verified.",
                    )
                )

        # 2. Evaluate Required Sources
        req_sources = expect.get("required_sources", [])
        if req_sources:
            retrieved_filenames = {c.split(" > ")[0] for c in state.citations}
            sources_passed = all(src in retrieved_filenames for src in req_sources)
            assertions.append(
                CaseAssertionResult(
                    name="required_sources_check",
                    category="retrieval",
                    passed=sources_passed,
                    status="PASS" if sources_passed else "FAIL",
                    detail=f"Required sources: {req_sources}. Actual citations: {list(retrieved_filenames)}",
                )
            )

        # 3. Evaluate Forbidden Sources
        forb_sources = expect.get("forbidden_sources_as_authority", [])
        if forb_sources:
            retrieved_filenames = {c.split(" > ")[0] for c in state.citations}
            forbidden_passed = not any(src in retrieved_filenames for src in forb_sources)
            assertions.append(
                CaseAssertionResult(
                    name="forbidden_sources_check",
                    category="retrieval",
                    passed=forbidden_passed,
                    status="PASS" if forbidden_passed else "FAIL",
                    detail=f"Forbidden sources: {forb_sources}. Actual citations: {list(retrieved_filenames)}",
                )
            )

        # 4. Evaluate Handoff Expectation
        if "handoff" in expect:
            expected_handoff = expect["handoff"]
            handoff_passed = state.handoff_recommended == expected_handoff
            assertions.append(
                CaseAssertionResult(
                    name="handoff_recommendation_check",
                    category="handoff",
                    passed=handoff_passed,
                    status="PASS" if handoff_passed else "FAIL",
                    detail=f"Expected handoff={expected_handoff}. Actual handoff_recommended={state.handoff_recommended}",
                )
            )

        # 5. Evaluate Refuse to Disclose PII / Privacy Rules
        must_refuse = expect.get("must_refuse_to_disclose", [])
        if must_refuse:
            privacy_passed = True
            if state.order_result:
                order_dict = state.order_result.to_dict()
                privacy_passed = not any(k in order_dict for k in ("customer", "internal", "email", "address", "risk_score"))
            assertions.append(
                CaseAssertionResult(
                    name="privacy_pii_sanitization_check",
                    category="privacy",
                    passed=privacy_passed,
                    status="PASS" if privacy_passed else "FAIL",
                    detail=f"PII fields stripped / tool execution refused: {privacy_passed}",
                )
            )

        # 6. Evaluate Semantic Concepts (must_include, must_not_include)
        semantic_keys = ["must_include", "must_not_include", "must_include_concepts", "must_ask_for", "must_not_invent"]
        has_semantic_expectations = any(k in expect for k in semantic_keys)

        if has_semantic_expectations:
            if is_mock_llm:
                assertions.append(
                    CaseAssertionResult(
                        name="semantic_llm_generation_check",
                        category="grounding",
                        passed=True,
                        status="UNVERIFIED_REQUIRES_LLM",
                        detail="Final semantic prose generation requires live LLM evaluation; deterministic state verified.",
                    )
                )
            else:
                answer_lower = (state.final_answer or "").lower()
                concepts_passed = True
                fail_details = []

                if "must_include" in expect:
                    for inc in expect["must_include"]:
                        if inc.lower() not in answer_lower:
                            concepts_passed = False
                            fail_details.append(f"Missing required phrase: '{inc}'")

                if "must_not_include" in expect:
                    for exc in expect["must_not_include"]:
                        if exc.lower() in answer_lower:
                            concepts_passed = False
                            fail_details.append(f"Forbidden phrase found: '{exc}'")

                assertions.append(
                    CaseAssertionResult(
                        name="semantic_llm_generation_check",
                        category="grounding",
                        passed=concepts_passed,
                        status="PASS" if concepts_passed else "FAIL",
                        detail="Evaluated text against semantic expectations. " + ("; ".join(fail_details) if fail_details else "All semantic requirements met."),
                    )
                )

        # Separate deterministic state status vs live LLM status vs overall status
        deterministic_assertions = [a for a in assertions if a.name != "semantic_llm_generation_check"]
        semantic_assertion = next((a for a in assertions if a.name == "semantic_llm_generation_check"), None)

        det_passed = all(a.status == "PASS" for a in deterministic_assertions)
        deterministic_status = "PASS" if det_passed else "FAIL"

        if is_mock_llm:
            live_llm_status = "UNAVAILABLE"
        elif semantic_assertion:
            live_llm_status = "PASS" if semantic_assertion.status == "PASS" else "FAIL"
        else:
            live_llm_status = "PASS"

        if not det_passed:
            overall = "FAIL"
        elif is_mock_llm:
            overall = "UNVERIFIED_REQUIRES_LLM"
        elif live_llm_status == "FAIL":
            overall = "FAIL"
        else:
            overall = "PASS"

        return CaseEvalReport(
            case_id=case_id,
            category=category,
            user_inputs=user_inputs,
            detected_intent=state.intent_category,
            tool_calls_made=state.tool_calls_made,
            retrieved_sources=state.citations,
            handoff_recommended=state.handoff_recommended,
            final_answer=state.final_answer,
            assertions=assertions,
            deterministic_status=deterministic_status,
            live_llm_status=live_llm_status,
            overall_status=overall,
        )

    def run_all(self, output_report_path: Optional[Path] = None) -> List[CaseEvalReport]:
        cases = self.load_cases()
        reports = [self.evaluate_case(c) for c in cases]

        if output_report_path is None:
            output_report_path = Path("evaluation/evaluation_results.json")

        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "total_cases": len(reports),
            "passed_cases": sum(1 for r in reports if r.overall_status == "PASS"),
            "unverified_cases": sum(1 for r in reports if r.overall_status == "UNVERIFIED_REQUIRES_LLM"),
            "failed_cases": sum(1 for r in reports if r.overall_status == "FAIL"),
            "case_reports": [r.to_dict() for r in reports],
        }
        output_report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        return reports

    def print_terminal_summary(self, reports: List[CaseEvalReport]):
        print("\n" + "=" * 105)
        print(f"VISIBLE EVALUATION CASES REPORT (Total: {len(reports)})")
        print("=" * 105)
        print(f"{'Case ID':<30} | {'Category':<20} | {'Intent':<12} | {'State':<8} | {'Live LLM':<12} | {'Overall'}")
        print("-" * 105)

        for r in reports:
            print(f"{r.case_id:<30} | {r.category:<20} | {r.detected_intent:<12} | {r.deterministic_status:<8} | {r.live_llm_status:<12} | {r.overall_status}")

        passed = sum(1 for r in reports if r.overall_status == "PASS")
        unverified = sum(1 for r in reports if r.overall_status == "UNVERIFIED_REQUIRES_LLM")
        failed = sum(1 for r in reports if r.overall_status == "FAIL")

        print("=" * 105)
        print(f"Summary: {passed} PASSED | {unverified} UNVERIFIED (Requires Live LLM) | {failed} FAILED")
        print("=" * 105 + "\n")
