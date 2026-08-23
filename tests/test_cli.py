import sys
import os
import pytest
from io import StringIO
from pathlib import Path
from src.cli import build_agent, print_formatted_response, run_cli
from src.llm import MockLLMProvider
from src.agent import SupportAgent


def test_cli_starts_offline_without_api_key(monkeypatch):
    """Test that CLI initializes offline by default without requiring an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = build_agent(use_live=False)
    assert isinstance(agent, SupportAgent)
    assert isinstance(agent.llm_provider, MockLLMProvider)


def test_live_mode_gracefully_handles_missing_api_key(monkeypatch):
    """Test that requesting --live mode without OPENAI_API_KEY exits gracefully with error code 1."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        build_agent(use_live=True)
    assert exc_info.value.code == 1


def test_one_policy_query_executes_successfully():
    """Test executing one policy query through CLI agent helper."""
    agent = build_agent(use_live=False)
    state = agent.process_turn("What is the return window?", session_id="cli_test_session")
    assert state.final_answer is not None
    assert len(state.citations) > 0


def test_one_order_lookup_executes_successfully():
    """Test executing one order lookup query through CLI agent helper."""
    agent = build_agent(use_live=False)
    state = agent.process_turn("Where is ORD-1007?", session_id="cli_test_order")
    assert state.order_result is not None
    assert state.order_result.order_id == "ORD-1007"
    assert "order_lookup" in state.tool_calls_made


def test_same_session_preserves_conversation_memory_across_turns():
    """Test multi-turn interaction in the same CLI session preserves conversation history."""
    agent = build_agent(use_live=False)
    session_id = "cli_multi_turn_session"

    state1 = agent.process_turn("Do you ship internationally?", session_id=session_id)
    assert state1.final_answer is not None

    state2 = agent.process_turn("What about Canada?", session_id=session_id)
    assert len(state2.history_turns) == 1
    assert state2.history_turns[0].user_query == "Do you ship internationally?"


def test_debug_mode_exposes_trace(capsys):
    """Test that print_formatted_response with debug=True outputs DEBUG EXECUTION TRACE."""
    agent = build_agent(use_live=False)
    state = agent.process_turn("What is your warranty policy?", session_id="cli_debug_test")
    
    print_formatted_response(state, debug=True)
    captured = capsys.readouterr().out

    assert "DEBUG EXECUTION TRACE:" in captured
    assert "TURN_STARTED" in captured
    assert "TURN_COMPLETED" in captured


def test_debug_mode_does_not_expose_pii(capsys):
    """Test that debug trace output excludes PII fields."""
    agent = build_agent(use_live=False)
    state = agent.process_turn("Where is ORD-1007?", session_id="cli_pii_test")
    
    print_formatted_response(state, debug=True)
    captured = capsys.readouterr().out

    assert "email" not in captured
    assert "address" not in captured
    assert "risk_score" not in captured
    assert "warehouse_note" not in captured


def test_exit_command_terminates_cleanly(monkeypatch, capsys):
    """Test that typing 'exit' terminates the interactive CLI loop cleanly."""
    monkeypatch.setattr("sys.stdin", StringIO("exit\n"))
    run_cli([])
    captured = capsys.readouterr().out
    assert "Thank you for contacting Aster & Row Support. Goodbye!" in captured
