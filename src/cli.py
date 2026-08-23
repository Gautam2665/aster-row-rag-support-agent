import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from src.ingestion import ingest_kb_directory
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider, OpenAILLMProvider
from src.planner import MockPlanner, LLMPlanner
from src.agent import SupportAgent
from src.memory import SessionMemoryStore


def build_agent(
    use_live: bool = False,
    kb_dir: Path = Path("knowledge-base"),
    data_path: Path = Path("data/orders.json"),
) -> SupportAgent:
    """
    Construct SupportAgent state machine by wiring existing application components.
    CLI acts strictly as an interface adapter without duplicating business logic.
    """
    chunks = ingest_kb_directory(kb_dir)
    vector_store = KBVectorStore(collection_name="cli_kb_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    order_tool = OrderLookupTool(data_path=data_path)

    if use_live:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("\n================================================================================")
            print("LIVE MODE ERROR: OPENAI_API_KEY is not configured in your environment or .env file.")
            print("Please add a valid OPENAI_API_KEY to your .env file to run with --live mode.")
            print("Example: OPENAI_API_KEY=sk-...\n")
            print("To run in offline development mode without an API key, omit the --live flag.")
            print("================================================================ algorithm\n")
            sys.exit(1)
        llm_provider = OpenAILLMProvider()
        planner = LLMPlanner(llm_provider=llm_provider)
    else:
        llm_provider = MockLLMProvider()
        planner = MockPlanner()

    memory_store = SessionMemoryStore(max_turns_per_session=5)

    return SupportAgent(
        vector_store=vector_store,
        order_tool=order_tool,
        llm_provider=llm_provider,
        planner=planner,
        memory_store=memory_store,
    )


def print_formatted_response(state, debug: bool = False):
    """Print clean user-facing response with citations and optional debug trace."""
    print(f"\nAssistant: {state.final_answer}\n")

    if state.citations:
        print("Sources:")
        for citation in state.citations:
            print(f"  - {citation}")
        print()

    if state.handoff_recommended:
        print("Note: [Human Support Escalation Recommended]\n")

    if debug and state.trace:
        print("-" * 75)
        print("DEBUG EXECUTION TRACE:")
        print("-" * 75)
        for ev in state.trace:
            ev_dict = ev.to_dict()
            action_str = f" | Action: {ev_dict['action_type']}" if ev_dict.get("action_type") else ""
            fail_str = f" | FailCategory: {ev_dict['failure_category']}" if ev_dict.get("failure_category") else ""
            print(f"  [{ev_dict['timestamp']}] Iter {ev_dict['iteration']}: {ev_dict['event_type']}{action_str}{fail_str} -> {ev_dict['summary']}")
        print("-" * 75 + "\n")


def run_cli(args_list: Optional[List[str]] = None):
    """Interactive command-line loop for Aster & Row support agent."""
    parser = argparse.ArgumentParser(description="Aster & Row AI Support Agent CLI")
    parser.add_argument("--live", action="store_true", help="Run with live OpenAILLMProvider (requires OPENAI_API_KEY)")
    parser.add_argument("--debug", action="store_true", help="Print sanitized structured execution trace after every turn")
    
    args = parser.parse_args(args_list)

    agent = build_agent(use_live=args.live)
    session_id = "cli_interactive_session"

    mode_name = "LIVE (OpenAI)" if args.live else "OFFLINE (Mock)"
    print("\n" + "=" * 75)
    print(f"  Aster & Row AI Customer Support Agent ({mode_name} Mode)")
    print("  Type 'exit' or 'quit' to end session. Type 'clear' to reset memory.")
    if args.debug:
        print("  [DEBUG MODE ENABLED: Sanitized execution traces will be printed]")
    print("=" * 75 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Thank you for contacting Aster & Row Support. Goodbye!")
            break

        if user_input.lower() == "clear":
            print("\n[Clearing conversation memory...]\n")
            agent.memory_store.clear_session(session_id)
            continue

        state = agent.process_turn(user_query=user_input, session_id=session_id)
        print_formatted_response(state, debug=args.debug)


if __name__ == "__main__":
    run_cli()
