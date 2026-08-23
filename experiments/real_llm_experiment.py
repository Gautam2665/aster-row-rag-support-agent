"""
Educational Experiment #4: Real LLM Inference Test
Sends retrieved knowledge-base evidence and query through OpenAILLMProvider and SupportAgent
to verify that live LLM generation correctly handles policy questions with source citations.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env file
load_dotenv()

from src.ingestion import ingest_kb_directory
from src.retrieval import KBVectorStore
from src.tools.order_lookup import OrderLookupTool
from src.llm import OpenAILLMProvider
from src.agent import SupportAgent

KB_DIR = Path("knowledge-base")
ORDERS_PATH = Path("data/orders.json")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("=" * 80)
        print("REAL LLM EXPERIMENT NOTICE")
        print("=" * 80)
        print("OPENAI_API_KEY is not configured in your local .env file.")
        print("Please add your OPENAI_API_KEY to .env to run live API calls:")
        print("  OPENAI_API_KEY=sk-...\n")
        print("The script is structured to invoke OpenAILLMProvider() as soon as the key is present.")
        print("=" * 80)
        return

    query = "How long do TrailPlus members have to return an item?"

    print("=" * 80)
    print("REAL LLM GENERATION EXPERIMENT WITH SUPPORT AGENT")
    print("=" * 80)
    print(f"User Query: \"{query}\"\n")

    # Step 1: Setup KB Vector Store & Order Tool
    print("[Step 1] Ingesting & vector indexing knowledge base...")
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="real_llm_exp_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)
    order_tool = OrderLookupTool(data_path=ORDERS_PATH)

    # Step 2: Initialize SupportAgent with OpenAILLMProvider
    print("[Step 2] Initializing SupportAgent with OpenAILLMProvider...")
    provider = OpenAILLMProvider(model_name="gpt-4o-mini")
    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=order_tool,
        llm_provider=provider,
    )

    # Step 3: Execute Turn
    print("[Step 3] Executing turn through agent state machine...")
    state = agent.process_turn(user_query=query, session_id="exp_real_llm")

    print("\n" + "=" * 80)
    print("AGENT FINAL ANSWER")
    print("=" * 80)
    print(f"Answer:\n{state.final_answer}\n")
    print(f"Citations: {state.citations}")
    print(f"Handoff Recommended: {state.handoff_recommended}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("SANITIZED AGENT TRACE")
    print("=" * 80)
    for event in state.trace:
        clean_event = event.to_dict()
        print(f"[{clean_event['timestamp']}] {clean_event['event_type']} (Iter {clean_event['iteration']}): {clean_event['summary']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
