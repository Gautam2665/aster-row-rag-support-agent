"""
Educational Experiment #4: Real LLM Inference Test
Sends retrieved knowledge-base evidence and query through OpenAILLMProvider to verify
that the live LLM correctly selects TrailPlus-specific 45-day evidence from retrieved context.
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
from src.llm import generate_grounded_response, OpenAILLMProvider

KB_DIR = Path("knowledge-base")


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
    print("REAL LLM GENERATION EXPERIMENT")
    print("=" * 80)
    print(f"User Query: \"{query}\"\n")

    # Step 1: Ingest KB & vector index
    print("[Step 1] Ingesting & vector indexing knowledge base...")
    chunks = ingest_kb_directory(KB_DIR)
    vector_store = KBVectorStore(collection_name="real_llm_exp_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    # Step 2: Native pre-filtered retrieval
    print("[Step 2] Retrieving evidence chunks from ChromaDB...")
    retrieved_chunks = vector_store.search(
        query=query,
        top_k=3,
        filter_customer_eligible=True
    )

    print(f"\n[Retrieved Evidence Chunks ({len(retrieved_chunks)})]")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"  {i}. {chunk.source_citation}")
        print(f"     Text: {chunk.text[:130]}...\n")

    # Step 3: Call OpenAILLMProvider with build_grounded_prompt
    print("[Step 3] Calling OpenAILLMProvider with grounded prompt...")
    provider = OpenAILLMProvider(model_name="gpt-4o-mini")
    response = generate_grounded_response(
        user_question=query,
        evidence_chunks=retrieved_chunks,
        provider=provider
    )

    print("\n" + "=" * 80)
    print("FINAL PROMPT SENT TO OPENAI")
    print("=" * 80)
    print(response.prompt_payload["full_prompt"])

    print("\n" + "=" * 80)
    print("RAW LLM ANSWER FROM OPENAI (gpt-4o-mini)")
    print("=" * 80)
    print(f"Answer:\n{response.answer}\n")
    print(f"Returned Citations: {response.source_citations}")
    print("=" * 80)


if __name__ == "__main__":
    main()
