"""
Educational Experiment #3: End-to-End RAG Demonstration
Ingests knowledge base -> Vector index -> Metadata-filtered retrieval -> Grounded Prompt Builder -> LLM Generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest_kb_directory
from src.retrieval import KBVectorStore
from src.llm import generate_grounded_response, MockLLMProvider, get_default_provider

KB_DIR = Path("knowledge-base")


def main():
    query = "How long do TrailPlus members have to return an item?"

    print("=" * 80)
    print("END-TO-END RAG DEMONSTRATION")
    print("=" * 80)
    print(f"User Query: \"{query}\"\n")

    # Step 1: Ingest knowledge base documents
    print("[Step 1] Ingesting knowledge-base directory...")
    chunks = ingest_kb_directory(KB_DIR)
    print(f"-> Extracted {len(chunks)} section chunks from 14 Markdown files.\n")

    # Step 2: Index in ChromaDB
    print("[Step 2] Indexing chunks in ChromaDB vector store...")
    vector_store = KBVectorStore(collection_name="demo_e2e_rag_kb")
    vector_store.clear()
    vector_store.index_chunks(chunks)
    print("-> Vector indexing complete with metadata attributes.\n")

    # Step 3: Native Pre-filtered Vector Search
    print("[Step 3] Executing vector search with native metadata pre-filtering...")
    retrieved_chunks = vector_store.search(
        query=query,
        top_k=3,
        filter_customer_eligible=True
    )
    print(f"-> Retrieved {len(retrieved_chunks)} eligible evidence chunks:\n")

    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"   [{i}] Citation : {chunk.source_citation}")
        print(f"       Status   : {chunk.metadata.status}")
        print(f"       Audience : {chunk.metadata.audience}")
        print(f"       Snippet  : {chunk.text[:120]}...\n")

    # Step 4: Prompt Building & LLM Generation
    print("[Step 4] Building grounded prompt and calling LLM Provider...")
    provider = get_default_provider()
    response = generate_grounded_response(
        user_question=query,
        evidence_chunks=retrieved_chunks,
        provider=provider
    )

    print("\n" + "=" * 80)
    print("FINAL PROMPT STRUCTURE (DELIMITED XML & SYSTEM DIRECTIVES)")
    print("=" * 80)
    print(response.prompt_payload["full_prompt"])

    print("\n" + "=" * 80)
    print("GENERATED GROUNDED RESPONSE")
    print("=" * 80)
    print(f"LLM Provider Used: {response.provider_name}")
    print(f"Answer:\n{response.answer}\n")
    print(f"Citations: {response.source_citations}")
    print("=" * 80)


if __name__ == "__main__":
    main()
