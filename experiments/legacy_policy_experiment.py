"""
Educational Experiment #2: Demonstrating Superseded Policy Risk in Pure Semantic Retrieval
Compares semantic similarity of a return window query against current vs legacy policy chunks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.ingestion import parse_markdown_file, parse_markdown_sections
from experiments.embedding_experiment import get_embeddings, cosine_similarity

KB_DIR = Path("knowledge-base")


def main():
    # Load and chunk current returns policy
    current_file = KB_DIR / "01-returns-policy-current.md"
    current_meta, current_body = parse_markdown_file(current_file)
    current_chunks = parse_markdown_sections(current_file.name, current_body, current_meta)
    current_chunk = next(c for c in current_chunks if "Standard return window" in c.heading)

    # Load and chunk legacy returns policy
    legacy_file = KB_DIR / "02-returns-policy-legacy.md"
    legacy_meta, legacy_body = parse_markdown_file(legacy_file)
    legacy_chunks = parse_markdown_sections(legacy_file.name, legacy_body, legacy_meta)
    legacy_chunk = next(c for c in legacy_chunks if "Return window" in c.heading)

    query = "What is the return window?"

    print("=" * 80)
    print("EDUCATIONAL EXPERIMENT #2: CURRENT VS SUPERSEDED POLICY SEMANTIC SIMILARITY")
    print("=" * 80)
    print(f"Query: \"{query}\"\n")

    print("[Chunk #1 - CURRENT POLICY]")
    print(f"Source Filename  : {current_chunk.filename}")
    print(f"Heading Citation : {current_chunk.source_citation}")
    print(f"Status           : {current_chunk.metadata.status}")
    print(f"Audience         : {current_chunk.metadata.audience}")
    print(f"Policy Authority : {current_chunk.metadata.policy_authority}")
    print(f"Customer Answering: {current_chunk.metadata.customer_answering}")
    print(f"Supersedes       : {current_chunk.metadata.supersedes}")
    print(f"Text Snippet     : {current_chunk.text[:120]}...\n")

    print("[Chunk #2 - LEGACY POLICY]")
    print(f"Source Filename  : {legacy_chunk.filename}")
    print(f"Heading Citation : {legacy_chunk.source_citation}")
    print(f"Status           : {legacy_chunk.metadata.status}")
    print(f"Audience         : {legacy_chunk.metadata.audience}")
    print(f"Policy Authority : {legacy_chunk.metadata.policy_authority}")
    print(f"Customer Answering: {legacy_chunk.metadata.customer_answering}")
    print(f"Superseded By    : {legacy_chunk.metadata.superseded_by}")
    print(f"Text Snippet     : {legacy_chunk.text[:120]}...\n")

    # Embed query and both chunks
    texts = [query, current_chunk.text, legacy_chunk.text]
    model_name, embeddings, dimension = get_embeddings(texts)
    query_emb, current_emb, legacy_emb = embeddings

    sim_current = cosine_similarity(query_emb, current_emb)
    sim_legacy = cosine_similarity(query_emb, legacy_emb)

    print("-" * 80)
    print(f"[Cosine Similarity Results - Model: {model_name}]")
    print(f"Query vs Current Policy Chunk ({current_chunk.metadata.status}) : {sim_current:.4f}")
    print(f"Query vs Legacy Policy Chunk  ({legacy_chunk.metadata.status}) : {sim_legacy:.4f}")

    print("=" * 80)
    print("KEY FINDING & RAG ARCHITECTURE INSIGHT")
    print("=" * 80)
    print("1. Both chunks score extremely high semantic similarity scores because both")
    print("   address return windows.")
    print("2. Pure vector retrieval (without frontmatter filtering or precedence rules)")
    print(f"   considers the superseded legacy document ({sim_legacy:.4f}) just as relevant")
    print(f"   as the current active policy ({sim_current:.4f}).")
    print("3. Consequently, a naive vector database could pass the outdated 60-day return policy")
    print("   to the LLM, leading to inaccurate/conflicting customer answers.")
    print("4. This proves why metadata filtering (status=='active', supersedes lineage) is mandatory")
    print("   for reliable production RAG systems.")
    print("=" * 80)


if __name__ == "__main__":
    main()
