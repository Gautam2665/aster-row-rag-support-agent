"""
Standalone Educational Experiment: Embedding Generation & Cosine Similarity Comparison
This script tests semantic similarity between a query and two document sentences.
"""

import os
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D numpy vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def get_embeddings(texts: List[str]) -> Tuple[str, List[np.ndarray], int]:
    """
    Generate embeddings using OpenAI if OPENAI_API_KEY is available,
    otherwise fallback to local sentence-transformers (all-MiniLM-L6-v2).
    Returns (model_name, embeddings_list, dimension).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model_name = "text-embedding-3-small"
            print(f"[Info] Using OpenAI Embedding API ({model_name})...")
            response = client.embeddings.create(input=texts, model=model_name)
            embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            dimension = len(embeddings[0])
            return model_name, embeddings, dimension
        except Exception as e:
            print(f"[Warning] OpenAI API failed ({e}). Falling back to local Sentence-Transformers model...")

    from sentence_transformers import SentenceTransformer
    model_name = "all-MiniLM-L6-v2"
    print(f"[Info] Using local SentenceTransformer model ({model_name})...")
    model = SentenceTransformer(model_name)
    raw_embeddings = model.encode(texts)
    embeddings = [np.array(emb, dtype=np.float32) for emb in raw_embeddings]
    dimension = len(embeddings[0])
    return model_name, embeddings, dimension


def main():
    doc1 = "Customers may return eligible products within 30 days of delivery."
    query = "How long do I have to send my product back?"
    doc3 = "Customers can track their shipment using the carrier tracking number."

    texts = [doc1, query, doc3]

    print("=" * 70)
    print("EMBEDDING & SEMANTIC SIMILARITY EXPERIMENT")
    print("=" * 70)
    print(f"Document #1: \"{doc1}\"")
    print(f"Query (#2):   \"{query}\"")
    print(f"Document #3: \"{doc3}\"")
    print("-" * 70)

    model_name, embeddings, dimension = get_embeddings(texts)
    emb_doc1, emb_query, emb_doc3 = embeddings

    sim_1_2 = cosine_similarity(emb_query, emb_doc1)
    sim_2_3 = cosine_similarity(emb_query, emb_doc3)

    print(f"\n[Model Details]")
    print(f"Model Name          : {model_name}")
    print(f"Embedding Dimension : {dimension}")

    print(f"\n[Cosine Similarity Results]")
    print(f"Query (#2) vs Document #1 (Return Policy)   : {sim_1_2:.4f}")
    print(f"Query (#2) vs Document #3 (Carrier Tracking): {sim_2_3:.4f}")

    print("\n" + "=" * 70)
    print("EXPLANATION & ANALYSIS")
    print("=" * 70)
    if sim_1_2 > sim_2_3:
        print(f"Result: Document #1 is MORE semantically similar to Query #2.")
        print(f"Reason: Document #1 ('return eligible products within 30 days') directly answers")
        print(f"        the semantic intent of Query #2 ('How long do I have to send my product back?'),")
        print(f"        whereas Document #3 is about shipment tracking.")
    else:
        print(f"Result: Document #3 scored higher or equal.")
    print("=" * 70)


if __name__ == "__main__":
    main()
