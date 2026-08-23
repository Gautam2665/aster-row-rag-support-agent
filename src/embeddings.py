import os
from typing import List
import numpy as np

class EmbeddingProvider:
    """
    Unified embedding generator ensuring identical model and dimensionality
    are used for indexing document chunks and embedding user queries.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._openai_client = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=api_key)
                self.model_name = "text-embedding-3-small"
            except Exception:
                self._openai_client = None

        if not self._openai_client:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings into float vectors."""
        if not texts:
            return []

        if self._openai_client:
            try:
                response = self._openai_client.embeddings.create(
                    input=texts, model=self.model_name
                )
                return [item.embedding for item in response.data]
            except Exception:
                # Graceful fallback to local SentenceTransformer if OpenAI API fails or key is invalid
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._openai_client = None

        embeddings = self._model.encode(texts)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]
