import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

from src.models import KBChunk, DocumentMetadata
from src.embeddings import EmbeddingProvider


class KBVectorStore:
    """
    ChromaDB-backed vector index for Knowledge Base chunks.
    Ensures identical embedding model is used for document indexing and query retrieval.
    """

    def __init__(
        self,
        collection_name: str = "aster_row_kb",
        persist_directory: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or EmbeddingProvider()

        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Create or get collection without built-in EF (we pass precomputed embeddings)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def clear(self):
        """Reset and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[KBChunk]) -> int:
        """Batch embed and index KBChunk objects in ChromaDB."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)

        metadatas = []
        for c in chunks:
            meta = c.metadata
            metadatas.append({
                "filename": c.filename,
                "heading": c.heading or "",
                "source_citation": c.source_citation,
                "document_id": meta.document_id,
                "title": meta.title,
                "status": meta.status,
                "audience": meta.audience,
                "policy_authority": meta.policy_authority,
                "customer_answering": meta.customer_answering,
                "supersedes": meta.supersedes or "",
                "superseded_by": meta.superseded_by or "",
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_customer_eligible: bool = True,
    ) -> List[KBChunk]:
        """
        Perform semantic retrieval against the Chroma index.
        If filter_customer_eligible=True, applies metadata filters to exclude
        superseded, draft, internal, or non-customer-answering documents.
        """
        query_embedding = self.embedding_provider.embed_query(query)

        where_filter = None
        if filter_customer_eligible:
            where_filter = {
                "$and": [
                    {"status": {"$eq": "active"}},
                    {"policy_authority": {"$eq": "official"}},
                    {"audience": {"$eq": "customer"}},
                    {"customer_answering": {"$eq": True}},
                ]
            }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks: List[KBChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return retrieved_chunks

        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        meta_list = results["metadatas"][0]

        for chunk_id, text, meta in zip(ids_list, docs_list, meta_list):
            doc_meta = DocumentMetadata(
                document_id=meta.get("document_id", ""),
                title=meta.get("title", ""),
                status=meta.get("status", "active"),
                audience=meta.get("audience", "customer"),
                policy_authority=meta.get("policy_authority", "official"),
                supersedes=meta.get("supersedes") or None,
                superseded_by=meta.get("superseded_by") or None,
                customer_answering=meta.get("customer_answering", True),
            )
            chunk = KBChunk(
                chunk_id=chunk_id,
                filename=meta.get("filename", ""),
                heading=meta.get("heading") or None,
                text=text,
                metadata=doc_meta,
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks
