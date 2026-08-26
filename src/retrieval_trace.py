from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RetrievalDiagnostic:
    """
    Structured, safe retrieval diagnostic capturing non-sensitive RAG metadata.
    Provides observability into vector search mechanics, candidate chunk counts,
    distance metrics, and metadata filtering without exposing customer PII, raw order data,
    or prompt text.
    """
    retrieval_query: str
    num_candidates_returned: int
    num_eligible_chunks: int
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_filenames: List[str] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    has_usable_evidence: bool = False
    failure_classification: Optional[str] = None
    filter_applied: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_query": self.retrieval_query,
            "num_candidates_returned": self.num_candidates_returned,
            "num_eligible_chunks": self.num_eligible_chunks,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_filenames": self.retrieved_filenames,
            "distances": [round(d, 4) for d in self.distances],
            "has_usable_evidence": self.has_usable_evidence,
            "failure_classification": self.failure_classification,
            "filter_applied": self.filter_applied,
        }
