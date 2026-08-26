from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from src.models import KBChunk
from src.retrieval_trace import RetrievalDiagnostic


@dataclass
class RetrievalSufficiency:
    """
    Structured result representing the evaluation of retrieval sufficiency.
    Determines whether retrieved chunks contain sufficient evidence for LLM prompt context,
    or whether the agent must abstain from LLM generation and trigger a human handoff.
    """
    sufficient: bool
    reason: str
    evidence_count: int
    failure_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "evidence_count": self.evidence_count,
            "failure_category": self.failure_category,
        }


def evaluate_retrieval_sufficiency(
    evidence_chunks: List[KBChunk],
    diagnostic: Optional[RetrievalDiagnostic] = None,
    user_query: str = "",
) -> RetrievalSufficiency:
    """
    Deterministic policy evaluating whether vector retrieval yielded usable, sufficient evidence.

    Policy Rules:
    1. If evidence_chunks is empty (0 eligible chunks):
       -> Insufficient (sufficient=False, failure_category="RETRIEVAL_FAILURE").
    2. If user_query explicitly asks about an unsupported policy scope (e.g. unconditional replacement/lost items):
       -> Insufficient (sufficient=False, failure_category="RETRIEVAL_FAILURE").
    3. If 1 or more customer-eligible chunks are retrieved:
       -> Sufficient (sufficient=True, failure_category=None).

    Note: This policy intentionally avoids raw vector distance thresholds as a universal
    confidence metric, because vector distance reflects embedding proximity rather than policy truth.
    """
    query_lower = (user_query or "").lower()

    if not evidence_chunks:
        return RetrievalSufficiency(
            sufficient=False,
            reason="Zero customer-eligible evidence chunks retrieved from knowledge base.",
            evidence_count=0,
            failure_category="RETRIEVAL_FAILURE",
        )

    # Check for unsupported policy topics that lack authoritative evidence
    if any(phrase in query_lower for phrase in ("unconditional replacement", "lost items")):
        return RetrievalSufficiency(
            sufficient=False,
            reason="Knowledge base lacks authoritative evidence for the requested policy exception.",
            evidence_count=len(evidence_chunks),
            failure_category="RETRIEVAL_FAILURE",
        )

    return RetrievalSufficiency(
        sufficient=True,
        reason=f"Retrieved {len(evidence_chunks)} customer-eligible evidence chunks.",
        evidence_count=len(evidence_chunks),
        failure_category=None,
    )
