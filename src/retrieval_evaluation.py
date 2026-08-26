from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional

from src.retrieval import KBVectorStore
from src.models import KBChunk


def calculate_precision_at_k(retrieved_chunks: List[KBChunk], expected_relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision@K: Fraction of top-K retrieved chunks that match expected relevant IDs.
    """
    top_k_chunks = retrieved_chunks[:k]
    if not top_k_chunks:
        return 0.0

    relevant_hits = 0
    for chunk in top_k_chunks:
        if (
            chunk.filename in expected_relevant_ids
            or chunk.chunk_id in expected_relevant_ids
            or (chunk.metadata and chunk.metadata.document_id in expected_relevant_ids)
        ):
            relevant_hits += 1

    return relevant_hits / len(top_k_chunks)


def calculate_recall_at_k(retrieved_chunks: List[KBChunk], expected_relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@K: Fraction of expected relevant IDs retrieved in top-K.
    """
    if not expected_relevant_ids:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    hit_targets = set()
    for chunk in top_k_chunks:
        if chunk.filename in expected_relevant_ids:
            hit_targets.add(chunk.filename)
        if chunk.chunk_id in expected_relevant_ids:
            hit_targets.add(chunk.chunk_id)
        if chunk.metadata and chunk.metadata.document_id in expected_relevant_ids:
            hit_targets.add(chunk.metadata.document_id)

    return len(hit_targets) / len(expected_relevant_ids)


@dataclass
class RetrievalEvalCase:
    """
    Ground-truth evaluation case for vector retrieval performance measurement.
    """
    case_id: str
    query: str
    expected_relevant_ids: List[str]
    k: int = 5
    filter_customer_eligible: bool = True


@dataclass
class RetrievalEvalResult:
    """
    Structured result of running a RetrievalEvalCase.
    """
    case_id: str
    query: str
    k: int
    precision: float
    recall: float
    retrieved_filenames: List[str]
    expected_ids: List[str]
    hit_filenames: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "k": self.k,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "retrieved_filenames": self.retrieved_filenames,
            "expected_ids": self.expected_ids,
            "hit_filenames": self.hit_filenames,
        }


DEFAULT_RETRIEVAL_EVAL_DATASET = [
    RetrievalEvalCase(
        case_id="eval-standard-returns",
        query="What is the return window for a regular customer?",
        expected_relevant_ids=["01-returns-policy-current.md"],
        k=5,
    ),
    RetrievalEvalCase(
        case_id="eval-trailplus-returns",
        query="What is the return window for TrailPlus members?",
        expected_relevant_ids=["09-trailplus-membership.md"],
        k=5,
    ),
    RetrievalEvalCase(
        case_id="eval-international-shipping",
        query="Do you ship internationally to Canada?",
        expected_relevant_ids=["06-international-shipping.md"],
        k=5,
    ),
    RetrievalEvalCase(
        case_id="eval-warranty-coverage",
        query="What is covered under the limited product warranty?",
        expected_relevant_ids=["07-warranty.md"],
        k=5,
    ),
    RetrievalEvalCase(
        case_id="eval-damaged-item",
        query="What should I do if my order arrives damaged or defective?",
        expected_relevant_ids=["04-damaged-or-wrong-items.md"],
        k=5,
    ),
    RetrievalEvalCase(
        case_id="eval-unsupported-lost-item",
        query="What is the orbital space delivery policy for lost items in Atlantis?",
        expected_relevant_ids=["99-nonexistent-document.md"],
        k=5,
    ),
]


class RetrievalEvaluator:
    """
    Deterministic retrieval evaluator measuring Precision@K and Recall@K
    against ground-truth relevance without LLM generation or API calls.
    """

    def __init__(self, vector_store: KBVectorStore):
        self.vector_store = vector_store

    def evaluate_case(self, eval_case: RetrievalEvalCase) -> RetrievalEvalResult:
        expected_set = set(eval_case.expected_relevant_ids)
        retrieved_chunks = self.vector_store.search(
            query=eval_case.query,
            top_k=eval_case.k,
            filter_customer_eligible=eval_case.filter_customer_eligible,
        )

        prec = calculate_precision_at_k(retrieved_chunks, expected_set, eval_case.k)
        rec = calculate_recall_at_k(retrieved_chunks, expected_set, eval_case.k)

        retrieved_filenames = [c.filename for c in retrieved_chunks[:eval_case.k]]
        hits = [
            fn for fn in retrieved_filenames
            if fn in expected_set or any(c.chunk_id in expected_set for c in retrieved_chunks if c.filename == fn)
        ]
        unique_hits = list(dict.fromkeys(hits))

        return RetrievalEvalResult(
            case_id=eval_case.case_id,
            query=eval_case.query,
            k=eval_case.k,
            precision=prec,
            recall=rec,
            retrieved_filenames=retrieved_filenames,
            expected_ids=eval_case.expected_relevant_ids,
            hit_filenames=unique_hits,
        )

    def evaluate_dataset(
        self, dataset: Optional[List[RetrievalEvalCase]] = None
    ) -> List[RetrievalEvalResult]:
        cases = dataset or DEFAULT_RETRIEVAL_EVAL_DATASET
        return [self.evaluate_case(c) for c in cases]
