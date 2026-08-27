from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.models import KBChunk


class EvidenceStatus:
    USABLE = "USABLE"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICT = "CONFLICT"


@dataclass
class EvidenceAssessment:
    """
    Structured outcome of deterministic evidence assessment.
    Distinguishes usable authoritative evidence, non-authoritative/insufficient evidence,
    and genuine active-source policy conflicts.
    """
    status: str
    usable_chunks: List[KBChunk] = field(default_factory=list)
    conflict_detected: bool = False
    reason: str = ""
    conflicting_documents: List[str] = field(default_factory=list)


def assess_evidence(chunks: Optional[List[KBChunk]]) -> EvidenceAssessment:
    """
    Deterministically assesses retrieved KB chunks for usability, non-authoritative filtering,
    and genuine active-source policy conflicts.
    """
    if not chunks:
        return EvidenceAssessment(
            status=EvidenceStatus.INSUFFICIENT,
            usable_chunks=[],
            conflict_detected=False,
            reason="No retrieved chunks provided.",
            conflicting_documents=[],
        )

    # 1. Filter active, customer-eligible authoritative chunks
    active_authoritative_chunks: List[KBChunk] = []
    for chunk in chunks:
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else (chunk.metadata.to_dict() if hasattr(chunk.metadata, "to_dict") else {})
        status = str(meta.get("status", "active"))
        audience = str(meta.get("audience", "customer"))
        authority = str(meta.get("policy_authority", "official"))

        # Exclude superseded, internal draft, or non-customer chunks
        if status.lower() == "active" and audience.lower() in ("customer", "all") and authority.lower() in ("official", "primary", "authoritative"):
            active_authoritative_chunks.append(chunk)

    if not active_authoritative_chunks:
        return EvidenceAssessment(
            status=EvidenceStatus.INSUFFICIENT,
            usable_chunks=[],
            conflict_detected=False,
            reason="No active, customer-eligible authoritative chunks found in retrieved evidence.",
            conflicting_documents=[],
        )

    # 2. Check for Genuine Active-Source Conflicts
    doc_chunks: Dict[str, List[KBChunk]] = {}
    for chunk in active_authoritative_chunks:
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else (chunk.metadata.to_dict() if hasattr(chunk.metadata, "to_dict") else {})
        doc_id = meta.get("filename") or meta.get("document_id") or "unknown_doc"
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(chunk)

    # If multiple distinct active official documents exist, check for contradictory policy directives
    if len(doc_chunks) >= 2:
        texts_by_doc: Dict[str, str] = {
            doc_id: " ".join([getattr(c, "text", getattr(c, "content", "")) for c in c_list]).lower()
            for doc_id, c_list in doc_chunks.items()
        }

        # Subject 1: Dishwasher safety directives (hand-wash body vs all components dishwasher safe)
        has_hand_wash = any("hand-wash" in t or "hand-washed" in t or "hand wash" in t for t in texts_by_doc.values())
        has_dishwasher_safe = any("dishwasher safe" in t or "all components are dishwasher" in t for t in texts_by_doc.values())

        if has_hand_wash and has_dishwasher_safe:
            conflicting_docs = [
                doc_id for doc_id, t in texts_by_doc.items()
                if ("hand-wash" in t or "hand-washed" in t or "hand wash" in t) or ("dishwasher safe" in t or "all components are dishwasher" in t)
            ]
            return EvidenceAssessment(
                status=EvidenceStatus.CONFLICT,
                usable_chunks=active_authoritative_chunks,
                conflict_detected=True,
                reason="Genuine active-source policy conflict detected between official documentation regarding dishwasher safety vs hand-washing.",
                conflicting_documents=conflicting_docs,
            )

        # Subject 2: Explicit contradictory policy assertions across distinct active documents
        for doc_a, text_a in texts_by_doc.items():
            for doc_b, text_b in texts_by_doc.items():
                if doc_a != doc_b:
                    if ("must not" in text_a and "permitted" in text_b) or ("not covered" in text_a and "covered" in text_b):
                        return EvidenceAssessment(
                            status=EvidenceStatus.CONFLICT,
                            usable_chunks=active_authoritative_chunks,
                            conflict_detected=True,
                            reason=f"Genuine active-source policy conflict detected between documents {doc_a} and {doc_b}.",
                            conflicting_documents=[doc_a, doc_b],
                        )

    # 3. Normal Usable Evidence
    return EvidenceAssessment(
        status=EvidenceStatus.USABLE,
        usable_chunks=active_authoritative_chunks,
        conflict_detected=False,
        reason="Evidence is active, authoritative, and non-conflicting.",
        conflicting_documents=[],
    )
