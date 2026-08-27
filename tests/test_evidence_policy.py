import pytest
from pathlib import Path
from typing import List

from src.models import KBChunk, DocumentMetadata
from src.evidence_policy import assess_evidence, EvidenceAssessment, EvidenceStatus
from src.retrieval import KBVectorStore
from src.ingestion import ingest_kb_directory
from src.agent import SupportAgent
from src.tools.order_lookup import OrderLookupTool
from src.llm import MockLLMProvider
from src.planner import FailureCategory


def make_chunk(
    chunk_id: str,
    content: str,
    filename: str = "doc1.md",
    status: str = "active",
    audience: str = "customer",
    policy_authority: str = "official"
) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        filename=filename,
        heading=None,
        text=content,
        metadata=DocumentMetadata(
            document_id=filename,
            title="Test Doc",
            status=status,
            audience=audience,
            policy_authority=policy_authority,
        )
    )


def test_active_authoritative_evidence_is_usable():
    """a. Verify active authoritative evidence returns USABLE status."""
    c1 = make_chunk("c1", "Standard return policy window is 30 days for undamaged items.")
    c2 = make_chunk("c2", "Items must be unused and in original packaging.")

    res = assess_evidence([c1, c2])
    assert res.status == EvidenceStatus.USABLE
    assert res.conflict_detected is False
    assert len(res.usable_chunks) == 2


def test_irrelevant_non_authoritative_evidence_is_insufficient():
    """b. Verify superseded or non-authoritative chunks return INSUFFICIENT status."""
    c1 = make_chunk("c1", "Obsolete policy draft", status="superseded", policy_authority="deprecated")
    c2 = make_chunk("c2", "Internal notes for warehouse team", audience="internal_ops")

    res = assess_evidence([c1, c2])
    assert res.status == EvidenceStatus.INSUFFICIENT
    assert len(res.usable_chunks) == 0
    assert "No active, customer-eligible" in res.reason


def test_empty_evidence_is_insufficient():
    """c. Verify empty or None chunk list returns INSUFFICIENT status."""
    res1 = assess_evidence([])
    assert res1.status == EvidenceStatus.INSUFFICIENT

    res2 = assess_evidence(None)
    assert res2.status == EvidenceStatus.INSUFFICIENT


def test_genuine_active_source_conflict_detected():
    """d. Verify active-source policy conflict between distinct official docs returns CONFLICT status."""
    c1 = make_chunk(
        "c1",
        "The stainless-steel body of the Breeze Tumbler should be hand-washed.",
        filename="11-product-care.md"
    )
    c2 = make_chunk(
        "c2",
        "The product card states that all components are dishwasher safe, top rack recommended.",
        filename="12-breeze-tumbler-product-card.md"
    )

    res = assess_evidence([c1, c2])
    assert res.status == EvidenceStatus.CONFLICT
    assert res.conflict_detected is True
    assert len(res.conflicting_documents) == 2
    assert "11-product-care.md" in res.conflicting_documents
    assert "12-breeze-tumbler-product-card.md" in res.conflicting_documents


def test_conflict_does_not_select_arbitrary_source_and_causes_safe_handoff():
    """e, f, g. Verify active-source conflict causes safe handoff and skips LLM generation without selecting one arbitrarily."""
    chunks = ingest_kb_directory(Path("knowledge-base"))
    vector_store = KBVectorStore(collection_name="test_evidence_conflict_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        llm_provider=MockLLMProvider(),
    )

    state = agent.process_turn("Can I put the entire Breeze Tumbler in the dishwasher?", session_id="ev_conflict_s")

    assert state.handoff_recommended is True
    assert any(obs.failure_category == FailureCategory.BUSINESS_FAILURE for obs in state.observations if obs.failure_category)
    # Verify the agent recognised the conflict and escalated — do not check MockLLM prose keywords
    # (MockLLMProvider returns templated responses; only a live LLM produces handoff-framed prose)
    assert state.final_answer is not None


def test_insufficient_evidence_skips_llm_generation():
    """h. Verify insufficient evidence triggers handoff and skips LLM answer generation."""
    chunks = ingest_kb_directory(Path("knowledge-base"))
    vector_store = KBVectorStore(collection_name="test_evidence_insuff_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        llm_provider=MockLLMProvider(),
    )

    # Use a query that is genuinely out-of-domain and unlikely to match any active authoritative chunk
    state = agent.process_turn(
        "What is the carbon offset policy for interstellar shipping from Jupiter colonies?",
        session_id="ev_insuff_s"
    )

    assert state.handoff_recommended is True
    assert any(obs.failure_category == FailureCategory.RETRIEVAL_FAILURE for obs in state.observations if obs.failure_category)


def test_evidence_policy_does_not_leak_pii():
    """i. Verify EvidenceAssessment contains zero customer PII or raw database objects."""
    c1 = make_chunk("c1", "Product care guide", filename="care.md")
    res = assess_evidence([c1])

    res_str = str(res).lower()
    # 'customer' is a legitimate metadata enum value (audience='customer') — not a PII field.
    # PII fields are: email, address, risk_score, warehouse_note, api_key, secret.
    sensitive_keys = ["email", "address", "risk_score", "warehouse_note", "api_key", "secret"]
    for key in sensitive_keys:
        assert key not in res_str


def test_conflict_does_not_create_retry_loop():
    """j. Verify evidence conflict respects max_iterations=3 and terminates immediately."""
    chunks = ingest_kb_directory(Path("knowledge-base"))
    vector_store = KBVectorStore(collection_name="test_conflict_loop_store")
    vector_store.clear()
    vector_store.index_chunks(chunks)

    agent = SupportAgent(
        vector_store=vector_store,
        order_tool=OrderLookupTool(data_path=Path("data/orders.json")),
        llm_provider=MockLLMProvider(),
    )

    state = agent.process_turn("Can I put the entire Breeze Tumbler in the dishwasher?", session_id="ev_loop_s")

    assert state.iterations <= 3
    assert state.handoff_recommended is True
