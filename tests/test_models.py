import pytest
from src.models import DocumentMetadata, KBChunk

def test_document_metadata_preservation():
    """Test that all key YAML frontmatter fields are properly retained."""
    frontmatter_active = {
        "document_id": "RET-2026-01",
        "title": "Returns Policy",
        "status": "active",
        "effective_date": "2026-04-01",
        "last_reviewed": "2026-07-15",
        "audience": "customer",
        "policy_authority": "official",
        "supersedes": "RET-2024-01"
    }
    meta = DocumentMetadata.from_dict(frontmatter_active)
    
    assert meta.document_id == "RET-2026-01"
    assert meta.status == "active"
    assert meta.audience == "customer"
    assert meta.policy_authority == "official"
    assert meta.supersedes == "RET-2024-01"
    assert meta.superseded_by is None
    assert meta.customer_answering is True
    assert meta.is_active_official is True


def test_legacy_superseded_metadata_preservation():
    """Test metadata for legacy/superseded document."""
    frontmatter_legacy = {
        "document_id": "RET-2024-01",
        "title": "Returns Policy — Legacy Version",
        "status": "superseded",
        "effective_date": "2024-01-01",
        "superseded_date": "2026-04-01",
        "last_reviewed": "2025-11-20",
        "audience": "customer",
        "policy_authority": "official",
        "superseded_by": "RET-2026-01"
    }
    meta = DocumentMetadata.from_dict(frontmatter_legacy)

    assert meta.status == "superseded"
    assert meta.superseded_by == "RET-2026-01"
    assert meta.is_active_official is False


def test_internal_migration_draft_metadata_preservation():
    """Test metadata for internal draft migration scratchpad with customer_answering=False."""
    frontmatter_draft = {
        "document_id": "MIG-TEST-04",
        "title": "Content Migration Scratchpad",
        "status": "draft",
        "effective_date": "2026-08-01",
        "last_reviewed": "2026-08-01",
        "audience": "internal",
        "policy_authority": "none",
        "customer_answering": False
    }
    meta = DocumentMetadata.from_dict(frontmatter_draft)

    assert meta.status == "draft"
    assert meta.audience == "internal"
    assert meta.policy_authority == "none"
    assert meta.customer_answering is False
    assert meta.is_active_official is False


def test_kb_chunk_creation_and_citation():
    """Test KBChunk model creation, source citation, and serialization."""
    meta = DocumentMetadata(
        document_id="RET-2026-01",
        title="Returns Policy",
        status="active",
        audience="customer",
        policy_authority="official",
        supersedes="RET-2024-01"
    )

    chunk = KBChunk(
        chunk_id="01-returns-policy-current.md#standard-return-window-0",
        filename="01-returns-policy-current.md",
        heading="Standard Return Window",
        text="Customers have 30 calendar days from delivery to initiate a return.",
        metadata=meta
    )

    assert chunk.filename == "01-returns-policy-current.md"
    assert chunk.heading == "Standard Return Window"
    assert chunk.source_citation == "01-returns-policy-current.md > Standard Return Window"
    assert chunk.metadata.status == "active"
    assert chunk.metadata.supersedes == "RET-2024-01"

    # Test dict serialization round-trip
    data_dict = chunk.to_dict()
    reconstructed = KBChunk.from_dict(data_dict)
    assert reconstructed.chunk_id == chunk.chunk_id
    assert reconstructed.source_citation == chunk.source_citation
    assert reconstructed.metadata.status == "active"
