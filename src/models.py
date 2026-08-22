from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

@dataclass
class DocumentMetadata:
    """Represents frontmatter metadata extracted from knowledge-base documents."""
    document_id: str
    title: str
    status: str  # e.g., "active", "superseded", "draft"
    audience: str  # e.g., "customer", "internal"
    policy_authority: str  # e.g., "official", "none"
    effective_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    superseded_date: Optional[str] = None
    customer_answering: bool = True  # Default True, set to False for internal draft notes

    @property
    def is_active_official(self) -> bool:
        """Returns True if the document is active, official, and customer-facing."""
        return (
            self.status == "active"
            and self.policy_authority == "official"
            and self.audience == "customer"
            and self.customer_answering is True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Construct metadata instance from dictionary (e.g. parsed YAML frontmatter)."""
        valid_keys = {
            "document_id", "title", "status", "audience", "policy_authority",
            "effective_date", "last_reviewed", "supersedes", "superseded_by",
            "superseded_date", "customer_answering"
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        # Handle string conversion for dates if needed
        for date_key in ("effective_date", "last_reviewed", "superseded_date"):
            if date_key in filtered_data and filtered_data[date_key] is not None:
                filtered_data[date_key] = str(filtered_data[date_key])
        return cls(**filtered_data)


@dataclass
class KBChunk:
    """Represents a chunk derived from a knowledge-base document for RAG indexing/retrieval."""
    chunk_id: str
    filename: str
    heading: Optional[str]
    text: str
    metadata: DocumentMetadata

    @property
    def source_citation(self) -> str:
        """Returns formatted source reference identifying filename and heading."""
        if self.heading:
            return f"{self.filename} > {self.heading}"
        return self.filename

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk and nested metadata to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBChunk":
        """Deserialize dictionary to KBChunk instance."""
        meta_data = data["metadata"]
        if isinstance(meta_data, dict):
            meta_data = DocumentMetadata.from_dict(meta_data)
        return cls(
            chunk_id=data["chunk_id"],
            filename=data["filename"],
            heading=data.get("heading"),
            text=data["text"],
            metadata=meta_data,
        )
