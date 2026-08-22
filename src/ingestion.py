import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import yaml

from src.models import DocumentMetadata, KBChunk


def parse_markdown_file(file_path: Path) -> Tuple[DocumentMetadata, str]:
    """Extract YAML frontmatter and body text from a Markdown file."""
    content = file_path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_frontmatter = parts[1]
            body_text = parts[2].strip()
            yaml_data = yaml.safe_load(raw_frontmatter) or {}
            metadata = DocumentMetadata.from_dict(yaml_data)
            return metadata, body_text

    # Default metadata fallback if frontmatter delimiter is missing
    metadata = DocumentMetadata(
        document_id=file_path.stem,
        title=file_path.stem,
        status="active",
        audience="customer",
        policy_authority="official",
    )
    return metadata, content.strip()


def slugify(text: str) -> str:
    """Convert heading text into a clean URL/ID-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def split_section_text(
    text: str, max_chars: int = 1000
) -> List[str]:
    """Subdivide a section text if it exceeds max_chars while preserving paragraph breaks."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    sub_chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        # If a single paragraph is larger than max_chars, split by line breaks or sentences
        if para_len > max_chars:
            if current_chunk:
                sub_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            lines = para.split("\n")
            for line in lines:
                if current_len + len(line) + 1 > max_chars and current_chunk:
                    sub_chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_len = len(line)
                else:
                    current_chunk.append(line)
                    current_len += len(line) + 1
            continue

        if current_len + para_len + 2 > max_chars and current_chunk:
            sub_chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + 2

    if current_chunk:
        sub_chunks.append("\n\n".join(current_chunk))

    return sub_chunks


def parse_markdown_sections(
    filename: str,
    body_text: str,
    metadata: DocumentMetadata,
    max_chunk_chars: int = 1000,
) -> List[KBChunk]:
    """Parse Markdown body into section-based KBChunk objects."""
    lines = body_text.split("\n")
    chunks: List[KBChunk] = []

    doc_title = metadata.title
    current_h2: Optional[str] = None
    current_lines: List[str] = []

    def flush_section():
        nonlocal current_lines, current_h2
        text = "\n".join(current_lines).strip()
        if not text:
            return

        # Heading representation for citations
        if current_h2 and current_h2 != doc_title:
            heading = f"{doc_title} > {current_h2}"
            heading_slug = slugify(current_h2)
        else:
            heading = doc_title
            heading_slug = "overview"

        sub_texts = split_section_text(text, max_chars=max_chunk_chars)
        for i, sub_text in enumerate(sub_texts):
            chunk_id = f"{filename}#{heading_slug}"
            if len(sub_texts) > 1:
                chunk_id += f"-part{i+1}"

            chunks.append(
                KBChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    heading=heading,
                    text=sub_text,
                    metadata=metadata,
                )
            )
        current_lines = []

    for line in lines:
        if line.startswith("## "):
            flush_section()
            current_h2 = line[3:].strip()
        elif line.startswith("# "):
            # Document level H1 header
            if not current_lines and not current_h2:
                current_h2 = line[2:].strip()
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    flush_section()
    return chunks


def ingest_kb_directory(
    directory_path: Path, max_chunk_chars: int = 1000
) -> List[KBChunk]:
    """Ingest all Markdown files from a knowledge-base directory into KBChunk objects."""
    all_chunks: List[KBChunk] = []
    kb_path = Path(directory_path)
    for file_path in sorted(kb_path.glob("*.md")):
        metadata, body_text = parse_markdown_file(file_path)
        file_chunks = parse_markdown_sections(
            filename=file_path.name,
            body_text=body_text,
            metadata=metadata,
            max_chunk_chars=max_chunk_chars,
        )
        all_chunks.extend(file_chunks)
    return all_chunks
