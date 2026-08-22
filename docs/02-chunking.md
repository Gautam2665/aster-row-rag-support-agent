# 02. Document Segmentation & Semantic Section Chunking

## 1. What is a Document?
A **document** is a full source file (such as `01-returns-policy-current.md`). It contains structural markup, metadata frontmatter, headings, lists, tables, and narrative paragraphs covering multiple sub-topics.

## 2. What is a Chunk?
A **chunk** is a discrete, semantically self-contained subsection extracted from a document. In our system, chunks are represented by the [`KBChunk`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py#L47-L86) class.

```text
Document (01-returns-policy-current.md)
  ├── Heading Section 1 ──► KBChunk (Standard return window)
  ├── Heading Section 2 ──► KBChunk (Item condition)
  └── Heading Section 3 ──► KBChunk (Return shipping and refunds)
```

## 3. Why Chunk?
Large Language Models and Vector Databases work best with focused, granular pieces of text. Passing an entire 10-page document for a single question dilute vector similarity and wastes context tokens.

## 4. The Whole-Document Embedding Problem
If an entire 10-page policy document is embedded into a single vector, its mathematical representation becomes an "average" of returns, shipping, warranty, and corporate history. The specific vector signal for *"30-day return limit"* gets diluted, making retrieval inaccurate.

---

## 5. Fixed-Size Chunking
Fixed-size chunking splits text every $N$ characters or tokens (e.g. 500 characters with 50 character overlap) blindly across the document string.

### Example:
```text
Original Text: "## Standard Return Window\nCustomers may return items within 30 days."
Fixed-Size Chunk 1: "## Standard Return Window\nCust"
Fixed-Size Chunk 2: "omers may return items within 30 days."
```

## 6. Problems with Fixed-Size Chunking
1. **Broken Structure**: Cuts sentences, lists, and markdown tables in half.
2. **Context Loss**: Separates headings from the paragraphs they describe.
3. **Citation Failure**: Cannot accurately attribute text to a specific heading section.

---

## 7. Semantic Chunking
Semantic chunking respects document structure and splits text along natural boundaries (such as paragraphs, sentences, or Markdown section headers) so that each chunk represents a single coherent concept.

## 8. Markdown Heading-Aware Chunking
Our ingestion engine ([`src/ingestion.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/ingestion.py#L82-L147)) uses heading-aware chunking. It parses Markdown headers (`#`, `##`, `###`) and groups all content under a header into a unified `KBChunk`.

## 9. Heading Hierarchy & Citation Format
When parsing `## Standard return window` inside `# Returns Policy`, our chunking algorithm builds an explicit citation heading:

$$\text{Heading Citation} = \text{Document Title} + \text{" > "} + \text{Section Heading}$$
$$\text{Example} = \text{"Returns Policy > Standard return window"}$$

This enables our support agent to present precise citations (`01-returns-policy-current.md > Standard return window`).

---

## 10. Large Sections & 11. Chunk Subdivision
If a single Markdown section is excessively long (e.g., > 1,000 characters), storing it as a single chunk degrades vector focus. 

Our algorithm ([`src/ingestion.py:split_section_text`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/ingestion.py#L38-L80)) subdivides large sections at paragraph boundaries (`\n\n`) or line breaks while **preserving parent heading context and document metadata across all sub-chunks**.

```text
Large Markdown Section (>1000 chars)
  ├── Sub-Chunk 1 (ID: doc.md#heading-part1) ──► Retains Heading + DocumentMetadata
  └── Sub-Chunk 2 (ID: doc.md#heading-part2) ──► Retains Heading + DocumentMetadata
```

## 12. Chunk Overlap
When subdividing long paragraphs, a small overlap (e.g. 50–100 characters) ensures boundary sentences are not cut off in a way that alters meaning.

## 13. Parent-Child Context
Sub-chunks retain reference to their parent document metadata (`document_id`, `status`, `supersedes`) so filtering rules apply equally to all chunks extracted from that document.

## 14. Metadata Attached to Chunks
Every [`KBChunk`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py#L47-L86) carries its full [`DocumentMetadata`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/models.py#L4-L44) payload:
* `filename`: `"01-returns-policy-current.md"`
* `heading`: `"Returns Policy > Standard return window"`
* `status`: `"active"`
* `audience`: `"customer"`
* `policy_authority`: `"official"`
* `supersedes`: `"RET-2024-01"`
* `customer_answering`: `True`

## 15. Multi-Document Retrieval
By chunking documents individually while attaching metadata, our vector store can seamlessly retrieve and rank chunks originating from multiple distinct files in a single search query.

---

## 16. How Our Implementation Works ([`src/ingestion.py`](file:///c:/Users/HP/OneDrive/Desktop/ai-intern-test/ai-agent-intern-test/src/ingestion.py))

```python
# 1. Parse frontmatter YAML delimiter
metadata, body_text = parse_markdown_file(file_path)

# 2. Extract sections by Markdown headers (##)
chunks = parse_markdown_sections(
    filename=file_path.name,
    body_text=body_text,
    metadata=metadata,
    max_chunk_chars=1000
)
```

---

## 17. Interview Questions & Answers

### Q1: What is the difference between fixed-size and semantic chunking?
> **Answer**: Fixed-size chunking splits text every $N$ characters without regard for structure, often cutting sentences or headings in half. Semantic section chunking splits along structural boundaries (e.g. Markdown headers `##`), keeping headings, paragraphs, and tables intact as logical units.

### Q2: How do you handle a section that is too large for a single chunk?
> **Answer**: We perform hierarchical subdivision. The section is split at paragraph boundaries (`\n\n`), while ensuring every sub-chunk inherits the parent section heading, source filename, and complete document metadata.
