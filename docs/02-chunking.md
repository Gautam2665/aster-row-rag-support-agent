# 02. Markdown Chunking & Heading Hierarchy

## Chunking Strategies: Fixed-Size vs. Semantic Section Chunking

| Strategy | Approach | Drawbacks |
|---|---|---|
| **Fixed-Character Window** | Splitting text every $N$ characters (e.g. 500 chars with 50 char overlap) regardless of structure. | Cuts sentences in half, separates headings from their paragraphs, splits tables/lists, loses context. |
| **Semantic Section Chunking** | Parsing document structure by Markdown headers (`#`, `##`, `###`). | Respects topic boundaries, keeps lists/tables together under their section heading. |

---

## Heading Hierarchy Preservation

In Markdown documents, headings define the topic scope. Our chunking engine parses Markdown headers to construct explicit section heading paths:

```text
Document Title: "Returns Policy"
Header: "## Standard return window"
Citation Heading: "Returns Policy > Standard return window"
```

### Why Preserving Heading Context Matters
1. **Citation Generation**: Enables the agent to output exact citations (`01-returns-policy-current.md > Standard return window`).
2. **Context Retention**: When a paragraph says *"Items must be unused"*, keeping the heading *"Standard return window"* ensures the vector embedding captures what topic the item condition applies to.

---

## Subdividing Large Sections

If a single Markdown section is excessively long (e.g., exceeding 1,000 characters):
* The algorithm subdivides the section text at natural paragraph boundaries (`\n\n`) or line breaks.
* **Crucially**, every sub-chunk inherits:
  1. The exact parent `heading` path.
  2. The source `filename`.
  3. The complete document `DocumentMetadata` (frontmatter attributes).
  4. A deterministic chunk ID (e.g. `01-returns-policy-current.md#standard-return-window-part1`).
