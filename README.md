# Multi-Vector Search — Overview and Implementation

## Overview
Multi-vector search, also known as **parent-child retrieval** or **hierarchical chunking search**, splits large documents into overlapping or nested chunks and embeds them separately. When searching, it retrieves semantically similar chunks and maps them back to their parent documents to return full context.

## What is Multi-Vector Search?
- Breaks large documents into smaller, semantically-focused child chunks.
- Embeds and indexes child chunks separately (each chunk gets its own vector).
- Parent documents are stored separately (one per original document) for context.
- Query retrieves top-N child chunks by semantic similarity, deduplicates by parent, and returns top-K unique parents with child text excerpts.

## Common Uses
- **Long-form document retrieval** (e.g., manuals, runbooks, research papers) where a small excerpt is more relevant than the whole document.
- **Passage-level ranking** in question-answering systems.
- **Technical documentation search** where specific sections matter more than full pages.
- **Reducing noise** when documents are large but queries are specific to sub-sections.
- **Hybrid retrieval** as a first-stage retriever before re-ranking with full documents.

## Limitations and Trade-offs
- **Indexing overhead**: Requires more storage and computation since each chunk becomes an independent vector.
- **Relevance fragmentation**: A query might match a child chunk from an irrelevant document, especially with loose chunking strategies.
- **Chunk boundary issues**: Important information split across chunk boundaries can be missed.
- **Memory cost**: Storing vectors for thousands of small chunks scales linearly with chunk count.
- **Staleness**: If parent documents are large, returning the full parent may include stale or outdated sections alongside the relevant child.
- **Hyperparameter sensitivity**: Chunk size and overlap significantly affect quality; tuning required per use case.

## How `multivector_search.py` Works (Implementation Details)

### 1. **Ingestion Phase: `ingestParentChildCorpus()`**
   - Splits the input corpus by double newlines (`"\n\n"`) into parent documents.
   - For each parent:
     - Uses `smart_chunking()` with `chunk_size=1000` to create parent chunks (rejoined as full parent text).
     - Further subdivides parent chunks into child chunks with `chunk_size=300`.
     - Encodes each child chunk into a dense vector using `SentenceTransformer('all-MiniLM-L6-V2')`.
   - Returns:
     - `parent_doc_store`: dict mapping `parent_id` → full parent text.
     - `child_vector_store`: list of dicts with `child_id`, `child_text`, `parent_id`, and `embeddings`.

### 2. **Smart Chunking: `smart_chunking()`**
   - Splits text by sentence boundaries (`.`, `!`, `?`).
   - Accumulates sentences into chunks while respecting `chunk_size` limit.
   - Handles edge cases where a single sentence exceeds chunk size (splits it word-by-word).
   - Returns a list of non-empty, stripped chunks.

### 3. **Search Phase: `parent_child_search()`**
   - **Step 1**: Encode the user query into a vector.
   - **Step 2**: Compute cosine similarity between query and all child vectors:
     $$\text{cosine}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|}$$
   - **Step 3**: Sort child chunks by similarity score (descending).
   - **Step 4**: Deduplicate by parent ID:
     - Iterate through top-N child results (default `fetch_child_n=10`).
     - For each unique parent, add one entry (with its matching child text and score).
     - Stop when top-K unique parents are retrieved (default `top_k=3`).
   - **Returns**: List of dicts with `parent_id`, `parent_text`, `child_text`, and `score`.

### 4. **Key Data Flow**
```
Input Corpus
    ↓
Split by "\n\n" into Parents
    ↓
Parent → smart_chunking(1000) → Parent Chunks (rejoined as full parent)
    ↓
Parent Chunks → smart_chunking(300) → Child Chunks
    ↓
Embed each Child Chunk → Child Vector Store
    ↓
Query → Encode → Cosine Similarity with all Child Vectors
    ↓
Sort Children by Score → Deduplicate by Parent → Return Top-K Parents
```

## Example Execution
The script includes an example in the `if __name__ == '__main__'` block:
- Ingests a corpus of technical runbooks (database failover, firewall config, resource diagnostics).
- Runs two queries:
  1. `"What is the exact CLI command to promote worker 09?"` — Retrieves child chunks about failover commands.
  2. `"database connection idle timeout and cluster cache flush instructions"` — Retrieves relevant database and cache sections.
- Each query returns top-3 unique parents with their matching child excerpts and cosine similarity scores.

## Practical Recommendations
- **Chunk size tuning**: Start with 300–500 tokens for child chunks; 1000–2000 for parents. Adjust based on query specificity.
- **Overlap strategy**: Add sliding-window overlap between chunks to preserve context at boundaries.
- **Batch encoding**: Use `model.encode(list_of_texts)` for faster bulk embedding during ingestion.
- **Deduplication strategy**: The current approach keeps the first (highest-scoring) child per parent. Alternative: average scores, or keep multiple child excerpts per parent.
- **Large-scale indexing**: Use FAISS or similar ANN indexes for retrieving top-N children efficiently.
- **Hybrid approach**: Combine BM25 (keyword filtering) with multi-vector search (semantic ranking) for robustness.
- **Metadata preservation**: Add metadata (source file, section header, timestamp) to child chunks for traceability.
