# Semantic Search — Overview and Implementation

## Overview
Semantic search finds documents that are relevant to a user's intent, not just documents that contain the exact keywords. Instead of matching words directly, it matches the meaning of the query and documents using vector representations (embeddings).

## What is Semantic Search?
- Converts text (queries and documents) into dense vectors using a pretrained embedding model.
- Measures similarity between query and document vectors (typically with cosine similarity) to rank results by semantic relevance.
- Unlike keyword search, semantic search captures synonyms, paraphrases, and contextual meaning.

## Common Uses
- Question answering and FAQ retrieval
- Document and knowledge-base search
- Recommendation and related-content ranking
- First-stage retrieval in hybrid systems (embedding-based retrieval + exact-match re-ranking)

## Limitations and Trade-offs
- Requires an embedding model and extra compute to encode text (CPU/GPU cost).
- Embeddings are approximate: they may miss fine-grained lexical signals (dates, IDs, exact phrases).
- Semantic models can hallucinate relevance in edge cases and may need domain-specific fine-tuning.
- Large collections need approximate nearest neighbor indexes (e.g., FAISS, Annoy) for speed and memory efficiency.
- Privacy and storage: embeddings may leak sensitive information if not handled securely.

## How `semantic_search.py` Works (implementation notes)

- Model: The script uses `SentenceTransformer('all-MiniLM-L6-v2')` to produce embeddings for both documents and queries.

- Embedding generation: `embeddingGeneration(raw_document)` encodes each document and builds a simple in-memory `vector_DB` with entries:

  - `doc_id`: integer id
  - `text`: original document text
  - `embeddings`: dense vector (NumPy array)

- Querying: `semanticSearch(user_query, vector_db, top_k=3)`:
  1. Encodes the `user_query` with the same model to a `query_vector`.
  2. Computes cosine similarity between `query_vector` and every document vector:

     $$\text{cosine}(q,d)=\frac{q\cdot d}{\|q\|\,\|d\|}$$

     The script computes this explicitly by calculating the dot product and magnitudes.
  3. Builds a scored list and returns the top-K results sorted by score (highest first).

## Practical Recommendations
- Use batch encoding (`model.encode(list_of_texts)`) when indexing many documents to speed up embedding generation.
- Pre-normalize vectors (divide by their L2-norm) so cosine similarity reduces to a dot product; this speeds up similarity computation.
- For larger datasets, persist embeddings and use an ANN index (FAISS, HNSW) instead of linear scan.
- Consider hybrid retrieval: run BM25 or keyword filters first, then re-rank a small candidate set with semantic similarity.
- Tune or fine-tune the embedding model for domain-specific language to improve accuracy.

## Example
- The script includes a small example in the `if __name__ == '__main__'` block that encodes three sample documents and runs a query `"database connection issue"` to demonstrate the end-to-end flow.

# Semantic Search Architecture & Mathematical Engine

## How Semantic Search Works Under the Hood

The semantic search architecture operates in two main phases: **Document Ingestion (Offline)** and **Query Execution (Online)**.

```
[ OFFLINE INGESTION PHASE ]
  Raw Document Corpus ──> [ Embedding Model ] ──> Dense Vectors ──> [ Vector Store / Database ]

[ ONLINE SEARCH PHASE ]
  User Query ─────────> [ Embedding Model ] ──> Query Vector ───┐
                                                                │
  Candidate Results <── [ Top-K Sort ] <── [ Similarity Math ] ◄┘
```

---

## Phase 1: Ingestion & Vector Indexing (Offline)

1. **Document Chunking:** Long documents are broken down into smaller, coherent text passages or sentences.
2. **Dense Vector Embedding Generation:** Each text chunk is passed through a deep learning encoder model (typically a **Transformer-based model** like `Sentence-BERT`, `OpenAI text-embedding-3`, or `all-MiniLM-L6-v2`).
   - The model converts the text string into a fixed-length list of floating-point numbers (e.g., an array of 384, 768, or 1536 dimensions).
   - Each dimension in the vector represents an abstract semantic feature (e.g., sentiment, subject matter, domain, grammar context).
3. **Indexing:** These vectors are stored alongside their raw text payload and metadata in a specialized **Vector Database** (e.g., ChromaDB, FAISS, Pinecone, or Qdrant).

---

## Phase 2: Query Processing & Matching (Online)

1. **Query Encoding:** The incoming user search query is passed through the **exact same embedding model** to produce a `query_vector` in the same vector space.
2. **Similarity Measurement:** The vector engine calculates the distance/angle between the `query_vector` and all indexed document vectors.
3. **Top-K Ranking:** The candidates are sorted by similarity score in descending order, and the top $K$ items are returned.

---

## The Mathematical Engine: Cosine Similarity

To determine how "close" or semantically aligned two pieces of text are, systems typically calculate **Cosine Similarity**. Cosine similarity measures the cosine of the angle $\theta$ between two vectors in multi-dimensional space:

$$\text{Cosine Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

* **Angle near $0^\circ$ ($\text{Similarity} \approx 1.0$):** High semantic alignment (very close meaning).
* **Angle near $90^\circ$ ($\text{Similarity} \approx 0.0$):** No semantic relationship.
* **Angle near $180^\circ$ ($\text{Similarity} \approx -1.0$):** Opposite meanings.
