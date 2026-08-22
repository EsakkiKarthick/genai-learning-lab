# ColBERT Reranking

## Overview

ColBERT (Contextualized Late Interaction over BERT) is a neural retrieval and ranking approach that represents the query and document separately, but keeps multiple vectors for each text. Instead of compressing an entire document into one vector, it preserves token-level information and compares the query with the document later, during search.

This project uses the ColBERT-compatible model `answerdotai/answerai-colbert-small-v1` through `MultiVectorEncoder` from `sentence-transformers`. The implementation is in `colbert_rerank.py`.

## How `colbert_rerank.py` Works

The `rerankwithcolbert()` function accepts a query and a list of documents. Each document must contain `doc_id` and `text` fields.

The flow is:

1. Encode the query into multiple token-level vectors.
2. Extract the text from each candidate document.
3. Encode every candidate document into multiple token-level vectors.
4. Compute ColBERT similarity between the query vectors and document vectors.
5. Attach each score to its original document.
6. Sort documents by score in descending order and return the first `top_k` results.

```text
User query
    |
    v
ColBERT query encoder
    |
    v
Query token vectors
    |
    +----------------------------+
    |                            |
    v                            v
Candidate document 1       Candidate document 2 ...
    |                            |
    v                            v
Document token vectors      Document token vectors
    \                            /
     +------ late interaction -+
                    |
                    v
             Ranked documents
```

### Late Interaction

For a query with token vectors $q_i$ and a document with token vectors $d_j$, ColBERT commonly calculates a MaxSim score like this:

$$
\mathrm{Score}(q,d) = \sum_i \max_j \mathrm{sim}(q_i,d_j)
$$

In other words, each query token finds its strongest matching document token, and those best matches are combined into a document score. This preserves more fine-grained matching information than a single vector for the whole text while allowing the query and documents to be encoded independently.

The call to `model.similarity(query_embedding, doc_embeddings)` delegates this multi-vector comparison to the model implementation.

## Search Pipeline

ColBERT is most useful as a second-stage ranker:

1. Use keyword search, semantic search, or hybrid search to retrieve a manageable candidate set.
2. Pass only those candidates to `rerankwithcolbert()`.
3. Use the returned ordering for the search response or as context for a downstream RAG system.

The current function also encodes the candidate documents every time it is called. For a real system, document vectors should be generated during indexing, stored, and reused for queries.

## ColBERT Compared with a Cross-Encoder

| Aspect | ColBERT | Cross-encoder |
|---|---|---|
| Encoding | Encodes query and document independently | Encodes the query and document together |
| Representation | Multiple vectors per text, usually one per token | A joint representation for the text pair |
| Interaction | Late interaction during scoring | Full token-level attention while encoding the pair |
| Document reuse | Document vectors can be precomputed and indexed | Each query-document pair must be evaluated again |
| Speed at scale | Faster and more scalable than a cross-encoder, though heavier than single-vector retrieval | Usually slower and more expensive per candidate |
| Ranking quality | Strong fine-grained matching with independent encoding | Often strongest for a small candidate set because it sees both texts jointly |
| Typical role | Retrieval or scalable second-stage ranking | Final reranking of a small candidate list |

A cross-encoder receives pairs such as `[query, document]` and directly predicts relevance. It can model interactions such as negation and detailed phrasing very well, but its document representation cannot be reused across different queries.

ColBERT separates encoding from interaction. This makes document representations reusable and supports large collections, while its token-level vectors retain more detail than a traditional single-vector bi-encoder. The trade-off is that ColBERT needs more storage and a more involved multi-vector index.

## Real-World Uses

### Retrieval-Augmented Generation

- Retrieve high-quality passages before sending context to an LLM.
- Improve the ordering of passages supplied to a question-answering prompt.
- Reduce irrelevant context when several documents contain similar terminology.

### Legal, Scientific, and Financial Research

- Find relevant clauses, passages, or evidence in long collections.
- Rank papers or reports against a detailed research question.
- Search filings and reports where important matches may be buried in a longer passage.

## Advantages

- **Fine-grained matching:** Token-level vectors can identify several important local matches within a passage.
- **Independent encoding:** Document vectors can be computed once and reused for many queries.
- **Better scalability than cross-encoding every document:** Candidate scoring can use an index rather than evaluating every pair with full joint attention.
- **Useful for long or detailed queries:** Different query terms can match different parts of a document.
- **Works with a first-stage retriever:** It can improve keyword, semantic, or hybrid retrieval results.

## Limitations

- **Higher storage requirements:** A document has multiple vectors instead of one, so the index can be much larger.
- **More complex infrastructure:** Efficient deployment usually needs a multi-vector index and specialized retrieval or scoring support.
- **Still requires candidate coverage:** If the first-stage retriever never returns a relevant document, ColBERT cannot rerank it into the result set.
- **More expensive than single-vector search:** Late interaction across many token vectors costs more than one dot product per document.
- **Candidate count affects latency:** Passing too many documents to the reranker reduces the benefit of a staged search pipeline.
- **Input length limits:** Long documents may need chunking. Chunk boundaries can separate a question from the context needed to answer it.
- **Model-domain mismatch:** A general-purpose checkpoint may perform poorly on specialized medical, legal, financial, or organization-specific language.
- **Scores need evaluation:** Similarity values are useful for ranking but are not automatically calibrated probabilities or universal relevance thresholds.
- **No explanation by default:** The score does not provide a human-readable reason for why a document was ranked highly.
- **Runtime dependency:** The model must be downloaded and loaded successfully, and production systems need monitoring for memory use, latency, and model changes.

## Practical Recommendations

- Use keyword, semantic, or hybrid retrieval to produce a limited candidate set before ColBERT scoring.
- Precompute and persist document multi-vectors during ingestion.
- Chunk long documents with enough context to preserve meaning across boundaries.
- Measure recall from the first-stage retriever separately from ranking quality after ColBERT.
- Tune candidate count and `top_k` against latency and relevance metrics.
- Evaluate on representative domain queries before deploying the general-purpose model.
- Consider a cross-encoder for a very small final candidate set when maximum ranking quality matters more than latency.
