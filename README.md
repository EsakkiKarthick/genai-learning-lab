# Cross-Encoder Search

## Overview

A **cross-encoder** is a neural ranking model that reads a pair of texts together and predicts how relevant one text is to the other. In search systems, the pair is usually:

- a user query
- a candidate document or passage

Unlike a bi-encoder, which embeds queries and documents independently, a cross-encoder can compare the words and meaning of both inputs directly. This usually produces more accurate relevance scores, especially when the query and document contain subtle relationships.

## Why the Encoder Matters

The encoder is the part of the model that converts language into a representation that can be used for prediction. In a cross-encoder, the query and document are processed together, allowing the model to pay attention to interactions such as:

- matching terms with their surrounding context
- negation and qualifiers, such as `not supported`
- whether a document actually answers the query
- relationships between concepts expressed with different wording

This interaction is the main reason cross-encoders often rank results better than simple keyword matching or independent vector similarity. The model is not just asking whether two texts look similar; it is estimating how well the document satisfies the query.

## How This Project Uses It

`cross_encoder.py` uses the `CrossEncoder` class from `sentence-transformers` with the model `cross-encoder/ms-marco-MiniLM-L6-v2`.

The search pipeline is:

1. Semantic search retrieves an initial set of candidate documents.
2. Each candidate is paired with the query as `[query, document_text]`.
3. The cross-encoder predicts a relevance score for every pair.
4. The raw score is passed through a sigmoid function to produce a value between 0 and 1.
5. Candidates are sorted by score and the top results are returned.

```text
User query
    |
    v
First-stage semantic search
    |
    v
Candidate documents
    |
    v
Cross-encoder scores [query, document]
    |
    v
Sorted top-k results
```

This is called **re-ranking**. The cross-encoder does not need to compare the query against every document in the collection; it only evaluates the smaller candidate set returned by the first search stage.

## Real-World Applications

### Search and Retrieval

- Improve the ordering of results in documentation, knowledge-base, and website search.
- Re-rank passages retrieved for retrieval-augmented generation (RAG).
- Find the most relevant policy, support article, or technical runbook for a question.

## Advantages

- **High ranking quality:** Direct query-document attention can capture meaning that independent embeddings miss.
- **Good for nuanced queries:** It can distinguish between similar passages with different answers or conditions.
- **Works well as a second stage:** It improves an existing retriever without requiring every document to be scored by the expensive model.
- **Simple integration:** The model accepts text pairs and returns relevance scores.

## Limitations

- **Slower than bi-encoders:** Every query-document pair requires a separate model evaluation. Scoring a large collection directly is usually too expensive.
- **Requires a candidate retriever:** It can only re-rank documents that the first-stage search returns. A relevant document missing from that set cannot be recovered.
- **Higher compute cost:** Inference uses more CPU/GPU time and memory than keyword scoring or precomputed vector similarity.
- **Latency grows with candidate count:** Increasing the number of candidates improves recall opportunities but also increases reranking latency.
- **Model and domain bias:** A model trained on a general search dataset may perform poorly on specialized legal, medical, financial, or internal terminology without evaluation or fine-tuning.
- **Input length limits:** Very long documents may need to be chunked. Important context can be lost if relevant information is split across chunks.
- **Scores are not universal probabilities:** The sigmoid output is useful for ordering, but it should not automatically be interpreted as a calibrated probability or compared across unrelated models.
- **No explanation by default:** The score indicates relevance, but the model does not explain which parts of the query or document affected the decision.
- **Operational dependency:** The model must be downloaded and available in the runtime environment, and production deployments need monitoring for model version, latency, and quality changes.

## Practical Guidance

Use a fast retriever such as keyword search, semantic search, or hybrid search to produce a moderate candidate set, then use the cross-encoder to re-rank it. Start with a small `top_k` and measure both relevance quality and response latency.

For production use, evaluate the model on representative queries. Track metrics such as recall at the candidate-retrieval stage, precision or NDCG after reranking, latency, and the rate of unanswered queries. Consider domain-specific fine-tuning when the general-purpose model does not understand the terminology of the target collection.
