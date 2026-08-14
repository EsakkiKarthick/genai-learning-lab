**Hybrid Search — README**

Overview
--------

Hybrid search combines lexical (term-based) and semantic (vector-based) retrieval to leverage the strengths of both: precise keyword matches from BM25-like systems and concept-level matching from dense vector search. The hybrid approach returns a merged ranking that improves recall for semantically relevant documents while preserving precision on exact matches.

Why hybrid search is required
---------------------------

- Keyword-only systems (BM25/BOW) excel at exact term matching and are fast, but miss paraphrases and semantic equivalence.
- Semantic/vector systems find conceptually similar texts even if surface terms differ, but can return loosely related or topically drifted items and sometimes miss exact, high-precision matches.
- Combining both reduces blindspots: get exact hits and semantically relevant results in a single ranked list.

How hybrid rankings are commonly combined
----------------------------------------

There are multiple strategies for combining lexical and semantic scores. Two common families are:

- Score fusion (normalize scores then combine linearly or nonlinearly).
- Rank fusion (use ranks from each method and combine them; Robust Rank Fusion is a popular rank fusion technique).

Robust Rank Fusion (RRF) — under the hood
-----------------------------------------

RRF is a simple but effective rank-based ensembling method. For each document, given its rank $r_i$ in each ranking source $i$, RRF computes:

$$\text{RRF\_score}(d) = \sum_{i=1}^m \frac{1}{k + r_i(d)}$$

where:
- $m$ is the number of ranking sources (e.g., BM25 and semantic)
- $r_i(d)$ is the 1-based rank of document $d$ in source $i$ (use a large value or ignore if not returned)
- $k$ is a constant that controls how much weight top ranks receive (common default: $k=60$)

Key properties and intuition:
- The harmonic-like term $1/(k+r)$ gives a strong boost to documents that are top-ranked in at least one source while still allowing contributions from other sources.
- Larger $k$ makes the function flatter (less emphasis on top ranks); smaller $k$ concentrates score on the very top results.
- RRF uses ranks instead of raw scores, so it’s robust to incompatible score scales between systems.

Practical RRF details
---------------------

- Handling missing docs: if a source does not return a doc, treat its rank as a large constant (e.g., 1e6) or skip that source for the doc.
- Ties: if two docs have the same rank from one source (rare for deterministic systems), pick any consistent tie-breaker (e.g., secondary rank from other source or doc id).
- Choosing `k`: try values between 10 and 100; 60 is a commonly used starting point. Lower `k` favours top results strongly; higher `k` smooths contributions.
- Number of top results to fuse: compute ranks only for top-N from each source (e.g., 100–1000) for efficiency; treat unseen docs as missing.

Implementation notes (score fusion vs RRF)
---------------------------------------

- Score fusion requires normalizing heterogeneous score ranges (min-max, z-score, logistic). Incorrect normalization can bias the ensemble.
- RRF avoids normalization issues by fusing ranks, making it simpler and more robust when sources have different scoring semantics.
- If you need to give semantic search more or less influence, you can either change its rank list size or use weighted RRF variants (multiply each 1/(k+r) term by a source weight).

Edge cases and gotchas
----------------------

- Score scale mismatch: if combining raw scores, always inspect distributions; outliers can dominate linear combinations.
- Empty or very short results from one source: if semantic retrieval returns very few hits, treat missing ranks carefully.
- Duplicate documents: deduplicate by canonical id before fusing ranks to avoid double-counting.
- Document set mismatch: when sources index different corpora or fields, ensure you align document ids.
- Stability: since RRF uses ranks, small score perturbations rarely change low-ranked docs, but can reorder close top results—use tie-breakers if reproducibility matters.
- Latency and cost: running two retrievals (BM25 + vector) doubles query-time work; consider caching, approximate nearest neighbor settings (ef/search_k), or using smaller top-N fusion windows.

Quick usage guidance
--------------------

- Run lexical search (BM25) to get top-N ranked doc ids and ranks.
- Run semantic/vector search to get top-M ranked doc ids and ranks.
- Compute RRF score per document using the ranks and chosen `k`.
- Sort by RRF score descending and return the merged top-K.

References & further reading
----------------------------

- Cormack, N., & Clarke, C. (2009). Fusion of ranked lists using reciprocal rank fusion (RRF).
- Practical posts and tutorials on hybrid search and rank fusion available in retrieval literature and applied-search blogs.
