# Keyword Search and BM25 Retrieval

## Overview
This repository implements a simple keyword search flow using the BM25 ranking function to retrieve and rank documents by relevance to a user query.

## What is Keyword Search?
Keyword search finds documents that contain the query terms and ranks them by how well they match the query. It's commonly used for:
- Search engines and site search
- Information retrieval over document collections
- FAQ and support ticket retrieval
- Code search and log search

Keyword search is fast and interpretable: matches are based on term overlap and term importance.

## Where It's Used
- Web search and site search
- Enterprise search across documents, emails, and knowledge-bases
- Retrieval in hybrid systems (first-stage retrieval before semantic re-ranking)

## BM25 Flow (How retrieval works)
BM25 is a probabilistic, term-weighted ranking function used to score documents for a query. The high-level flow:

1. Preprocess documents and queries (tokenize, normalize, optional stopword removal).
2. Build an inverted index with term frequencies and document lengths.
3. For a query, compute BM25 scores for each candidate document.
4. Rank documents by score and return the top-K results.

### BM25 Formula
## 1. The Core Basis: Document Frequency ($DF$)

Before calculating scores, the system counts **how many individual documents contain a specific term at least once**. This count is defined as $DF$:

$$\text{Document Frequency } (DF) = \text{Count of distinct documents where term } t \text{ exists}$$

* **High $DF$** $\rightarrow$ Term appears in many documents $\rightarrow$ **Common / Non-informative term**
* **Low $DF$** $\rightarrow$ Term appears in very few documents $\rightarrow$ **Rare / Highly informative term**

---

## 2. Walkthrough of the Logic using the Previous Example

In the dataset from the previous example, we had **$N = 4$ total documents** in the corpus:

* **Doc 0:** `"Error ERR_902: Database connection pool exhausted in production server."`
* **Doc 1:** `"Database backup process completed successfully without any error."`
* **Doc 2:** `"Error ERR_404: User dashboard failed to load due to network error."`
* **Doc 3:** `"Critical production alert: Database CPU utilization reached 98 percent."`

When the query `"Database error production"` is executed, the algorithm evaluates $DF$ for each term across all 4 documents:

| Query Term | Documents Containing the Term | Document Frequency ($DF$) | Rarity Assessment |
| --- | --- | --- | --- |
| `"database"` | Doc 0, Doc 1, Doc 3 | **$DF = 3$** (appears in 3 out of 4 docs) | Common term |
| `"error"` | Doc 0, Doc 1, Doc 2 | **$DF = 3$** (appears in 3 out of 4 docs) | Common term |
| `"production"` | Doc 0, Doc 3 | **$DF = 2$** (appears in 2 out of 4 docs) | **Rarest term** |

---

## 3. How the Mathematical Formula Converts $DF$ into a Score

Because $DF$ is in the **denominator** of the IDF formula, a smaller $DF$ produces a larger fraction, which yields a higher final IDF score:

$$\text{IDF}(t) = \ln\left(\frac{N - DF + 0.5}{DF + 0.5} + 1.0\right)$$

Plugging in the numbers for $N = 4$:

### For `"database"` and `"error"` ($DF = 3$):

$$\text{IDF} = \ln\left(\frac{4 - 3 + 0.5}{3 + 0.5} + 1.0\right) = \ln(1.428) \approx \mathbf{0.3567}$$

### For `"production"` ($DF = 2$):

$$\text{IDF} = \ln\left(\frac{4 - 2 + 0.5}{2 + 0.5} + 1.0\right) = \ln(2.0) \approx \mathbf{0.6931}$$

## Implementation Notes
- The main entry for this flow is [keyword_search.py](keyword_search.py).
- Results and quick outputs may be written to `output.txt` for inspection.

## Running the Example
Activate your virtual environment and run:

```bash
python keyword_search.py
```

The script will load or index the sample documents, accept a query (or use a hard-coded example), compute BM25 scores, and print the top matches.

## Tuning and Extensions
- Tune `$k_1$` and `$b$` to your collection (short documents prefer lower $b$).
- Add query expansion, stemming, or synonyms to improve recall.
- Use BM25 as the first-stage retriever, then apply embedding-based re-ranking for semantic correctness.

## References
- Robertson, S., Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond".
- Practical IR guides and libraries (e.g., Lucene, Whoosh, Elasticsearch) for production-grade implementations.
