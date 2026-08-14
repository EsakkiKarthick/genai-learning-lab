from semantic_search import semanticSearch, embeddingGeneration
from keyword_search import keyword_search

def hybrid_seach(user_query: str, document_corpus: list[str], top_k: int=5, rrf_k: int = 60, fetch_top_n: int = 20):
    #keyword search
    sparse_results = keyword_search(user_query, document_corpus, fetch_top_n) 
    #Semantic Search
    vector_db = embeddingGeneration(document_corpus) #vector data generation for document corpus
    dense_results = semanticSearch(user_query, vector_db, fetch_top_n)

    scores = {}

    for rank_index, record in enumerate(sparse_results):
        doc_id = record['doc_id']
        doc_text = record['text']

        rank = rank_index +1
        rrf_weight = round(1.0 / (rrf_k + rank), 6)

        scores[doc_id]= {
            "doc_id": doc_id,
            "text": doc_text,
            "rrf_score": rrf_weight
            }

    for rank_index, record in enumerate(dense_results):
        doc_id = record['doc_id']
        doc_text = record['text']

        rank = rank_index+1
        rrf_weight = round(1.0 / (rrf_k + rank), 6)
        #doc_id_exists = any(d.get('doc_id')==doc_id for d in scores)
        if doc_id in scores:
            scores[doc_id]['rrf_score'] = scores[doc_id]['rrf_score'] + rrf_weight
        else:
            scores[doc_id]= {
                "doc_id": doc_id,
                "text": doc_text,
                "rrf_score": rrf_weight
            }
    list_scores = list(scores.values())
    results = sorted(list_scores, key=lambda item:item['rrf_score'], reverse= True)
    return results[:top_k]

if __name__ == '__main__':
    document_corpus = [
    # Doc 0: Exact Technical SKU & Alphanumeric Code (Keyword Edge Case)
    "Critical System Error: Code ERR_502_DB_CONN occurred on cluster node worker-09 during query execution.",

    # Doc 1: Semantic Intent without Overlapping Keywords (Dense Vector Edge Case)
    "The primary database server ran out of memory, causing all active client connections to drop immediately.",

    # Doc 2: High Keyword Overlapping False Positive (BM25 Distractor Edge Case)
    "How to configure connection parameters for the database pool in production server settings.",

    # Doc 3: Compound Technical Query (Hybrid Match Required)
    "To resolve connection drops on worker nodes, restart the background query daemon and clear the database cache.",

    # Doc 4: Vocabulary Mismatch / Synonyms Only (Dense Vector Edge Case)
    "High memory consumption caused the primary analytical storage node to terminate unexpectedly.",

    # Doc 5: Punctuation & Sub-word Tokenization Edge Case (Regex / Normalization)
    "Error ERR-502-DB-CONN: Database connection pool exhausted."
    ]

    query = ['ERR_502_DB_CONN worker-09','server out of RAM causing client disconnects', 'ERR_502_DB_CONN connection drops']
    #query = 'ERR_502_DB_CONN connection drops'
    #results = hybrid_seach(query, document_corpus)
    #print(results)
    for q in query:
        results = hybrid_seach(q, document_corpus)
        print(results)