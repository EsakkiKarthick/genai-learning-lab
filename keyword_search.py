import re
import math

def keyword_search(user_query: str, document_corpus: list[str], top_k: int) -> list[dict]:
    #initialize values
    k1 = 1.2
    b = 0.75
    N = len(document_corpus)
    clean_corpus = []
    doc_lengths = []
    total_corpus_tokens = 0

    for idx, doc in enumerate(document_corpus):
        clean_text = re.sub(r"[^\w\s]",'',doc.strip().lower())
        tokens = clean_text.split(" ")
        clean_corpus.append(tokens)
        doc_lengths.append(len(tokens))
        total_corpus_tokens = total_corpus_tokens + len(tokens)
        #print(f'Tokens:{tokens}')

    avg_doc_length = total_corpus_tokens / N
    #input query normalization
    clean_query = re.sub(r"[^\w\s]",'',user_query.strip().lower())
    query_tokens = clean_query.split(" ")
    print(f'Clean Text:{clean_query}')
    document_scores ={}
    
    for query_token in query_tokens:
        doc_frequency = 0
        for doc in clean_corpus:
            if query_token in doc:
                doc_frequency+= 1
        if doc_frequency == 0:
            continue

        idf = math.log(((N- doc_frequency + 0.5)/(doc_frequency+ 0.5)) + 1.0)

        for idx in range(N):
            doc_tokens = clean_corpus[idx]
            term_frequency = doc_tokens.count(query_token)

            if term_frequency>0:
                doc_length = doc_lengths[idx]

                numerator = term_frequency * (k1 + 1.0)
                denominator = term_frequency + k1 * (1.0-b + b * (doc_length / avg_doc_length) )
                bm25_weight = idf * (numerator/denominator)

                if idx not in document_scores:
                    document_scores[idx] = 0.0
                document_scores[idx] = document_scores[idx] + bm25_weight

    results =[]
    for doc_idx, final_score in document_scores.items():
        results.append({'doc_id': doc_idx, "text": document_corpus[doc_idx], "score": final_score})

    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    #print(f'Results:{sorted_results}')
    return sorted_results

if __name__=='__main__':
    docs = [
        "Error ERR_902: Database connection pool exhausted in production server.",  # Doc 0
        "Database backup process completed successfully without any error.",       # Doc 1
        "Error ERR_404: User dashboard failed to load due to network error.",     # Doc 2
        "Critical production alert: Database CPU utilization reached 98 percent."    # Doc 3
    ]

    results = keyword_search("Database error production",docs, 3)
    print("Keyword Search Results", results)
