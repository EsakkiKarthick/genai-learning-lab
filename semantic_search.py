from sentence_transformers import SentenceTransformer
import numpy as np
import math

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
model = SentenceTransformer(EMBEDDING_MODEL)

def embeddingGeneration(raw_document: list[str]):
    #model = SentenceTransformer(embedding_model)
    vector_DB = []
    for doc_id, text in enumerate(raw_document):
        doc_vector= model.encode(text)
        vector_data = {
                "doc_id": doc_id,
                "text": text,
                "embeddings": doc_vector

            }
        vector_DB.append(vector_data)
    return vector_DB

def semanticSearch(user_query: str, vector_db: list[dict], top_k:int = 3)-> list[dict]:
    #model = SentenceTransformer(embedding_model)
    query_vector = model.encode(user_query)
    #results = []
    search_results = []

    for vector_data in vector_db:
        doc_vector = vector_data['embeddings']
        
        dot_product = np.dot(query_vector, doc_vector)
        #Custom logic
        #dot_product = sum(query_vector[i] * doc_vector[i] for i in range(len(query_vector)))
        query_magnitude = math.sqrt(sum(query_vector[i]**2 for i in range(len(query_vector))))
        doc_magnitude = math.sqrt(sum(doc_vector[i]**2 for i in range(len(doc_vector))))
        cosine_similarity = dot_product / (query_magnitude * doc_magnitude)

        score = float(cosine_similarity)
        search_results.append({
            "doc_id": vector_data["doc_id"],
            "text" : vector_data["text"],
            "score" : round(score,4)
            })    

    results = sorted(search_results, key=lambda item: item['score'], reverse = True)[:top_k]
    return results

if __name__ == '__main__':
    raw_documents = [
    "The server crashed because the database connection pool was exhausted.",  # Doc 0
    "Baking a cake requires flour, sugar, eggs, and butter.",                   # Doc 1
    "Network connection timeout while trying to query primary cluster."         # Doc 2
    ]

    vector_data = embeddingGeneration(raw_documents)
    query = "database connection issue"
    output = semanticSearch(query, vector_data)
    print(f'Output:{output}')
