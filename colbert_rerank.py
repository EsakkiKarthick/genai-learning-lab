from sentence_transformers import MultiVectorEncoder
from semantic_search import embeddingGeneration, semanticSearch
from multivector_search import smart_chunking
import json

print('model loading started')
model = MultiVectorEncoder("answerdotai/answerai-colbert-small-v1")
print('model loading completed')


def rerankwithcolbert(query: str, documentdata: list[dict], top_k: int=3):
    print('query embedding started')
    query_embedding = model.encode(query)
    print('query embedding completed')
    result=[]
    documents = [doc['text'] for doc in documentdata]
    doc_embeddings = model.encode(documents)
    #sim_matrix = query_embedding @ doc_embedding.T
    scores = model.similarity(query_embedding, doc_embeddings)
    #max_sim_score = 0.0
    #for query_token_vector_sims in sim_matrix:
    #    max_sim_score= max_sim_score + max(query_token_vector_sims)
    for doc, score in zip(documentdata, scores[0]):
        result.append(
            {
                "id": doc['doc_id'],
                "text": doc['text'],
                "score": float(score)
            }
        )

    reranked_result = sorted(result, key=lambda item:item['score'], reverse=True)
    return reranked_result[:top_k]

if __name__ == '__main__':
    """with open("input.txt", "r") as file:
        raw_documents = file.read()"""

    raw_documents =[
        {
            "doc_id": "D1",
            "text": """
            Employees in certain approved roles may be eligible
            for a monthly internet allowance. Eligibility depends
            on job role, employment agreement, and manager approval.
            """
        },
    
        {
            "doc_id": "D2",
            "text": """
            The company does not normally reimburse employees for
            home internet connections.
            """
        },
    
        {
            "doc_id": "D3",
            "text": """
            Employees may receive a monthly internet allowance if
            they are in an approved role and have manager approval.
            """
        },
    
        {
            "doc_id": "D4",
            "text": """
            Employees can claim reimbursement for mobile phone
            expenses incurred for business purposes.
            """
        },
    
        {
            "doc_id": "D5",
            "text": """
            Employees are responsible for their own home internet
            costs unless specifically approved under the company's
            internet allowance policy.
            """
        }
    ]

    chunk_data = []
    #paragraphs = raw_documents.split("\n\n")
    for doc in raw_documents:
        temp_data = smart_chunking(raw_text=doc['text'], chunk_size=300)
        chunk_data.append(" ".join(temp_data))

    vector_data = embeddingGeneration(chunk_data)
    
    query = ["Can I get an internet allowance??",
                "Who is eligible for internet reimbursement?",
                "Does everyone receive an internet allowance?",
                "Will the company pay for my home WiFi?",
                "What conditions must I satisfy to receive internet support?",
                "Is home internet automatically reimbursed for all employees?"
        ]
    for q in query:
        print(f'Query:{q}')
        output = semanticSearch(q, vector_data, 5)
        print(f'Search Results:{json.dumps(output, indent=4)}\n')

        reranking_result = rerankwithcolbert(q, output)
        print(f'\nReranked Output:{json.dumps(reranking_result, indent=4)}\n')