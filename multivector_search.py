import re
from sentence_transformers import SentenceTransformer
import math

model = SentenceTransformer("all-MiniLM-L6-V2")
def ingestParentChildCorpus(input_data: str, chunk_size: int=1000):
    parent_doc_store = {}
    child_vector_store = []

    raw_documents = input_data.split("\n\n")
    for index, doc in enumerate(raw_documents):
        parent_id = "parent_" + str(index)
        parent_chunk = smart_chunking(doc, chunk_size)
        parent_doc_store[parent_id]= " ".join(parent_chunk)
        
        child_chunks = smart_chunking(raw_text=" ".join(parent_chunk), chunk_size=300)

        for index, child_chunk in enumerate(child_chunks):
            child_vector = model.encode(child_chunk, convert_to_tensor=True)
            child_vector_store.append({
                "child_id": parent_id + "_child_"+ str(index),
                "child_text": child_chunk,
                "parent_id": parent_id,
                "embeddings": child_vector
            })

    return parent_doc_store, child_vector_store

def parent_child_search(user_query: str, parent_doc_store: list[dict], child_vector_store: list[dict], top_k: int=3, fetch_child_n: int=10):

    query_vector = model.encode(user_query)

    scored_children = []

    for record in child_vector_store:
        doc_vector = record['embeddings']

        dot_product = sum(query_vector[i] * doc_vector[i] for i in range(len(query_vector)))
        q_mag = math.sqrt(sum(query_vector[i] **2 for i in range(len(query_vector))))
        d_mag = math.sqrt(sum(doc_vector[i] **2 for i in range(len(doc_vector))))
        sim_score = dot_product / (q_mag * d_mag)

        scored_children.append({
            "parent_id": record['parent_id'],
            "child_text": record['child_text'],
            "score": float(sim_score)
        })
    top_children = sorted(scored_children, key=lambda item: item['score'], reverse=True)[:fetch_child_n]
    #Deduplicate parent data
    retrieved_parents = []
    seen_parent_ids= set()

    for children in top_children:
        parent_id = children['parent_id']
        if not children['parent_id'] in seen_parent_ids:
            retrieved_parents.append({
                "parent_id": parent_id,
                "parent_text": parent_doc_store[parent_id],
                "child_text": children['child_text'],
                "score": round(children['score'],4)
            })
            seen_parent_ids.add(parent_id)
        if len(retrieved_parents)> top_k:
            break
    return retrieved_parents

def smart_chunking(raw_text: str, chunk_size: int):
    current_chunk= []
    chunks=[]
    current_length = 0
    sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())

    for sentence in sentences:
        if len(sentence) > chunk_size:
            for word in sentence:
                sub_chunk=[]
                sub_len = 0
                if sub_len + len(word) > chunk_size and sub_chunk:
                    current_chunk.append(" ".join(sub_chunk))
                    sub_chunk=[]
                    sub_len=0
                    
                sub_chunk.append(" ".join(word))
                sub_len+= len(word)
                
                if sub_chunk:
                    current_chunk.append(" ".join(sub_chunk))
                continue

        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk=[sentence]
            current_length= len(sentence)#len(current_chunk)
        else:
            current_chunk.append(sentence)
            current_length+= len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    #print("Chunks Length:", len(chunks))
    #print("Chunks:", chunks)
    return [c.strip() for c in chunks if c.strip()]

if __name__ == '__main__':
    document_corpus = """
        "Database Architecture & Failover Runbook:"
        "Connection Pool: Default pool size is 50 with a max overflow of 10. Idle timeout is set to 300 seconds."
        "1) Node Health: If worker node heartbeats drop below 3/sec, standby replica promotion triggers."
        "2) Failover Command: Run 'failover-ctl promote --node worker-09 --force' to promote node worker-09."
        "3) Cache Invalidation: Flush Redis cluster cache using 'redis-cli flushall async' immediately post-failover.\n\n"
        "Corporate Network Firewall Configuration:"
        "1) Port Ingress Rules: Open port 443 for external HTTPS traffic. Open internal port 5432 strictly for database replication."
        "2) Rate Limiting: Limit inbound API queries to 100 req/sec per IP to mitigate DDoS."
        "3) SSL Certificates: Certificates auto-renew every 90 days via Let's Encrypt bot daemon.\n\n"
        "Server Resource Outage Diagnostics:"
        "1) Memory Exhaustion: When available RAM drops below 5%, the Linux kernel OOM-killer terminates high-memory processes."
        "2) CPU Spikes: A CPU spike >95% sustained for 5 minutes triggers an auto-scaling event in Kubernetes cluster us-east-1."
        "3) Core Dumps: Dump files are saved to /var/log/crashdumps/ for post-mortem analysis."""
    parent_doc_store, child_vector_store = ingestParentChildCorpus(document_corpus)

    query = ["What is the exact CLI command to promote worker 09?","database connection idle timeout and cluster cache flush instructions"]
    for q in query:
        results = parent_child_search(q, parent_doc_store,child_vector_store )
        print(f'Query:{q}Result:{results}')