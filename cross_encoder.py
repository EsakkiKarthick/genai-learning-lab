from semantic_search import semanticSearch, embeddingGeneration
from multivector_search import smart_chunking
from sentence_transformers import CrossEncoder
import math
import json

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
def reranking(query: str, retrieved_docs: list[dict], top_k: int=3, min_score: int=0.25):
    if len(retrieved_docs)==0:
        print(f'No document data retrieved')
        return []

    input_pairs={}
    reranked_results = []
    input_pairs = [[query, doc['text']] for doc in retrieved_docs]
    #for index, doc in enumerate(retrieved_docs):
    #    input_pairs[index]=[query, doc['text']]
            #"query": query,
            #"doc_text": doc['text']

    raw_scores = model.predict(input_pairs, batch_size=32, show_progress_bar=False)
    #model.predict()
    for index, doc in enumerate(retrieved_docs):
        raw_score = raw_scores[index]
        normalized_score = 1.0 / (1.0 + math.exp(-raw_score))
        print(f'\nReranking Index {index}:{doc['text']} \nEncoder Score:{round(normalized_score,4)}\n')
        reranked_results.append({
            "id": doc['doc_id'],
            "text": doc['text'],
            "rerank_score": normalized_score
        })


    sorted_results = sorted(reranked_results, key=lambda item: item['rerank_score'], reverse=True)
    print(f'Results Count:{len(sorted_results)}')

    return sorted_results[:top_k]

if __name__ == '__main__':
    """raw_documents = [
  "PostgreSQL Performance: Increase work_mem and maintenance_work_mem in postgresql.conf to accelerate complex analytical JOIN queries and sorting operations.",
  "PostgreSQL Connection Pooling: Use PgBouncer in transaction pooling mode to mitigate high connection overhead and avoid hitting max_connections limits.",
  "MySQL Performance Tuning: Configure innodb_buffer_pool_size to 70-80% of total system RAM to optimize read-heavy transaction workloads.",
  "Kubernetes Ingress Troubleshooting: Error 502 Bad Gateway typically occurs when the ingress controller cannot reach the upstream backend Pod IP on the defined target port.",
  "Kubernetes Pod CrashLoopBackOff: Check container logs with 'kubectl logs --previous' and verify livenessProbe and readinessProbe misconfigurations.",
  "Kubernetes HPA: Horizontal Pod Autoscaler scales replication controllers based on observed CPU and memory metrics reported by metrics-server.",
  "AWS IAM Best Practices: Never use root account credentials for day-to-day operations; enforce MFA and assign granular least-privilege IAM roles.",
  "AWS S3 Error 403 Access Denied: Verify bucket policies, IAM role permissions, and ensure KMS key decrypt permissions exist if SSE-KMS is enabled.",
  "AWS S3 Lifecycle Policies: Automatically transition infrequently accessed objects from Standard to S3 Glacier Flexible Retrieval after 90 days to minimize storage costs.",
  "Redis Caching Strategies: Implement Cache-Aside pattern with explicit TTL expirations to prevent cache penetration and stale data drift.",
  "Redis Cluster Failover: Redis Sentinel provides high availability by automatically electing a new replica when the master instance becomes unresponsive.",
  "Docker Image Optimization: Use multi-stage builds and Alpine base images to minimize container attack surface and reduce image transfer latency.",
  "OAuth 2.0 Authorization Code Flow: Recommended for web applications with a secure backend; exchange authorization code for access tokens via POST request.",
  "OAuth 2.0 PKCE Extension: Mandatory for Single Page Applications (SPA) and mobile apps to prevent authorization code interception without client secrets.",
  "SAML 2.0 SSO Integration: Service Provider initiates authentication by redirecting the user browser to the Identity Provider with a SAMLAuthnRequest XML payload.",
  "Linux Memory Management: Understand OOM (Out Of Memory) Killer; adjust /proc/sys/vm/overcommit_memory and vm.swappiness to tune memory allocation behavior.",
  "Linux File System Tuning: Use noatime mount option in /etc/fstab to eliminate write overhead on read-heavy SSD storage arrays.",
  "GraphQL API Security: Implement query depth limiting and query complexity scoring to prevent denial-of-service (DoS) attacks via deeply nested queries.",
  "REST API Rate Limiting: Implement Token Bucket or Leaky Bucket algorithms in the API Gateway (e.g., Kong, Nginx) returning HTTP 429 Too Many Requests.",
  "Terraform State Management: Always use remote backend storage with S3 and DynamoDB state locking to prevent race conditions in CI/CD pipelines.",
  "Terraform Drift Detection: Run 'terraform plan -refresh-only' to reconcile external infrastructure changes without applying structural modifications.",
  "CI/CD Security: Scan container images during pipeline build steps using Trivy or Grype before pushing artifacts to production registries.",
  "Kafka Partition Optimization: Ensure partition keys evenly distribute consumer load across brokers to eliminate partition hot-spotting.",
  "Zero Trust Architecture: Authenticate and authorize every access request continuously based on device health, user context, and dynamic policy verification.",
  "TLS 1.3 Configuration: Disable legacy CBC-mode ciphers and older protocols (SSLv3, TLS 1.0, TLS 1.1) to enforce forward secrecy and fast 1-RTT handshakes."
        ]"""
    with open("input.txt", "r") as file:
        raw_documents = file.read()
    chunk_data = []
    paragraphs = raw_documents.split("\n\n")
    for paragraph in paragraphs:
        temp_data = smart_chunking(raw_text=paragraph, chunk_size=300)
        chunk_data.append(" ".join(temp_data))

    vector_data = embeddingGeneration(chunk_data)
    query = ["How many days per week can an employee work from home?",
             "Who must approve an employee's work-from-home request??",
             "Does the company pay for my home Wi-Fi when I work remotely?",
             "Can I get money from the company for my internet connection?",
             "Does the company always reimburse home internet expenses?"
        ]
    for q in query:
        print(f'Query:{q}')
        output = semanticSearch(q, vector_data, 5)
        print(f'Search Results:{json.dumps(output, indent=4)}\n')

        reranking_result = reranking(q, output)
        print(f'\nReranked Output:{json.dumps(reranking_result, indent=4)}\n')