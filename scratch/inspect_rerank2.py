"""
Inspect reranker behavior including truncation and sanity check.
"""
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query

TARGET_QUERIES = [
    {
        "q": "What is the procedure for filing an FIR under BNSS?",
        "kws": ["FIR", "first information report", "cognizable", "police", "section"]
    },
    {
        "q": "Can a landlord refuse to return a tenant's security deposit?",
        "kws": ["landlord", "tenant", "deposit", "refund", "rent"]
    },
    {
        "q": "What are the RBI guidelines on digital payment fraud?",
        "kws": ["RBI", "digital payment", "fraud", "liability", "unauthorized"]
    }
]

print("Loading indexes...")
index_manager = LegalIndexManager()
if not index_manager.load_indexes():
    raise SystemExit("Failed to load indexes.")

pipeline = LegalRAGPipeline(index_manager)
reranker = pipeline.load_reranker()

# Sanity Check
sanity_pairs = [
    ["What is an apple?", "A computer is a machine."],
    ["What is an apple?", "An apple is a sweet, edible fruit produced by an apple tree."]
]
scores = reranker.predict(sanity_pairs, batch_size=2)
print("Sanity Check Scores:", scores)
if scores[1] > scores[0]:
    print("Sanity check PASS")
else:
    print("Sanity check FAIL")

for item in TARGET_QUERIES:
    q = item["q"]
    keywords = [kw.lower() for kw in item["kws"]]
    namespace = route_query(q)
    candidates = pipeline.retrieve(q, target_namespace=namespace, top_n=20)
    print(f"\n{'='*70}\nQUERY: {q}\nnamespace_searched: {namespace}")

    if not candidates:
        print("  NO CANDIDATES RETURNED")
        continue

    target_idx = -1
    truncation_status = ""
    for rrf_idx, c in enumerate(candidates):
        chunk_text = c["text"].lower()
        if any(kw in chunk_text for kw in keywords):
            target_idx = rrf_idx
            chunk_8192 = chunk_text[:8192]
            if any(kw in chunk_8192 for kw in keywords):
                truncation_status = "within first 8192 chars"
            else:
                truncation_status = "BEYOND 8192 TRUNCATION"
            break

    pairs = [[q, c["text"][:8192]] for c in candidates]
    scores = reranker.predict(pairs, batch_size=4)
    for c, s in zip(candidates, scores):
        c["relevance_score"] = float(s)
        
    for rrf_idx, c in enumerate(candidates):
        c["rrf_rank"] = rrf_idx + 1
        
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    print("Rankings:")
    for i, c in enumerate(candidates):
        is_target = (c["rrf_rank"] == target_idx + 1)
        flag = f"<-- CONTAINS ({truncation_status})" if is_target else ""
        if i < 5 or is_target:
            if i == 0 and q in ["What is the procedure for filing an FIR under BNSS?", "What are the RBI guidelines on digital payment fraud?"]:
                print(f"  Rerank #{i+1} (RRF#{c['rrf_rank']}) score={c['relevance_score']:.4f} {flag}\nFULL TEXT:\n{c['text']}\n" + "-"*40)
            else:
                snippet = c["text"][:100].replace("\n", " ")
                print(f"  Rerank #{i+1} (RRF#{c['rrf_rank']}) score={c['relevance_score']:.4f}  {snippet}... {flag}")
