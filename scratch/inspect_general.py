"""
Inspect the 2 general queries (general_01, general_02) that failed context_recall.
"""
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query
import re

TARGET_QUERIES = [
    {
        "id": "general_01",
        "q": "Someone stole my identity online and took a loan in my name. What laws apply?",
        "expected_ns": "all",
        "kws": ["identity theft", "fraud", "IT Act", "BNS", "cheating"]
    },
    {
        "id": "general_02",
        "q": "Someone sent me a fake UPI payment screenshot to scam me. What legal action can I take?",
        "expected_ns": "all",
        "kws": ["UPI", "fraud", "cheating", "FIR", "IT Act"]
    }
]

print("Loading indexes...")
index_manager = LegalIndexManager()
if not index_manager.load_indexes():
    raise SystemExit("Failed to load indexes.")

pipeline = LegalRAGPipeline(index_manager)
reranker = pipeline.load_reranker()

for item in TARGET_QUERIES:
    q = item["q"]
    keywords = [kw.lower() for kw in item["kws"]]
    
    # 1. Routing check
    routed_ns = route_query(q)
    match_status = "MATCH" if routed_ns == item["expected_ns"] else f"MISMATCH (routed={routed_ns}, expected={item['expected_ns']})"
    print(f"\n{'='*70}\nQUERY: {q}\nRouting: {match_status}")
    
    # 2. Retrieve candidates
    candidates = pipeline.retrieve(q, target_namespace=routed_ns, top_n=20)
    print(f"Candidates returned from retrieve(): {len(candidates)}")
    
    # 3. Check if target keywords exist in the candidate pool
    found_idx = -1
    found_chunk = None
    for idx, c in enumerate(candidates):
        chunk_text = c["text"].lower()
        if any(kw in chunk_text for kw in keywords):
            found_idx = idx
            found_chunk = c
            break
            
    if found_idx == -1:
        print("  --> OUTCOME: NOT FOUND in top-20")
        continue
        
    print(f"  Found in top-20 at RRF#{found_idx + 1} (Pre-rerank)")
    
    # 4. Rerank the candidates
    pairs = [[q, pipeline.smart_truncate(c["text"], q, 2048)] for c in candidates]
    scores = reranker.predict(pairs, batch_size=4)
    for c, s in zip(candidates, scores):
        c["relevance_score"] = float(s)
        
    # Keep track of original RRF rank
    for idx, c in enumerate(candidates):
        c["rrf_rank"] = idx + 1
        
    # Sort descending
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Find new rank post-rerank
    new_rank = -1
    for rank_idx, c in enumerate(candidates):
        if c["rrf_rank"] == found_idx + 1:
            new_rank = rank_idx + 1
            break
            
    score = candidates[new_rank - 1]["relevance_score"]
    print(f"  Reranked position: #{new_rank} (RRF#{found_idx + 1}) with score={score:.4f}")
    if item["id"] == "general_02":
        print(f"\nFULL TEXT OF MATCHING CHUNK FOR general_02:\n{found_chunk['text']}\n" + "-"*40)
    
    # Determine outcome
    # Outcome 1: NOT FOUND in top-20 (Handled above)
    # Outcome 2: Found in top-20, but ranked below 5 originally (RRF rank > 5)
    # Outcome 3: Found in top-20 (originally in top-5), and reranking pushes it down further (Rerank rank > 5)
    
    pre_rerank_rank = found_idx + 1
    if pre_rerank_rank > 5:
        print(f"  --> OUTCOME: Found in top-20, but ranked below 5 (RRF#{pre_rerank_rank})")
    elif pre_rerank_rank <= 5 and new_rank > 5:
        print(f"  --> OUTCOME: Found, and reranking pushes it down further (originally RRF#{pre_rerank_rank}, reranked to #{new_rank})")
    else:
        print(f"  --> OUTCOME: Found in top-5 (RRF#{pre_rerank_rank}) and reranked in top-5 (#{new_rank}) with score={score:.4f}")
