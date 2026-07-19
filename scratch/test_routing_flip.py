from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline
import re

print("Loading indexes...")
index_manager = LegalIndexManager()
if not index_manager.load_indexes():
    raise SystemExit("Failed to load indexes.")

pipeline = LegalRAGPipeline(index_manager)

queries = [
    {
        "id": "general_01",
        "q": "Someone stole my identity online and took a loan in my name. What laws apply?",
        "kws": ["identity theft", "fraud", "IT Act", "BNS", "cheating"]
    },
    {
        "id": "general_02",
        "q": "Someone sent me a fake UPI payment screenshot to scam me. What legal action can I take?",
        "kws": ["UPI", "fraud", "cheating", "FIR", "IT Act"]
    }
]

for item in queries:
    q = item["q"]
    kws = item["kws"]
    print(f"\nQUERY: {q}")
    
    # Force 'all' namespace
    candidates = pipeline.retrieve(q, target_namespace="all", top_n=20)
    print(f"  Candidates returned from 'all': {len(candidates)}")
    
    if not candidates:
        print("  Confidence: 0.0 (refused=True) - NO CANDIDATES FOUND")
        continue
        
    MAX_RERANK_CHARS = 2048
    reranker = pipeline.load_reranker()
    pairs = [[q, pipeline.smart_truncate(cand["text"], q, MAX_RERANK_CHARS)] for cand in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=4)
    for cand, score in zip(candidates, rerank_scores):
        cand["relevance_score"] = float(score)
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_chunks = candidates[:5]
    confidence_all = top_chunks[0]["relevance_score"]
    
    # Check if target context is recalled in top 20 candidates
    found_recall = False
    recalled_rank = -1
    keywords = [kw.lower() for kw in kws]
    for idx, c in enumerate(candidates):
        chunk_text = c["text"].lower()
        if any(re.search(r"\b" + re.escape(kw) + r"\b", chunk_text) for kw in keywords):
            found_recall = True
            recalled_rank = idx + 1
            break
            
    print(f"  Context recalled (top-20): {found_recall} (at rank #{recalled_rank if found_recall else 'N/A'})")
    print(f"  Confidence under 'all':    {confidence_all:.4f}")
    print(f"  Refused (threshold=0.02):  {confidence_all < 0.02}")
    print(f"  Refused (threshold=0.65):  {confidence_all < 0.65}")
    
    if found_recall:
         print(f"  Rerank #1 snippet: {candidates[0]['text'][:200].replace(chr(10), ' ')}...")
