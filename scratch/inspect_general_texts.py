from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline
import re

index_manager = LegalIndexManager()
index_manager.load_indexes()
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
    print(f"\n{'='*70}\nQUERY: {q}")
    candidates = pipeline.retrieve(q, target_namespace="all", top_n=20)
    
    # Rerank
    MAX_RERANK_CHARS = 2048
    reranker = pipeline.load_reranker()
    pairs = [[q, pipeline.smart_truncate(cand["text"], q, MAX_RERANK_CHARS)] for cand in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=4)
    for cand, score in zip(candidates, rerank_scores):
        cand["relevance_score"] = float(score)
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Find matching chunk rank
    keywords = [kw.lower() for kw in kws]
    target_idx = -1
    for idx, c in enumerate(candidates):
        chunk_text = c["text"].lower()
        if any(re.search(r"\b" + re.escape(kw) + r"\b", chunk_text) for kw in keywords):
            target_idx = idx
            break

    print(f"Top 3 Reranked Winners:")
    for rank_idx in range(min(3, len(candidates))):
        c = candidates[rank_idx]
        is_target = (rank_idx == target_idx)
        print(f"\n--- Rerank #{rank_idx+1} (RRF#{c.get('rrf_rank', 'N/A')}) score={c['relevance_score']:.4f} {'<-- TARGET' if is_target else ''} ---")
        print(f"Metadata: {c['metadata']}")
        print("TEXT:")
        print(c["text"])
        
    if target_idx >= 3:
        c = candidates[target_idx]
        print(f"\n--- Rerank #{target_idx+1} (RRF#{c.get('rrf_rank', 'N/A')}) score={c['relevance_score']:.4f} <-- TARGET ---")
        print(f"Metadata: {c['metadata']}")
        print("TEXT:")
        print(c["text"])
    
    print("="*70)
