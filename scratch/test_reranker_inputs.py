import sys, io, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipe = LegalRAGPipeline(im)

queries = [
    ("my boss is verbally abusing me and making me work overtime", "BNS criminal intimidation workplace harassment Code on Wages overtime working hours employer employee"),
    ("I received a defective product and the seller is refusing a refund", "Consumer Protection Act deficiency in service defective goods e-commerce refund District Commission")
]

for orig_q, kw_expanded in queries:
    print(f"\n==========================================")
    print(f"QUERY: {orig_q}")
    print(f"==========================================")
    
    # 1. Retrieve candidates using keyword expansion
    cand_kw = pipe.retrieve(kw_expanded, target_namespace="consumer" if "refund" in orig_q else "general", top_n=10)
    print(f"Retrieved {len(cand_kw)} candidates with keywords.")
    
    reranker = pipe.load_reranker()
    
    # Test scoring with original query vs keyword query
    pairs_orig = [[orig_q, pipe.smart_truncate(c["text"], orig_q, 2048)] for c in cand_kw[:3]]
    scores_orig = reranker.predict(pairs_orig)
    
    pairs_kw = [[kw_expanded, pipe.smart_truncate(c["text"], kw_expanded, 2048)] for c in cand_kw[:3]]
    scores_kw = reranker.predict(pairs_kw)
    
    for i, (c, s_orig, s_kw) in enumerate(zip(cand_kw[:3], scores_orig, scores_kw)):
        doc_name = c.get("metadata", {}).get("document_name")
        print(f"[{i+1}] {doc_name} -> Score(orig_q): {s_orig:.4f} | Score(kw): {s_kw:.4f}")
