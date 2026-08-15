import os, sys
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

q = "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary."
expanded_q = "Indian Contract Act Payment of Wages Act breach of contract compensation unpaid salary employee dues notice period"

res = pipeline.retrieve(expanded_q, target_namespace="general", top_n=20)
print(f"Retrieved {len(res)} candidates from general:")
for i, c in enumerate(res[:10]):
    print(f"  {i+1}: [{c['metadata'].get('document_name')}] (score={c.get('rrf_score', 0):.4f}) -> {c['text'][:100]}...")

scored = pipeline.rerank(q, res, max_candidates=20)
print(f"\nReranked scores for query: '{q}'")
for i, (chunk, score) in enumerate(scored[:5]):
    print(f"  Rank {i+1}: [{chunk['metadata'].get('document_name')}] score={score:.4f} -> {chunk['text'][:100]}...")
