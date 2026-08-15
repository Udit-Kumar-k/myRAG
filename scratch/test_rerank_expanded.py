import os
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

q = "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary."
expanded_q = "Indian Contract Act Payment of Wages Act breach of contract compensation unpaid salary employee dues notice period"

candidates = pipeline.retrieve(expanded_q, target_namespace="general", top_n=20)
print(f"Top 5 retrieved chunks:")
for i, c in enumerate(candidates[:5]):
    print(f"  {i+1}: [{c['metadata'].get('document_name')}] -> {c['text'][:120]}...")

reranker = pipeline.load_reranker()

pairs_raw = [[q, pipeline.smart_truncate(c["text"], q, 2048)] for c in candidates[:5]]
scores_raw = reranker.predict(pairs_raw, batch_size=4)
print(f"\nRerank with raw query: '{q}'")
for i, (c, s) in enumerate(zip(candidates[:5], scores_raw)):
    print(f"  {i+1}: [{c['metadata'].get('document_name')}] score={float(s):.4f}")

pairs_exp = [[expanded_q, pipeline.smart_truncate(c["text"], expanded_q, 2048)] for c in candidates[:5]]
scores_exp = reranker.predict(pairs_exp, batch_size=4)
print(f"\nRerank with expanded query: '{expanded_q}'")
for i, (c, s) in enumerate(zip(candidates[:5], scores_exp)):
    print(f"  {i+1}: [{c['metadata'].get('document_name')}] score={float(s):.4f}")

combined_q = f"{q} {expanded_q}"
pairs_comb = [[combined_q, pipeline.smart_truncate(c["text"], combined_q, 2048)] for c in candidates[:5]]
scores_comb = reranker.predict(pairs_comb, batch_size=4)
print(f"\nRerank with combined query: '{combined_q}'")
for i, (c, s) in enumerate(zip(candidates[:5], scores_comb)):
    print(f"  {i+1}: [{c['metadata'].get('document_name')}] score={float(s):.4f}")
