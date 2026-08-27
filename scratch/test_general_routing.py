import sys, io, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipe = LegalRAGPipeline(im)

q = "my boss is verbally abusing me and making me work overtime"
res = pipe.query(q)
print("Namespace:", res["namespace_searched"])
print(f"Confidence: {res['confidence_score']:.4f}")
print("Refused:", res["refused"])
print("Top Chunks Count:", len(res["retrieved_chunks"]))
for i, c in enumerate(res["retrieved_chunks"][:3]):
    meta = c.get("metadata", {})
    doc_name = meta.get("document_name")
    score = c.get("relevance_score", 0)
    print(f"[{i+1}] {doc_name} | Score: {score:.4f}")
