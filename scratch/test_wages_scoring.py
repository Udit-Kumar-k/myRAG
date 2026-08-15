import os, pickle
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)
reranker = pipeline.load_reranker()

with open("data/indexes/general_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Chunk 21 from Payment of Wages Act
chunk_21 = chunks[18063] if len(chunks) > 18063 else None
for c in chunks:
    if "payment of wages has been delayed" in c.get("text", "").lower():
        chunk_21 = c
        break

print(f"Found Chunk 21: [{chunk_21['metadata'].get('document_name')}] -> {chunk_21['text'][:150]}...")

queries_to_test = [
    "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary.",
    "Indian Contract Act Payment of Wages Act breach of contract compensation unpaid salary employee dues notice period",
    "Payment of Wages Act delayed payment of wages deduction from wages unpaid salary employer employee",
    "unpaid salary delayed payment of wages employer employee resignation notice period",
    "Payment of Wages Act unpaid salary employee resignation notice period"
]

for q in queries_to_test:
    truncated = pipeline.smart_truncate(chunk_21["text"], q, 2048)
    score = reranker.predict([[q, truncated]])[0]
    print(f"\nQuery: '{q[:70]}...'")
    print(f"  Truncated starts with: {truncated[:80]}...")
    print(f"  Score: {float(score):.4f}")
