import os, sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline
from src.backend.chain import LegalRAGChain

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)
rag_chain = LegalRAGChain()

q = "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary."
expanded_q = rag_chain.expand_query(q)

print("Original query:", q)
print("Expanded query:", expanded_q)

res = pipeline.query(expanded_q)

print("\nConfidence:", res["confidence_score"])
print("Namespace:", res["namespace_searched"])
print("Refused:", res["refused"])
print(f"\nRetrieved {len(res['retrieved_chunks'])} Chunks:")

for i, c in enumerate(res["retrieved_chunks"]):
    meta = c["metadata"]
    text = c.get("text", "")
    print(f"\n==================== CHUNK {i+1} ====================")
    print("Document:", meta.get("document_name"))
    print("Act Name:", meta.get("act_name"))
    print("Relevance Score:", c.get("relevance_score"))
    print("--- Full Text ---")
    print(text)
    print("-----------------")
    print("Contains 'Payment of Wages':", "Payment of Wages" in text or "payment of wages" in text.lower())
    print("Contains 'Section 15' or '15':", "15" in text)
