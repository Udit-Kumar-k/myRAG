"""Diagnose why confidence scores are too low for clearly in-scope queries."""
import os, sys, json
os.environ["EMBEDDING_DEVICE"] = "cpu"
sys.path.insert(0, os.path.abspath("."))

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()

# Use threshold=0 to see raw scores without refusal
pipeline = LegalRAGPipeline(im, confidence_threshold=0.0)

test_queries = [
    "What is the punishment for murder under BNS?",
    "What does BNS say about cybercrime?",
    "Someone sent me a fake UPI screenshot, what law applies?",
    "What are the penalties for hacking under the IT Act?",
]

# Also check chunk sizes
print("=== CHUNK SIZE ANALYSIS ===")
for ns, chunks in im.chunks.items():
    lengths = [len(c["text"]) for c in chunks]
    print(f"  {ns}: {len(chunks)} chunks, avg={sum(lengths)//len(lengths)} chars, "
          f"max={max(lengths)}, min={min(lengths)}")

print("\n=== QUERY RESULTS ===")
for q in test_queries:
    res = pipeline.query(q)
    print(f"\nQ: {q}")
    print(f"  confidence: {res['confidence_score']:.4f}")
    print(f"  namespace:  {res['namespace_searched']}")
    print(f"  refused:    {res['refused']}")
    print(f"  chunks:     {len(res['retrieved_chunks'])}")
    if res["retrieved_chunks"]:
        top = res["retrieved_chunks"][0]
        print(f"  top chunk text (first 200 chars): {top['text'][:200]}")
        print(f"  top chunk metadata: {json.dumps({k:v for k,v in top['metadata'].items() if k != 'text'}, default=str)}")
