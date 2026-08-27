import sys, os, traceback
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

print("1. Initializing IndexManager...")
im = LegalIndexManager()
im.load_indexes()
im.load_embedding_model()

print("2. Initializing LegalRAGPipeline...")
pipe = LegalRAGPipeline(im, confidence_threshold=0.55)

q = "my boss is verbally abusing me and making me work overtime"
print(f"3. Calling pipe.query('{q}')...")
try:
    res = pipe.query(q)
    print("4. Query returned successfully!")
    print("Refused:", res.get("refused"))
    print("Confidence:", res.get("confidence_score"))
    print("Namespace:", res.get("namespace_searched"))
    print("Chunks count:", len(res.get("retrieved_chunks", [])))
    for c in res.get("retrieved_chunks", [])[:3]:
        print(" -", c.get("metadata", {}).get("document_name"), "Score:", c.get("relevance_score"))
except Exception as e:
    print("CAUGHT EXCEPTION IN pipe.query:")
    traceback.print_exc()
