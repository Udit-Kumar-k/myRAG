"""Full pipeline diagnostic — runs each step of the query path in isolation to find the exact failure."""
import os, sys, traceback

sys.path.insert(0, os.path.abspath("."))

os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"
os.environ["MOCK_AUTH"] = "true"

from dotenv import load_dotenv
load_dotenv()

print("=== STEP 1: Load indexes ===")
from src.backend.indexing import LegalIndexManager
im = LegalIndexManager()
ok = im.load_indexes()
print(f"Indexes loaded: {ok}")
if not ok:
    print("FATAL: No indexes found")
    sys.exit(1)

print("\n=== STEP 2: Create pipeline ===")
from src.backend.retrieval import LegalRAGPipeline
threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.02))
pipeline = LegalRAGPipeline(im, confidence_threshold=threshold)
print(f"Pipeline created OK with confidence_threshold={threshold}")

print("\n=== STEP 3: Run retrieval query ===")
try:
    result = pipeline.query("What is murder under BNS?")
    refused = result["refused"]
    confidence = result["confidence_score"]
    ns = result["namespace_searched"]
    chunks = result["retrieved_chunks"]
    print(f"refused={refused}, confidence={confidence:.4f}, namespace={ns}, chunks={len(chunks)}")
    if chunks:
        c0 = chunks[0]
        print(f"chunk keys: {list(c0.keys())}")
        print(f"metadata keys: {list(c0['metadata'].keys())}")
        print(f"text preview: {c0['text'][:120]}...")
except Exception:
    traceback.print_exc()
    print("\n*** RETRIEVAL FAILED ***")
    sys.exit(1)

print("\n=== STEP 4: Build source metadata (what main.py does) ===")
try:
    for chunk in chunks:
        meta = chunk["metadata"]
        src = {
            "document_name": meta["document_name"],
            "legal_domain": meta.get("legal_domain", meta.get("geography_iso", "general")),
            "pub_year": meta["pub_year"],
            "namespace": meta["namespace"],
            "source_url": meta["source_url"],
            "relevance_score": chunk["relevance_score"]
        }
        print(f"  Source OK: {src['document_name']} domain={src['legal_domain']}")
except Exception:
    traceback.print_exc()
    print("\n*** SOURCE METADATA BUILD FAILED ***")
    sys.exit(1)

print("\n=== STEP 5: Run LLM chain ===")
try:
    from src.backend.chain import LegalRAGChain
    chain = LegalRAGChain()
    answer = chain.run("What is murder under BNS?", chunks, history=[])
    print(f"Answer preview: {answer[:300]}...")
except Exception:
    traceback.print_exc()
    print("\n*** CHAIN/LLM FAILED ***")
    sys.exit(1)

print("\n=== ALL STEPS PASSED ===")
