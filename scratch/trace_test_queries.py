"""Simulate the 5 original test queries through retrieval only to check routing, confidence, and source acts."""
import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query

# The 5 test queries (paraphrased from the original test)
TEST_QUERIES = [
    "It's been 3 months since I filed my FIR and the police haven't told me anything about the investigation progress.",
    "My client gave me a bounced cheque due to insufficient funds. Is this a consumer dispute or criminal offence?",
    "Someone sent me a malicious link, hacked my net banking, and transferred money. What laws apply?",
    "My employer is withholding my last month's salary after I gave proper notice. What can I do?",
    "My father died without a will. How is his self-acquired property divided among my mother, brother, and me?",
]

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

for i, q in enumerate(TEST_QUERIES, 1):
    print(f"\n{'='*70}")
    print(f"QUERY {i}: {q[:80]}...")
    
    # Check routing
    ns = route_query(q)
    print(f"  Routed to: {ns}")
    
    # Run full pipeline
    result = pipeline.query(q)
    print(f"  Confidence: {result['confidence_score']:.4f}")
    print(f"  Refused: {result['refused']}")
    print(f"  Namespace searched: {result['namespace_searched']}")
    
    if result['retrieved_chunks']:
        print(f"  Top {len(result['retrieved_chunks'])} sources:")
        for j, chunk in enumerate(result['retrieved_chunks']):
            meta = chunk['metadata']
            print(f"    [{j+1}] {meta['document_name']} (ns={meta['namespace']}, score={chunk['relevance_score']:.4f})")
            print(f"        text preview: {chunk['text'][:100]}...")
    else:
        print(f"  No chunks returned (refused)")
