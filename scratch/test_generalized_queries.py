"""Test 5 completely rephrased legal scenarios to verify true structural generalization."""
import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query

# 5 Completely Rephrased Scenarios (never seen verbatim in prompt or old tests)
REPHRASED_QUERIES = [
    "It's been four months since I gave my statement at the station and nobody will give me an update on what's happening with my complaint.",
    "A buyer paid for my goods using a cheque that bounced because there wasn't enough balance in their account. Can I take legal action?",
    "I clicked a malicious link in a suspicious text message and lost money from my bank account.",
    "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary.",
    "My mother passed away without leaving a legal document for her self-earned house. How will it be split among her children?",
]

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

for i, q in enumerate(REPHRASED_QUERIES, 1):
    print(f"\n{'='*70}")
    print(f"REPHRASED QUERY {i}: {q}")
    ns = route_query(q)
    print(f"  Routed to: {ns}")
    
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
