"""Memory-optimized generalization test script for rephrased queries."""
import os, sys, gc
sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query

# Rephrased legal scenarios (no prompt overlap)
REPHRASED_QUERIES = [
    ("Scenario 1 (FIR Progress)", "It's been four months since I gave my statement at the station and nobody will give me an update on what's happening with my complaint."),
    ("Scenario 2 (Cheque Dishonour)", "A buyer paid for my goods using a cheque that bounced because there wasn't enough balance in their account. Can I take legal action?"),
    ("Scenario 3 (Cyber Fraud)", "I clicked a malicious link in a suspicious text message and lost money from my bank account."),
    ("Scenario 4 (Salary Withholding)", "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary."),
    ("Scenario 5 (Intestate Succession)", "My mother passed away without leaving a legal document for her self-earned house. How will it be split among her children?"),
]

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

# Pre-load models step-by-step
print("Loading reranker...")
pipeline.load_reranker()

for label, q in REPHRASED_QUERIES:
    gc.collect()
    print(f"\n{'='*70}")
    print(f"[{label}]")
    print(f"Query: \"{q}\"")
    ns = route_query(q)
    print(f"Routed namespace: {ns}")
    
    try:
        result = pipeline.query(q)
        print(f"Confidence score: {result['confidence_score']:.4f}")
        print(f"Refused: {result['refused']}")
        print(f"Namespace searched: {result['namespace_searched']}")
        
        if result['retrieved_chunks']:
            print("Top Retrieved Sources:")
            for j, chunk in enumerate(result['retrieved_chunks'][:3]):
                meta = chunk['metadata']
                print(f"  [{j+1}] {meta['document_name']} (score={chunk['relevance_score']:.4f})")
                print(f"      Preview: {chunk['text'][:100]}...")
        else:
            print("  No chunks returned (refused)")
    except Exception as e:
        print(f"  Error running query: {e}")
