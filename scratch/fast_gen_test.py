"""Test generalized rephrased queries with lazy reranker initialization."""
import os, sys, gc
sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query

# 5 Rephrased legal scenarios (no prompt overlap)
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

print("Starting Query Evaluation...")
for label, q in REPHRASED_QUERIES:
    print(f"\n{'='*70}")
    print(f"[{label}]")
    print(f"Query: \"{q}\"")
    ns = route_query(q)
    print(f"Routed namespace: {ns}")
    
    try:
        # Run retrieval directly using the pipeline's retrieve method
        expanded_query = q
        try:
            from src.backend.chain import LegalRAGChain
            chain = LegalRAGChain()
            expanded_query = chain.expand_query(q)
            print(f"Expanded Query: {expanded_query}")
        except Exception as ex:
            print(f"Expansion warning: {ex}")
            
        candidates = pipeline.retrieve(expanded_query, target_namespace=ns, top_n=10)
        print(f"Candidates retrieved: {len(candidates)}")
        if candidates:
            for k, cand in enumerate(candidates[:3]):
                print(f"  Candidate {k+1}: {cand['metadata']['document_name']} ({cand['metadata']['namespace']})")
                print(f"    Preview: {cand['text'][:120]}...")
    except Exception as e:
        print(f"  Error: {e}")
