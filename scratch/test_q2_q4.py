"""Test score improvements for Query 2 and Query 4."""
import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

# Test 1: Query 2 routed to 'all' vs 'banking' vs 'criminal'
q2 = "My client gave me a bounced cheque due to insufficient funds. Is this a consumer dispute or criminal offence?"
for ns in ["criminal", "banking", "all"]:
    res = pipeline.retrieve("dishonour of cheque Negotiable Instruments Act Section 138 BNS cheating insufficient funds", target_namespace=ns, top_n=5)
    reranker = pipeline.load_reranker()
    pairs = [[q2, pipeline.smart_truncate(cand["text"], q2, 2048)] for cand in res]
    scores = reranker.predict(pairs, batch_size=4)
    max_score = max(scores) if len(scores) > 0 else 0
    print(f"Query 2 target_namespace={ns}: max_rerank_score={max_score:.4f}")
    if res:
        print(f"   Top chunk doc: {res[0]['metadata']['document_name']} (score={float(scores[0]):.4f})")

# Test 2: Query 4 with focused expansion
q4 = "My employer is withholding my last month's salary after I gave proper notice. What can I do?"
q4_exp = "Indian Contract Act 1872 Section 73 breach of contract compensation damages withholding salary notice period employment"
res4 = pipeline.retrieve(q4_exp, target_namespace="all", top_n=5)
pairs4 = [[q4, pipeline.smart_truncate(cand["text"], q4, 2048)] for cand in res4]
scores4 = reranker.predict(pairs4, batch_size=4)
max_score4 = max(scores4) if len(scores4) > 0 else 0
print(f"\nQuery 4 (focused expansion) target_namespace=all: max_rerank_score={max_score4:.4f}")
if res4:
    for i, (c, s) in enumerate(zip(res4[:3], scores4[:3])):
        print(f"   Top [{i+1}] {c['metadata']['document_name']} score={float(s):.4f}")
