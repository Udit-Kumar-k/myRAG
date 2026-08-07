"""Test smart_truncate with expanded_query vs original query."""
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
reranker = pipeline.load_reranker()

# Query 4 with expanded query passed to smart_truncate
q4_orig = "My employer is withholding my last month's salary after I gave proper notice. What can I do?"
q4_exp = "breach of contract Indian Contract Act Section 73 compensation salary withholding employment notice period"

res4 = pipeline.retrieve(q4_exp, target_namespace="all", top_n=10)

print("\n--- QUERY 4 RETRIEVAL RESULTS ---")
pairs_exp = [[q4_orig, pipeline.smart_truncate(cand["text"], q4_exp, 2048)] for cand in res4]
scores_exp = reranker.predict(pairs_exp, batch_size=4)

for i, (cand, score) in enumerate(zip(res4, scores_exp)):
    print(f"[{i+1}] {cand['metadata']['document_name']} | score: {float(score):.4f}")
    print(f"    Text snippet: {cand['text'][:120]}...\n")
