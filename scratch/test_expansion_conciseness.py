import os, sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)
reranker = pipeline.load_reranker()

query = "Someone broke into my shop at night and stole goods worth several lakhs."

expansions = [
    ("Verbose (current)", "BNS theft housebreaking burglary dishonest misappropriation property night criminal trespass BNSS FIR investigation commercial establishment shop breaking"),
    ("Concise focused", "BNS theft housebreaking shop breaking property theft"),
    ("Statutory terms", "BNS theft dwelling house building housebreaking lurking house-trespass"),
]

for label, exp in expansions:
    retrieved = pipeline.retrieve(exp, target_namespace="criminal", top_n=20)
    pairs = [[exp, c['text']] for c in retrieved[:20]]
    scores = reranker.predict(pairs)
    for i, c in enumerate(retrieved[:20]):
        c['relevance_score'] = float(scores[i])
    
    reranked = sorted(retrieved[:20], key=lambda x: x['relevance_score'], reverse=True)
    top_score = reranked[0]['relevance_score']
    top_doc = reranked[0]['metadata'].get('document_name', 'Unknown')
    top_sec = reranked[0]['metadata'].get('section_number', reranked[0]['metadata'].get('section', 'N/A'))
    top_snippet = reranked[0]['text'][:100].replace('\n', ' ')
    
    print(f"\n[{label}] score={top_score:.4f} (passed={top_score >= 0.65})")
    print(f"  Expansion: {exp}")
    print(f"  Top hit: {top_doc} (Sec {top_sec}) -> {top_snippet}...")
