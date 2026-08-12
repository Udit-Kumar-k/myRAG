import os, sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline
from src.backend.chain import LegalRAGChain

query = "Someone broke into my shop at night and stole goods worth several lakhs."

im = LegalIndexManager()
im.load_indexes()

rag = LegalRAGChain()
expanded_query = rag.expand_query(query)
print(f"Original query: {query}")
print(f"Expanded query: {expanded_query}")

pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

# Call pipeline.retrieve
retrieved = pipeline.retrieve(expanded_query, target_namespace="criminal", top_n=20)
print(f"\n--- Retrieved RRF Chunks ({len(retrieved)}) ---")
for rank, c in enumerate(retrieved[:10], 1):
    doc = c['metadata'].get('document_name', 'Unknown')
    sec = c['metadata'].get('section_number', c['metadata'].get('section', 'N/A'))
    snippet = c['text'][:120].replace('\n', ' ')
    print(f"  [{rank}] {doc} (Sec {sec}) rrf_score={c['rrf_score']:.4f} -> {snippet}...")

# Rerank
reranker = pipeline.load_reranker()
pairs = [[expanded_query, c['text']] for c in retrieved[:20]]
scores = reranker.predict(pairs)
for i, c in enumerate(retrieved[:20]):
    c['relevance_score'] = float(scores[i])

reranked = sorted(retrieved[:20], key=lambda x: x['relevance_score'], reverse=True)

print(f"\n--- Reranked Top 10 ---")
for rank, chunk in enumerate(reranked[:10], 1):
    doc = chunk['metadata'].get('document_name', 'Unknown')
    sec = chunk['metadata'].get('section_number', chunk['metadata'].get('section', 'N/A'))
    score = chunk['relevance_score']
    snippet = chunk['text'][:150].replace('\n', ' ')
    print(f"  [{rank}] {doc} (Sec {sec}) score={score:.4f} -> {snippet}...")
