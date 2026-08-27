import sys, io, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline

im = LegalIndexManager()
im.load_indexes()
pipe = LegalRAGPipeline(im)

q = "my boss is verbally abusing me and making me work overtime"
hyde_text = "Where an employer compels an employee to work beyond the prescribed maximum statutory hours of work, such employee shall be entitled to overtime wages calculated at twice the ordinary rate of wages in accordance with the Code on Wages, 2019. Further, any employer or person in authority who intentionally insults, uses abusive language, or criminally intimidates an employee causing distress commits an offence punishable under Section 351 and Section 352 of the Bharatiya Nyaya Sanhita, 2023."

print("--- Testing candidates from HyDE dense retrieval ---")
candidates = pipe.retrieve(hyde_text, target_namespace="all", top_n=10)
print(f"Retrieved candidates count: {len(candidates)}")
for i, c in enumerate(candidates[:5]):
    doc_name = c.get("metadata", {}).get("document_name")
    print(f"[{i+1}] {doc_name} | RRF: {c.get('rrf_score', 0):.4f}")

print("\n--- Scoring with Reranker using (1) original query vs (2) hyde text ---")
reranker = pipe.load_reranker()
pairs_orig = [[q, c["text"][:1024]] for c in candidates[:5]]
scores_orig = reranker.predict(pairs_orig)

pairs_hyde = [[hyde_text, c["text"][:1024]] for c in candidates[:5]]
scores_hyde = reranker.predict(pairs_hyde)

for i, (c, s_orig, s_hyde) in enumerate(zip(candidates[:5], scores_orig, scores_hyde)):
    doc_name = c.get("metadata", {}).get("document_name")
    print(f"[{i+1}] {doc_name} -> Score with orig query: {s_orig:.4f} | Score with hyde: {s_hyde:.4f}")
