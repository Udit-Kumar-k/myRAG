"""
Inspect what the reranker actually does with the 3 queries that were
refused despite context_recalled=true (criminal_04, consumer_03, banking_02).

pipeline.query() discards retrieved_chunks when refused, so this replicates
its internal steps (route -> retrieve -> rerank) without discarding, and
prints the full top-5 post-rerank so we can see whether the right chunk
made it to #1 with a genuinely low score, or got outranked by something
irrelevant.
"""
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query

TARGET_QUERIES = [
    "What is the procedure for filing an FIR under BNSS?",
    "Can a landlord refuse to return a tenant's security deposit?",
    "What are the RBI guidelines on digital payment fraud?",
]

print("Loading indexes...")
index_manager = LegalIndexManager()
if not index_manager.load_indexes():
    raise SystemExit("Failed to load indexes.")

pipeline = LegalRAGPipeline(index_manager)
reranker = pipeline.load_reranker()

for q in TARGET_QUERIES:
    namespace = route_query(q)
    candidates = pipeline.retrieve(q, target_namespace=namespace, top_n=20)
    print(f"\n{'='*70}\nQUERY: {q}\nnamespace_searched: {namespace}\ncandidates from retrieve(): {len(candidates)}")

    if not candidates:
        print("  NO CANDIDATES RETURNED — retrieval itself found nothing.")
        continue

    pairs = [[q, c["text"][:2048]] for c in candidates]
    scores = reranker.predict(pairs, batch_size=4)
    for c, s in zip(candidates, scores):
        c["relevance_score"] = float(s)
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    print("Top 5 after rerank:")
    for i, c in enumerate(candidates[:5]):
        snippet = c["text"][:180].replace("\n", " ")
        print(f"  #{i+1} score={c['relevance_score']:.4f}  {snippet}...")
