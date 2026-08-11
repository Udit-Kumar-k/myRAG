"""
Lightweight salary diagnostic — embedding similarity only, no reranker.
Avoids the BAAI/bge-reranker-v2-m3 CPU load that causes os-error-1455 OOM crashes.

Tests whether Indian Contract Act chunks appear in embedding top-k for:
  T1: expanded query (current pipeline) 
  T2: raw query (bypass expand_query)
  T3: raw query, general namespace only
  T4: expanded query, general namespace only

The embedding cosine scores are not identical to reranker scores, but the
presence/absence of Indian Contract Act in top-k is the key signal either way.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query
from src.backend.chain import LegalRAGChain

SALARY_QUERY = "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary."

print("Loading indexes...")
im = LegalIndexManager()
im.load_indexes()
# threshold=0.0 ensures we never refuse — we want to see raw scores
pipeline = LegalRAGPipeline(im, confidence_threshold=0.0)

print(f"\nQuery: \"{SALARY_QUERY}\"")

# Expand once
print("\nExpanding query...")
rag = LegalRAGChain()
expanded = rag.expand_query(SALARY_QUERY)
print(f"Expansion: {expanded}")

def run_embedding_only(label, query, namespace):
    """Retrieve top-20 by embedding similarity (no reranker), show top-5."""
    print(f"\n{'='*60}")
    print(f"{label}")
    # retrieve() runs dual BM25+embedding search + RRF merge; returns
    # candidates sorted by RRF score (embedding cosine is the primary signal).
    candidates = pipeline.retrieve(query, target_namespace=namespace, top_n=20)
    if not candidates:
        print("  No candidates.")
        return []
    top = candidates[:5]
    for i, c in enumerate(top):
        doc = c["metadata"].get("document_name", c["metadata"].get("act_name", "?"))
        # RRF score is not in [0,1] but higher = more relevant
        rrf = c.get("rrf_score", c.get("score", "?"))
        print(f"  [{i+1}] {doc} (rrf={rrf})")
    # Check if Indian Contract Act appears anywhere in top-20
    ica_hits = [c for c in candidates if "Contract" in c["metadata"].get("document_name", "")]
    print(f"  Indian Contract Act in top-20: {len(ica_hits)} chunk(s)")
    if ica_hits:
        for h in ica_hits[:3]:
            print(f"    -> rank {candidates.index(h)+1}: {h['metadata'].get('document_name','?')}")
    return candidates

# Note: retrieve() needs the embedding model; load happens on first call
print("\n[First call loads BGE-M3 on CUDA]")
c1 = run_embedding_only("TEST 1: Expanded / all",     expanded,     "all")
c2 = run_embedding_only("TEST 2: Raw      / all",     SALARY_QUERY, "all")
c3 = run_embedding_only("TEST 3: Raw      / general", SALARY_QUERY, "general")
c4 = run_embedding_only("TEST 4: Expanded / general", expanded,     "general")

print(f"\n{'='*60}")
print("SUMMARY (embedding/RRF scores only — no reranker)")
print(f"  T1 ICA in top-20: {len([c for c in c1 if 'Contract' in c['metadata'].get('document_name','')])} chunks")
print(f"  T2 ICA in top-20: {len([c for c in c2 if 'Contract' in c['metadata'].get('document_name','')])} chunks")
print(f"  T3 ICA in top-20: {len([c for c in c3 if 'Contract' in c['metadata'].get('document_name','')])} chunks")
print(f"  T4 ICA in top-20: {len([c for c in c4 if 'Contract' in c['metadata'].get('document_name','')])} chunks")
print()
print("If ICA appears in T2/T3 but not T1 → expansion is burying the ICA chunks")
print("If ICA absent in all → corpus genuinely lacks salary-relevant ICA text")
