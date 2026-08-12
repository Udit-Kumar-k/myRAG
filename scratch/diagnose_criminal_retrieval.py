"""
Diagnostic: arrest/detention and property crimes — unexpanded vs expanded, side by side.

Rationale (from salary case): the gap between unexpanded and expanded scores is what
distinguishes RRF dilution from corpus gap. A baseline alone can't tell you that.

Hypothesis for criminal namespace: BNS (substantive offences) likely outnumbers BNSS
(procedure/rights) in the criminal index, same way BNS outnumbered ICA in 'all'.
If so, arrest/detention queries hitting BNSS sections (47, 58) could be getting
buried by BNS murder/theft chunks in the RRF merge — the same mechanism that
buried ICA at rank 13 in 'all', now showing up inside the criminal namespace.

Uses pipeline.query() exclusively (known-working path). Monkey-patches expand_query()
to return raw query for the unexpanded variant.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from collections import Counter
import src.backend.retrieval as retrieval_module
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query
from src.backend.chain import LegalRAGChain

QUERIES = {
    "arrest_detention": (
        "The police arrested my brother without telling him why "
        "and haven't produced him before a judge in over 24 hours."
    ),
    "property_crime": (
        "Someone broke into my shop at night and stole goods worth several lakhs."
    ),
}

# Expected: BNSS ss.47+58 for arrest; BNS burglary/theft for property crime

print("Loading indexes...")
im = LegalIndexManager()
im.load_indexes()

# --- Namespace chunk distribution ---
print("\n" + "="*60)
print("NAMESPACE CHUNK DISTRIBUTION")
print("="*60)
for ns in im.namespaces:
    chunks = im.chunks.get(ns, [])
    doc_counts = Counter(c["metadata"].get("document_name", "?") for c in chunks)
    print(f"\n  [{ns}] — {len(chunks)} total chunks")
    for doc, count in sorted(doc_counts.items(), key=lambda x: -x[1])[:6]:
        print(f"    {count:4d}  {doc}")

pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)

def run_pair(label, query):
    """Run unexpanded then expanded for the same query; compare results."""
    print(f"\n{'='*60}")
    print(f"DOMAIN: {label}")
    print(f"Query: {query}")
    print(f"Raw routing: {route_query(query)}")

    # ── Unexpanded: monkey-patch expand_query to return raw query ──────────
    orig_expand = LegalRAGChain.expand_query
    LegalRAGChain.expand_query = lambda self, q: q
    try:
        r_raw = pipeline.query(query)
    finally:
        LegalRAGChain.expand_query = orig_expand

    conf_raw = r_raw["confidence_score"]
    ns_raw   = r_raw["namespace_searched"]
    refused_raw = r_raw["refused"]
    chunks_raw  = r_raw["retrieved_chunks"]

    print(f"\n  [UNEXPANDED]  confidence={conf_raw:.4f}  namespace={ns_raw}  refused={refused_raw}")
    for i, c in enumerate((chunks_raw or [])[:5]):
        doc = c["metadata"].get("document_name", "?")
        print(f"    [{i+1}] {doc} (score={c['relevance_score']:.4f})")

    # ── Expanded: standard pipeline call ──────────────────────────────────
    r_exp = pipeline.query(query)

    conf_exp = r_exp["confidence_score"]
    ns_exp   = r_exp["namespace_searched"]
    refused_exp = r_exp["refused"]
    chunks_exp  = r_exp["retrieved_chunks"]

    # Grab expanded text from pipeline stdout already logged, but also re-run
    # expand_query once just for display.
    try:
        rag = LegalRAGChain()
        expanded_text = rag.expand_query(query)
    except Exception:
        expanded_text = "(expansion failed)"

    print(f"\n  [EXPANDED]    confidence={conf_exp:.4f}  namespace={ns_exp}  refused={refused_exp}")
    print(f"  Expansion: {expanded_text}")
    for i, c in enumerate((chunks_exp or [])[:5]):
        doc = c["metadata"].get("document_name", "?")
        print(f"    [{i+1}] {doc} (score={c['relevance_score']:.4f})")

    # ── Interpretation ────────────────────────────────────────────────────
    delta = conf_exp - conf_raw
    print(f"\n  DELTA (expanded - raw): {delta:+.4f}")
    if abs(delta) < 0.05:
        interp = "expansion is neutral — routing/namespace is the dominant variable"
    elif delta > 0:
        interp = "expansion HELPS — vocabulary bridge is doing real work"
    else:
        interp = "expansion HURTS — fabricated/mismatched terms degrading score"
    print(f"  Interpretation: {interp}")

    return conf_raw, conf_exp

# ── Run both domains ──────────────────────────────────────────────────────
scores = {}
print("\n[Models load on first pipeline.query() call]")
for domain, query in QUERIES.items():
    scores[domain] = run_pair(domain, query)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'Domain':<20} {'Raw':>8} {'Expanded':>10} {'Delta':>8} {'Gate(0.65)':>12}")
print("-"*60)
for domain, (raw, exp) in scores.items():
    gate_raw = "PASS" if raw >= 0.65 else "FAIL"
    gate_exp = "PASS" if exp >= 0.65 else "FAIL"
    delta = exp - raw
    print(f"  {domain:<18} {raw:>8.4f} {exp:>10.4f} {delta:>+8.4f}  {gate_raw}->{gate_exp}")

print(f"\nDilution hypothesis check:")
print("  If arrest_detention raw > expanded:")
print("  -> BNSS chunks are reachable from plain language, expansion injects BNS terms that outrank them")
print("  If arrest_detention raw < expanded:")
print("  -> expansion is doing real bridging work (BNS keyword advantage is helping BNSS retrieval)")
print("  Either way, compare against salary result: raw=0 chunks ICA, expanded+general=rank2 ICA")
