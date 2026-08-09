"""
End-to-end generation benchmark across 5 rephrased legal scenarios.
Tests BOTH retrieval (confidence, routing) AND generation (actual LLM answer text).
Specifically designed to catch generation-time failures:
  - Fabricated state amendments (e.g. invented "Telangana Amendment")
  - Succession arithmetic errors (N heirs must receive 1/N shares)
  - Hallucinated section numbers not present in retrieved context
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query
from src.backend.chain import LegalRAGChain, format_context

# 5 rephrased scenarios — no verbatim overlap with any few-shot example in the prompt.
# Each includes a GENERATION AUDIT NOTE explaining what to check in the LLM answer.
SCENARIOS = [
    {
        "label": "Scenario 1 — FIR Progress Update (4 months, no update)",
        "query": "It's been four months since I gave my statement at the station and nobody will give me an update on what's happening with my complaint.",
        "audit": [
            "Must cite BNSS (not CrPC) for the police notification obligation",
            "Must NOT mention any state-specific (Telangana/AP) amendment unless it appears in a retrieved chunk",
            "Must recommend consulting a lawyer",
        ],
    },
    {
        "label": "Scenario 2 — Cheque Dishonour (commercial buyer)",
        "query": "A buyer paid for my goods using a cheque that bounced because there wasn't enough balance in their account. Can I take legal action?",
        "audit": [
            "Must cite Negotiable Instruments Act Section 138 (not 'BNS Section 138')",
            "Must mention the 30-day notice requirement before filing complaint",
            "Must NOT attribute Section 138 to BNS or BNSS",
        ],
    },
    {
        "label": "Scenario 3 — Cyber Fraud via Phishing SMS",
        "query": "I clicked a malicious link in a suspicious text message and lost money from my bank account.",
        "audit": [
            "Must cite IT Act (Information Technology Act, 2000) Sections — not IPC",
            "Must NOT cite BNS Section 103 (murder) or any irrelevant BNS provision",
        ],
    },
    {
        "label": "Scenario 4 — Withheld Final Salary",
        "query": "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary.",
        # FALSIFIABLE: refused is NOT a passing outcome.
        # Root cause of refusal is BNS Section 317 contamination in expansion prompt (now fixed).
        # If this still refuses after the fix, retrieval for salary disputes is broken.
        "audit": [
            "MUST be answered (not refused) — confidence must reach >= 0.65",
            "Must cite Indian Contract Act Section 73 and/or Payment of Wages Act — not BNS criminal law",
            "Must NOT cite BNS Sections (e.g. 317, 272, 340) for a salary dispute",
        ],
    },
    {
        "label": "Scenario 5 — Mother's Intestate Succession",
        # Headcount NOT given — model must identify heirs from context and state equal division.
        "query": "My mother passed away last year without making a will. She owned a house she had bought herself. Her children want to know how the property will be divided among them.",
        "audit": [
            "Must cite Hindu Succession Act Section 15 (female Hindu intestate) — NOT Section 8 (male)",
            "Must state children inherit equally — must NOT fabricate unequal shares between sons and daughter",
            "Must NOT invent a Telangana or AP state amendment",
        ],
    },
]

# Load indexes
print("Loading indexes...")
im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)
rag_chain = LegalRAGChain()

print("\n" + "="*70)
print("END-TO-END GENERATION BENCHMARK")
print("="*70)

for s in SCENARIOS:
    label = s["label"]
    query = s["query"]
    audit_points = s["audit"]
    
    print(f"\n{'='*70}")
    print(f"[{label}]")
    print(f"Query: \"{query}\"")
    
    # --- Step 1: Retrieval ---
    routed_ns = route_query(query)
    print(f"\nRetrieval:")
    print(f"  Routed namespace: {routed_ns}")
    result = pipeline.query(query)
    conf = result["confidence_score"]
    refused = result["refused"]
    ns_searched = result["namespace_searched"]
    chunks = result["retrieved_chunks"]
    
    print(f"  Confidence score: {conf:.4f}")
    print(f"  Refused: {refused}")
    print(f"  Namespace searched: {ns_searched}")
    
    if not refused and chunks:
        print(f"  Retrieved sources:")
        for j, chunk in enumerate(chunks[:3]):
            meta = chunk["metadata"]
            print(f"    [{j+1}] {meta['document_name']} (score={chunk['relevance_score']:.4f})")
    
    # --- Step 2: Generation ---
    if refused:
        print(f"\nGeneration: SKIPPED (refused by confidence gate at {conf:.4f})")
    else:
        print(f"\nGeneration (calling LegalRAGChain.run):")
        try:
            answer = rag_chain.run(
                question=query,
                context_chunks=chunks,
                history=[],
            )
            print(f"\n--- ANSWER ---")
            print(answer)
            print(f"--- END ANSWER ---")
        except Exception as e:
            print(f"  Generation error: {e}")
    
    # --- Step 3: Audit checklist ---
    print(f"\nAudit checklist (MANUAL REVIEW REQUIRED):")
    for point in audit_points:
        print(f"  [ ] {point}")

print(f"\n{'='*70}")
print("Benchmark complete.")
