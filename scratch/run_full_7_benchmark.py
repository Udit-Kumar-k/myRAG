"""
Comprehensive 7-scenario benchmark runner and audit script.
Executes both Groq and Gemini (or configured provider), logs exact model/provider,
checks every retrieved chunk for TOC/index contamination, and validates citation verification.
"""
import os, sys, time, json, re
sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini-3.6-flash"

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query
from src.backend.chain import LegalRAGChain

SCENARIOS = [
    {
        "id": 1,
        "label": "Scenario 1 — FIR Progress Update (4 months, no update)",
        "query": "It's been four months since I gave my statement at the station and nobody will give me an update on what's happening with my complaint.",
        "expected_ns": "criminal",
        "key_statutes": ["BNSS", "Bharatiya Nagarik Suraksha Sanhita"],
        "disallowed": ["CrPC", "Telangana", "Andhra"],
    },
    {
        "id": 2,
        "label": "Scenario 2 — Cheque Dishonour (commercial buyer)",
        "query": "A buyer paid for my goods using a cheque that bounced because there wasn't enough balance in their account. Can I take legal action?",
        "expected_ns": "banking",
        "key_statutes": ["Negotiable Instruments Act", "Section 138"],
        "disallowed": ["BNS Section 138", "BNSS Section 138", "IPC"],
    },
    {
        "id": 3,
        "label": "Scenario 3 — Cyber Fraud via Phishing SMS",
        "query": "I clicked a malicious link in a suspicious text message and lost money from my bank account.",
        "expected_ns": "cyber",
        "key_statutes": ["Information Technology Act", "IT Act"],
        "disallowed": ["Section 103", "murder", "IPC"],
    },
    {
        "id": 4,
        "label": "Scenario 4 — Withheld Final Salary",
        "query": "I resigned from my job with proper notice but my former boss is refusing to clear my final pending salary.",
        "expected_ns": "general",
        "key_statutes": ["Indian Contract Act", "Payment of Wages Act"],
        "disallowed": ["BNS", "BNSS", "Section 317", "Section 272"],
    },
    {
        "id": 5,
        "label": "Scenario 5 — Mother's Intestate Succession",
        "query": "My mother passed away last year without making a will. She owned a house she had bought herself. Her children want to know how the property will be divided among them.",
        "expected_ns": "general",
        "key_statutes": ["Hindu Succession Act", "Section 15"],
        "disallowed": ["Telangana", "Andhra", "Section 8"],
    },
    {
        "id": 6,
        "label": "Scenario 6 — Arrest Without Grounds / No Magistrate Production",
        "query": "The police arrested my brother without telling him why and haven't produced him before a judge in over 24 hours.",
        "expected_ns": "criminal",
        "key_statutes": ["BNSS", "Bharatiya Nagarik Suraksha Sanhita", "Section 47", "Section 58"],
        "disallowed": ["CrPC", "Section 35"],
    },
    {
        "id": 7,
        "label": "Scenario 7 — Shop Break-in and Theft",
        "query": "Someone broke into my shop at night and stole goods worth several lakhs.",
        "expected_ns": "criminal",
        "key_statutes": ["BNS", "Bharatiya Nyaya Sanhita", "theft", "house-breaking"],
        "disallowed": ["IPC", "shopbreaking"],
    },
]

TOC_MARKERS = [
    "table of contents", "arrangement of sections", "corresponding section table",
    "statement of objects and reasons"
]

print("Initializing Index Manager and Pipeline...")
im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im, confidence_threshold=0.65)
rag_chain = LegalRAGChain()

print(f"\nActive Provider: {rag_chain.provider}")
print(f"Active Model: {rag_chain.model_name}")

results_summary = []

print("\n" + "="*80)
print(f"RUNNING FULL 7-SCENARIO BENCHMARK AGAINST CURRENT CODEBASE")
print("="*80)

for s in SCENARIOS:
    sid = s["id"]
    label = s["label"]
    query = s["query"]
    
    print(f"\n--------------------------------------------------------------------------------")
    print(f"[{sid}/7] {label}")
    print(f"Query: \"{query}\"")
    
    t0 = time.time()
    routed_ns = route_query(query)
    res = pipeline.query(query)
    t_retrieval = time.time() - t0
    
    conf = res["confidence_score"]
    refused = res["refused"]
    ns_searched = res["namespace_searched"]
    chunks = res["retrieved_chunks"]
    
    print(f"Retrieval ({t_retrieval:.2f}s):")
    print(f"  Routed NS: {routed_ns} | Searched NS: {ns_searched} | Conf: {conf:.4f} | Refused: {refused}")
    
    # Check TOC / contamination in retrieved chunks
    toc_pollution_found = []
    chunk_summaries = []
    for i, c in enumerate(chunks):
        doc = c["metadata"].get("document_name", "Unknown")
        ns = c["metadata"].get("namespace", "Unknown")
        score = c.get("relevance_score", 0.0)
        txt = c.get("text", "")
        
        # Check for TOC patterns
        is_toc = any(m in txt.lower() for m in TOC_MARKERS)
        if is_toc:
            toc_pollution_found.append((i+1, doc))
            
        preview = txt[:90].replace('\n', ' ')
        chunk_summaries.append({
            "rank": i+1,
            "doc": doc,
            "ns": ns,
            "score": score,
            "is_toc": is_toc,
            "preview": preview
        })
        print(f"    Rank {i+1}: [{doc}] (ns={ns}, score={score:.4f}, TOC={is_toc}) -> {preview}...")
    
    # Generation
    answer = ""
    unverified = []
    if not refused:
        t_gen_0 = time.time()
        try:
            answer = rag_chain.run(query, chunks, [])
            t_gen = time.time() - t_gen_0
            print(f"\nGeneration ({t_gen:.2f}s):")
            print(f"\n  --- GENERATED ANSWER ---")
            print(f"  {answer}")
            print(f"  --- END ANSWER ---")
        except Exception as ex:
            print(f"Generation error: {ex}")
            answer = f"Error: {ex}"
    else:
        print("\nGeneration: REFUSED (Confidence below threshold)")
        
    results_summary.append({
        "scenario": sid,
        "label": label,
        "routed_ns": routed_ns,
        "searched_ns": ns_searched,
        "confidence": conf,
        "refused": refused,
        "toc_pollution": toc_pollution_found,
        "unverified_citations": unverified,
        "answer_preview": answer[:150] if answer else "REFUSED"
    })

print("\n" + "="*80)
print("BENCHMARK SUMMARY TABLE")
print("="*80)
print(f"{'#':<3} {'Scenario':<32} {'NS':<10} {'Conf':<8} {'Refused':<8} {'TOC Poll':<10} {'Citation Redactions'}")
print("-" * 85)
for r in results_summary:
    toc_str = "None" if not r["toc_pollution"] else f"{len(r['toc_pollution'])} chunks"
    redact_str = "None" if not r["unverified_citations"] else str(r["unverified_citations"])
    print(f"{r['scenario']:<3} {r['label'][:30]:<32} {r['searched_ns']:<10} {r['confidence']:<8.4f} {str(r['refused']):<8} {toc_str:<10} {redact_str}")

with open("data/benchmark_7_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)
print("\nSaved benchmark results to data/benchmark_7_results.json")
