import os, sys, time
sys.path.insert(0, os.path.abspath("."))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini-3.6-flash"

from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline, route_query
from src.backend.chain import LegalRAGChain

print("1. Initializing IndexManager and Pipeline...", flush=True)
im = LegalIndexManager()
im.load_indexes()
pipeline = LegalRAGPipeline(im)
chain = LegalRAGChain()

print(f"2. Chain initialized: Provider={chain.provider}, Model={chain.model_name}", flush=True)

test_queries = [
    ("Cheque Dishonour", "A buyer paid for my goods using a cheque that bounced because there wasn't enough balance in their account. Can I take legal action?"),
    ("Cyber Phishing", "I clicked a malicious link in a suspicious text message and lost money from my bank account."),
    ("Mother Succession", "My mother passed away without a will. How is her property divided among children?")
]

for label, q in test_queries:
    t0 = time.time()
    print(f"\n--- Running: {label} ---", flush=True)
    exp = chain.expand_query(q)
    print(f"Expansion: {exp[:60]}...", flush=True)
    res = pipeline.query(q)
    chunks = res["chunks"]
    conf = res["confidence_score"]
    ns = res["namespace_searched"]
    print(f"Retrieved {len(chunks)} chunks from ns='{ns}', confidence={conf:.4f}", flush=True)
    ans = chain.run(q, chunks, [])
    elapsed = time.time() - t0
    print(f"Answer ({elapsed:.2f}s):\n{ans[:200]}...\n", flush=True)

print("ALL BENCHMARK TESTS COMPLETED SUCCESSFULLY!", flush=True)
