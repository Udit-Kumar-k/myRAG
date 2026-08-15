"""
Re-indexes and rebuilds FAISS & BM25 indexes for cyber, banking, and general namespaces
using the cleaned assign_namespace function.
"""
import os, sys, pickle, glob
import numpy as np

sys.path.insert(0, os.path.abspath("."))
os.environ["EMBEDDING_DEVICE"] = "cpu"

from src.backend.ingestion import assign_namespace
from src.backend.indexing import LegalIndexManager, tokenize_for_bm25
from rank_bm25 import BM25Okapi
import faiss

print("=== Re-partitioning chunks across namespaces ===")
all_chunks = []
for p in glob.glob("data/indexes/*_chunks.pkl"):
    with open(p, "rb") as f:
        chunks = pickle.load(f)
        all_chunks.extend(chunks)

print(f"Total existing chunks across all files: {len(all_chunks)}")

# Re-assign namespaces
ns_chunks = {"criminal": [], "cyber": [], "consumer": [], "banking": [], "general": []}
for c in all_chunks:
    doc = c["metadata"].get("act_name", c["metadata"].get("document_name", ""))
    new_ns = assign_namespace(doc)
    c["metadata"]["namespace"] = new_ns
    c["metadata"]["legal_domain"] = new_ns
    ns_chunks[new_ns].append(c)

for ns, chunks in ns_chunks.items():
    acts = set(c["metadata"]["document_name"] for c in chunks)
    print(f"\nNamespace '{ns}': {len(chunks)} chunks, {len(acts)} acts")
    if ns in ["cyber", "banking", "consumer"]:
        for a in sorted(acts):
            cnt = sum(1 for c in chunks if c["metadata"]["document_name"] == a)
            print(f"  - {a} ({cnt} chunks)")

im = LegalIndexManager()
model = im.load_embedding_model()

# Rebuild cyber, banking, and general
for ns in ["cyber", "banking"]:
    chunks = ns_chunks[ns]
    print(f"\n--- Building clean indexes for {ns} ({len(chunks)} chunks) ---")
    
    # 1. BM25
    tokenized = [tokenize_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    
    # 2. FAISS
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    
    faiss_idx = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_idx.add(embeddings)
    
    im.save_namespace(ns, chunks, faiss_idx, bm25)

print("\nSuccessfully updated cleaned indexes for cyber and banking.")
