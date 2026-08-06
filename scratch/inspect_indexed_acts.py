"""Inspect which acts are indexed across all namespaces."""
import os, sys, pickle
from collections import Counter

sys.path.insert(0, os.path.abspath("."))

INDEX_DIR = "data/indexes"
NAMESPACES = ["criminal", "cyber", "consumer", "banking", "general"]

for ns in NAMESPACES:
    chunks_path = os.path.join(INDEX_DIR, f"{ns}_chunks.pkl")
    if not os.path.exists(chunks_path):
        print(f"  {ns}: NOT FOUND")
        continue
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    act_counts = Counter(c["metadata"].get("document_name", "Unknown") for c in chunks)
    print(f"\n=== {ns.upper()} namespace ({len(chunks)} chunks) ===")
    for act, count in act_counts.most_common():
        print(f"  {act}: {count} chunks")
