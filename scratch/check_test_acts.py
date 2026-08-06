"""Check if specific acts from the 5 test queries are in the indexed corpus."""
import os, sys, pickle
from collections import Counter

sys.path.insert(0, os.path.abspath("."))

INDEX_DIR = "data/indexes"
NAMESPACES = ["criminal", "cyber", "consumer", "banking", "general"]

# Acts we need for the 5 failing queries
SEARCH_TERMS = [
    "negotiable instrument",
    "hindu succession",
    "indian contract",
    "nagarik suraksha",  # BNSS
    "nyaya sanhita",     # BNS 
]

all_act_names = set()
for ns in NAMESPACES:
    chunks_path = os.path.join(INDEX_DIR, f"{ns}_chunks.pkl")
    if not os.path.exists(chunks_path):
        continue
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    for c in chunks:
        all_act_names.add(c["metadata"].get("document_name", "Unknown"))

print(f"Total unique acts across all namespaces: {len(all_act_names)}")
print()

for term in SEARCH_TERMS:
    matches = [a for a in all_act_names if term.lower() in a.lower()]
    if matches:
        print(f"FOUND '{term}': {matches}")
    else:
        print(f"NOT FOUND: '{term}'")
