import pickle

with open("data/indexes/general_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Total general chunks: {len(chunks)}")
wages_chunks = [c for c in chunks if "payment of wages" in c.get("metadata", {}).get("document_name", "").lower() or "wages act" in c.get("metadata", {}).get("act_name", "").lower()]
print(f"Total Payment of Wages chunks: {len(wages_chunks)}")

for i, c in enumerate(wages_chunks):
    text = c["text"]
    first_line = text.split("\n")[0][:100]
    print(f"  Chunk {i+1}: len={len(text)} | {first_line}")
    if any(k in text.lower() for k in ["section 15", "15. claims", "deduction", "delayed wages", "delayed payment", "section 5", "time of payment"]):
        print(f"    -> FOUND KEY SECTION: {text[:200]}...\n")
