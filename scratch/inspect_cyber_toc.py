import pickle

with open('data/indexes/cyber_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

for i, c in enumerate(chunks):
    t = c['text']
    doc = c['metadata'].get('document_name', '')
    if any(m in t.lower() for m in ['the first schedule', 'the second schedule', 'table of contents', 'arrangement of sections']):
        print(f"Chunk [{i}] ({doc}):")
        print(t[:400])
        print("="*60)
