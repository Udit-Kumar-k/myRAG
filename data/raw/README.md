# NyayBot — PDF Corpus

Place the following 3 PDF files in this directory before running ingestion:

1. **`BNSag.pdf`** — Bharatiya Nyaya Sanhita 2023 (replaces IPC)
2. **`BNSSrag.pdf`** — Bharatiya Nagarik Suraksha Sanhita 2023 (replaces CrPC)
3. **`BSArag.pdf`** — Bharatiya Sakshya Adhiniyam 2023 (replaces Indian Evidence Act)

## Where to download

- Official gazette: [egazette.gov.in](https://egazette.gov.in)
- India Code: [indiacode.nic.in](https://www.indiacode.nic.in)
- Legislative Department: [legislative.gov.in](https://legislative.gov.in)

## After placing files

Run the ingestion pipeline:

```bash
python -m src.backend.indexing
```

This will:
1. Extract text from each PDF using pymupdf
2. Split on section boundaries (`Section 1.`, `Section 2.`, etc.)
3. Load supplementary acts from HuggingFace (`geekyrakshit/indian-legal-acts`)
4. Filter out repealed acts (IPC, CrPC, Evidence Act)
5. Build FAISS and BM25 indexes per namespace
