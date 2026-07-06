# NyayBot

> Hybrid RAG legal awareness assistant over Indian statutory law — BNS, BNSS, BSA, IT Act, Consumer Protection Act.

NyayBot is a Retrieval-Augmented Generation (RAG) system that answers legal questions grounded exclusively in Indian statutory law. It uses a hybrid retrieval pipeline combining dense semantic search with sparse keyword matching, fused via Reciprocal Rank Fusion and validated through cross-encoder reranking with a confidence gate.

**This is a legal awareness tool, not a lawyer.** Always consult a qualified legal professional for specific legal advice.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         NyayBot Pipeline                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Query                                                     │
│       │                                                          │
│       ▼                                                          │
│   ┌──────────────┐                                               │
│   │ Namespace    │──→ criminal | cyber | consumer | banking | all│
│   │ Router       │                                               │
│   └──────┬───────┘                                               │
│          │                                                       │
│    ┌─────┴─────┐                                                 │
│    │           │                                                 │
│    ▼           ▼                                                 │
│  ┌─────────┐ ┌─────────┐                                        │
│  │ FAISS   │ │ BM25    │                                        │
│  │ Dense   │ │ Sparse  │                                        │
│  │ (BGE-M3)│ │ (Okapi) │                                        │
│  └────┬────┘ └────┬────┘                                        │
│       │           │                                              │
│       └─────┬─────┘                                              │
│             ▼                                                    │
│   ┌──────────────────┐                                           │
│   │ RRF Fusion       │ Reciprocal Rank Fusion + temporal boost   │
│   └────────┬─────────┘                                           │
│            ▼                                                     │
│   ┌──────────────────┐                                           │
│   │ Cross-Encoder    │ BAAI/bge-reranker-v2-m3                   │
│   │ Reranker         │                                           │
│   └────────┬─────────┘                                           │
│            ▼                                                     │
│   ┌──────────────────┐                                           │
│   │ Confidence Gate  │ Threshold: 0.65 (calibrated)              │
│   └────────┬─────────┘                                           │
│            │                                                     │
│     ┌──────┴──────┐                                              │
│     │ PASS        │ REFUSE                                       │
│     ▼             ▼                                              │
│   ┌────────┐  ┌────────────────────┐                             │
│   │ LLM    │  │ "Insufficient      │                             │
│   │ Answer │  │  evidence" refusal │                             │
│   └────────┘  └────────────────────┘                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Corpus

| Act | Namespace | Source | Why Included |
|-----|-----------|--------|-------------|
| Bharatiya Nyaya Sanhita 2023 (BNS) | `criminal` | PDF | Replaces IPC — substantive criminal law, in force July 1 2024 |
| Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS) | `criminal` | PDF | Replaces CrPC — criminal procedure |
| Bharatiya Sakshya Adhiniyam 2023 (BSA) | `criminal` | PDF | Replaces Indian Evidence Act — law of evidence |
| Information Technology Act 2000 | `cyber` | HuggingFace | Cybercrime, data protection, intermediary liability |
| Consumer Protection Act 2019 | `consumer` | HuggingFace | Consumer rights, product liability, e-commerce |
| Negotiable Instruments Act 1881 | `banking` | HuggingFace | Cheque bounce, promissory notes |
| Other central/state acts | `general` | HuggingFace | Various Indian legal acts from indiacode.nic.in |

> **Excluded:** Indian Penal Code, Code of Criminal Procedure, Indian Evidence Act (all repealed July 1 2024). Filtered out during ingestion.

## Chunking Strategy

- **PDFs (BNS, BNSS, BSA):** Extracted with `pymupdf`, split on section boundaries using regex `(?=\b[Ss]ection\s+\d+[A-Z]?\.)`. Chunks under 50 characters are discarded as noise.
- **HuggingFace dataset (`geekyrakshit/indian-legal-acts`):** Auto-detects whether rows are full acts (>5000 chars → apply section-boundary chunking) or pre-chunked sections (use directly). Repealed acts filtered out.

## Retrieval Pipeline

1. **BGE-M3 Embeddings** — Dense vector representations via `BAAI/bge-m3`
2. **FAISS IndexFlatIP** — Inner product search over normalized embeddings (equivalent to cosine similarity)
3. **BM25 (Okapi)** — Sparse keyword matching via `rank-bm25`
4. **Reciprocal Rank Fusion (RRF)** — Merges dense and sparse result lists with temporal boost for newer legislation
5. **Cross-Encoder Reranking** — `BAAI/bge-reranker-v2-m3` rescores top candidates
6. **Confidence Gate** — Top-1 reranker score must exceed calibrated threshold (0.65) or query is refused

## Tech Stack

- **Backend:** FastAPI + LangChain + Gemini/Groq LLMs
- **Frontend:** React + Vite
- **Database:** Supabase (PostgreSQL) with local JSON fallback
- **Auth:** Supabase Auth (Google OAuth + email/password)
- **CI/CD:** GitHub Actions
- **Evaluation:** RAGAS-style context recall + confidence calibration

## Quick Start

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Place PDFs in data/raw/ (see data/raw/README.md)
# Set API keys in .env (copy from .env.template)

# Build indexes
python -m src.backend.indexing

# Start server
uvicorn src.backend.main:app --host 0.0.0.0 --port 8001
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Known Limitations

- **State-specific laws** are not indexed — only central acts
- **Court judgments and case law** are not included — statutory text only
- **Pre-July 2024 corpus** for non-criminal acts (IT Act, Consumer Protection Act etc.) may not reflect latest amendments
- **Tax law** is not covered
- **Legal procedure advice** (e.g., "how to file a case") is limited to what's in the statutory text
- The system is a legal awareness tool — **not a substitute for professional legal advice**

## License

See [LICENSE](LICENSE) for details.
