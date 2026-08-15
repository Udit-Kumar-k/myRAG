---
title: NyayBot
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

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

## Corpus Scope & Supported Domains

| Subject Domain | Primary Acts Covered | Namespace | Status |
|----------------|----------------------|-----------|--------|
| **Substantive Criminal Law** | Bharatiya Nyaya Sanhita 2023 (BNS) | `criminal` | Verified & Hardened |
| **Criminal Procedure & Arrest** | Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS) | `criminal` | Verified & Hardened |
| **Law of Evidence** | Bharatiya Sakshya Adhiniyam 2023 (BSA) | `criminal` | Verified & Hardened |
| **Cyber Law & Online Offences** | Information Technology Act, 2000 | `cyber` | Verified & Hardened |
| **Consumer Protection** | Consumer Protection Act, 2019 | `consumer` | Verified & Hardened |
| **Commercial Cheque Dishonour** | Negotiable Instruments Act, 1881 (s.138) | `banking` | Verified & Hardened |
| **Banking Regulation & RBI** | RBI Act 1934, Banking Regulation Act 1949 | `banking` | Verified & Hardened |
| **Civil Contracts & Unpaid Salary** | Indian Contract Act 1872, Payment of Wages Act 1936 | `general` | Verified & Hardened |
| **Intestate Property Succession** | Hindu Succession Act 1956 (s.15/s.8) | `general` | Verified & Hardened |
| **Other Central Legislation** | ~860 central statutes from indiacode.nic.in | `general` | Available via general namespace |

### Explicit Out-of-Scope & Untested Domains

The system enforces strict confidence gating (threshold: 0.65) and will **refuse** queries outside indexed statutory law. The following areas are explicitly **OUT OF SCOPE**:

1. **State-Specific Amendments & Local Enactments:** State tenancy acts (e.g. Maharashtra Rent Control, Delhi Rent Control), local municipal bylaws, and state-specific amendments unless explicitly indexed.
2. **Case Law & Judicial Precedents:** Supreme Court and High Court judgments, ratio decidendi, and citations (e.g. *AIR*, *SCC*, *SCR*).
3. **Motor Vehicle Accidents:** Motor Vehicles Act third-party compensation claims, MACT tribunal procedures, and traffic challan appeals.
4. **Matrimonial & Family Dispute Procedures:** Contested divorce trial proceedings, child custody battles, and maintenance applications under state family court rules.
5. **Taxation & Corporate Filings:** Income Tax Act 1961 assessment procedures, GST tribunal appeals, Customs disputes, and ROC compliance.
6. **Ongoing Case Strategy & Procedural Litigation:** Procedural timelines, drafting formats, and legal strategy for pending court cases.

> **Repealed Acts Filter:** The Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), and Indian Evidence Act were repealed on July 1, 2024. All queries referencing repealed acts are automatically mapped to their BNS/BNSS/BSA counterparts and never cited as current law.

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
