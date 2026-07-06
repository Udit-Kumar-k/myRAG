import os
import re
from typing import List, Dict, Any, Optional


# ── Namespace assignment ─────────────────────────────────────────────────────

def assign_namespace(act_name: str) -> str:
    """Maps an Indian legal act name to one of five subject namespaces."""
    act = act_name.lower()
    if any(x in act for x in ["nyaya sanhita", "nagarik suraksha", "sakshya"]):
        return "criminal"
    elif any(x in act for x in ["information technology", "it act"]):
        return "cyber"
    elif "consumer" in act:
        return "consumer"
    elif any(x in act for x in ["rbi", "reserve bank", "payment"]):
        return "banking"
    else:
        return "general"


# ── Repealed acts filter ─────────────────────────────────────────────────────

REPEALED_ACT_KEYWORDS = [
    "indian penal code",
    "code of criminal procedure",
    "indian evidence act",
]


def is_repealed(act_name: str) -> bool:
    """Returns True if the act is one of the three repealed statutes."""
    name_lower = act_name.lower()
    return any(kw in name_lower for kw in REPEALED_ACT_KEYWORDS)


# ── PDF chunking by section boundary ─────────────────────────────────────────

SECTION_BOUNDARY_RE = re.compile(r'(?=\b[Ss]ection\s+\d+[A-Z]?\.)')


def chunk_pdf(pdf_path: str, act_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF using pymupdf (fitz) and splits on section
    boundaries. Each chunk carries metadata for downstream indexing.
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Split on section boundary regex
    raw_chunks = SECTION_BOUNDARY_RE.split(full_text)

    namespace = assign_namespace(act_name)
    chunks: List[Dict[str, Any]] = []

    for raw in raw_chunks:
        text = raw.strip()
        if len(text) < 50:
            continue  # Skip noise/headers

        chunks.append({
            "text": text,
            "metadata": {
                "document_name": act_name,
                "act_name":      act_name,
                "namespace":     namespace,
                "source":        "pdf",
                "pub_year":      2023,
                "source_url":    "https://indiacode.nic.in",
                "geography_iso": namespace,
            }
        })

    print(f"  {act_name}: {len(chunks)} chunks from PDF")
    return chunks


# ── HuggingFace dataset loading ──────────────────────────────────────────────

def load_hf_legal_acts(
    dataset_name: str = "geekyrakshit/indian-legal-acts",
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Loads the Indian legal acts dataset from HuggingFace.
    Filters out repealed acts (IPC, CrPC, Evidence Act).
    Auto-detects whether rows need section-boundary chunking or are pre-chunked.
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split="central", token=hf_token)
    print(f"Loaded {len(ds)} rows from HuggingFace dataset.")

    # Inspect row structure to decide chunking strategy
    sample_text = (ds[0].get("Markdown") or ds[0].get("text") or "") if len(ds) > 0 else ""
    needs_chunking = len(sample_text) > 5000
    if needs_chunking:
        print("  Rows are full acts (>5000 chars) — applying section-boundary chunking.")
    else:
        print("  Rows are pre-chunked sections (<= 5000 chars) — using directly.")

    chunks: List[Dict[str, Any]] = []
    namespace_counts: Dict[str, int] = {}
    filtered_count = 0

    for row in ds:
        act_name = row.get("Short Title") or row.get("act_name") or row.get("title", "Unknown Act")
        text = row.get("Markdown") or row.get("text") or row.get("content") or row.get("contents", "")

        # Filter out repealed acts
        if is_repealed(act_name):
            filtered_count += 1
            continue

        if not text or not text.strip():
            continue

        namespace = assign_namespace(act_name)

        if needs_chunking:
            # Large rows need section-boundary chunking
            raw_chunks = SECTION_BOUNDARY_RE.split(text)
            for raw in raw_chunks:
                chunk_text = raw.strip()
                if len(chunk_text) < 50:
                    continue
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "document_name": act_name,
                        "act_name":      act_name,
                        "namespace":     namespace,
                        "source":        "huggingface",
                        "pub_year":      2020,
                        "source_url":    "https://huggingface.co/datasets/geekyrakshit/indian-legal-acts",
                        "geography_iso": namespace,
                    }
                })
                namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1
        else:
            # Pre-chunked rows can be used directly
            if len(text.strip()) < 50:
                continue
            chunks.append({
                "text": text.strip(),
                "metadata": {
                    "document_name": act_name,
                    "act_name":      act_name,
                    "namespace":     namespace,
                    "source":        "huggingface",
                    "pub_year":      2020,
                    "source_url":    "https://huggingface.co/datasets/geekyrakshit/indian-legal-acts",
                    "geography_iso": namespace,
                }
            })
            namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1

    print(f"  Filtered out {filtered_count} repealed act rows (IPC/CrPC/Evidence Act).")
    print(f"  HuggingFace namespace distribution:")
    for ns, count in sorted(namespace_counts.items()):
        print(f"    {ns}: {count} chunks")
    print(f"  Total HF chunks: {len(chunks)}")
    return chunks


# ── Main entry point ─────────────────────────────────────────────────────────

# PDF corpus definition: 3 primary statutes
PDF_CORPUS = [
    ("data/raw/BNSag.pdf",  "Bharatiya Nyaya Sanhita 2023"),
    ("data/raw/BNSSrag.pdf", "Bharatiya Nagarik Suraksha Sanhita 2023"),
    ("data/raw/BSArag.pdf",  "Bharatiya Sakshya Adhiniyam 2023"),
]


def process_corpus(
    hf_token: Optional[str] = None,
    **_kwargs,
) -> List[Dict[str, Any]]:
    """
    Main ingestion entry point — loads and chunks all legal corpus sources.
    Drop-in replacement for the old MedRAG process_corpus.
    """
    all_chunks: List[Dict[str, Any]] = []

    # STEP 1: Chunk PDFs by section boundary
    print("\n=== STEP 1: Chunking PDFs by section boundary ===")
    for pdf_path, act_name in PDF_CORPUS:
        if os.path.exists(pdf_path):
            pdf_chunks = chunk_pdf(pdf_path, act_name)
            all_chunks.extend(pdf_chunks)
        else:
            print(f"  WARNING: {pdf_path} not found. Skipping.")

    # STEP 2: Load and filter HuggingFace dataset
    print("\n=== STEP 2: Loading HuggingFace legal acts dataset ===")
    try:
        hf_chunks = load_hf_legal_acts(hf_token=hf_token)
        all_chunks.extend(hf_chunks)
    except Exception as e:
        print(f"  ERROR loading HuggingFace dataset: {e}")
        print("  Continuing with PDF chunks only.")

    # Summary
    print(f"\n=== CORPUS SUMMARY ===")
    namespace_totals: Dict[str, int] = {}
    for chunk in all_chunks:
        ns = chunk["metadata"]["namespace"]
        namespace_totals[ns] = namespace_totals.get(ns, 0) + 1

    for ns, count in sorted(namespace_totals.items()):
        print(f"  {ns}: {count} chunks")
    print(f"  TOTAL: {len(all_chunks)} chunks")

    return all_chunks
