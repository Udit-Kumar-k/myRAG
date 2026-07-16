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


# ── PDF chunking — zone-based extraction for Telangana Police format PDFs ────

# Section boundary: line starts with 1-3 digit number, optional capital letter
# (e.g. "34A"), a period, then a space.  Tested against all three PDFs.
SECTION_RE = re.compile(r'^(\d{1,3}[A-Z]?)\.\s', re.MULTILINE)

# Schedule markers per act (used to find end of Zone 3)
_SCHEDULE_MARKERS: Dict[str, str] = {
    "Bharatiya Sakshya Adhiniyam 2023":            "THE SCHEDULE",
    "Bharatiya Nagarik Suraksha Sanhita 2023":      "THE FIRST SCHEDULE",
    # BNS has no schedule — runs to end of document
}


def _find_nth(text: str, needle: str, n: int = 1) -> int:
    """Return the char-index of the *n*-th occurrence of *needle*, or -1."""
    start = -1
    for _ in range(n):
        start = text.find(needle, start + 1)
        if start == -1:
            return -1
    return start


def _extract_zone3(full_text: str, act_name: str) -> str:
    """
    Extracts Zone 3 (Chapters and Sections — the actual legal text) from
    the full extracted PDF text.

    Zone layout of all three Telangana Police PDFs:
      1. Index (TOC)
      2. Corresponding Section Table (old↔new mapping)
      3. Chapters and Sections   ← this is what we want
      4. Schedule(s) (if any)

    Start: second occurrence of "Chapters and Sections"
    End:   schedule marker (act-specific) or end of document
    """
    # --- locate start ---
    start = _find_nth(full_text, "Chapters and Sections", 2)
    if start == -1:
        # Fallback: try first occurrence (may include some TOC noise)
        start = full_text.find("Chapters and Sections")
    if start == -1:
        # No zone markers at all — fall back to full text
        return full_text

    # --- locate end ---
    marker = _SCHEDULE_MARKERS.get(act_name)
    if marker is None:
        # BNS — no schedule, run to end
        end = len(full_text)
    else:
        # BSA: "THE SCHEDULE" appears once → take first occurrence
        # BNSS: "THE FIRST SCHEDULE" appears twice → take second occurrence
        if act_name == "Bharatiya Nagarik Suraksha Sanhita 2023":
            end = _find_nth(full_text, marker, 2)
        else:
            end = full_text.find(marker)
        if end == -1:
            end = len(full_text)

    return full_text[start:end]


def _split_sections(zone3_text: str) -> List[str]:
    """
    Splits Zone 3 text into per-section chunks using SECTION_RE boundaries.

    Handles the BSA formatting quirk where each section number appears twice
    (once on the caption line, once on the operative-text restart) within
    ~200 chars — these are collapsed into a single boundary.
    """
    matches = list(SECTION_RE.finditer(zone3_text))
    if not matches:
        # No section boundaries found — return the whole zone as one chunk
        return [zone3_text] if zone3_text.strip() else []

    # Collapse duplicate section-number matches within 200 chars (BSA quirk)
    collapsed: List[re.Match] = []
    i = 0
    while i < len(matches):
        m = matches[i]
        # Check if the next match is a duplicate within 200 chars
        if (i + 1 < len(matches)
                and matches[i + 1].group(1) == m.group(1)
                and (matches[i + 1].start() - m.start()) <= 200):
            # Keep the first occurrence as the boundary, skip the duplicate
            collapsed.append(m)
            i += 2
        else:
            collapsed.append(m)
            i += 1

    # Split text at collapsed boundary positions
    chunks: List[str] = []
    for idx, m in enumerate(collapsed):
        start = m.start()
        end = collapsed[idx + 1].start() if idx + 1 < len(collapsed) else len(zone3_text)
        chunk_text = zone3_text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_pdf(pdf_path: str, act_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF using pymupdf (fitz) and splits using
    zone-based extraction tuned for the Telangana Police PDF format.

    Zone 3 (Chapters and Sections) is split by section-number boundaries.
    Zones 1 (Index), 2 (Corresponding Section Table), and 4 (Schedules)
    are skipped for v1.
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Extract Zone 3 (the actual legal text)
    zone3 = _extract_zone3(full_text, act_name)

    # Split into per-section chunks
    raw_chunks = _split_sections(zone3)

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
                "legal_domain":  namespace,
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
        if namespace == "general" and namespace_counts.get("general", 0) >= 1000:
            continue

        if needs_chunking:
            # Large rows need section-boundary chunking
            raw_chunks = SECTION_RE.split(text)
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
                        "legal_domain":  namespace,
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
                    "legal_domain":  namespace,
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
    ("data/raw/BNSrag.pdf", "Bharatiya Nyaya Sanhita 2023"),
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
