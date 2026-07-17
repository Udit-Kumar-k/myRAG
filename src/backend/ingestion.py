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
    elif any(x in act for x in ["rbi", "reserve bank", "banking", "negotiable instrument",
                                  "digital payment"]):
        # NOTE: bare "payment" is intentionally excluded — it matches
        # "Payment of Wages Act", "Payment of Bonus Act", "Payment of
        # Gratuity Act", etc. which are labour-law acts, not banking law.
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


# Maximum chunk size before flagging as oversized (table extraction produces
# multi-thousand-character fragments of disconnected word/cell-per-line text).
_MAX_CHUNK_CHARS = 4000


def _caption_boundary(zone3_text: str, match_start: int) -> int:
    """
    Given the start offset of a section-number match (e.g. 'N. '), walk
    backwards through the text to find where the caption/title line that
    precedes this section begins.

    Real BNS/BNSS/BSA PDF structure (verified against actual files):
        ...body of section N-1...
        [optional blank lines]
        Caption title of section N      ← belongs to section N
        [optional blank lines]
        N. (1) Operative text begins    ← SECTION_RE match here

    We walk backwards from match_start, skipping blank lines, and return
    the start of the first non-blank line we find.  That line is the caption
    for the current section and must NOT be included in the previous chunk.

    If there is no non-blank line before the match (e.g. section 1 at the
    very start), we return match_start unchanged so the previous chunk keeps
    its full content.
    """
    pos = match_start - 1
    # Step 1: skip the newline(s) immediately before the match
    while pos >= 0 and zone3_text[pos] in ('\n', '\r', ' '):
        pos -= 1
    if pos < 0:
        return match_start  # nothing before — leave boundary as-is

    # pos is now on the last char of the line immediately before the match.
    # That line is the caption. Find its start (the char after the preceding \n).
    line_end = pos
    while pos >= 0 and zone3_text[pos] not in ('\n', '\r'):
        pos -= 1
    caption_start = pos + 1  # first char of the caption line

    # Sanity check: the caption should be reasonably short (< 200 chars).
    # If it is very long the line is body text, not a caption — leave boundary.
    caption_len = line_end - caption_start + 1
    if caption_len > 200:
        return match_start

    return caption_start


def _split_sections(zone3_text: str) -> List[str]:
    """
    Splits Zone 3 text into per-section chunks using SECTION_RE boundaries.

    Handles three quirks of the Telangana Police PDFs:

    1. BSA duplicate-number quirk: each section number appears twice within
       ~200 chars (once on the bare caption line, once on the operative-text
       restart).  The second occurrence is collapsed into the first.

    2. Spurious backward matches: a stray "90." appearing ~180,000 chars after
       the real Section 90 would otherwise create a garbage chunk boundary.
       We reject any match whose section number is less than the running
       maximum already seen (genuine section numbering is monotone-increasing).

    3. Caption-boundary placement: each section has a short title line
       (e.g. "Short title, commencement, and application") that appears on
       the line immediately before the "N. " regex match.  The old boundary
       used match.start() as the end of the previous chunk, which left the
       caption stranded at the tail of the wrong section's chunk.
       Fix: _caption_boundary() walks backwards from each match to find the
       start of its caption line, and uses *that* as the end of the previous
       chunk so every caption travels with its own section.
    """
    matches = list(SECTION_RE.finditer(zone3_text))
    if not matches:
        # No section boundaries found — return the whole zone as one chunk
        return [zone3_text] if zone3_text.strip() else []

    # ── Pass 1: Collapse BSA duplicate section-number matches (within 200 chars) ──
    collapsed: List[re.Match] = []
    i = 0
    while i < len(matches):
        m = matches[i]
        # Check if the next match is a duplicate (same section number) within 200 chars
        if (i + 1 < len(matches)
                and matches[i + 1].group(1) == m.group(1)
                and (matches[i + 1].start() - m.start()) <= 200):
            # Keep the first occurrence as the boundary; the second is a
            # formatting repeat. Skip both and add only the first.
            collapsed.append(m)
            i += 2
        else:
            collapsed.append(m)
            i += 1

    # ── Pass 2: Reject spurious backward-numbered matches ──────────────────
    # A genuine statute numbers sections monotone-increasing (1, 2, 3, ...).
    # Any match whose section number is *less than* the running maximum is a
    # stray match (e.g. "90." appearing in footnote text 180k chars after the
    # real Section 90) and must be dropped to prevent garbage chunk boundaries.
    filtered: List[re.Match] = []
    max_section_seen = -1
    for m in collapsed:
        try:
            sec_num = int(''.join(filter(str.isdigit, m.group(1))))
        except (ValueError, TypeError):
            sec_num = max_section_seen + 1  # non-numeric suffix (e.g. "34A"): treat as in-sequence
        if sec_num >= max_section_seen:
            filtered.append(m)
            max_section_seen = sec_num
        else:
            print(f"  [_split_sections] Rejected spurious backward match: "
                  f"section {m.group(1)!r} at offset {m.start()} "
                  f"(max seen so far: {max_section_seen})")

    if not filtered:
        return [zone3_text] if zone3_text.strip() else []

    # ── Pass 3: Slice into per-section chunks ──────────────────────────────
    # Each chunk's START is the caption boundary of *that* section (i.e. the
    # start of the title line that precedes the section number in the PDF).
    # Each chunk's END is the caption boundary of the *next* section.
    # The first chunk starts at filtered[0].start() (the section number itself,
    # or wherever the zone begins before the first caption).
    chunks: List[str] = []
    # Pre-compute caption boundaries for all filtered matches
    caption_starts = [_caption_boundary(zone3_text, m.start()) for m in filtered]

    for idx, m in enumerate(filtered):
        start = caption_starts[idx]
        end = caption_starts[idx + 1] if idx + 1 < len(filtered) else len(zone3_text)
        chunk_text = zone3_text[start:end].strip()
        if not chunk_text:
            continue
        if len(chunk_text) > _MAX_CHUNK_CHARS:
            print(f"  [_split_sections] WARNING: oversized chunk ({len(chunk_text)} chars) "
                  f"starting near offset {start} — likely a table or multi-section block. "
                  f"Flagged with is_oversized=True for downstream filtering.")
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

        is_oversized = len(text) > _MAX_CHUNK_CHARS
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
                "is_oversized":  is_oversized,
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
            # Large rows need section-boundary chunking.
            # Use _split_sections (finditer + slicing) rather than
            # SECTION_RE.split(text): re.split() with a capturing group
            # returns each captured group (e.g. "34", "90") as its own
            # fragment, separate from the body text it introduces.  These
            # tiny fragments fall below the 50-char noise filter and are
            # silently discarded, causing every HF chunk to lose its section
            # number.  _split_sections reuses the already-correct PDF path.
            raw_chunks = _split_sections(text)
            for raw in raw_chunks:
                chunk_text = raw.strip()
                if len(chunk_text) < 50:
                    continue
                is_oversized = len(chunk_text) > _MAX_CHUNK_CHARS
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
                        "is_oversized":  is_oversized,
                    }
                })
                namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1
        else:
            # Pre-chunked rows can be used directly
            if len(text.strip()) < 50:
                continue
            is_oversized = len(text.strip()) > _MAX_CHUNK_CHARS
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
                    "is_oversized":  is_oversized,
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
