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


# Maximum chunk size before sub-chunking.  Chunks larger than this are split
# further by _sub_chunk() rather than indexed as-is.  Table extraction via
# page.get_text() and long definition sections (e.g. §2 BNS Definitions) both
# produce oversized text — sub-chunking recovers the real legal content from
# them instead of discarding it.
_MAX_CHUNK_CHARS = 4000
# Sliding-window parameters used by _sub_chunk's fallback pass.
_SUB_CHUNK_SIZE    = 800   # target size for each window sub-chunk (chars)
_SUB_CHUNK_OVERLAP = 150   # overlap between consecutive windows (chars)


def _sub_chunk(text: str) -> List[str]:
    """
    Splits a text block that is too large for a single embedding into
    smaller sub-chunks that each fit within _MAX_CHUNK_CHARS.

    Strategy (two passes):

    Pass 1 — Paragraph split: split on double-newlines ("\\n\\n"), which
    preserves the structure of numbered definitions, sub-clauses, and
    explanatory paragraphs.  Groups consecutive paragraphs into a running
    buffer; when adding the next paragraph would exceed _MAX_CHUNK_CHARS,
    flush the buffer as a sub-chunk and start a new one.

    Pass 2 — Sliding window: any paragraph that is *itself* longer than
    _MAX_CHUNK_CHARS (e.g. a 2000-char run-on sentence) is split by a
    sliding window of _SUB_CHUNK_SIZE with _SUB_CHUNK_OVERLAP chars of
    overlap so no context is lost at boundaries.

    Returns a list of non-empty strings, each <= _MAX_CHUNK_CHARS chars
    (except in the degenerate case of a single token > _MAX_CHUNK_CHARS,
    which is left as-is and will be caught by the is_oversized flag).
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    # ── Pass 1: Paragraph-aware grouping ──────────────────────────────────
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        # No paragraph breaks — fall straight through to sliding window
        paragraphs = [text]

    sub_chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for para in paragraphs:
        if len(para) > _MAX_CHUNK_CHARS:
            # This paragraph is itself oversized — flush buffer first, then
            # apply sliding window to the paragraph.
            if buffer:
                sub_chunks.append("\n\n".join(buffer))
                buffer, buffer_len = [], 0
            # ── Pass 2: Sliding window on the overlong paragraph ──────────
            start = 0
            while start < len(para):
                end = start + _SUB_CHUNK_SIZE
                sub_chunks.append(para[start:end].strip())
                if end >= len(para):
                    break
                start = end - _SUB_CHUNK_OVERLAP
        else:
            # Would adding this paragraph overflow the buffer?
            joining_len = buffer_len + len(para) + (2 if buffer else 0)  # "\n\n" = 2
            if buffer and joining_len > _MAX_CHUNK_CHARS:
                sub_chunks.append("\n\n".join(buffer))
                buffer, buffer_len = [para], len(para)
            else:
                buffer.append(para)
                buffer_len += len(para) + (2 if len(buffer) > 1 else 0)

    if buffer:
        sub_chunks.append("\n\n".join(buffer))

    return [s for s in sub_chunks if s.strip()]


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
            # Sub-chunk instead of discarding — real legal content (e.g. §2
            # Definitions with 50+ entries) is recovered this way.
            chunks.extend(_sub_chunk(chunk_text))
        else:
            chunks.append(chunk_text)

    return chunks


def _extract_page_text_table_aware(page: Any) -> str:
    """
    Extracts text from a single PDF page while replacing raw table regions
    with a clean structured representation.

    Problem: page.get_text() turns table cells into one word/cell per line
    (PDF reading-order stream), producing multi-thousand-char blobs of
    disconnected tokens for sections like BNSS §359 schedule tables.

    Fix: detect table regions with page.find_tables(), render each table as
    compact 'col: val | col: val' rows, and suppress the raw table blocks
    from the page's text output so they don't overlap.

    Falls back to plain page.get_text() on any exception (e.g. older PyMuPDF
    without find_tables support), so this is non-breaking.
    """
    try:
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables]

        if not table_bboxes:
            return page.get_text()

        # Format each detected table as compact rows keyed by y0 position
        table_text_by_y0: Dict[float, str] = {}
        for table in tables:
            rows = table.extract()
            if not rows:
                continue
            header = rows[0]
            formatted_rows: List[str] = []
            for row in rows[1:]:
                cells = [
                    f"{str(h).strip()}: {str(c).strip()}"
                    for h, c in zip(header, row)
                    if str(h).strip() or str(c).strip()
                ]
                if cells:
                    formatted_rows.append(" | ".join(cells))
            if formatted_rows:
                table_text_by_y0[table.bbox[1]] = "\n".join(formatted_rows)

        # Walk text blocks in reading order; skip blocks inside table bboxes,
        # inserting the formatted table text in their place (once per table).
        page_text_parts: List[str] = []
        inserted_tables: set = set()
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        for block in sorted(blocks, key=lambda b: b[1]):
            bx0, by0, bx1, by1 = block[:4]
            block_text = block[4].strip()
            if not block_text:
                continue
            in_table = any(
                bx0 >= tbx0 - 2 and by0 >= tby0 - 2 and bx1 <= tbx1 + 2 and by1 <= tby1 + 2
                for (tbx0, tby0, tbx1, tby1) in table_bboxes
            )
            if in_table:
                # Insert the formatted version of whichever table contains this block
                for ty0, table_text in table_text_by_y0.items():
                    if ty0 not in inserted_tables and abs(ty0 - by0) < 20:
                        page_text_parts.append(table_text)
                        inserted_tables.add(ty0)
                        break
            else:
                page_text_parts.append(block_text)

        # Any tables not yet inserted (no overlapping block found by position)
        for ty0, table_text in table_text_by_y0.items():
            if ty0 not in inserted_tables:
                page_text_parts.append(table_text)

        return "\n".join(page_text_parts)

    except Exception:
        # Fallback: plain text extraction (e.g. older PyMuPDF builds)
        return page.get_text()


def chunk_pdf(pdf_path: str, act_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF using pymupdf (fitz) and splits using
    zone-based extraction tuned for the Telangana Police PDF format.

    Text extraction uses _extract_page_text_table_aware() so that table
    regions (e.g. BNSS §359 schedule tables) are rendered as compact
    structured rows instead of word-per-cell vertical dumps — fixing the
    root cause of oversized chunks for table-heavy sections.

    Zone 3 (Chapters and Sections) is split by section-number boundaries.
    Zones 1 (Index), 2 (Corresponding Section Table), and 4 (Schedules)
    are skipped for v1.
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += _extract_page_text_table_aware(page)
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

        # Sub-chunk any genuinely long section (e.g. §2 Definitions) so each
        # sub-chunk fits within the embedding model's context window.
        # is_oversized=True on sub-chunks that remain large is informational
        # only — not used as a filter or retrieval penalty.
        sub_texts = _sub_chunk(text)
        for sub_text in sub_texts:
            if len(sub_text) < 50:
                continue
            is_oversized = len(sub_text) > _MAX_CHUNK_CHARS
            chunks.append({
                "text": sub_text,
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
    Auto-detects per-row whether rows need section-boundary chunking or are
    pre-chunked (checked per-row so mixed datasets are handled correctly).

    The optional HF_GENERAL_LIMIT env var caps general-namespace chunks when
    set to a positive integer (useful for development/testing).  Defaults to 0
    (unlimited) so no content is silently dropped in production.
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split="central", token=hf_token)
    print(f"Loaded {len(ds)} rows from HuggingFace dataset.")

    # Optional cap on the "general" namespace — 0 means unlimited.
    # Set HF_GENERAL_LIMIT=1000 in .env for a quick dev/test run.
    general_limit = int(os.environ.get("HF_GENERAL_LIMIT", "0"))
    if general_limit > 0:
        print(f"  HF_GENERAL_LIMIT={general_limit}: general-namespace chunks will be "
              f"capped at {general_limit}. Set HF_GENERAL_LIMIT=0 (or unset) to index all.")

    chunks: List[Dict[str, Any]] = []
    namespace_counts: Dict[str, int] = {}
    filtered_count = 0
    general_skipped = 0

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

        # Apply the optional general-namespace cap.
        # When the limit is active and reached, count every skipped row so
        # the summary line below can report the total — nothing is silently lost.
        if general_limit > 0 and namespace == "general" and namespace_counts.get("general", 0) >= general_limit:
            general_skipped += 1
            continue

        # ── Per-row chunking strategy ────────────────────────────────────────
        # Each row's own text length decides whether it goes through
        # _split_sections() or is used as-is.  A single global flag computed
        # from ds[0] would mis-handle rows that don't share ds[0]'s shape
        # (e.g. a dataset mixing full-act rows with pre-chunked section rows).
        needs_chunking = len(text) > 5000

        if needs_chunking:
            raw_chunks = _split_sections(text)  # already sub-chunks oversized sections internally
            for raw in raw_chunks:
                chunk_text = raw.strip()
                if len(chunk_text) < 50:
                    continue
                # _split_sections already called _sub_chunk, so additional
                # sub-chunks are only needed for the direct HF path below.
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
            # Pre-chunked rows: sub-chunk any that are oversized before indexing.
            sub_texts = _sub_chunk(text.strip())
            for sub_text in sub_texts:
                if len(sub_text) < 50:
                    continue
                is_oversized = len(sub_text) > _MAX_CHUNK_CHARS
                chunks.append({
                    "text": sub_text,
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
    if general_skipped > 0:
        print(f"  Skipped {general_skipped} general-namespace rows due to HF_GENERAL_LIMIT={general_limit}. "
              f"Set HF_GENERAL_LIMIT=0 to index all.")
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
