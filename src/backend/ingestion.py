import os
from typing import List, Dict, Any, Optional

# Maps textbook title keywords to subject namespaces.
# Checked against lowercased title (with underscores/spaces normalized); first match wins.
TEXTBOOK_NAMESPACE_MAP = {
    # Basic Sciences
    "anatomy":            "basic_sciences",
    "gray":               "basic_sciences",
    "biochemistry":       "basic_sciences",
    "lippinc":            "basic_sciences",
    "cell_biology":       "basic_sciences",
    "alberts":            "basic_sciences",
    "histology":          "basic_sciences",
    "ross":               "basic_sciences",
    "physiology":         "basic_sciences",
    "levy":               "basic_sciences",
    "step1":              "basic_sciences",
    "step_1":             "basic_sciences",

    # Pharmacology & Pathology
    "pharmacology":       "pharmacology",
    "katzung":            "pharmacology",
    "pathology":          "pharmacology",
    "robbins":            "pharmacology",
    "pathoma":            "pharmacology",
    "husain":             "pharmacology",
    "immunology":         "pharmacology",
    "janeway":            "pharmacology",

    # Clinical Medicine
    "harrison":           "clinical_medicine",
    "internalmed":        "clinical_medicine",
    "internal medicine":  "clinical_medicine",
    "step2":              "clinical_medicine",
    "step_2":             "clinical_medicine",
    "step3":              "clinical_medicine",
    "step_3":             "clinical_medicine",
    "first aid":          "clinical_medicine",
    "first_aid":          "clinical_medicine",
    "surgery":            "clinical_medicine",
    "schwartz":           "clinical_medicine",
    "pediatrics":         "clinical_medicine",
    "nelson":             "clinical_medicine",
    "obstetrics":         "clinical_medicine",
    "obstentrics":        "clinical_medicine",  # matches typo in dataset: Obstentrics_Williams
    "williams":           "clinical_medicine",
    "gynecology":         "clinical_medicine",
    "novak":              "clinical_medicine",
    "psychiatry":         "clinical_medicine",
    "psichiatry":         "clinical_medicine",  # matches typo in dataset: Psichiatry_DSM-5
    "neurology":          "clinical_medicine",
    "adams":              "clinical_medicine",
}


def derive_namespace(title: str) -> str:
    """Maps a textbook title to one of three subject namespaces."""
    title_lower = title.lower().replace(" ", "_")
    for keyword, namespace in TEXTBOOK_NAMESPACE_MAP.items():
        # Normalize keyword spaces to underscores for robust matching
        kw_norm = keyword.lower().replace(" ", "_")
        if kw_norm in title_lower:
            return namespace
    return "clinical_medicine"  # safe default


def load_medrag_textbooks(
    dataset_name: str = "MedRAG/textbooks",
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Downloads MedRAG/textbooks from HuggingFace.
    Every row is already a pre-chunked snippet (~182 tokens, ≤1000 chars).
    SemanticChunker is not needed — no chunking risk.
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name} from HuggingFace...")
    dataset = load_dataset(dataset_name, split="train", token=hf_token)
    print(f"Loaded {len(dataset)} pre-chunked textbook snippets.")

    chunks: List[Dict[str, Any]] = []
    namespace_counts: Dict[str, int] = {
        "basic_sciences": 0,
        "pharmacology": 0,
        "clinical_medicine": 0,
    }

    for row in dataset:
        # MedRAG/textbooks fields: id, title, content / contents
        text = row.get("contents") or row.get("content") or ""
        title = row.get("title") or "Unknown Textbook"
        chunk_id = str(row.get("id") or "")

        if not text.strip():
            continue

        namespace = derive_namespace(title)
        namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1

        chunks.append({
            "text": text,
            "metadata": {
                "document_name":   title,
                "textbook_title":  title,
                "chunk_id":        chunk_id,
                "namespace":       namespace,
                # Conservative approximation; MedRAG textbooks are ~2018-2022 eds.
                # Temporal boost will still rank more-recent editions higher if pub_year
                # metadata ever becomes available per-chunk.
                "pub_year":        2020,
                "source_url":      "https://huggingface.co/datasets/MedRAG/textbooks",
                # geography_iso kept for schema compatibility with ClimateIndexManager;
                # set to namespace label so calibrate_threshold recall-by-namespace works.
                "geography_iso":   namespace,
            }
        })

    print("Namespace distribution:")
    for ns, count in namespace_counts.items():
        print(f"  {ns}: {count} chunks")
    print(f"Total: {len(chunks)} chunks")
    return chunks


# Entry point: call from indexing.py instead of process_corpus
def process_corpus(
    hf_token: Optional[str] = None,
    **_kwargs,          # absorbs max_docs and tokenizer args for drop-in compatibility
) -> List[Dict[str, Any]]:
    """
    Drop-in replacement for the old ClimateRAG process_corpus.
    Ignores max_docs because the dataset is bounded and safe to load fully.
    """
    return load_medrag_textbooks(hf_token=hf_token)
