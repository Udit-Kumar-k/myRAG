import re
import os
from typing import Generator, List, Dict, Any, Optional

# List of G20 ISO country/jurisdiction codes as specified in the plan
G20_ISO_CODES = {
    "ARG", "AUS", "BRA", "CAN", "CHN", "DEU", "FRA", "GBR", "IDN", "IND",
    "ITA", "JPN", "KOR", "MEX", "RUS", "SAU", "TUR", "USA", "ZAF", 
    "EU", "EUR", "EUE" # Supporting variations of European Union
}

def extract_year_from_ts(ts: Optional[str]) -> Optional[int]:
    """Extracts the publication year from the ISO timestamp string."""
    if not ts:
        return None
    try:
        # Match a 4 digit year at the start of the string
        match = re.match(r"^(\d{4})", ts)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None

def is_heading(text: str) -> bool:
    """
    Determines if a block is likely a heading:
    - Under 15 words
    - Title Case or ALL CAPS
    """
    words = text.strip().split()
    if not words or len(words) >= 15:
        return False
    
    clean_text = text.strip()
    # Check for title case or all caps
    is_caps = clean_text.isupper()
    is_title = clean_text.istitle()
    
    return is_caps or is_title

def clean_text_block(text: str) -> str:
    """Cleans up text whitespace."""
    return re.sub(r"\s+", " ", text).strip()

class SemanticChunker:
    def __init__(self, tokenizer: Optional[Any] = None, target_chunk_size: int = 400, overlap_size: int = 50):
        self.tokenizer = tokenizer
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size

    def count_tokens(self, text: str) -> int:
        """Counts tokens using tokenizer, or falls back to word count approximation if tokenizer is not available."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback: average english word is ~1.3 tokens
        return int(len(text.split()) * 1.3)

    def chunk_document(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of text blocks from a single document and groups them semantically.
        
        Each block is expected to have 'text' and other metadata.
        """
        chunks = []
        buffer_blocks = []
        buffer_tokens = 0
        
        for i, block in enumerate(blocks):
            text = clean_text_block(block.get("text", ""))
            if not text:
                continue
                
            block_tokens = self.count_tokens(text)
            
            # Check if this block is a heading and we already have content in the buffer
            if is_heading(text) and buffer_blocks:
                # Flush the buffer as a completed chunk
                chunks.append(self._create_chunk_from_buffer(buffer_blocks, blocks[0]))
                
                # Keep last few blocks for overlap (approx 50 tokens)
                buffer_blocks = self._get_overlap_blocks(buffer_blocks)
                buffer_tokens = sum(self.count_tokens(b["text"]) for b in buffer_blocks)
            
            # Add current block to buffer
            buffer_blocks.append({"text": text, "index": i})
            buffer_tokens += block_tokens
            
            # Flush if the buffer exceeds our target chunk size
            if buffer_tokens >= self.target_chunk_size:
                chunks.append(self._create_chunk_from_buffer(buffer_blocks, blocks[0]))
                buffer_blocks = self._get_overlap_blocks(buffer_blocks)
                buffer_tokens = sum(self.count_tokens(b["text"]) for b in buffer_blocks)
                
        # Flush remaining blocks at the end of the document
        if buffer_blocks:
            chunks.append(self._create_chunk_from_buffer(buffer_blocks, blocks[0]))
            
        return chunks

    def _create_chunk_from_buffer(self, buffer_blocks: List[Dict[str, Any]], metadata_ref: Dict[str, Any]) -> Dict[str, Any]:
        """Combines buffer blocks into a single chunk dictionary with aggregated metadata."""
        combined_text = "\n".join(b["text"] for b in buffer_blocks)
        
        # Determine the namespace based on corpus type and title
        corpus_type = metadata_ref.get("corpus_type_name", "")
        doc_name = metadata_ref.get("document_name", "")
        
        namespace = "international_agreements"
        if corpus_type == "Laws and Policies":
            namespace = "national_laws"
        elif corpus_type == "International Agreements":
            if "ndc" in doc_name.lower():
                namespace = "ndc_commitments"
            else:
                namespace = "international_agreements"
                
        pub_ts = metadata_ref.get("publication_ts")
        pub_year = extract_year_from_ts(pub_ts)
        
        # Build strict metadata schema
        chunk_metadata = {
            "geography_iso": metadata_ref.get("geography_iso", "UNK"),
            "namespace": namespace,
            "pub_year": pub_year if pub_year is not None else 2000, # default to 2000 if missing
            "document_name": doc_name,
            "source_url": metadata_ref.get("source_url", ""),
            "language": "en"
        }
        
        return {
            "text": combined_text,
            "metadata": chunk_metadata
        }

    def _get_overlap_blocks(self, buffer_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Selects the trailing blocks from the buffer that sum to approximately the overlap size."""
        overlap_blocks = []
        current_tokens = 0
        for block in reversed(buffer_blocks):
            block_tokens = self.count_tokens(block["text"])
            if current_tokens + block_tokens > self.overlap_size and overlap_blocks:
                break
            overlap_blocks.insert(0, block)
            current_tokens += block_tokens
        return overlap_blocks

def filter_and_stream_dataset(hf_token: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """Streams the dataset, filters by G20 + English, and yields raw records."""
    from datasets import load_dataset
    
    # Load dataset in streaming mode
    # Some datasets require hf_token to be set if they are private or gated
    dataset = load_dataset(
        "ClimatePolicyRadar/all-document-text-data", 
        split="train", 
        streaming=True,
        token=hf_token
    )
    
    for row in dataset:
        doc_metadata = row.get("document_metadata", {})
        language = doc_metadata.get("language_iso", "")
        geography = doc_metadata.get("geography_iso", "")
        
        # Keep G20 countries and English language only
        if language == "en" and geography in G20_ISO_CODES:
            # Yield record flat structure with essential fields
            yield {
                "text": row.get("text_block_text", ""),
                "document_name": row.get("document_name", ""),
                "corpus_type_name": doc_metadata.get("corpus_type_name", ""),
                "publication_ts": doc_metadata.get("publication_ts", ""),
                "geography_iso": geography,
                "source_url": row.get("document_source_url", ""),
            }

def process_corpus(hf_token: Optional[str] = None, tokenizer: Optional[Any] = None, max_docs: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Main function to stream the dataset, group blocks by document,
    and generate semantic chunks.
    """
    print("Starting dataset streaming and chunking...")
    chunker = SemanticChunker(tokenizer=tokenizer)
    
    current_doc_name = None
    current_doc_blocks = []
    all_chunks = []
    doc_count = 0
    
    for row in filter_and_stream_dataset(hf_token):
        doc_name = row.get("document_name")
        if not doc_name:
            continue
            
        # Check if we have transitioned to a new document
        if current_doc_name is not None and doc_name != current_doc_name:
            # Chunk the completed document
            doc_chunks = chunker.chunk_document(current_doc_blocks)
            all_chunks.extend(doc_chunks)
            doc_count += 1
            
            if doc_count % 100 == 0:
                print(f"Processed {doc_count} documents. Total chunks so far: {len(all_chunks)}")
                
            if max_docs and doc_count >= max_docs:
                break
                
            current_doc_blocks = []
            
        current_doc_name = doc_name
        current_doc_blocks.append(row)
        
    # Chunk the last document in the stream
    if current_doc_blocks and (not max_docs or doc_count < max_docs):
        doc_chunks = chunker.chunk_document(current_doc_blocks)
        all_chunks.extend(doc_chunks)
        doc_count += 1
        
    print(f"Ingestion completed. Total processed documents: {doc_count}. Total semantic chunks: {len(all_chunks)}")
    return all_chunks
