import os
import pickle
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

# We import faiss inside functions or check if it is available to make it robust
# We import sentence_transformers inside functions as well to allow lazy loading

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text into lowercase words for BM25 keyword matching."""
    return re.findall(r"\b\w{2,}\b", text.lower())

class ClimateIndexManager:
    def __init__(self, index_dir: str = "data/indexes", model_name: str = "BAAI/bge-m3"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.namespaces = ["basic_sciences", "pharmacology", "clinical_medicine"]
        
        # In-memory storage for loaded indexes
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.faiss_indexes: Dict[str, Any] = {}
        self.bm25_indexes: Dict[str, BM25Okapi] = {}
        self.model = None

        os.makedirs(self.index_dir, exist_ok=True)

    def load_embedding_model(self):
        """Lazy loads the BGE-M3 embedding model."""
        if self.model is None:
            print(f"Loading embedding model {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            # Using CPU by default, sentence_transformers will auto-detect CUDA if available
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully.")
        return self.model

    def build_indexes(self, all_chunks: List[Dict[str, Any]], batch_size: int = 256):
        """
        Builds FAISS and BM25 indexes for all namespaces from the list of chunks.
        """
        # Separate chunks by namespace
        ns_chunks: Dict[str, List[Dict[str, Any]]] = {ns: [] for ns in self.namespaces}
        for chunk in all_chunks:
            ns = chunk["metadata"]["namespace"]
            if ns in ns_chunks:
                ns_chunks[ns].append(chunk)
            else:
                # Fallback to clinical_medicine if namespace is unrecognized
                ns_chunks["clinical_medicine"].append(chunk)

        model = self.load_embedding_model()
        import faiss

        for ns in self.namespaces:
            chunks = ns_chunks[ns]
            print(f"\n--- Building indexes for namespace: {ns} ({len(chunks)} chunks) ---")
            if not chunks:
                print(f"No chunks found for namespace {ns}. Skipping.")
                continue

            # 1. Build BM25 Index
            print(f"Building BM25 index for {ns}...")
            tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 2. Build FAISS Index
            print(f"Generating dense embeddings for {ns} (batch size: {batch_size})...")
            texts = [chunk["text"] for chunk in chunks]
            
            # Encode in batches to prevent OOM
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # Normalize embeddings so cosine similarity = inner product (dot product)
                batch_embeds = model.encode(
                    batch_texts, 
                    normalize_embeddings=True, 
                    show_progress_bar=False
                )
                embeddings.append(batch_embeds)
                if (i + len(batch_texts)) % 1024 == 0 or i + len(batch_texts) == len(texts):
                    print(f"Embedded {i + len(batch_texts)}/{len(texts)} chunks...")
            
            embeddings = np.vstack(embeddings).astype('float32')
            
            # Create FAISS IndexFlatIP (Inner Product)
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(embeddings)
            
            # Save namespace artifacts
            self.save_namespace(ns, chunks, faiss_index, bm25)

    def save_namespace(self, namespace: str, chunks: List[Dict[str, Any]], faiss_index: Any, bm25: BM25Okapi):
        """Saves a namespace's chunks, FAISS index, and BM25 index to disk."""
        import faiss
        
        # Save chunks metadata
        chunks_path = os.path.join(self.index_dir, f"{namespace}_chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
            
        # Save FAISS index
        faiss_path = os.path.join(self.index_dir, f"{namespace}_faiss.index")
        faiss.write_index(faiss_index, faiss_path)
        
        # Save BM25 index (we save tokenized corpus and weights)
        bm25_path = os.path.join(self.index_dir, f"{namespace}_bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
            
        print(f"Successfully saved {namespace} index files to {self.index_dir}")

    def load_indexes(self) -> bool:
        """
        Loads all namespace indexes from disk into memory.
        Returns True if successful, False otherwise.
        """
        import faiss
        
        for ns in self.namespaces:
            chunks_path = os.path.join(self.index_dir, f"{ns}_chunks.pkl")
            faiss_path = os.path.join(self.index_dir, f"{ns}_faiss.index")
            bm25_path = os.path.join(self.index_dir, f"{ns}_bm25.pkl")
            
            if not (os.path.exists(chunks_path) and os.path.exists(faiss_path) and os.path.exists(bm25_path)):
                print(f"Missing index files for namespace: {ns}. Looked in {self.index_dir}")
                return False
                
            print(f"Loading index files for namespace: {ns}...")
            # Load chunks
            with open(chunks_path, "rb") as f:
                self.chunks[ns] = pickle.load(f)
                
            # Load FAISS index
            self.faiss_indexes[ns] = faiss.read_index(faiss_path)
            
            # Load BM25 index
            with open(bm25_path, "rb") as f:
                self.bm25_indexes[ns] = pickle.load(f)
                
        print("All namespace indexes loaded successfully.")
        return True

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.backend.ingestion import process_corpus  # now loads MedRAG/textbooks

    print("=== MedRAG Index Generation Pipeline ===")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token == "your_huggingface_token_here" or not hf_token:
        hf_token = None

    try:
        chunks = process_corpus(hf_token=hf_token)
        if not chunks:
            print("No chunks generated. Check dataset access.")
        else:
            print(f"Generated {len(chunks)} chunks. Building indexes...")
            # Drop batch_size to 32 for RTX 2050 (4 GB VRAM)
            manager = ClimateIndexManager()
            manager.build_indexes(chunks, batch_size=32)
            print("=== Index Generation Complete ===")
    except Exception as e:
        print(f"Error: {e}")
