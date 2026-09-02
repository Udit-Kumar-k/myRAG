import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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


class HFEmbeddingAPI:
    """Drop-in replacement for SentenceTransformer that calls the
    HuggingFace Inference API instead of running a local model.

    huggingface_hub.InferenceClient.feature_extraction() returns
    the pooled sentence embedding (same pooling as SentenceTransformer),
    so the existing FAISS index built with local BGE-M3 stays valid.

    If HF Inference API times out or is unreachable, it automatically
    falls back to a local SentenceTransformer model if available.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        token = os.environ.get("HF_TOKEN", "").strip() or None
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(token=token, timeout=60)
        self._local_fallback = None
        print(f"HF Inference API client ready for {model_name} (timeout: 60s)")

    def _get_local_model(self):
        if self._local_fallback is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[EMBEDDING FALLBACK] Loading local SentenceTransformer ({self.model_name})...")
                self._local_fallback = SentenceTransformer(self.model_name, device="cpu")
                self._local_fallback.max_seq_length = 512
            except Exception as e:
                print(f"[EMBEDDING FALLBACK] Could not load local model: {e}")
        return self._local_fallback

    def encode(self, texts, normalize_embeddings: bool = False, batch_size: int = 8, **kwargs) -> np.ndarray:
        """Encode texts via HF Inference API with automatic local fallback."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embs = self._client.feature_extraction(batch, model=self.model_name)
                all_embs.append(np.array(embs, dtype="float32"))

            result = np.vstack(all_embs)  # (N, dim)
            if normalize_embeddings:
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                result = result / np.maximum(norms, 1e-9)
            return result
        except Exception as e:
            print(f"[HF_EMBEDDING_API ERROR] {e}. Attempting local model fallback...")
            local_model = self._get_local_model()
            if local_model is not None:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("show_progress_bar", None)
                return local_model.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=False, **fallback_kwargs)
            raise e

    # Compatibility shim: SentenceTransformer exposes max_seq_length
    @property
    def max_seq_length(self) -> int:
        return 512

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        pass  # no-op — sequence length is handled server-side


class LegalIndexManager:
    def __init__(self, index_dir: str = "data/indexes", model_name: str = "BAAI/bge-m3"):
        # Resolve index_dir relative to current working directory or package root
        if not os.path.exists(index_dir):
            alt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "indexes")
            if os.path.exists(alt_dir):
                index_dir = os.path.abspath(alt_dir)
        self.index_dir = index_dir
        self.model_name = model_name
        self.namespaces = ["criminal", "cyber", "consumer", "banking", "general"]
        
        # In-memory storage for loaded indexes
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.faiss_indexes: Dict[str, Any] = {}
        self.bm25_indexes: Dict[str, BM25Okapi] = {}
        self.model = None

        os.makedirs(self.index_dir, exist_ok=True)

    def load_embedding_model(self):
        """Lazy loads the embedding model.
        
        When EMBEDDING_PROVIDER=api (default) → uses HuggingFace Inference API (free, no GPU needed).
        When EMBEDDING_PROVIDER=local        → loads model locally with GPU when available.
        """
        if self.model is None:
            provider = os.environ.get("EMBEDDING_PROVIDER", "api").lower()
            if provider == "api":
                self.model = HFEmbeddingAPI(self.model_name)
            else:
                try:
                    import torch
                    device = os.environ.get("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
                    print(f"Loading embedding model {self.model_name} on device={device}...")
                    from sentence_transformers import SentenceTransformer
                    try:
                        self.model = SentenceTransformer(self.model_name, device=device)
                    except Exception as e:
                        if device == "cuda":
                            print(f"CUDA initialization failed ({e}). Falling back to CPU...")
                            self.model = SentenceTransformer(self.model_name, device="cpu")
                        else:
                            raise e
                    self.model.max_seq_length = 512
                    print("Model loaded successfully.")
                except (ImportError, Exception) as e:
                    print(f"Local embedding model unavailable ({e}). Falling back to HuggingFace Inference API...")
                    self.model = HFEmbeddingAPI(self.model_name)
        return self.model

    def build_indexes(self, all_chunks: List[Dict[str, Any]], batch_size: int = 256, force_rebuild: bool = False):
        """
        Builds FAISS and BM25 indexes for all namespaces from the list of chunks.

        Args:
            all_chunks: Flat list of chunk dicts produced by ingestion.
            batch_size: Embedding batch size (reduce for low-VRAM GPUs).
            force_rebuild: When True, existing index files on disk are
                overwritten instead of being skipped.  Use this after any
                ingestion or chunking change to ensure fresh indexes.
        """
        # Separate chunks by namespace
        ns_chunks: Dict[str, List[Dict[str, Any]]] = {ns: [] for ns in self.namespaces}
        for chunk in all_chunks:
            ns = chunk["metadata"]["namespace"]
            if ns in ns_chunks:
                ns_chunks[ns].append(chunk)
            else:
                # Fallback to general if namespace is unrecognized
                ns_chunks["general"].append(chunk)

        model = self.load_embedding_model()
        import faiss

        # torch.no_grad() is only needed when running a local SentenceTransformer.
        # When EMBEDDING_PROVIDER=api, torch isn't installed (removed from
        # requirements.txt to save ~2 GB), so we fall back to a no-op context.
        try:
            import torch
            _inference_ctx = torch.no_grad
        except ImportError:
            from contextlib import nullcontext
            _inference_ctx = nullcontext

        rebuild_ns_env = os.environ.get("REBUILD_NAMESPACES", "").strip()
        target_namespaces = [n.strip() for n in rebuild_ns_env.split(",") if n.strip()] if rebuild_ns_env else self.namespaces

        for ns in self.namespaces:
            if ns not in target_namespaces:
                print(f"\n--- Namespace: {ns} not in REBUILD_NAMESPACES. Preserving existing index. ---")
                continue

            # Check if index files already exist to support resuming.
            # Skip only when force_rebuild is False — after any ingestion or
            # chunking fix, pass force_rebuild=True to avoid serving stale data.
            chunks_path = os.path.join(self.index_dir, f"{ns}_chunks.pkl")
            faiss_path = os.path.join(self.index_dir, f"{ns}_faiss.index")
            bm25_path = os.path.join(self.index_dir, f"{ns}_bm25.pkl")
            if not force_rebuild and os.path.exists(chunks_path) and os.path.exists(faiss_path) and os.path.exists(bm25_path):
                print(f"\n--- Namespace: {ns} already indexed. Skipping (pass force_rebuild=True to override). ---")
                continue

            chunks = ns_chunks[ns]
            print(f"\n--- Building indexes for namespace: {ns} ({len(chunks)} chunks) ---")
            if not chunks:
                print(f"No chunks found for namespace {ns}. Skipping.")
                continue

            # Note: is_oversized=True chunks are included in the index.
            # The flag is informational only (for dashboards/QA) — genuinely
            # long statutory sections (e.g. BNS §2 Definitions) must remain
            # retrievable. Table-extraction garbage is fixed at source in
            # chunk_pdf() via page.find_tables(), not by filtering here.

            # 1. Build BM25 Index
            print(f"Building BM25 index for {ns}...")
            tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 2. Build FAISS Index
            print(f"Generating dense embeddings for {ns} (batch size: {batch_size})...")
            texts = [chunk["text"] for chunk in chunks]
            
            # Encode in batches to prevent OOM
            embeddings = []
            with _inference_ctx():
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
        Loads all available namespace indexes from disk into memory.
        Returns True if at least one namespace was loaded, False otherwise.
        """
        import faiss
        
        loaded_count = 0
        for ns in self.namespaces:
            chunks_path = os.path.join(self.index_dir, f"{ns}_chunks.pkl")
            faiss_path = os.path.join(self.index_dir, f"{ns}_faiss.index")
            bm25_path = os.path.join(self.index_dir, f"{ns}_bm25.pkl")
            
            if not (os.path.exists(chunks_path) and os.path.exists(faiss_path) and os.path.exists(bm25_path)):
                print(f"Namespace '{ns}' index files not found on disk. Skipping.")
                continue

            # Guard against Git LFS pointer text files (< 400 bytes containing 'git-lfs')
            is_lfs_pointer = False
            for p in [chunks_path, faiss_path, bm25_path]:
                if os.path.getsize(p) < 400:
                    try:
                        with open(p, "rb") as f:
                            head = f.read(100)
                            if b"git-lfs" in head or b"oid sha256" in head:
                                is_lfs_pointer = True
                                break
                    except Exception:
                        pass
            if is_lfs_pointer:
                print(f"WARNING: Namespace '{ns}' index files are unresolved Git LFS pointer files (not real binaries). Skipping.")
                continue
                
            print(f"Loading index files for namespace: {ns}...")
            try:
                # Load chunks
                with open(chunks_path, "rb") as f:
                    self.chunks[ns] = pickle.load(f)
                    
                # Load FAISS index
                self.faiss_indexes[ns] = faiss.read_index(faiss_path)
                
                # Load BM25 index
                with open(bm25_path, "rb") as f:
                    self.bm25_indexes[ns] = pickle.load(f)
                    
                loaded_count += 1
            except Exception as e:
                print(f"ERROR: Failed to load index files for namespace '{ns}': {e}. Skipping.")
                continue
                
        if loaded_count > 0:
            print(f"Successfully loaded {loaded_count} namespace index(es) into memory.")
            return True
        else:
            print("No valid namespace indexes found on disk.")
            return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.backend.ingestion import process_corpus

    print("=== NyayBot Index Generation Pipeline ===")

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    hf_token = hf_token or None

    # Default FORCE_REBUILD=1 when executed as a script so rebuilding always overwrites stale indexes.
    force_rebuild = os.environ.get("FORCE_REBUILD", "1").strip() in ("1", "true", "yes")
    if force_rebuild:
        print("FORCE_REBUILD=1 detected — existing index files will be overwritten.")

    try:
        chunks = process_corpus(hf_token=hf_token)
        if not chunks:
            print("No chunks generated. Check dataset access.")
        else:
            print(f"Generated {len(chunks)} chunks. Building indexes...")
            # Drop batch_size to 16 for RTX 2050 (4 GB VRAM) to prevent CUDA OOM
            manager = LegalIndexManager()
            manager.build_indexes(chunks, batch_size=16, force_rebuild=force_rebuild)
            print("=== Index Generation Complete ===")
    except Exception as e:
        print(f"Error: {e}")
