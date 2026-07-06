import os
import sys

print("=== STARTING DIAGNOSTIC MODEL LOAD ===")
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Memory Total:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
except ImportError as e:
    print("Error importing PyTorch:", e)
    sys.exit(1)

# Step 1: Try loading embedding model
try:
    print("\n1. Loading embedding model BAAI/bge-m3 on GPU...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.max_seq_length = 512
    print("Successfully loaded embedding model on GPU.")
except Exception as e:
    print("Failed to load embedding model on GPU:", e)

# Step 2: Try loading reranker model on CPU
try:
    print("\n2. Loading reranker model BAAI/bge-reranker-v2-m3 on CPU...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    print("Successfully loaded reranker model on CPU.")
except Exception as e:
    print("Failed to load reranker model on CPU:", e)

print("\n=== DIAGNOSTIC COMPLETE ===")
