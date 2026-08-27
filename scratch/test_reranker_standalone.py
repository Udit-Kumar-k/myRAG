import sys, os, traceback
sys.path.insert(0, os.path.abspath("."))
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("VRAM Allocated (MB):", torch.cuda.memory_allocated() / 1024 / 1024)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("Loading embedding model on CUDA...")
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    print("Embedder loaded. VRAM Allocated (MB):", torch.cuda.memory_allocated() / 1024 / 1024)
    
    print("Loading reranker model on CPU...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    print("Reranker loaded successfully on CPU!")
    
    # Test a prediction
    print("Testing prediction...")
    res = reranker.predict([["test query", "test passage"]], batch_size=1)
    print("Prediction result:", res)
except Exception as e:
    print("CAUGHT EXCEPTION:")
    traceback.print_exc()
