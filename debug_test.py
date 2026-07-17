from unittest.mock import MagicMock
import numpy as np
from src.backend.retrieval import LegalRAGPipeline

mock_index_manager = MagicMock()
mock_index_manager.namespaces = ["criminal", "cyber", "consumer", "banking", "general"]
pipeline = LegalRAGPipeline(mock_index_manager, confidence_threshold=0.65)

pipeline.load_reranker = MagicMock()
mock_reranker = MagicMock()
pipeline.reranker = mock_reranker

# Mock retrieve to return a candidate chunk
candidates = [
    {"text": "Sample context", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}
]
pipeline.retrieve = MagicMock(return_value=candidates)

# Mock predict to return a numpy array of shape (1,).
# bge-reranker-v2-m3 already applies sigmoid inside .predict() — scores are
# already in [0, 1] range, so we simulate a realistic post-sigmoid value.
mock_reranker.predict.return_value = np.array([0.88])

print(f"Before query: candidates = {candidates}")
print(f"mock_reranker.predict.return_value = {mock_reranker.predict.return_value} (type: {type(mock_reranker.predict.return_value)})")

# Let's see what happens inside query() manually
candidates_retrieved = pipeline.retrieve("What is the punishment for murder under BNS?", target_namespace="all", top_n=20)
print(f"Candidates retrieved: {candidates_retrieved} (len: {len(candidates_retrieved)})")

pairs = [["What is the punishment for murder under BNS?", cand["text"]] for cand in candidates_retrieved]
rerank_scores = mock_reranker.predict(pairs)
print(f"Rerank scores: {rerank_scores} (len: {len(rerank_scores) if hasattr(rerank_scores, '__len__') else 'N/A'}, type: {type(rerank_scores)})")

for cand, score in zip(candidates_retrieved, rerank_scores):
    # bge-reranker-v2-m3 already returns sigmoid-normalized scores from .predict()
    cand["relevance_score"] = float(score)
    print(f"Added relevance_score to: {cand}")

print(f"Sorting candidates...")
candidates_retrieved.sort(key=lambda x: x["relevance_score"], reverse=True)
print(f"Sorted candidates: {candidates_retrieved}")

