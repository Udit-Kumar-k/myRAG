from unittest.mock import MagicMock
import numpy as np
from src.backend.retrieval import MedicalRAGPipeline

mock_index_manager = MagicMock()
mock_index_manager.namespaces = ["basic_sciences", "pharmacology", "clinical_medicine"]
pipeline = MedicalRAGPipeline(mock_index_manager, confidence_threshold=0.65)

pipeline.load_reranker = MagicMock()
mock_reranker = MagicMock()
pipeline.reranker = mock_reranker

# Mock retrieve to return a candidate chunk
candidates = [
    {"text": "Sample context", "metadata": {"document_name": "Doc A", "geography_iso": "pharmacology", "pub_year": 2020, "namespace": "pharmacology"}}
]
pipeline.retrieve = MagicMock(return_value=candidates)

# Mock predict to return a numpy array of shape (1,)
mock_reranker.predict.return_value = np.array([2.0])

print(f"Before query: candidates = {candidates}")
print(f"mock_reranker.predict.return_value = {mock_reranker.predict.return_value} (type: {type(mock_reranker.predict.return_value)})")

# Let's see what happens inside query() manually
candidates_retrieved = pipeline.retrieve("What is the mechanism of action of metformin?", target_namespace="all", top_n=20)
print(f"Candidates retrieved: {candidates_retrieved} (len: {len(candidates_retrieved)})")

pairs = [["What is the mechanism of action of metformin?", cand["text"]] for cand in candidates_retrieved]
rerank_scores = mock_reranker.predict(pairs)
print(f"Rerank scores: {rerank_scores} (len: {len(rerank_scores) if hasattr(rerank_scores, '__len__') else 'N/A'}, type: {type(rerank_scores)})")

for cand, score in zip(candidates_retrieved, rerank_scores):
    prob = 1.0 / (1.0 + np.exp(-score))
    cand["relevance_score"] = float(prob)
    print(f"Added relevance_score to: {cand}")

print(f"Sorting candidates...")
candidates_retrieved.sort(key=lambda x: x["relevance_score"], reverse=True)
print(f"Sorted candidates: {candidates_retrieved}")
