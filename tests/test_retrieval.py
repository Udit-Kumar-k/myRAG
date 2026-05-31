import unittest
from unittest.mock import MagicMock
from src.backend.retrieval import ClimateRAGPipeline

class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        # Create pipeline with mocked index manager
        self.mock_index_manager = MagicMock()
        self.mock_index_manager.namespaces = ["national_laws", "ndc_commitments", "international_agreements"]
        self.pipeline = ClimateRAGPipeline(self.mock_index_manager, confidence_threshold=0.65)

    def test_rrf_merge_ranking(self):
        # Create sample dense and sparse results
        dense_results = [
            ({"text": "Doc A content", "metadata": {"document_name": "Doc A", "geography_iso": "USA", "pub_year": 2020, "namespace": "ndc_commitments"}}, 0),
            ({"text": "Doc B content", "metadata": {"document_name": "Doc B", "geography_iso": "IND", "pub_year": 2015, "namespace": "ndc_commitments"}}, 1),
        ]
        sparse_results = [
            ({"text": "Doc B content", "metadata": {"document_name": "Doc B", "geography_iso": "IND", "pub_year": 2015, "namespace": "ndc_commitments"}}, 0),
            ({"text": "Doc A content", "metadata": {"document_name": "Doc A", "geography_iso": "USA", "pub_year": 2020, "namespace": "ndc_commitments"}}, 1),
        ]
        
        # Merge with 0 temporal boost to isolate RRF
        merged = self.pipeline.rrf_merge(dense_results, sparse_results, k=60, temporal_boost=0.0)
        
        # Verify no duplicates
        self.assertEqual(len(merged), 2)
        # Verify sorting (Doc A and Doc B should have identical RRF scores before temporal boost because rank sum is identical: 1/(60+0) + 1/(60+1))
        self.assertAlmostEqual(merged[0]["rrf_score"], merged[1]["rrf_score"])

    def test_rrf_merge_temporal_boost(self):
        # Create sample dense results
        # Doc A is 2022 (newer)
        # Doc B is 2015 (older)
        # Both are ranked at the same position in dense and sparse lists
        dense_results_a = [({"text": "Doc A content", "metadata": {"document_name": "Doc A", "geography_iso": "USA", "pub_year": 2022, "namespace": "ndc_commitments"}}, 0)]
        dense_results_b = [({"text": "Doc B content", "metadata": {"document_name": "Doc B", "geography_iso": "USA", "pub_year": 2015, "namespace": "ndc_commitments"}}, 0)]
        
        merged_a = self.pipeline.rrf_merge(dense_results_a, [], k=60, temporal_boost=0.1)
        merged_b = self.pipeline.rrf_merge(dense_results_b, [], k=60, temporal_boost=0.1)
        
        # Since ranks are identical (rank 0), the newer document (Doc A) must have a higher final score due to temporal boost
        self.assertGreater(merged_a[0]["rrf_score"], merged_b[0]["rrf_score"])

    def test_confidence_gate(self):
        # We test routing and gating with mocked reranker
        mock_reranker = MagicMock()
        self.pipeline.load_reranker = MagicMock(return_value=mock_reranker)
        self.pipeline.reranker = mock_reranker
        
        # Mock search retrieval to return a candidate chunk
        self.pipeline.retrieve = MagicMock(return_value=[
            {"text": "Sample context", "metadata": {"document_name": "Doc A", "geography_iso": "USA", "pub_year": 2020, "namespace": "ndc_commitments"}}
        ])
        
        # 1. Test query passes confidence gate (high rerank score)
        mock_reranker.predict.return_value = [2.0] # raw logit, sigmoid(2.0) is approx 0.88 >= 0.65
        res = self.pipeline.query("What is the USA NDC target?")
        self.assertFalse(res["refused"])
        self.assertGreaterEqual(res["confidence_score"], 0.65)
        self.assertEqual(len(res["retrieved_chunks"]), 1)

        # 2. Test query is blocked by confidence gate (low rerank score)
        mock_reranker.predict.return_value = [-2.0] # raw logit, sigmoid(-2.0) is approx 0.12 < 0.65
        res = self.pipeline.query("Who is the prime minister of France?")
        self.assertTrue(res["refused"])
        self.assertEqual(len(res["retrieved_chunks"]), 0)

if __name__ == "__main__":
    unittest.main()
