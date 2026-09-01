import unittest
from unittest.mock import MagicMock
from src.backend.retrieval import LegalRAGPipeline

class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        # Create pipeline with mocked index manager
        self.mock_index_manager = MagicMock()
        self.mock_index_manager.namespaces = ["criminal", "cyber", "consumer", "banking", "general"]
        self.pipeline = LegalRAGPipeline(self.mock_index_manager, confidence_threshold=0.65)

    def test_rrf_merge_ranking(self):
        # Create sample dense and sparse results
        dense_results = [
            ({"text": "Section 103 of BNS defines punishment for murder.", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}, 0),
            ({"text": "Section 303 of BNS covers theft provisions.", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}, 1),
        ]
        sparse_results = [
            ({"text": "Section 303 of BNS covers theft provisions.", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}, 0),
            ({"text": "Section 103 of BNS defines punishment for murder.", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}, 1),
        ]
        
        # Merge with 0 temporal boost to isolate RRF
        merged = self.pipeline.rrf_merge(dense_results, sparse_results, k=60, temporal_boost=0.0)
        
        # Verify no duplicates
        self.assertEqual(len(merged), 2)
        # Verify sorting (both docs should have identical RRF scores before temporal boost because rank sum is identical: 1/(60+0) + 1/(60+1))
        self.assertAlmostEqual(merged[0]["rrf_score"], merged[1]["rrf_score"])
 
    def test_rrf_merge_temporal_boost(self):
        # Create sample dense results
        # Doc A is 2023 (newer)
        # Doc B is 2015 (older)
        # Both are ranked at the same position in dense and sparse lists
        dense_results_a = [({"text": "BNS Section 103 content", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}, 0)]
        dense_results_b = [({"text": "IT Act Section 66 content", "metadata": {"document_name": "Information Technology Act 2000", "legal_domain": "cyber", "pub_year": 2000, "namespace": "cyber"}}, 0)]
        
        merged_a = self.pipeline.rrf_merge(dense_results_a, [], k=60, temporal_boost=0.1)
        merged_b = self.pipeline.rrf_merge(dense_results_b, [], k=60, temporal_boost=0.1)
        
        # Since ranks are identical (rank 0), the newer document (Doc A) must have a higher final score due to temporal boost
        self.assertGreater(merged_a[0]["rrf_score"], merged_b[0]["rrf_score"])

    def test_confidence_gate(self):
        from unittest.mock import patch
        
        with patch('src.backend.chain.LegalRAGChain.expand_query', side_effect=lambda q: q):
            # 1. Test query passes confidence gate (high rerank score >= 0.65)
            self.pipeline.retrieve = MagicMock(return_value=[
                {"text": "Section 103 of BNS prescribes punishment for murder.", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}
            ])
            self.pipeline.rerank_with_cohere = MagicMock(return_value=[
                {"text": "Section 103 of BNS prescribes punishment for murder.", "relevance_score": 0.85, "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023", "legal_domain": "criminal", "pub_year": 2023, "namespace": "criminal"}}
            ])
            res = self.pipeline.query("What is the punishment for murder under BNS?")
            self.assertFalse(res["refused"])
            self.assertGreaterEqual(res["confidence_score"], 0.65)
            self.assertEqual(len(res["retrieved_chunks"]), 1)

            # 2. Test query is blocked by confidence gate (low rerank score < 0.65)
            self.pipeline.retrieve = MagicMock(return_value=[
                {"text": "Unrelated statutory provision.", "metadata": {"document_name": "General Act", "legal_domain": "general", "pub_year": 2000, "namespace": "general"}}
            ])
            self.pipeline.rerank_with_cohere = MagicMock(return_value=[
                {"text": "Unrelated statutory provision.", "relevance_score": 0.25, "metadata": {"document_name": "General Act", "legal_domain": "general", "pub_year": 2000, "namespace": "general"}}
            ])
            res = self.pipeline.query("Who is the prime minister of France?")
            self.assertTrue(res["refused"])
            self.assertEqual(len(res["retrieved_chunks"]), 0)

if __name__ == "__main__":
    unittest.main()
