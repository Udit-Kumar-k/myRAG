import os
# Prevent connection failures to database placeholders during tests
os.environ["DATABASE_URL"] = ""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# We mock the startup_event to prevent loading heavy models and indexes during web client tests
with patch("src.backend.main.startup_event") as mock_startup:
    from src.backend.main import app
    client = TestClient(app)

class TestIntegrationEndpoints(unittest.TestCase):
    
    @patch("src.backend.main.rag_pipeline")
    @patch("src.backend.main.rag_chain")
    def test_query_endpoint_success(self, mock_chain, mock_pipeline):
        # Configure pipeline mock for a successful query
        mock_pipeline.query.return_value = {
            "query": "What is Germany's 2030 target?",
            "refused": False,
            "confidence_score": 0.85,
            "namespace_searched": "national_laws",
            "retrieved_chunks": [
                {
                    "text": "Germany targets 65% reduction by 2030.",
                    "relevance_score": 0.85,
                    "metadata": {
                        "document_name": "Climate Protection Act",
                        "geography_iso": "DEU",
                        "pub_year": 2021,
                        "namespace": "national_laws",
                        "source_url": "https://bundesregierung.de/law"
                    }
                }
            ]
        }
        
        # Configure chain mock
        mock_chain.run.return_value = "Germany targets 65% reduction by 2030 under its Climate Protection Act."
        
        # Make the request to FastAPI
        # Header has bearer token, which activates mock auth test_user_123
        response = client.post(
            "/query",
            json={
                "question": "What is Germany's 2030 target?",
                "conversation_id": "test_conv_abc"
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["refused"])
        self.assertEqual(data["confidence_score"], 0.85)
        self.assertIn("65%", data["answer"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["geography_iso"], "DEU")

    @patch("src.backend.main.rag_pipeline")
    def test_query_endpoint_refusal(self, mock_pipeline):
        # Configure pipeline mock for a blocked query
        mock_pipeline.query.return_value = {
            "query": "Who is the prime minister of France?",
            "refused": True,
            "confidence_score": 0.15,
            "namespace_searched": "all",
            "retrieved_chunks": []
        }
        
        response = client.post(
            "/query",
            json={
                "question": "Who is the prime minister of France?",
                "conversation_id": "test_conv_abc"
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["refused"])
        self.assertEqual(data["confidence_score"], 0.15)
        self.assertEqual(len(data["sources"]), 0)
        self.assertIn("Insufficient evidence found", data["answer"])

    def test_health_endpoint(self):
        # Patch index manager to say indexes are loaded
        with patch("src.backend.main.index_manager") as mock_manager:
            mock_manager.faiss_indexes = {"national_laws": {}, "ndc_commitments": {}, "international_agreements": {}}
            mock_manager.namespaces = ["national_laws", "ndc_commitments", "international_agreements"]
            
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["indexes_loaded"])

if __name__ == "__main__":
    unittest.main()
