import os
# Prevent connection failures to database placeholders during tests
os.environ["DATABASE_URL"] = ""

import unittest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# Patch the lifespan context manager so tests don't load heavy models/indexes.
# We replace it with a no-op async context manager before importing the app.
@asynccontextmanager
async def _noop_lifespan(app):
    yield

with patch("src.backend.main.lifespan", _noop_lifespan):
    from src.backend.main import app
    client = TestClient(app)

class TestIntegrationEndpoints(unittest.TestCase):
    
    @patch("src.backend.main.rag_pipeline")
    @patch("src.backend.main.rag_chain")
    def test_query_endpoint_success(self, mock_chain, mock_pipeline):
        # Configure pipeline mock for a successful query
        mock_pipeline.query.return_value = {
            "query": "What is the first-line treatment for pneumonia?",
            "refused": False,
            "confidence_score": 0.85,
            "namespace_searched": "clinical_medicine",
            "retrieved_chunks": [
                {
                    "text": "First-line treatment for community-acquired pneumonia is amoxicillin.",
                    "relevance_score": 0.85,
                    "metadata": {
                        "document_name": "Nelson Pediatrics",
                        "geography_iso": "clinical_medicine",
                        "pub_year": 2021,
                        "namespace": "clinical_medicine",
                        "source_url": "https://ncbi.nlm.nih.gov/books"
                    }
                }
            ]
        }
        
        # Configure chain mock
        mock_chain.run.return_value = "First-line treatment for community-acquired pneumonia is amoxicillin as stated in Nelson Pediatrics."
        
        # Make the request to FastAPI
        # Header has bearer token, which activates mock auth test_user_123
        response = client.post(
            "/query",
            json={
                "question": "What is the first-line treatment for pneumonia?",
                "conversation_id": "test_conv_abc"
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["refused"])
        self.assertEqual(data["confidence_score"], 0.85)
        self.assertIn("pneumonia", data["answer"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["geography_iso"], "clinical_medicine")

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
        self.assertIn("Insufficient evidence found in indexed medical textbooks", data["answer"])

    def test_health_endpoint(self):
        # Patch index manager to say indexes are loaded
        with patch("src.backend.main.index_manager") as mock_manager:
            mock_manager.faiss_indexes = {"basic_sciences": {}, "pharmacology": {}, "clinical_medicine": {}}
            mock_manager.namespaces = ["basic_sciences", "pharmacology", "clinical_medicine"]
            
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["indexes_loaded"])

if __name__ == "__main__":
    unittest.main()
