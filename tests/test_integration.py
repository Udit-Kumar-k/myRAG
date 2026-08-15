import os
# Prevent connection failures to database placeholders during tests
os.environ["DATABASE_URL"] = ""
# Enable mock authentication so tests can send "Bearer mock-token" without
# hitting real Supabase JWT validation. This mirrors the MOCK_AUTH=true
# flag used in local dev — the tests explicitly opt into mock mode rather
# than relying on the (now-removed) unconditional mock-token string bypass.
os.environ["MOCK_AUTH"] = "true"

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
            "query": "What is the punishment for murder under BNS?",
            "refused": False,
            "confidence_score": 0.85,
            "namespace_searched": "criminal",
            "retrieved_chunks": [
                {
                    "text": "Section 103 of BNS prescribes punishment for murder with imprisonment for life or death.",
                    "relevance_score": 0.85,
                    "metadata": {
                        "document_name": "Bharatiya Nyaya Sanhita 2023",
                        "legal_domain": "criminal",
                        "pub_year": 2023,
                        "namespace": "criminal",
                        "source_url": "https://indiacode.nic.in"
                    }
                }
            ]
        }
        
        # Configure chain mock
        mock_chain.run.return_value = "Under BNS Section 103, the punishment for murder is imprisonment for life or death, along with a fine."
        
        # Make the request to FastAPI
        # Header has bearer token, which activates mock auth test_user_123
        response = client.post(
            "/query",
            json={
                "question": "What is the punishment for murder under BNS?",
                "conversation_id": "test_conv_abc"
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["refused"])
        self.assertEqual(data["confidence_score"], 0.85)
        self.assertIn("BNS", data["answer"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["legal_domain"], "criminal")

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
        self.assertIn("does not contain sufficient information", data["answer"])

    def test_feedback_endpoint(self):
        response = client.post(
            "/feedback",
            json={
                "conversation_id": "test_conv_abc",
                "message_id": "msg_123",
                "query": "What is Section 138?",
                "rating": "thumbs_up",
                "category": "other",
                "comment": "Accurate response"
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

    def test_telemetry_endpoint(self):
        response = client.get("/telemetry")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_queries", data)
        self.assertIn("refusal_rate", data)
        self.assertIn("average_confidence", data)
        self.assertIn("namespaces", data)
        self.assertIn("feedback", data)

if __name__ == "__main__":
    unittest.main()
