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
                "conversation_id": "conv_abc1234"  # valid pattern: conv_[a-z0-9]{7}
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
                "conversation_id": "conv_ref1234"  # valid pattern: conv_[a-z0-9]{7}
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["refused"])
        self.assertEqual(data["confidence_score"], 0.15)
        self.assertEqual(len(data["sources"]), 0)
        self.assertIn("does not contain sufficient information", data["answer"])

    @patch("src.backend.main.rag_pipeline")
    @patch("src.backend.main.rag_chain")
    def test_followup_query_memory(self, mock_chain, mock_pipeline):
        # Configure chain mock return value
        mock_chain.run.return_value = "Under BNS Section 103, murder is punishable with death or life imprisonment."
        # Configure pipeline mock with fully typed return so the normal path
        # also works — test ordering means prior history may not exist yet.
        mock_pipeline.query.return_value = {
            "query": "could you be more clear on this?",
            "refused": False,
            "confidence_score": 0.75,
            "namespace_searched": "criminal",
            "retrieved_chunks": [
                {
                    "text": "Section 103 of BNS prescribes punishment for murder.",
                    "relevance_score": 0.75,
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

        response = client.post(
            "/query",
            json={
                "question": "could you be more clear on this?",
                "conversation_id": "conv_abc1234"
            },
            headers={"Authorization": "Bearer mock-token"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["refused"])
        self.assertIsInstance(data["confidence_score"], float)
        self.assertGreaterEqual(data["confidence_score"], 0.0)
        self.assertIn("Section 103", data["answer"])

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
        # Telemetry now requires authentication
        response = client.get("/telemetry", headers={"Authorization": "Bearer mock-token"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_queries", data)
        self.assertIn("refusal_rate", data)
        self.assertIn("average_confidence", data)
        self.assertIn("namespaces", data)
        self.assertIn("feedback", data)

    def test_telemetry_requires_auth(self):
        # In production (MOCK_AUTH=false), unauthenticated access returns 401.
        # In the test environment MOCK_AUTH=true lets all requests through, so
        # we verify the handler is WIRED to authenticate_user instead of checking
        # the HTTP status (which can't be 401 while mock auth is active).
        import inspect
        from src.backend.main import get_telemetry_metrics, authenticate_user
        sig = inspect.signature(get_telemetry_metrics)
        deps = [
            p.default for p in sig.parameters.values()
            if hasattr(p.default, 'dependency')
        ]
        dep_fns = [d.dependency for d in deps]
        self.assertIn(authenticate_user, dep_fns,
            "get_telemetry_metrics must depend on authenticate_user")

    def test_telemetry_blocks_guest_user(self):
        # Guests should be blocked from seeing sensitive telemetry metrics
        response = client.get("/telemetry", headers={"Authorization": "Bearer guest-token-test1234"})
        self.assertEqual(response.status_code, 403)
        self.assertIn("detail", response.json())

    def test_get_client_ip_headers(self):
        from src.backend.main import get_client_ip
        from starlette.requests import Request

        # 1. Cloudflare header
        scope_cf = {
            "type": "http",
            "headers": [(b"cf-connecting-ip", b"198.51.100.4")]
        }
        req_cf = Request(scope_cf)
        self.assertEqual(get_client_ip(req_cf), "198.51.100.4")

        # 2. X-Forwarded-For multi-hop (leftmost is original client)
        scope_xff = {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"203.0.113.195, 70.41.3.18, 10.0.0.1")]
        }
        req_xff = Request(scope_xff)
        self.assertEqual(get_client_ip(req_xff), "203.0.113.195")

    @patch("src.backend.main.rag_pipeline")
    @patch("src.backend.main.rag_chain")
    def test_stream_query_endpoint(self, mock_chain, mock_pipeline):
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

        async def mock_tokens(*args, **kwargs):
            for t in ["Under ", "BNS ", "Section 103..."]:
                yield t

        mock_chain.astream_run.side_effect = mock_tokens
        mock_chain.verify_citations.return_value = ("Under BNS Section 103...", [])

        response = client.post(
            "/stream-query",
            json={
                "question": "What is the punishment for murder under BNS?",
                "conversation_id": "conv_abc1234"
            },
            headers={"Authorization": "Bearer mock-token"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        body = response.text
        self.assertIn('data: {"token": "Under ", "done": false}', body)
        self.assertIn('"done": true', body)
        self.assertIn('"confidence_score": 0.85', body)
        self.assertIn('"sources":', body)

if __name__ == "__main__":
    unittest.main()
