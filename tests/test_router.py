import unittest
from src.backend.retrieval import route_query


class TestLegalQueryRouter(unittest.TestCase):

    def test_criminal_routing(self):
        queries = [
            "What is the punishment for murder under BNS?",
            "How do I file an FIR for theft?",
            "What are the bail provisions under BNSS?",
            "What does BSA say about confession evidence?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "criminal")

    def test_cyber_routing(self):
        queries = [
            "What are the penalties for hacking under IT Act?",
            "Someone committed identity theft using my email.",
            "What is cybercrime under Indian law?",
            "Is phishing punishable under the IT Act?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "cyber")

    def test_consumer_routing(self):
        queries = [
            "How do I file a consumer complaint for a defective product?",
            "The seller delivered a damaged item and won't give a refund.",
            "What are consumer rights regarding refund of goods?",
            "Can I sue for misleading advertisement under consumer protection?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "consumer")

    def test_general_routing(self):
        queries = [
            "My landlord won't return my deposit.",
            "Can the lessor evict the tenant without notice under the lease?",
            "What are my rights regarding unpaid salary and wages?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "general")

    def test_banking_routing(self):
        queries = [
            "What are RBI guidelines on UPI transaction limits?",
            "What happens in case of a cheque bounce?",
            "Is my bank loan interest rate legal?",
            "What are the NEFT transfer limits set by the reserve bank?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "banking")

    def test_all_routing_fallback(self):
        queries = [
            "Tell me about Indian law.",
            "What are my rights?",
            "How does the legal system work?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "all")


if __name__ == "__main__":
    unittest.main()
