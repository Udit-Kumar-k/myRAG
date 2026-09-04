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

    def test_hybrid_tenancy_lockout(self):
        q = "My landlord locked me out of the apartment and cut off electricity."
        self.assertEqual(route_query(q), "general,criminal")

    def test_hybrid_cyber_extortion(self):
        q = "Someone is blackmailing me on WhatsApp demanding money to delete morphed photos."
        self.assertEqual(route_query(q), "cyber,criminal")

    def test_direct_statutory_query_detection(self):
        from src.backend.retrieval import is_direct_statutory_query
        # Direct section queries
        self.assertTrue(is_direct_statutory_query("What is Section 138 of the Negotiable Instruments Act?"))
        self.assertTrue(is_direct_statutory_query("Explain Section 308 of BNS 2023"))
        self.assertTrue(is_direct_statutory_query("FIR under Sec 173 BNSS"))
        self.assertTrue(is_direct_statutory_query("Compensation under Consumer Protection Act 2019"))
        # Colloquial queries without explicit section or statute
        self.assertFalse(is_direct_statutory_query("My boss didn't pay me"))
        self.assertFalse(is_direct_statutory_query("Someone stole my phone"))
        # Follow-up with history should never skip HyDE
        self.assertFalse(is_direct_statutory_query("What is the punishment for Section 308?", history=[{"role": "user", "content": "hi"}]))

    def test_expanded_follow_up_intent_detection(self):
        from src.backend.chain import LegalRAGChain
        # Natural contextual follow-up phrases
        self.assertEqual(LegalRAGChain.detect_query_intent("Can police arrest without warrant in such cases?", has_prior_history=True), "FOLLOW_UP")
        self.assertEqual(LegalRAGChain.detect_query_intent("What is the limitation period to file this?", has_prior_history=True), "FOLLOW_UP")
        self.assertEqual(LegalRAGChain.detect_query_intent("Can my landlord do this?", has_prior_history=True), "FOLLOW_UP")
        self.assertEqual(LegalRAGChain.detect_query_intent("Who should I send the notice to?", has_prior_history=True), "FOLLOW_UP")
        self.assertEqual(LegalRAGChain.detect_query_intent("Can they also seize my phone?", has_prior_history=True), "FOLLOW_UP")
        # Standalone legal query with prior history
        self.assertEqual(LegalRAGChain.detect_query_intent("What is the punishment for murder under BNS?", has_prior_history=True), "LEGAL_QUERY")


if __name__ == "__main__":
    unittest.main()

