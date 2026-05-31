import unittest
from src.backend.retrieval import route_query

class TestQueryRouter(unittest.TestCase):
    def test_law_routing(self):
        queries = [
            "What does Germany's climate law state?",
            "Show me national legislation in Brazil.",
            "Are there domestic regulations in China?",
            "What was passed by the GBR parliament?"
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "national_laws")

    def test_ndc_routing(self):
        queries = [
            "What is India's NDC commitment?",
            "Show me the USA Nationally Determined Contribution submission.",
            "What did South Africa pledge under its NDC target?",
            "What are South Korea's NDC commitments?"
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "ndc_commitments")

    def test_international_routing(self):
        queries = [
            "What does Article 6 of the Paris Agreement state?",
            "Show me COP decisions on climate finance.",
            "What are the outcomes of the global stocktake?",
            "Is there an international treaty on emissions?"
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "international_agreements")

    def test_all_routing_fallback(self):
        queries = [
            "How does G20 climate policy work?", # no specific namespace keywords
            "Does Germany's Climate Protection Act match its NDC?", # contains Act (law) and NDC (ndc) - tie
            "What are the domestic targets under the Paris Agreement?", # contains domestic (law) and Paris Agreement (int) - tie
            "Tell me about global warming." # general
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "all")

if __name__ == "__main__":
    unittest.main()
