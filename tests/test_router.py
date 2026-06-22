import unittest
from src.backend.retrieval import route_query


class TestMedicalQueryRouter(unittest.TestCase):

    def test_basic_sciences_routing(self):
        queries = [
            "What enzyme catalyzes the rate-limiting step of glycolysis?",
            "Describe the anatomy of the brachial plexus.",
            "How does action potential propagation work in neurons?",
            "What is the role of mitochondria in cellular respiration?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "basic_sciences")

    def test_pharmacology_routing(self):
        queries = [
            "What is the mechanism of action of metformin?",
            "Which antibiotic inhibits bacterial cell wall synthesis?",
            "What are the side effects of statins?",
            "Explain the mechanism of penicillin resistance.",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "pharmacology")

    def test_clinical_medicine_routing(self):
        queries = [
            "A patient presents with crushing chest pain and ST elevation.",
            "What is the first-line treatment for community-acquired pneumonia?",
            "How do you manage a patient with diabetic ketoacidosis?",
            "What are the clinical signs of appendicitis?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "clinical_medicine")

    def test_all_routing_fallback(self):
        queries = [
            "Tell me about medicine.",
            "What causes disease?",
            "How does the body work?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(route_query(q), "all")


if __name__ == "__main__":
    unittest.main()
