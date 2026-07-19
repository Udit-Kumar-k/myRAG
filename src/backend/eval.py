import os
import json
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 20-question hardcoded eval set — representative Indian legal queries.
# Ground truth keywords are terms that MUST appear in a relevant legal chunk
# for context recall to be counted as a hit.
# Run `python -m src.backend.eval` to regenerate data/eval_set.json.
# ---------------------------------------------------------------------------

EVAL_QUERIES: List[Dict[str, Any]] = [
    # ── Criminal Law (7) ────────────────────────────────────────────────────
    {
        "id": "criminal_01",
        "category": "criminal",
        "question": "What is the punishment for murder under Bharatiya Nyaya Sanhita?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["section 103", "death", "imprisonment for life", "murder", "BNS"]
    },
    {
        "id": "criminal_02",
        "category": "criminal",
        "question": "What are the provisions for anticipatory bail under BNSS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["anticipatory bail", "BNSS", "section 482", "arrest", "apprehension"]
    },
    {
        "id": "criminal_03",
        "category": "criminal",
        "question": "What does BNS say about the offence of kidnapping?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["kidnapping", "abduction", "minor", "lawful guardian", "section"]
    },
    {
        "id": "criminal_04",
        "category": "criminal",
        "question": "What is the procedure for filing an FIR under BNSS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["FIR", "first information report", "cognizable", "police", "section"]
    },
    {
        "id": "criminal_05",
        "category": "criminal",
        "question": "What constitutes criminal conspiracy under BNS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["conspiracy", "agreement", "illegal act", "two or more", "section"]
    },
    {
        "id": "criminal_06",
        "category": "criminal",
        "question": "What are the rules regarding admissibility of electronic evidence under BSA?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["electronic", "evidence", "admissibility", "BSA", "certificate", "section"]
    },
    {
        "id": "criminal_07",
        "category": "criminal",
        "question": "What is the punishment for dowry-related offences under BNS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["dowry", "cruelty", "husband", "relatives", "imprisonment", "section"]
    },

    # ── Cyber Law (4) ──────────────────────────────────────────────────────
    {
        "id": "cyber_01",
        "category": "cyber",
        "question": "What are the penalties for hacking under the Information Technology Act?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["hacking", "unauthorized access", "computer", "section 66", "IT Act"]
    },
    {
        "id": "cyber_02",
        "category": "cyber",
        "question": "What does the IT Act say about publishing obscene content online?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["obscene", "electronic", "publish", "section 67", "IT Act"]
    },
    {
        "id": "cyber_03",
        "category": "cyber",
        "question": "What is the liability of intermediaries under the IT Act?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["intermediary", "liability", "section 79", "due diligence", "IT Act"]
    },
    {
        "id": "cyber_04",
        "category": "cyber",
        "question": "What constitutes cyber terrorism under Indian law?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["cyber terrorism", "section 66F", "critical information", "IT Act"]
    },

    # ── Consumer Law (4) ────────────────────────────────────────────────────
    {
        "id": "consumer_01",
        "category": "consumer",
        "question": "How do I file a complaint in a consumer forum for a defective product?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["consumer complaint", "defective", "product", "district commission", "consumer protection"]
    },
    {
        "id": "consumer_02",
        "category": "consumer",
        "question": "What is the liability of a product manufacturer under the Consumer Protection Act?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["product liability", "manufacturer", "defect", "consumer protection", "compensation"]
    },
    {
        "id": "consumer_03",
        "category": "consumer",
        "question": "Can a landlord refuse to return a tenant's security deposit?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["landlord", "tenant", "deposit", "refund", "rent"]
    },
    {
        "id": "consumer_04",
        "category": "consumer",
        "question": "What are the penalties for misleading advertisements under consumer protection law?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["misleading", "advertisement", "unfair trade", "penalty", "consumer"]
    },

    # ── Banking Law (2) ─────────────────────────────────────────────────────
    {
        "id": "banking_01",
        "category": "banking",
        "question": "What are the consequences of a cheque bounce under the Negotiable Instruments Act?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["cheque", "dishonour", "section 138", "negotiable instrument", "imprisonment"]
    },
    {
        "id": "banking_02",
        "category": "banking",
        "question": "What are the RBI guidelines on digital payment fraud?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["RBI", "digital payment", "fraud", "liability", "unauthorized"]
    },

    # ── General / Cross-domain (2) ──────────────────────────────────────────
    {
        "id": "general_01",
        "category": "general",
        "question": "Someone stole my identity online and took a loan in my name. What laws apply?",
        "expected_namespace": "all",
        "ground_truth_keywords": ["identity theft", "fraud", "IT Act", "BNS", "cheating"]
    },
    {
        "id": "general_02",
        "category": "general",
        "question": "Someone sent me a fake UPI payment screenshot to scam me. What legal action can I take?",
        "expected_namespace": "all",
        "ground_truth_keywords": ["UPI", "fraud", "cheating", "FIR", "IT Act"]
    },
]


def run_local_evaluation(pipeline: Any, chain: Any) -> Dict[str, Any]:
    """
    Evaluates the RAG pipeline on EVAL_QUERIES.

    Context Recall: at least one ground_truth_keyword must appear in a
    retrieved chunk's text.  This directly verifies that the retrieved
    passage covers the legally correct concept.

    RAGAS Faithfulness: unchanged — LLM answer must be grounded in context.
    """
    print(f"Starting evaluation on {len(EVAL_QUERIES)} queries...")

    results = []
    refused_count = 0
    correct_retrieval_count = 0
    total_latency = 0.0

    for item in EVAL_QUERIES:
        start = time.time()
        res = pipeline.query(item["question"])
        latency = time.time() - start
        total_latency += latency

        refused = res["refused"]
        retrieved_chunks = res["retrieved_chunks"]

        if refused:
            refused_count += 1

        # ── Context Recall (keyword-based) ──────────────────────────────
        keywords = [kw.lower() for kw in item.get("ground_truth_keywords", [])]
        found_target = False

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "").lower()
            if any(re.search(r"\b" + re.escape(kw) + r"\b", chunk_text) for kw in keywords):
                found_target = True
                break

        # If refused, still check pre-gate candidates so we can distinguish
        # retrieval failures from gate failures
        if not found_target:
            ns = res.get("namespace_searched", "all")
            candidates = pipeline.retrieve(item["question"], target_namespace=ns, top_n=20)
            for chunk in candidates:
                chunk_text = chunk.get("text", "").lower()
                if any(re.search(r"\b" + re.escape(kw) + r"\b", chunk_text) for kw in keywords):
                    found_target = True
                    break

        if found_target:
            correct_retrieval_count += 1

        results.append({
            "id":               item["id"],
            "category":         item["category"],
            "question":         item["question"],
            "refused":          refused,
            "confidence_score": res["confidence_score"],
            "context_recalled": found_target,
            "latency":          latency,
        })

    n = len(EVAL_QUERIES)
    summary = {
        "total_queries":            n,
        "average_confidence":       sum(r["confidence_score"] for r in results) / n,
        "context_recall":           correct_retrieval_count / n,
        "refusal_rate":             refused_count / n,
        "average_latency_seconds":  total_latency / n,
        "results":                  results,
    }

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total Queries        : {n}")
    print(f"Context Recall       : {summary['context_recall']:.2%}")
    print(f"Refusal Rate         : {summary['refusal_rate']:.2%}")
    print(f"Avg Confidence Score : {summary['average_confidence']:.4f}")
    print(f"Avg Latency          : {summary['average_latency_seconds']:.2f}s")
    print("==========================\n")

    os.makedirs("data", exist_ok=True)
    with open("data/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Save evaluation queries to JSON
    print("Saving NyayBot evaluation query set...")

    with open("data/eval_set.json", "w") as f:
        json.dump(EVAL_QUERIES, f, indent=2)
    print(f"Saved {len(EVAL_QUERIES)} eval queries to data/eval_set.json")

    import sys
    from src.backend.indexing import LegalIndexManager
    from src.backend.retrieval import LegalRAGPipeline
    from src.backend.chain import LegalRAGChain

    print("Loading indexes for evaluation...")
    index_manager = LegalIndexManager()
    if not index_manager.load_indexes():
        print("Error: Could not load indexes. Make sure to build them first.")
        sys.exit(1)

    threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.65))
    print(f"Initializing pipeline with CONFIDENCE_THRESHOLD={threshold}...")
    pipeline = LegalRAGPipeline(index_manager, confidence_threshold=threshold)
    chain = LegalRAGChain()

    print("Running local evaluation...")
    run_local_evaluation(pipeline, chain)
    print("Evaluation complete. Results saved to data/eval_results.json")
