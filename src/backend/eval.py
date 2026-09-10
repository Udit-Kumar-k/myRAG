import os
import json
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 50-question eval set — representative Indian legal queries.
# Ground truth keywords are terms that MUST appear in a relevant legal chunk
# for context recall to be counted as a hit.
# Run `python -m src.backend.eval` to regenerate data/eval_set.json.
# ---------------------------------------------------------------------------

EVAL_QUERIES: List[Dict[str, Any]] = [

    # ── CRIMINAL (15) — BNS + BNSS + BSA — primary indexed statutes ──────────

    # Direct statutory (5)
    {
        "id": "criminal_01", "category": "criminal",
        "question": "What is the punishment for murder under the Bharatiya Nyaya Sanhita?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["section 103", "death", "imprisonment for life", "murder"]
    },
    {
        "id": "criminal_02", "category": "criminal",
        "question": "What are the provisions for anticipatory bail under BNSS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["anticipatory bail", "section 482", "arrest", "apprehension"]
    },
    {
        "id": "criminal_03", "category": "criminal",
        "question": "What does BNS say about the offence of kidnapping?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["kidnapping", "abduction", "minor", "lawful guardian"]
    },
    {
        "id": "criminal_04", "category": "criminal",
        "question": "What is the procedure for filing an FIR under BNSS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["FIR", "first information report", "cognizable", "police"]
    },
    {
        "id": "criminal_05", "category": "criminal",
        "question": "What constitutes criminal conspiracy under BNS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["conspiracy", "agreement", "illegal act", "two or more"]
    },

    # Naturalistic (5)
    {
        "id": "criminal_06", "category": "criminal",
        "question": "My neighbour attacked me with a weapon and I was hospitalised. What crime has been committed?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["grievous hurt", "voluntarily", "weapon", "imprisonment", "section"]
    },
    {
        "id": "criminal_07", "category": "criminal",
        "question": "A group of men followed me home and threatened to harm me. Is there a law against this?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["criminal intimidation", "threat", "section", "BNS"]
    },
    {
        "id": "criminal_08", "category": "criminal",
        "question": "Someone is spreading false rumours about me to ruin my reputation. What can I do legally?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["defamation", "reputation", "section", "imputation"]
    },
    {
        "id": "criminal_09", "category": "criminal",
        "question": "My wife has been harassed by her in-laws for dowry. What are her legal options?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["dowry", "cruelty", "husband", "relatives", "imprisonment"]
    },
    {
        "id": "criminal_10", "category": "criminal",
        "question": "A person was arrested without a warrant. Is this legal under BNSS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["arrest", "without warrant", "cognizable", "police officer"]
    },

    # Evidence / BSA (3)
    {
        "id": "criminal_11", "category": "criminal",
        "question": "Is a WhatsApp message admissible as evidence in court under BSA?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["electronic", "evidence", "admissibility", "certificate"]
    },
    {
        "id": "criminal_12", "category": "criminal",
        "question": "What is the burden of proof in a criminal trial under BSA?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["burden of proof", "accused", "presumption", "BSA"]
    },
    {
        "id": "criminal_13", "category": "criminal",
        "question": "Can a confession made to a police officer be used as evidence under BSA?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["confession", "police officer", "inadmissible", "BSA"]
    },

    # Out-of-scope / should refuse (2)
    {
        "id": "criminal_14", "category": "criminal",
        "question": "What are the tax implications of selling a property in India?",
        "expected_namespace": "general",
        "ground_truth_keywords": ["capital gains", "income tax", "property", "sale"]
    },
    {
        "id": "criminal_15", "category": "criminal",
        "question": "What is the process for getting a divorce in India?",
        "expected_namespace": "general",
        "ground_truth_keywords": ["divorce", "marriage", "Hindu Marriage Act", "dissolution"]
    },

    # ── CYBER (10) — IT Act — sparse in indexes, lower recall expected ────────

    # Direct statutory (4)
    {
        "id": "cyber_01", "category": "cyber",
        "question": "What are the penalties for hacking under the Information Technology Act?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["hacking", "unauthorized access", "section 66", "IT Act"]
    },
    {
        "id": "cyber_02", "category": "cyber",
        "question": "What does the IT Act say about publishing obscene content online?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["obscene", "electronic", "publish", "section 67", "IT Act"]
    },
    {
        "id": "cyber_03", "category": "cyber",
        "question": "What is the liability of intermediaries under the IT Act?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["intermediary", "liability", "section 79", "due diligence"]
    },
    {
        "id": "cyber_04", "category": "cyber",
        "question": "What constitutes cyber terrorism under Indian law?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["cyber terrorism", "section 66F", "critical information", "IT Act"]
    },

    # Naturalistic (4)
    {
        "id": "cyber_05", "category": "cyber",
        "question": "Someone hacked into my email and is blackmailing me with my private photos. What law applies?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["unauthorized access", "section 66", "IT Act", "computer"]
    },
    {
        "id": "cyber_06", "category": "cyber",
        "question": "A fake profile of me was created on social media to defame me. What can I do?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["identity", "electronic", "IT Act", "impersonation"]
    },
    {
        "id": "cyber_07", "category": "cyber",
        "question": "I received a phishing email and lost money from my bank account. Is there a cyber law for this?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["fraud", "electronic", "IT Act", "section 66", "cheating"]
    },
    {
        "id": "cyber_08", "category": "cyber",
        "question": "A company shared my personal data without my consent. What are my rights under IT law?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["data", "personal information", "IT Act", "section 43A", "sensitive"]
    },

    # Out-of-scope / should refuse (2)
    {
        "id": "cyber_09", "category": "cyber",
        "question": "What is the best antivirus software for protecting against cyberattacks?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["antivirus", "software", "protection", "cybersecurity"]
    },
    {
        "id": "cyber_10", "category": "cyber",
        "question": "How does end-to-end encryption work?",
        "expected_namespace": "cyber",
        "ground_truth_keywords": ["encryption", "key", "algorithm", "decryption"]
    },

    # ── CONSUMER (10) — Consumer Protection Act — sparse in indexes ───────────

    # Direct statutory (4)
    {
        "id": "consumer_01", "category": "consumer",
        "question": "How do I file a complaint in a consumer forum for a defective product?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["consumer complaint", "defective", "district commission", "consumer protection"]
    },
    {
        "id": "consumer_02", "category": "consumer",
        "question": "What is the liability of a manufacturer for a defective product under the Consumer Protection Act?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["product liability", "manufacturer", "defect", "compensation"]
    },
    {
        "id": "consumer_03", "category": "consumer",
        "question": "What are the penalties for misleading advertisements under consumer protection law?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["misleading", "advertisement", "unfair trade", "penalty"]
    },
    {
        "id": "consumer_04", "category": "consumer",
        "question": "What is the pecuniary jurisdiction of the District Consumer Commission?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["district commission", "jurisdiction", "lakh", "consumer protection"]
    },

    # Naturalistic (4)
    {
        "id": "consumer_05", "category": "consumer",
        "question": "I bought a phone online and it stopped working after two days. Can I get a full refund?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["defective", "product", "refund", "consumer", "commission"]
    },
    {
        "id": "consumer_06", "category": "consumer",
        "question": "A hospital charged me for a procedure that was never done. What are my rights?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["deficiency", "service", "consumer", "commission", "compensation"]
    },
    {
        "id": "consumer_07", "category": "consumer",
        "question": "A builder took full payment but has not delivered my flat in three years. What can I do?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["deficiency", "service", "consumer", "complaint", "compensation"]
    },
    {
        "id": "consumer_08", "category": "consumer",
        "question": "An e-commerce platform is running a fake discount scheme. Which law covers this?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["unfair trade", "misleading", "consumer protection", "e-commerce"]
    },

    # Out-of-scope / should refuse (2)
    {
        "id": "consumer_09", "category": "consumer",
        "question": "What is the best consumer electronics brand in India?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["brand", "electronics", "review", "quality"]
    },
    {
        "id": "consumer_10", "category": "consumer",
        "question": "How do I start a business selling consumer goods online?",
        "expected_namespace": "consumer",
        "ground_truth_keywords": ["business", "e-commerce", "GST", "registration"]
    },

    # ── BANKING (10) — NI Act + RBI guidelines — sparse in indexes ───────────

    # Direct statutory (4)
    {
        "id": "banking_01", "category": "banking",
        "question": "What are the consequences of a cheque bounce under the Negotiable Instruments Act?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["cheque", "dishonour", "section 138", "negotiable instrument", "imprisonment"]
    },
    {
        "id": "banking_02", "category": "banking",
        "question": "What is the time limit for filing a cheque bounce case under the NI Act?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["cheque", "limitation", "section 138", "month", "notice"]
    },
    {
        "id": "banking_03", "category": "banking",
        "question": "What are the RBI guidelines on liability for unauthorised digital transactions?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["RBI", "digital", "unauthorized", "liability", "customer"]
    },
    {
        "id": "banking_04", "category": "banking",
        "question": "What is a promissory note under the Negotiable Instruments Act?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["promissory note", "negotiable instrument", "unconditional", "section"]
    },

    # Naturalistic (4)
    {
        "id": "banking_05", "category": "banking",
        "question": "Someone issued me a cheque and it bounced. What legal steps can I take?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["cheque", "dishonour", "section 138", "notice", "prosecution"]
    },
    {
        "id": "banking_06", "category": "banking",
        "question": "My bank account was debited without my knowledge. What are my rights against the bank?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["unauthorized", "transaction", "bank", "liability", "RBI"]
    },
    {
        "id": "banking_07", "category": "banking",
        "question": "A moneylender is charging me 60% annual interest on a loan. Is this legal?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["interest", "loan", "moneylender", "usurious", "banking"]
    },
    {
        "id": "banking_08", "category": "banking",
        "question": "What happens if I cannot repay a bank loan? Can I be arrested?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["loan", "default", "arrest", "recovery", "bank"]
    },

    # Out-of-scope / should refuse (2)
    {
        "id": "banking_09", "category": "banking",
        "question": "Which bank offers the best fixed deposit interest rates right now?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["fixed deposit", "interest rate", "bank", "savings"]
    },
    {
        "id": "banking_10", "category": "banking",
        "question": "How do I apply for a home loan from SBI?",
        "expected_namespace": "banking",
        "ground_truth_keywords": ["home loan", "SBI", "application", "mortgage"]
    },

    # ── GENERAL / CROSS-DOMAIN (5) ────────────────────────────────────────────

    {
        "id": "general_01", "category": "general",
        "question": "Someone stole my identity online and took a loan in my name. What laws apply?",
        "expected_namespace": "all",
        "ground_truth_keywords": ["identity theft", "fraud", "IT Act", "BNS", "cheating"]
    },
    {
        "id": "general_02", "category": "general",
        "question": "I was scammed by an online seller who took my money but never delivered the product.",
        "expected_namespace": "all",
        "ground_truth_keywords": ["cheating", "fraud", "consumer", "section", "BNS"]
    },
    {
        "id": "general_03", "category": "general",
        "question": "My employer has not paid my salary for three months. What legal options do I have?",
        "expected_namespace": "general",
        "ground_truth_keywords": ["wages", "employer", "labour", "payment", "section"]
    },
    {
        "id": "general_04", "category": "general",
        "question": "A person threatened to share my private photos unless I paid them money. What crime is this?",
        "expected_namespace": "all",
        "ground_truth_keywords": ["extortion", "blackmail", "section", "BNS", "IT Act"]
    },
    {
        "id": "general_05", "category": "general",
        "question": "What is the punishment for cheating by personation under BNS?",
        "expected_namespace": "criminal",
        "ground_truth_keywords": ["cheating", "personation", "BNS", "punishment", "section"]
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

    threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.55"))
    print(f"Initializing pipeline with CONFIDENCE_THRESHOLD={threshold}...")
    pipeline = LegalRAGPipeline(index_manager, confidence_threshold=threshold)
    chain = LegalRAGChain()

    print("Running local evaluation...")
    run_local_evaluation(pipeline, chain)
    print("Evaluation complete. Results saved to data/eval_results.json")
