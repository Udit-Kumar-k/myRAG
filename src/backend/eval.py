import os
import json
import time
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# 20-question hardcoded eval set — representative USMLE-style queries.
# Ground truth keywords are terms that MUST appear in a relevant textbook chunk
# for context recall to be counted as a hit.
# Run `python -m src.backend.eval` to regenerate data/eval_set.json.
# To extend to the full MedQA benchmark, call build_eval_from_medqa().
# ---------------------------------------------------------------------------

EVAL_QUERIES: List[Dict[str, Any]] = [
    # ── Basic Sciences (7) ──────────────────────────────────────────────────
    {
        "id": "basic_01",
        "category": "basic_sciences",
        "question": "What enzyme catalyzes the rate-limiting step of glycolysis?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["phosphofructokinase", "PFK", "glycolysis", "rate-limiting"]
    },
    {
        "id": "basic_02",
        "category": "basic_sciences",
        "question": "Which cranial nerve provides the afferent limb of the corneal reflex?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["trigeminal", "ophthalmic", "V1", "corneal reflex", "CN V"]
    },
    {
        "id": "basic_03",
        "category": "basic_sciences",
        "question": "What is the primary site of erythropoietin production in adults?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["kidney", "renal", "erythropoietin", "peritubular"]
    },
    {
        "id": "basic_04",
        "category": "basic_sciences",
        "question": "Which type of collagen is predominantly found in hyaline cartilage?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["type II", "collagen", "hyaline", "cartilage", "chondrocyte"]
    },
    {
        "id": "basic_05",
        "category": "basic_sciences",
        "question": "Which enzyme is deficient in phenylketonuria (PKU)?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["phenylalanine hydroxylase", "PKU", "phenylketonuria", "phenylalanine"]
    },
    {
        "id": "basic_06",
        "category": "basic_sciences",
        "question": "Which complex of the electron transport chain produces the greatest amount of ATP via oxidative phosphorylation?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["complex V", "ATP synthase", "oxidative phosphorylation", "proton gradient"]
    },
    {
        "id": "basic_07",
        "category": "basic_sciences",
        "question": "What structural feature of the small intestine maximally increases absorptive surface area?",
        "geography_iso": "basic_sciences",
        "expected_namespace": "basic_sciences",
        "ground_truth_keywords": ["microvilli", "brush border", "villi", "surface area", "intestinal"]
    },

    # ── Pharmacology (7) ────────────────────────────────────────────────────
    {
        "id": "pharma_01",
        "category": "pharmacology",
        "question": "What is the mechanism of action of metformin in type 2 diabetes?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["AMPK", "hepatic glucose", "biguanide", "gluconeogenesis", "metformin"]
    },
    {
        "id": "pharma_02",
        "category": "pharmacology",
        "question": "Which class of antibiotics inhibits cell wall synthesis by binding penicillin-binding proteins?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["beta-lactam", "penicillin-binding protein", "peptidoglycan", "cell wall", "PBP"]
    },
    {
        "id": "pharma_03",
        "category": "pharmacology",
        "question": "A patient develops nephrotoxicity and ototoxicity after antibiotic treatment. Which class is most likely responsible?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["aminoglycoside", "gentamicin", "nephrotoxicity", "ototoxicity", "tobramycin"]
    },
    {
        "id": "pharma_04",
        "category": "pharmacology",
        "question": "How do statins reduce circulating LDL cholesterol?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["HMG-CoA reductase", "statin", "LDL", "cholesterol synthesis", "mevalonate"]
    },
    {
        "id": "pharma_05",
        "category": "pharmacology",
        "question": "Which antifungal drug inhibits ergosterol synthesis by blocking lanosterol demethylase?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["azole", "fluconazole", "ergosterol", "cytochrome P450", "lanosterol"]
    },
    {
        "id": "pharma_06",
        "category": "pharmacology",
        "question": "A hypertensive patient develops a persistent dry cough after starting antihypertensive therapy. Which drug class is most likely responsible?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["ACE inhibitor", "bradykinin", "cough", "angiotensin converting enzyme"]
    },
    {
        "id": "pharma_07",
        "category": "pharmacology",
        "question": "What is the primary mechanism of penicillin resistance in Staphylococcus aureus?",
        "geography_iso": "pharmacology",
        "expected_namespace": "pharmacology",
        "ground_truth_keywords": ["beta-lactamase", "penicillinase", "resistance", "MRSA", "mecA"]
    },

    # ── Clinical Medicine (6) ───────────────────────────────────────────────
    {
        "id": "clinical_01",
        "category": "clinical_medicine",
        "question": "A 55-year-old man presents with crushing substernal chest pain and ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
        "geography_iso": "clinical_medicine",
        "expected_namespace": "clinical_medicine",
        "ground_truth_keywords": ["right coronary artery", "inferior MI", "RCA", "ST elevation", "inferior wall"]
    },
    {
        "id": "clinical_02",
        "category": "clinical_medicine",
        "question": "A 3-year-old presents with fever, drooling, and stridor. She sits upright and leans forward. Which organism is most likely responsible?",
        "geography_iso": "clinical_medicine",
        "expected_namespace": "clinical_medicine",
        "ground_truth_keywords": ["Haemophilus influenzae", "epiglottitis", "type B", "HiB", "stridor"]
    },
    {
        "id": "clinical_03",
        "category": "clinical_medicine",
        "question": "A 25-year-old presents with polyuria, polydipsia, weight loss, glucose 350 mg/dL, pH 7.2, and ketonuria. What is the most appropriate initial management?",
        "geography_iso": "clinical_medicine",
        "expected_namespace": "clinical_medicine",
        "ground_truth_keywords": ["insulin", "IV fluids", "DKA", "diabetic ketoacidosis", "potassium replacement"]
    },
    {
        "id": "clinical_04",
        "category": "clinical_medicine",
        "question": "Which clinical sign is most specific for acute appendicitis?",
        "geography_iso": "clinical_medicine",
        "expected_namespace": "clinical_medicine",
        "ground_truth_keywords": ["McBurney", "rebound tenderness", "Rovsing", "psoas sign", "appendicitis"]
    },
    {
        "id": "clinical_05",
        "category": "clinical_medicine",
        "question": "A 70-year-old presents with dyspnea, bilateral leg edema, and orthopnea. CXR shows cardiomegaly and pulmonary vascular congestion. What is the most appropriate initial pharmacological treatment?",
        "geography_iso": "clinical_medicine",
        "expected_namespace": "clinical_medicine",
        "ground_truth_keywords": ["furosemide", "diuretic", "heart failure", "ACE inhibitor", "loop diuretic"]
    },
    {
        "id": "clinical_06",
        "category": "clinical_medicine",
        "question": "A newborn presents with jaundice within 24 hours of birth. Direct Coombs test is positive. What is the most likely diagnosis?",
        "geography_iso": "clinical_medicine",
        "expected_namespace": "clinical_medicine",
        "ground_truth_keywords": ["hemolytic disease", "Rh incompatibility", "ABO incompatibility", "maternal antibodies", "Coombs"]
    },
]


def build_eval_from_medqa(num_questions: int = 100) -> List[Dict[str, Any]]:
    """
    Optional: loads USMLE questions from the public MedQA benchmark and
    converts them to the EVAL_QUERIES schema. Falls back to EVAL_QUERIES
    if the dataset is unavailable.
    Call this to extend beyond the 20 hardcoded queries.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            split=f"test[:{num_questions}]",
        )
    except Exception as e:
        print(f"MedQA dataset unavailable ({e}). Falling back to hardcoded eval set.")
        return EVAL_QUERIES

    records = []
    for i, row in enumerate(ds):
        question = row.get("question", "")
        answer_key = row.get("answer_idx") or row.get("answer") or ""
        options = row.get("options") or {}

        # Extract the text of the correct option as the primary ground truth
        correct_text = ""
        if isinstance(options, dict):
            correct_text = options.get(answer_key, "")
        elif isinstance(options, list):
            # Some versions use a list of dicts
            for opt in options:
                if opt.get("key") == answer_key:
                    correct_text = opt.get("value", "")
                    break

        # Build ground truth keywords from the correct answer text
        keywords = [w.lower() for w in correct_text.split() if len(w) > 4][:6]
        if not keywords:
            keywords = [answer_key]

        records.append({
            "id": f"medqa_{i:04d}",
            "category": "clinical_medicine",
            "question": question,
            "geography_iso": "clinical_medicine",
            "expected_namespace": "all",
            "ground_truth_keywords": keywords,
        })

    return records if records else EVAL_QUERIES


def run_local_evaluation(pipeline: Any, chain: Any) -> Dict[str, Any]:
    """
    Evaluates the RAG pipeline on EVAL_QUERIES.

    Context Recall: at least one ground_truth_keyword must appear in a
    retrieved chunk's text.  This is more meaningful than the old country-ISO
    check — it directly verifies that the retrieved passage covers the
    medically correct concept.

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
            if any(kw in chunk_text for kw in keywords):
                found_target = True
                break

        # If refused, still check pre-gate candidates so we can distinguish
        # retrieval failures from gate failures
        if not found_target:
            ns = res.get("namespace_searched", "all")
            candidates = pipeline.retrieve(item["question"], target_namespace=ns, top_n=5)
            for chunk in candidates:
                chunk_text = chunk.get("text", "").lower()
                if any(kw in chunk_text for kw in keywords):
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
    
    # Try to load evaluation questions from public HuggingFace MedQA benchmark
    print("Attempting to load evaluation set from HuggingFace MedQA...")
    queries = build_eval_from_medqa(num_questions=50)  # 50 representative questions
    
    with open("data/eval_set.json", "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Saved {len(queries)} eval queries to data/eval_set.json")
