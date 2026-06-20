import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any

# Enforce headless mode for matplotlib before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.eval import EVAL_QUERIES

OUT_OF_SCOPE_QUERIES = [
    "Who is the current prime minister of the United Kingdom?",
    "What is the capital city of Australia?",
    "How do you bake a sourdough bread at home?",
    "What are the symptoms of common cold?",
    "Explain the theory of general relativity by Einstein.",
    "What is the stock price of Apple today?",
    "Who won the FIFA World Cup in 2022?",
    "What is the population of Tokyo?",
    "How does a database transaction index work?",
    "What are the best tourist spots in Italy?",
    "How do you write a quicksort algorithm in Python?",
    "What is the formula for calculating compound interest?",
    "Who is the author of Harry Potter books?",
    "What is the distance between Earth and Mars?",
    "How do you clean a laptop keyboard safely?",
    "What is the history of the Eiffel Tower?",
    "Explain the rules of cricket.",
    "What are the benefits of cardiovascular exercise?",
    "How does a nuclear power plant generate electricity?",
    "What are the primary colors in painting?"
]

def run_real_calibration() -> List[Dict[str, Any]]:
    """Runs queries through the actual pipeline and collects confidence scores."""
    print("Running calibration using actual indexed RAG pipeline...")
    from src.backend.indexing import ClimateIndexManager
    from src.backend.retrieval import ClimateRAGPipeline
    
    index_manager = ClimateIndexManager()
    if not index_manager.load_indexes():
        raise ValueError("Failed to load indexes. Build them first.")
        
    pipeline = ClimateRAGPipeline(index_manager, confidence_threshold=0.0) # threshold 0 to get all scores
    
    records = []
    print("Running in-scope queries...")
    for item in EVAL_QUERIES:
        res = pipeline.query(item["question"])
        records.append({
            "question": item["question"],
            "confidence_score": res["confidence_score"],
            "is_in_scope": True
        })
        
    print("Running out-of-scope queries...")
    for q in OUT_OF_SCOPE_QUERIES:
        res = pipeline.query(q)
        records.append({
            "question": q,
            "confidence_score": res["confidence_score"],
            "is_in_scope": False
        })
        
    return records

def main():
    print("=== ClimateRAG Confidence Threshold Calibration ===")
    
    index_exists = os.path.exists("data/indexes/national_laws_chunks.pkl")
    if not index_exists:
        raise ValueError("Failed to load indexes. Build them first using: python -m src.backend.indexing")
        
    try:
        records = run_real_calibration()
    except Exception as e:
        print(f"Error running real calibration: {e}")
        sys.exit(1)
        
    # Calibrate thresholds from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []
    f1s = []
    
    for T in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for r in records:
            score = r["confidence_score"]
            is_in_scope = r["is_in_scope"]
            
            # If score >= T, it passes (not refused)
            # If score < T, it is blocked (refused)
            if score >= T:
                if is_in_scope:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_in_scope:
                    fn += 1
                else:
                    tn += 1
                    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    # Find optimal threshold that maximizes F1 score
    opt_idx = np.argmax(f1s)
    opt_threshold = thresholds[opt_idx]
    opt_f1 = f1s[opt_idx]
    opt_precision = precisions[opt_idx]
    opt_recall = recalls[opt_idx]
    
    print(f"\n--- Optimization Results ---")
    print(f"Optimal Confidence Threshold: {opt_threshold:.2f}")
    print(f"Max F1-Score: {opt_f1:.4f}")
    print(f"Precision at Optimal: {opt_precision:.4f}")
    print(f"Recall at Optimal: {opt_recall:.4f}")
    print(f"----------------------------\n")
    
    # Save the calibration plot
    os.makedirs("data", exist_ok=True)
    plot_path = "data/confidence_calibration.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='#1f77b4', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', color='#ff7f0e', linewidth=2)
    plt.plot(thresholds, f1s, label='F1-Score', color='#2ca02c', linewidth=2.5)
    plt.axvline(x=opt_threshold, color='red', linestyle='--', label=f'Optimal Threshold ({opt_threshold:.2f})')
    
    plt.title('Confidence Gate Threshold Calibration', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Confidence Score Threshold', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    
    # Add text annotation
    plt.text(opt_threshold + 0.02, 0.5, 
             f"Optimal T = {opt_threshold:.2f}\nF1 = {opt_f1:.3f}\nPrecision = {opt_precision:.3f}\nRecall = {opt_recall:.3f}",
             color='red', weight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
             
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Confidence calibration plot saved successfully to {plot_path}")
    
    # Save threshold details to data/calibration_details.json
    details = {
        "optimal_threshold": float(opt_threshold),
        "max_f1_score": float(opt_f1),
        "precision_at_optimal": float(opt_precision),
        "recall_at_optimal": float(opt_recall),
        "calibration_timestamp": float(time.time())
    }
    with open("data/calibration_details.json", "w") as f:
        json.dump(details, f, indent=2)

if __name__ == "__main__":
    main()
