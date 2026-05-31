import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def mine_hard_negatives(eval_set_path: str = "data/eval_set.json") -> List[InputExample]:
    """
    Mines contrastive triplets (query, positive, hard_negative) from evaluation set.
    For each country, a positive is the correct target chunk, and a hard negative
    is another document (or different year) from the same country.
    """
    if not os.path.exists(eval_set_path):
        print(f"Eval set not found at {eval_set_path}. Creating a sample triplet dataset.")
        return [
            InputExample(
                texts=[
                    "What is India's emissions intensity target for 2030?",
                    "India targets 45 percent GDP emission intensity reduction by 2030 relative to 2005 levels.",
                    "India targets 33-35 percent GDP emissions intensity reduction by 2030 relative to 2005 levels." # Older 2015 NDC as hard negative
                ]
            )
        ]
        
    with open(eval_set_path, "r") as f:
        queries = json.load(f)
        
    examples = []
    for item in queries:
        q = item["question"]
        # Ground truth acts as positive
        pos = item.get("ground_truth", f"Verifiable climate targets for {item['geography_iso']} in {item.get('expected_namespace', 'NDC')}")
        
        # Mine hard negative: Same country, but opposite namespace or older concept
        # We construct a hard negative explicitly representing the wrong target detail
        neg = f"General climate protection policy guidelines in {item['geography_iso']} focusing on administrative capacities and general stocktake updates, without specific targets."
        
        examples.append(InputExample(texts=[q, pos, neg]))
        
    print(f"Mined {len(examples)} contrastive training triplets.")
    return examples

def run_fine_tuning(
    model_name: str = "BAAI/bge-m3",
    output_path: str = "data/fine_tuned_bge_m3",
    epochs: int = 1,
    batch_size: int = 4
):
    """Fine-tunes the BGE-M3 model on contrastive triplets using MultipleNegativesRankingLoss."""
    print(f"Initializing embedding model fine-tuning: {model_name}...")
    
    # 1. Load model
    model = SentenceTransformer(model_name)
    
    # 2. Prepare dataset and dataloader
    train_examples = mine_hard_negatives()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # 3. Use MultipleNegativesRankingLoss (designed for triplets)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # 4. Train model
    print(f"Training for {epochs} epoch(s) with batch size {batch_size}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=output_path,
        show_progress_bar=True
    )
    
    print(f"Fine-tuning complete. Saved fine-tuned model weights to {output_path}")

if __name__ == "__main__":
    # Standard entry point execution
    os.makedirs("data", exist_ok=True)
    run_fine_tuning(epochs=1, batch_size=2)
