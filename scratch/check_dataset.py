import os
from datasets import load_dataset
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token == "your_huggingface_token_here" or not hf_token:
    hf_token = None

print("Loading MedRAG/textbooks dataset split=train...")
dataset = load_dataset("MedRAG/textbooks", split="train", token=hf_token)

print(f"Total rows: {len(dataset)}")
print("Scanning unique textbook titles and chunk counts:")

counts = Counter(dataset["title"])
for title, count in counts.items():
    print(f"  - {title}: {count} chunks")
