import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token == "your_huggingface_token_here" or not hf_token:
    hf_token = None

print("Loading geekyrakshit/indian-legal-acts split=central...")
dataset = load_dataset("geekyrakshit/indian-legal-acts", split="central", token=hf_token)

print(f"Total rows: {len(dataset)}")
print("First row keys:", dataset[0].keys())
print("First row sample:")
for k, v in dataset[0].items():
    print(f"  {k}: {str(v)[:200]}")
