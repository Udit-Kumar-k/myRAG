import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if hf_token == "your_huggingface_token_here" or not hf_token:
    hf_token = None

print("Loading dataset in streaming mode...")
dataset = load_dataset(
    "ClimatePolicyRadar/all-document-text-data", 
    split="train", 
    streaming=True,
    token=hf_token
)

print("Fetching first row...")
for row in dataset:
    print("KEYS in row:", list(row.keys()))
    print("SAMPLE ROW:")
    # Print the row but truncate long text to keep output readable
    sample_print = {}
    for k, v in row.items():
        if isinstance(v, str) and len(v) > 200:
            sample_print[k] = v[:200] + "..."
        else:
            sample_print[k] = v
    import pprint
    pprint.pprint(sample_print)
    break
