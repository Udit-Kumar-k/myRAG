import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token == "your_huggingface_token_here" or not hf_token:
    hf_token = None

dataset = load_dataset("ClimatePolicyRadar/all-document-text-data", split="train", streaming=True, token=hf_token)

print("Scanning for G20 countries and languages...")
count = 0
found_g20 = 0
for row in dataset:
    count += 1
    geographies = row.get("document_metadata.geographies") or []
    languages = row.get("document_metadata.languages") or []
    
    g20_matches = [g for g in geographies if g in ["ARG", "AUS", "BRA", "CAN", "CHN", "DEU", "FRA", "GBR", "IDN", "IND", "ITA", "JPN", "KOR", "MEX", "RUS", "SAU", "TUR", "USA", "ZAF", "EU", "EUR", "EUE"]]
    if g20_matches:
        found_g20 += 1
        if found_g20 <= 15:
            print(f"Row {count}: G20 matches {g20_matches}, Languages: {languages}, Doc Title: {row.get('document_metadata.document_title')}")
    
    if count >= 30000:
        break

print(f"Scanned 30000 rows. Found G20 documents: {found_g20}")
