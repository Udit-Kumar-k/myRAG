import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini-3.6-flash"

from src.backend.chain import LegalRAGChain

chain = LegalRAGChain()
print(f"Testing LegalRAGChain with Provider: {chain.provider} | Model: {chain.model_name}")

# 1. Test Query Expansion
expanded = chain.expand_query("A customer gave me a bounced cheque because of low balance in their account.")
print(f"\nExpanded Query:\n  {expanded}")

# 2. Test Generation
chunks = [
    {
        "text": "138. Dishonour of cheque for insufficiency, etc., of funds in the account. Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge, in whole or in part, of any debt or other liability, is returned by the bank unpaid...",
        "metadata": {"document_name": "The Negotiable Instruments Act, 1881", "act_name": "The Negotiable Instruments Act, 1881"}
    }
]

ans = chain.run("What can I do if a cheque bounced?", chunks, [])
print(f"\nGenerated Answer:\n  {ans}")
