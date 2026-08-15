import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

from dotenv import load_dotenv
load_dotenv()
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini-3.6-flash"

from src.backend.chain import LegalRAGChain

chain = LegalRAGChain()
print(f"Provider: {chain.provider}, Model: {chain.model_name}")

# Test direct generation with context
q = "What are the legal remedies for a bounced cheque?"
chunks = [
    {
        "text": "138. Dishonour of cheque for insufficiency, etc., of funds in the account. Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge... is returned by the bank unpaid...",
        "metadata": {"document_name": "The Negotiable Instruments Act, 1881", "act_name": "The Negotiable Instruments Act, 1881"}
    }
]

ans = chain.run(q, chunks, [])
print("\n--- GENERATED ANSWER ---")
print(ans)
