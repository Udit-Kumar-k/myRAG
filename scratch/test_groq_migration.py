import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_key = os.environ.get("GROQ_API_KEY")
print(f"Testing Groq API key: {groq_key[:8]}...")

models_to_test = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "qwen-2.5-32b",
    "meta-llama/llama-guard-3-8b"
]

for m in models_to_test:
    try:
        llm = ChatGroq(model=m, api_key=groq_key, temperature=0.0)
        res = llm.invoke("Hi")
        print(f"Model '{m}': SUCCESS -> {res.content.strip()[:60]}")
    except Exception as e:
        print(f"Model '{m}': FAILED -> {e}")
