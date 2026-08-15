import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_key = os.environ.get("GROQ_API_KEY")
print(f"Testing Groq models with key: {groq_key[:8]}...")

for model in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"]:
    try:
        llm = ChatGroq(model=model, api_key=groq_key, temperature=0.0)
        res = llm.invoke("Convert this to 5 Indian legal keywords: Someone broke into my shop at night and stole goods.")
        print(f"\nModel: {model} -> SUCCESS:\n  {res.content.strip()[:100]}")
    except Exception as e:
        print(f"\nModel: {model} -> FAILED: {e}")
