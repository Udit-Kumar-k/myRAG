import os
from dotenv import load_dotenv
load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
print(f"Testing Gemini API key: {gemini_key[:10]}...")

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-pro"
]

for m in models_to_test:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=m, google_api_key=gemini_key, temperature=0.0)
        res = llm.invoke("Say 'Gemini is working' in 3 words.")
        print(f"Model '{m}': SUCCESS -> {res.content.strip()[:60]}")
    except Exception as e:
        print(f"Model '{m}': FAILED -> {e}")
