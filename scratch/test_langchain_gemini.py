import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_key = os.environ.get("GEMINI_API_KEY")

for m in ["gemini-3.6-flash", "gemini-flash-latest"]:
    try:
        llm = ChatGoogleGenerativeAI(model=m, google_api_key=gemini_key, temperature=0.0)
        res = llm.invoke("Hi")
        print(f"LangChain Gemini '{m}': SUCCESS -> {res.content.strip()[:60]}")
    except Exception as e:
        print(f"LangChain Gemini '{m}': FAILED -> {e}")
