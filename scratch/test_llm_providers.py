import os
from dotenv import load_dotenv
load_dotenv()

print("--- Testing Groq ---")
try:
    from langchain_groq import ChatGroq
    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0
    )
    res_groq = groq_llm.invoke("Say 'Groq OK' in 2 words.")
    print("Groq response:", res_groq.content)
except Exception as e:
    print("Groq error:", e)

print("\n--- Testing Gemini ---")
for model_candidate in ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-3.6-flash"]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_llm = ChatGoogleGenerativeAI(
            model=model_candidate,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.0
        )
        res_gemini = gemini_llm.invoke("Say 'Gemini OK' in 2 words.")
        print(f"Gemini ({model_candidate}) response: {res_gemini.content}")
    except Exception as e:
        print(f"Gemini ({model_candidate}) error: {e}")
