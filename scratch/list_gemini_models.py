import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

gemini_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)

print("Listing available models for Gemini API key:")
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name} (display: {m.display_name})")
except Exception as e:
    print(f"Error listing models: {e}")
