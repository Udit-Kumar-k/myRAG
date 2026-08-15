import os, requests
from dotenv import load_dotenv
load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")

for model_name in ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-flash-lite", "gemma-4-31b-it"]:
    gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
    payload = {"contents": [{"parts": [{"text": "Say 'Gemini is alive and working' in 5 words."}]}]}
    res = requests.post(gen_url, json=payload)
    if res.status_code == 200:
        ans = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"\nModel '{model_name}': SUCCESS -> {ans.strip()}")
    else:
        print(f"\nModel '{model_name}': HTTP {res.status_code} -> {res.text[:150]}")
