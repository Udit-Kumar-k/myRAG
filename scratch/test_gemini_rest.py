import os, requests
from dotenv import load_dotenv
load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
print(f"Testing Gemini REST API with key: {gemini_key[:10]}...")

# 1. List models
url_models = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
r = requests.get(url_models)
if r.status_code == 200:
    models = r.json().get("models", [])
    print(f"Found {len(models)} Gemini models available:")
    for m in models:
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            print(f"  - {m.get('name')}")
else:
    print(f"List models failed: {r.status_code} - {r.text}")

# 2. Test generation on standard models
for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]:
    gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
    payload = {"contents": [{"parts": [{"text": "Say 'Gemini working' in 2 words."}]}]}
    res = requests.post(gen_url, json=payload)
    if res.status_code == 200:
        ans = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"\nModel '{model_name}': SUCCESS -> {ans.strip()}")
    else:
        print(f"\nModel '{model_name}': HTTP {res.status_code} -> {res.text[:150]}")
