import sys, io, requests, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

headers = {'Authorization': 'Bearer guest-token', 'Content-Type': 'application/json'}
q = 'my boss is verbally abusing me and making me work overtime'

print(f"QUERY: {q}")
r = requests.post(
    'http://127.0.0.1:8001/query',
    json={'question': q, 'conversation_id': 'conv_abc1234'},
    headers=headers,
    timeout=60
)
print(f"HTTP Status: {r.status_code}")
if r.ok:
    data = r.json()
    print(f"Refused: {data.get('refused')}")
    print(f"Confidence Score: {data.get('confidence_score')}")
    print(f"Provider / Model: {data.get('provider')} / {data.get('model')}")
    print(f"Sources Count: {len(data.get('sources', []))}")
    for s in data.get('sources', [])[:3]:
        print(f" - {s.get('document_name')} | Score: {s.get('relevance_score')}")
    print(f"\nAnswer:\n{data.get('answer')[:600]}...")
else:
    print(f"Error: {r.text}")
