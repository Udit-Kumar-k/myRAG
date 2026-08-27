import sys, io, requests, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

headers = {'Authorization': 'Bearer guest-token', 'Content-Type': 'application/json'}

queries = [
    'hi',
    'my boss is verbally abusing me and making me work overtime',
    'I received a defective product and the seller is refusing a refund'
]

for q in queries:
    print(f"\n==========================================")
    print(f"QUERY: {q}")
    print(f"==========================================")
    try:
        r = requests.post(
            'http://127.0.0.1:8001/query',
            json={'question': q, 'conversation_id': 'conv_live123'},
            headers=headers,
            timeout=120
        )
        print(f"HTTP Status: {r.status_code}")
        if r.ok:
            data = r.json()
            print(f"Refused: {data.get('refused')}")
            print(f"Confidence Score: {data.get('confidence_score')}")
            print(f"Sources Count: {len(data.get('sources', []))}")
            for s in data.get('sources', [])[:2]:
                print(f" - {s.get('document_name')} | Score: {s.get('relevance_score')}")
            print(f"\nAnswer:\n{data.get('answer')[:400]}...")
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Request Exception: {e}")
