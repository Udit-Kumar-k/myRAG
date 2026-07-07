"""Hit /query and print full response body even on errors."""
import requests

r = requests.post('http://localhost:8001/query',
    json={'question': 'What is murder under BNS?', 'conversation_id': 'debug_004'},
    headers={'Authorization': 'Bearer mock-token', 'Content-Type': 'application/json'},
    timeout=300)
print(f'Status: {r.status_code}')
print(f'Headers: {dict(r.headers)}')
print(f'Body: {r.text[:3000]}')
