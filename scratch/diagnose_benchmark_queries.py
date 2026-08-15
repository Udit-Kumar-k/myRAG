from src.backend.retrieval import route_query, CRIMINAL_KEYWORDS, CYBER_KEYWORDS, CONSUMER_KEYWORDS, BANKING_KEYWORDS, CIVIL_KEYWORDS
import re

queries = [
    ("Scenario 1", "It's been four months since I gave my statement at the station and nobody will give me an update on what's happening with my complaint."),
    ("Scenario 2", "A buyer paid for my goods using a cheque that bounced because there wasn't enough balance in their account. Can I take legal action?"),
]

for name, q in queries:
    q_lower = q.lower()
    words = set(re.findall(r"\b\w+\b", q_lower))
    
    def get_score(keywords):
        score = 0
        matched = []
        for kw in keywords:
            if " " in kw:
                if kw in q_lower:
                    score += 1
                    matched.append(kw)
            else:
                if kw in words:
                    score += 1
                    matched.append(kw)
        return score, matched

    c_s, c_m = get_score(CRIMINAL_KEYWORDS)
    cy_s, cy_m = get_score(CYBER_KEYWORDS)
    co_s, co_m = get_score(CONSUMER_KEYWORDS)
    b_s, b_m = get_score(BANKING_KEYWORDS)
    civ_s, civ_m = get_score(CIVIL_KEYWORDS)
    
    print(f"\n{name}: '{q}'")
    print(f"  criminal={c_s} {c_m}")
    print(f"  cyber={cy_s} {cy_m}")
    print(f"  consumer={co_s} {co_m}")
    print(f"  banking={b_s} {b_m}")
    print(f"  general={civ_s} {civ_m}")
