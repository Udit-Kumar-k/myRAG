from src.backend.retrieval import route_query, CRIMINAL_KEYWORDS, CYBER_KEYWORDS, CONSUMER_KEYWORDS, BANKING_KEYWORDS, CIVIL_KEYWORDS
import re

queries = [
    ("Scenario 1", "I filed a police complaint 4 months ago about my stolen vehicle and haven't received any update or copy of the FIR."),
    ("Scenario 2", "A commercial buyer issued a cheque for Rs 5 lakh that bounced due to insufficient funds, and 15 days have passed since my legal notice."),
    ("Scenario 3", "I received a phishing SMS with a link claiming my bank account was blocked, and Rs 50,000 was debited after clicking it."),
    ("Scenario 4", "My employer has withheld my final settlement and 2 months of salary after I resigned with full notice."),
    ("Scenario 5", "My mother passed away without a will, leaving a self-acquired house. How will the property be divided between my father, my brother, and me?"),
    ("Scenario 6", "The police arrested my brother without telling him why and haven't produced him before a judge in over 24 hours."),
    ("Scenario 7", "Someone broke into my shop at night and stole goods worth several lakhs.")
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
    print(f"  -> ROUTED TO: {route_query(q)}")
