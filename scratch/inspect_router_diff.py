import os
import re
from src.backend.eval import EVAL_QUERIES
from src.backend.retrieval import route_query, CRIMINAL_KEYWORDS, CYBER_KEYWORDS, CONSUMER_KEYWORDS, BANKING_KEYWORDS

def route_query_new(query: str) -> str:
    q_lower = query.lower()
    words = set(re.findall(r"\b\w+\b", q_lower))

    def get_score(keywords):
        score = 0
        for kw in keywords:
            if " " in kw:
                if kw in q_lower:
                    score += 1
            else:
                if kw in words:
                    score += 1
        return score

    criminal_score = get_score(CRIMINAL_KEYWORDS)
    cyber_score    = get_score(CYBER_KEYWORDS)
    consumer_score = get_score(CONSUMER_KEYWORDS)
    banking_score  = get_score(BANKING_KEYWORDS)

    scores = {
        "criminal": criminal_score,
        "cyber":    cyber_score,
        "consumer": consumer_score,
        "banking":  banking_score,
    }

    max_score = max(scores.values())
    if max_score <= 1: # NEW RULE: fallback to all if weak match
        return "all"

    tied = [ns for ns, s in scores.items() if s == max_score]
    if len(tied) > 1:
        return "all"

    return tied[0]

print(f"{'ID':<15} | {'Query':<50} | {'Old':<10} | {'New':<10} | {'Status':<10}")
print("-" * 105)

for item in EVAL_QUERIES:
    q = item["question"]
    old_ns = route_query(q)
    new_ns = route_query_new(q)
    status = "FLIPPED" if old_ns != new_ns else "SAME"
    # truncate query for printing
    q_trunc = q[:47] + "..." if len(q) > 50 else q
    print(f"{item['id']:<15} | {q_trunc:<50} | {old_ns:<10} | {new_ns:<10} | {status:<10}")
