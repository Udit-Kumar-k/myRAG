import os, sys, pickle, re
import numpy as np

sys.path.insert(0, os.path.abspath("."))

def assign_namespace_cleaned(act_name: str) -> str:
    act = act_name.lower()
    if any(x in act for x in ["nyaya sanhita", "nagarik suraksha", "sakshya"]):
        return "criminal"
    if (re.search(r'\binformation technology\b', act) or re.search(r'\bit act\b', act) or "data protection" in act):
        if "institutes of information technology" not in act and "institute of information technology" not in act:
            return "cyber"
    if "consumer" in act:
        return "consumer"
    if re.search(r'\b(rbi|reserve bank|banking|negotiable instruments?|digital payments?)\b', act):
        return "banking"
    return "general"

# Test cleaned namespace function on tricky act names
test_cases = [
    ("The Information Technology Act, 2000", "cyber"),
    ("The Indian Institutes of Information Technology Act, 2014", "general"),
    ("The Indian Institutes of Information Technology (Public-private Partnership) Act, 2017", "general"),
    ("The Sir Dinshaw Manockjee Petit Act, 1893", "general"),
    ("The Maternity Benefit Act, 1961", "general"),
    ("The Arbitration and Conciliation Act, 1996", "general"),
    ("The Reserve Bank of India Act, 1934", "banking"),
    ("The Banking Regulation Act, 1949", "banking"),
    ("The Negotiable Instruments Act, 1881", "banking"),
    ("The Consumer Protection Act, 2019", "consumer"),
    ("Bharatiya Nyaya Sanhita 2023", "criminal"),
]

print("=== Testing assign_namespace_cleaned ===")
all_pass = True
for name, expected in test_cases:
    actual = assign_namespace_cleaned(name)
    status = "PASS" if actual == expected else "FAIL"
    print(f"[{status}] {name} -> {actual} (expected: {expected})")
    if actual != expected:
        all_pass = False

print(f"\nAll tests passed: {all_pass}")
