import re
from typing import List, Dict, Any, Tuple

def verify_and_clean_citations(answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Robust post-generation citation verifier:
    1. Extracts (Section Number, Act Name) pairs and standalone Act names from answer text.
    2. Verifies whether cited Acts and Sections co-occur in retrieved context chunks.
    3. Strips unverified section numbers directly inline from body text.
    4. Redacts fabricated Act names and state-amendment modifications (e.g. Telangana Amendment Act).
    """
    if not context_chunks or not answer:
        return answer, []

    retrieved_texts = [c.get("text", "") for c in context_chunks]
    retrieved_acts = []
    for c in context_chunks:
        meta = c.get("metadata", {})
        if "act_name" in meta:
            retrieved_acts.append(str(meta["act_name"]).lower())
        if "document_name" in meta:
            retrieved_acts.append(str(meta["document_name"]).lower())
    
    combined_retrieved_text = " ".join(retrieved_texts).lower()
    combined_act_metadata = " ".join(retrieved_acts)

    unverified_claims = []
    verified_sections = set()

    # --- CHECK 1: Act-and-Section Pair Verification ---
    pair_pattern = r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?([A-Z][A-Za-z0-9\s,\(\)]+?(?:Act|Sanhita|Adhiniyam|Code)(?:,\s*\d{4})?)'
    
    matches = list(re.finditer(pair_pattern, answer))
    for match in matches:
        full_phrase = match.group(0)
        sec_num = match.group(1)
        act_name = match.group(2).strip()

        verified = False
        act_words = [w.lower() for w in re.findall(r'\b[A-Za-z]+\b', act_name) if len(w) > 3 and w.lower() not in ["the", "act", "with", "from", "that", "code", "sanhita", "adhiniyam"]]
        modifiers = [w for w in act_words if w in ["telangana", "andhra", "pradesh", "amendment", "karnataka", "maharashtra", "tamil", "nadu", "delhi"]]
        
        for chunk_text in retrieved_texts:
            c_text_lower = chunk_text.lower()
            has_sec = bool(re.search(r'\b(?:section|sec|u/s|s\.)?\s*' + re.escape(sec_num) + r'\b', c_text_lower))
            has_act = any(w in c_text_lower or w in combined_act_metadata for w in act_words) if act_words else True
            has_modifiers = all(m in c_text_lower or m in combined_act_metadata for m in modifiers)
            if has_sec and has_act and has_modifiers:
                verified = True
                break
        
        if verified:
            verified_sections.add(sec_num)
        else:
            unverified_claims.append(full_phrase)
            clean_act = act_name
            for m in modifiers:
                if m not in combined_retrieved_text and m not in combined_act_metadata:
                    clean_act = re.sub(r'\(?\b' + re.escape(m) + r'\b\s*(?:amendment)?\)?', '', clean_act, flags=re.IGNORECASE).strip()
            
            replacement = f"the {clean_act}"
            answer = answer.replace(full_phrase, replacement)

    # --- CHECK 2: Standalone Section Number Verification ---
    sec_pattern = r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\b'
    for match in re.finditer(sec_pattern, answer, flags=re.IGNORECASE):
        full_sec = match.group(0)
        sec_num = match.group(1)
        
        # If already verified in Check 1, skip
        if sec_num in verified_sections:
            continue
        
        # Check against bare act text where section number appears with optional prefix
        if not re.search(r'\b(?:section|sec|u/s|s\.)?\s*' + re.escape(sec_num) + r'\b', combined_retrieved_text):
            if full_sec not in unverified_claims:
                unverified_claims.append(full_sec)
                answer = re.sub(r'\b' + re.escape(full_sec) + r'\b', 'the applicable provisions', answer)

    # --- CHECK 3: Standalone State Amendment Verification ---
    state_modifiers = ["telangana", "andhra pradesh", "andhra", "karnataka", "maharashtra", "tamil nadu", "delhi"]
    for mod in state_modifiers:
        if mod in answer.lower() and mod not in combined_retrieved_text and mod not in combined_act_metadata:
            if mod not in unverified_claims:
                unverified_claims.append(f"State Amendment: {mod}")
                answer = re.sub(r'\b' + re.escape(mod) + r'\b\s*(?:amendment)?', '', answer, flags=re.IGNORECASE)

    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer, unverified_claims


# ======================================================================
# ADVERSARIAL TEST SUITE
# ======================================================================

print("--- TEST 1: Grounded Section with Bare Act Header (Should NOT be stripped) ---")
test_chunks_grounded = [
    {"text": "**58. Person arrested not to be detained more than twenty-four hours.** No police officer shall detain...", "metadata": {"act_name": "The Bharatiya Nagarik Suraksha Sanhita, 2023"}}
]
ans_grounded = "According to Section 58 of the Bharatiya Nagarik Suraksha Sanhita, 2023, a person arrested cannot be detained..."
cleaned_g, unv_g = verify_and_clean_citations(ans_grounded, test_chunks_grounded)
print("Unverified:", unv_g)
print("Output:\n", cleaned_g)

print("\n--- TEST 2: Grounded Section with Sanhita / Act name variants (Section 331 BNS) ---")
test_chunks_bns = [
    {"text": "**331. House-breaking after sunset and before sunrise.** Whoever commits lurking house-trespass...", "metadata": {"act_name": "Bharatiya Nyaya Sanhita 2023"}}
]
ans_bns = "Under Section 331 of the Bharatiya Nyaya Sanhita, 2023, house breaking is punishable..."
cleaned_bns, unv_bns = verify_and_clean_citations(ans_bns, test_chunks_bns)
print("Unverified:", unv_bns)
print("Output:\n", cleaned_bns)

print("\n--- TEST 3: Unverified Section (Payment of Wages Sec 15 absent from context) ---")
test_chunks_salary = [
    {"text": "The Code on Wages, 2019 provides for prohibition of discrimination in wages.", "metadata": {"act_name": "The Code on Wages, 2019"}}
]
ans_salary = "According to Section 15 of the Payment of Wages Act, 1936, an employee may apply to the authority."
cleaned_sal, unv_sal = verify_and_clean_citations(ans_salary, test_chunks_salary)
print("Unverified:", unv_sal)
print("Cleaned Body Output:\n", cleaned_sal)

print("\n--- TEST 4: Fabricated State Amendment (Telangana Amendment) ---")
test_chunks_succ = [
    {"text": "**15. General rules of succession in the case of female Hindus.** (1) The property of a female Hindu...", "metadata": {"act_name": "The Hindu Succession Act, 1956"}}
]
ans_succ = "Under Section 15 of the Hindu Succession (Telangana Amendment) Act, 2017, the property will be divided."
cleaned_succ, unv_succ = verify_and_clean_citations(ans_succ, test_chunks_succ)
print("Unverified:", unv_succ)
print("Cleaned Body Output:\n", cleaned_succ)
