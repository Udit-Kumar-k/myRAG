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

    # --- CHECK 1: Act-and-Section Pair Verification ---
    pair_pattern = r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?([A-Z][A-Za-z0-9\s,\(\)]+?Act(?:,\s*\d{4})?)'
    
    matches = list(re.finditer(pair_pattern, answer))
    for match in matches:
        full_phrase = match.group(0)
        sec_num = match.group(1)
        act_name = match.group(2).strip()

        verified = False
        act_words = [w.lower() for w in re.findall(r'\b[A-Za-z]+\b', act_name) if len(w) > 3 and w.lower() not in ["the", "act", "with", "from", "that", "code", "sanhita"]]
        
        # Every modifier word in act_name (especially state names or 'amendment') MUST exist in retrieved context
        modifiers = [w for w in act_words if w in ["telangana", "andhra", "pradesh", "amendment", "karnataka", "maharashtra", "tamil", "nadu", "delhi"]]
        
        for chunk_text in retrieved_texts:
            c_text_lower = chunk_text.lower()
            has_sec = bool(re.search(r'\b(?:section|sec|u/s|s\.)?\s*' + re.escape(sec_num) + r'\b', c_text_lower))
            has_act = any(w in c_text_lower or w in combined_act_metadata for w in act_words)
            has_modifiers = all(m in c_text_lower or m in combined_act_metadata for m in modifiers)
            if has_sec and has_act and has_modifiers:
                verified = True
                break
        
        if not verified:
            unverified_claims.append(full_phrase)
            # Strip section number inline; if state/amendment modifier is fabricated, scrub the modifier too
            clean_act = act_name
            for m in modifiers:
                if m not in combined_retrieved_text and m not in combined_act_metadata:
                    # Strip fabricated modifier (e.g. "(Telangana Amendment)")
                    clean_act = re.sub(r'\(?\b' + re.escape(m) + r'\b\s*(?:amendment)?\)?', '', clean_act, flags=re.IGNORECASE).strip()
            
            replacement = f"the {clean_act}"
            answer = answer.replace(full_phrase, replacement)

    # --- CHECK 2: Standalone Section Number Verification ---
    sec_pattern = r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\b'
    for match in re.finditer(sec_pattern, answer, flags=re.IGNORECASE):
        full_sec = match.group(0)
        sec_num = match.group(1)
        
        if not re.search(r'\b(?:section|sec|u/s|s\.)\s*' + re.escape(sec_num) + r'\b', combined_retrieved_text):
            if full_sec not in unverified_claims:
                unverified_claims.append(full_sec)
                answer = re.sub(r'\b' + re.escape(full_sec) + r'\b', 'the applicable provisions', answer)

    # --- CHECK 3: Standalone Act & State Amendment Verification ---
    state_modifiers = ["telangana", "andhra pradesh", "andhra", "karnataka", "maharashtra", "tamil nadu", "delhi"]
    for mod in state_modifiers:
        if mod in answer.lower() and mod not in combined_retrieved_text and mod not in combined_act_metadata:
            if mod not in unverified_claims:
                unverified_claims.append(f"State Amendment: {mod}")
                # Strip the fabricated state name from answer text
                answer = re.sub(r'\b' + re.escape(mod) + r'\b\s*(?:amendment)?', '', answer, flags=re.IGNORECASE)

    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer, unverified_claims


# ======================================================================
# ADVERSARIAL TEST SUITE
# ======================================================================

print("--- TEST 1: Inline Redaction of Unverified Section ---")
test_chunks_1 = [
    {"text": "The Code on Wages, 2019 provides for prohibition of discrimination in wages.", "metadata": {"act_name": "The Code on Wages, 2019"}}
]
ans_1 = "According to Section 15 of the Payment of Wages Act, 1936, an employee may apply to the authority."
cleaned_1, unv_1 = verify_and_clean_citations(ans_1, test_chunks_1)
print("Unverified:", unv_1)
print("Cleaned Body Answer:\n", cleaned_1)

print("\n--- TEST 2: Coincidental Numeral Collision (Adversarial Case) ---")
test_chunks_2 = [
    {"text": "Section 12 of LIC Act: salary not exceeding 15 hundred rupees per mensem or notice of 15 days.", "metadata": {"act_name": "The Life Insurance Corporation Act, 1956"}}
]
ans_2 = "According to Section 15 of the Payment of Wages Act, 1936, the employer must clear dues."
cleaned_2, unv_2 = verify_and_clean_citations(ans_2, test_chunks_2)
print("Unverified:", unv_2)
print("Cleaned Body Answer:\n", cleaned_2)

print("\n--- TEST 3: Act Name & State Amendment Hallucination (Telangana Amendment) ---")
test_chunks_3 = [
    {"text": "Section 15 of the Hindu Succession Act, 1956 states that property of a female Hindu dying intestate shall devolve upon her children.", "metadata": {"act_name": "The Hindu Succession Act, 1956"}}
]
ans_3 = "Under Section 15 of the Hindu Succession (Telangana Amendment) Act, 2017, the property will be divided."
cleaned_3, unv_3 = verify_and_clean_citations(ans_3, test_chunks_3)
print("Unverified:", unv_3)
print("Cleaned Body Answer:\n", cleaned_3)
