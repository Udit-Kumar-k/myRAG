import re
from typing import List, Dict, Any, Tuple

def _has_section_in_text(sec_num: str, text: str) -> bool:
    """
    Matches a section number in text only if:
    1. Preceded by an explicit section keyword: 'Section 58', 'Sec. 58', 'u/s 58', 's. 58'
    2. OR appears as a statutory bare act header: line start/bold/header followed by period & title,
       e.g. '58. Person arrested', '**331. House-breaking', '15. General rules', '331. (1)'
    Explicitly avoids matching incidental digits like '2 days', '15 hundred rupees', or sub-clause '(2)'.
    """
    # Pattern 1: Explicit keyword prefix
    p1 = r'\b(?:section|sec|u/s|s\.)\s*' + re.escape(sec_num) + r'\b'
    if re.search(p1, text, flags=re.IGNORECASE):
        return True
    # Pattern 2: Bare act section header at line start, bold (**), header (###), or sentence boundary
    # Must be followed by a period and space then capital letter, bracket, bold, or quote
    p2 = r'(?:^|[\n\r]|\*\*|#+)\s*' + re.escape(sec_num) + r'\.\s+(?:[A-Z\(\"]|\*\*)'
    if re.search(p2, text):
        return True
    return False

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

    combined_retrieved_text = " ".join(retrieved_texts)
    combined_act_metadata = " ".join(retrieved_acts)

    unverified_claims = []
    verified_act_sections = set()  # set of (act_keyword, sec_num)
    verified_section_nums = set()

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

        for chunk in context_chunks:
            c_text = chunk.get("text", "")
            c_meta_doc = str(chunk.get("metadata", {}).get("document_name", "")).lower()
            c_meta_act = str(chunk.get("metadata", {}).get("act_name", "")).lower()
            c_meta = f"{c_meta_doc} {c_meta_act}"

            has_sec = _has_section_in_text(sec_num, c_text)
            has_act = any(w in c_text.lower() or w in c_meta for w in act_words) if act_words else True
            has_modifiers = all(m in c_text.lower() or m in c_meta for m in modifiers)

            if has_sec and has_act and has_modifiers:
                verified = True
                break

        if verified:
            for w in act_words:
                verified_act_sections.add((w, sec_num))
            verified_section_nums.add(sec_num)
        else:
            unverified_claims.append(full_phrase)
            clean_act = act_name
            for m in modifiers:
                if m not in combined_retrieved_text.lower() and m not in combined_act_metadata:
                    clean_act = re.sub(r'\(?\b' + re.escape(m) + r'\b\s*(?:amendment)?\)?', '', clean_act, flags=re.IGNORECASE).strip()

            replacement = f"the {clean_act}"
            answer = answer.replace(full_phrase, replacement)

    # --- CHECK 2: Standalone Section Number Verification ---
    sec_pattern = r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\b'
    for match in re.finditer(sec_pattern, answer, flags=re.IGNORECASE):
        full_sec = match.group(0)
        sec_num = match.group(1)

        # If already verified in Check 1 as a valid grounded act+section pair, skip
        if sec_num in verified_section_nums:
            continue

        # Check against context chunks using robust section header / keyword matcher
        is_grounded = any(_has_section_in_text(sec_num, c_text) for c_text in retrieved_texts)

        if not is_grounded:
            if full_sec not in unverified_claims:
                unverified_claims.append(full_sec)
                answer = re.sub(r'\b' + re.escape(full_sec) + r'\b', 'the applicable provisions', answer)

    # --- CHECK 3: Standalone State Amendment Verification ---
    state_modifiers = ["telangana", "andhra pradesh", "andhra", "karnataka", "maharashtra", "tamil nadu", "delhi"]
    for mod in state_modifiers:
        if mod in answer.lower() and mod not in combined_retrieved_text.lower() and mod not in combined_act_metadata:
            if mod not in unverified_claims:
                unverified_claims.append(f"State Amendment: {mod}")
                answer = re.sub(r'\b' + re.escape(mod) + r'\b\s*(?:amendment)?', '', answer, flags=re.IGNORECASE)

    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer, unverified_claims

if __name__ == "__main__":
    print("=== ADVERSARIAL VERIFIER TEST SUITE V2 ===")

    # Test 1: Grounded Bare Act Header
    test_chunks_1 = [
        {"text": "Person arrested not to be detained more than twenty-four hours.\n58. No police officer shall detain in custody...", "metadata": {"document_name": "Bharatiya Nagarik Suraksha Sanhita 2023"}}
    ]
    ans_1 = "Under Section 58 of the Bharatiya Nagarik Suraksha Sanhita, 2023, detention cannot exceed 24 hours."
    cleaned_1, unv_1 = verify_and_clean_citations(ans_1, test_chunks_1)
    print(f"Test 1 (Grounded Bare Act): unverified={unv_1} -> {cleaned_1}")
    assert unv_1 == [], f"Test 1 failed: {unv_1}"

    # Test 2: Grounded Markdown Bold Header
    test_chunks_2 = [
        {"text": "**331. House-breaking after sunset and before sunrise.** Whoever commits lurking house-trespass...", "metadata": {"document_name": "Bharatiya Nyaya Sanhita 2023"}}
    ]
    ans_2 = "According to Section 331 of the Bharatiya Nyaya Sanhita 2023, house breaking is punishable."
    cleaned_2, unv_2 = verify_and_clean_citations(ans_2, test_chunks_2)
    print(f"Test 2 (Grounded Bold Header): unverified={unv_2} -> {cleaned_2}")
    assert unv_2 == [], f"Test 2 failed: {unv_2}"

    # Test 3: Incidental Digit Collision (LIC Act mentions '15 hundred rupees' and '15 days', but NOT Section 15 of Payment of Wages Act)
    test_chunks_3 = [
        {"text": "Section 12 of LIC Act: salary not exceeding 15 hundred rupees per mensem or notice of 15 days.", "metadata": {"document_name": "The Life Insurance Corporation Act, 1956"}}
    ]
    ans_3 = "According to Section 15 of the Payment of Wages Act, 1936, the employee must apply."
    cleaned_3, unv_3 = verify_and_clean_citations(ans_3, test_chunks_3)
    print(f"Test 3 (Incidental Digit Collision): unverified={unv_3} -> {cleaned_3}")
    assert "Section 15 of the Payment of Wages Act, 1936" in unv_3, f"Test 3 failed: {unv_3}"

    # Test 4: Sub-clause numbering (2) in text vs Section 2
    test_chunks_4 = [
        {"text": "Section 47. (1) Every person arrested shall be informed. (2) Where a person is arrested for bailable offence...", "metadata": {"document_name": "Bharatiya Nagarik Suraksha Sanhita 2023"}}
    ]
    ans_4 = "Under Section 2 of the Bharatiya Nagarik Suraksha Sanhita 2023, bail is granted."
    cleaned_4, unv_4 = verify_and_clean_citations(ans_4, test_chunks_4)
    print(f"Test 4 (Subclause (2) vs Section 2): unverified={unv_4} -> {cleaned_4}")
    assert any("Section 2" in u for u in unv_4), f"Test 4 failed: {unv_4}"

    # Test 5: Fabricated State Amendment
    test_chunks_5 = [
        {"text": "15. General rules of succession in the case of female Hindus.\n(1) The property of a female Hindu...", "metadata": {"document_name": "The Hindu Succession Act, 1956"}}
    ]
    ans_5 = "Under Section 15 of the Hindu Succession (Telangana Amendment) Act, 2017, property divides equally."
    cleaned_5, unv_5 = verify_and_clean_citations(ans_5, test_chunks_5)
    print(f"Test 5 (Fabricated State Amendment): unverified={unv_5} -> {cleaned_5}")
    assert len(unv_5) > 0 and any("Telangana" in u for u in unv_5), f"Test 5 failed: {unv_5}"

    # Test 6: Standalone Hallucinated Section (Section 66D when only Section 66 is retrieved)
    test_chunks_6 = [
        {"text": "Section 66. Computer related offences. If any person, dishonestly or fraudulently...", "metadata": {"document_name": "The Information Technology Act, 2000"}}
    ]
    ans_6 = "The offender is liable under Section 66 for computer offences, as well as Section 66D for cheating by personation."
    cleaned_6, unv_6 = verify_and_clean_citations(ans_6, test_chunks_6)
    print(f"Test 6 (Standalone Hallucinated Section 66D): unverified={unv_6} -> {cleaned_6}")
    assert "Section 66D" in unv_6, f"Test 6 failed: {unv_6}"
    assert "Section 66 for" in cleaned_6, f"Test 6 grounded section stripped: {cleaned_6}"

    print("\nALL 6 ADVERSARIAL TESTS PASSED!")

