import re

def get_sentence_for_match(text, match_start, match_end):
    # Find sentence boundaries (period followed by space, or newline, or start/end of text)
    # Search backwards for sentence start
    pre = text[:match_start]
    post = text[match_end:]
    
    # Preceding sentence boundary
    m_pre = list(re.finditer(r'(?:[\.\n\r;]\s+|\A)', pre))
    start_idx = m_pre[-1].end() if m_pre else 0
    
    # Following sentence boundary
    m_post = re.search(r'(?:[\.\n\r;]|\Z)', post)
    end_idx = match_end + m_post.start() if m_post else len(text)
    
    return text[start_idx:end_idx]

ans2 = "Under the Hindu Succession Act, 1956, Section 15 applies to female intestate succession. In contrast, under the Indian Contract Act, 1872, Section 15 defines coercion."

for m in re.finditer(r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\b', ans2, flags=re.IGNORECASE):
    sent = get_sentence_for_match(ans2, m.start(), m.end())
    act_words = [w for w in re.findall(r'\b[a-zA-Z]+\b', sent.lower()) if len(w) > 3 and w not in ["the", "with", "from", "that", "code", "sanhita", "adhiniyam", "section", "under", "this", "also", "applies", "defines", "contrast"]]
    print(f"Match: '{m.group(0)}' at {m.start()}-{m.end()}")
    print(f"  Sentence: '{sent}'")
    print(f"  Act words: {act_words}")
