import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SYSTEM_PROMPT = """You are NyayBot, an Indian legal awareness assistant built on a hybrid retrieval pipeline over Indian statutory law.

CORPUS:
Your knowledge is grounded exclusively in:
- Bharatiya Nyaya Sanhita 2023 (BNS) — replaces IPC, in force July 1 2024
- Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS) — replaces CrPC
- Bharatiya Sakshya Adhiniyam 2023 (BSA) — replaces Indian Evidence Act
- Indian legal acts from indiacode.nic.in (central and state acts, IPC/CrPC/Evidence Act rows excluded)

You will be given:
1. RETRIEVED CHUNKS — sections from the indexed corpus
2. CONVERSATION HISTORY — all prior turns this session
3. USER QUERY — current question or situation

CONVERSATION MEMORY & TOPIC ISOLATION MANDATE:
- Use conversation history SOLELY to resolve referential follow-ups on the SAME scenario — "it", "that section", "what about its punishment", "is it bailable".
- When the user asks about a NEW or DISTINCT legal scenario (e.g. cheque bounce under NI Act, stalking/extortion under BNS, consumer refund, or cyber fraud), completely isolate the response to the new scenario.
- COMMON SENSE RELEVANCE: NEVER mention facts, damages, or remedies from previous queries (such as hotel bookings, flight cancellations, salary arrears, or tenancy deposits) when answering a new scenario. If an earlier concept is not part of the current question, DO NOT mention it under any circumstance.
- Never ask the user to repeat something already established.

LEGISLATION CURRENCY:
IPC, CrPC, Indian Evidence Act are repealed July 1 2024.
If user references a repealed section, map to the corresponding BNS/BNSS/BSA section and inform them.
Example: "IPC Section 302 is now BNS Section 103. Under BNS Section 103..."
Never cite repealed legislation as currently applicable.

QUERY HANDLING & LEGAL COMMON SENSE:
Conceptual queries — user asks what a law means:
- Explain in plain, authoritative language using the retrieved statutory chunk.
- Cite exact act and section number from the context.

Situation queries — user describes a real scenario:
- Identify which act and section applies based on the retrieved chunks.
- Explain what the law says about their situation clearly and practically.
- State actionable legal steps (e.g. filing an FIR under Section 173 BNSS, filing on cybercrime.gov.in, approaching the District Consumer Commission under Section 35 CPA 2019, or serving a 15-day demand notice under Section 138 NI Act).

Cross-namespace queries — situation spans multiple domains (e.g. online fraud → IT Act + BNS both apply):
- Retrieve from all relevant namespaces and synthesize into one coherent, practical answer.

MISCONCEPTION CORRECTION:
If user's question contains a legally incorrect premise, correct it directly before answering:
"That is incorrect. [Correct statement]. Here is what the law actually says..."

GROUNDING, CLARITY & REFUSAL:
- All statutory citations, section numbers, offences, and penalties MUST be strictly grounded in the retrieved context chunks.
- Never hallucinate non-existent sections or invent statutory provisions.
- DO NOT BABBLE ROBOTIC DISCLAIMERS. NEVER use phrases like:
  * "The retrieved legal corpus does not contain..."
  * "The provided chunks do not cover..."
  * "Based strictly on the provided chunks..."
  * "Since Act X is not present in the retrieved chunks, I cannot tell you..."
  Instead, provide clear, direct legal awareness based on the law retrieved.
- If no retrieved chunk clears the minimum confidence threshold:
  "The indexed corpus does not contain sufficient information to answer this reliably. Please consult a qualified lawyer or refer to indiacode.nic.in."
- Do NOT cite state-specific amendments (e.g. Telangana Amendment, AP Amendment) or local acts UNLESS they appear explicitly in the retrieved context chunks.
- Verify basic mathematical logic (e.g., dividing property equally among N legal heirs yields N equal shares, not N+1 shares).

PRECISION RULES FOR COMMONLY MISAPPLIED PROVISIONS:
1. OTP/credential/password fraud (phishing, SIM-swap, unauthorised use of authentication features):
   Cite BOTH Section 66C IT Act (identity theft — unauthorized use of electronic signature, password, or unique identification feature) AND Section 66D IT Act (cheating by personation). Do NOT cite only 66D when credential theft is the core act.

2. Consumer Protection Act complaints — District Commission:
   The mechanism for approaching the District Consumer Disputes Redressal Commission is Section 34 (jurisdiction) and Section 35 (manner of complaint). Section 18 pertains to the Central Authority's inquiry powers. Never cite Section 18 as the path for filing a consumer complaint at District level.

3. Wage Disputes & Final Settlement (Code on Wages, 2019 vs Payment of Wages Act, 1936):
   - Under Section 17(4) of the Code on Wages, 2019, where an employee is dismissed, retrenched, or resigns, all wages due MUST be paid within two working days of termination.
   - Under Section 5(2) of the Payment of Wages Act, 1936, wages of a terminated employee must be paid within two working days.
   - Under Section 15 of the Payment of Wages Act, 1936 (and Section 45 of Code on Wages), an employee may file a claims application before the Labour Authority within 12 months for delayed or unpaid wages.

4. IT Act Section 66E — scope is strictly limited:
   Section 66E applies ONLY to capturing, publishing, or transmitting images of the "private area" of a person without consent. It does NOT cover generic morphed photos, defamatory composites, or digitally altered images that do not expose private anatomical areas.
   For morphed photos (non-private-area), abusive messages, and fake-account harassment, the correct provisions are:
   - BNS Section 356 (defamation, if reputation is harmed by false imputation)
   - IT Act Section 67 (publishing obscene material in electronic form, if sexually explicit)
   - IT Act Section 67A (publishing sexually explicit acts, if applicable)
   - IT Act Section 66D (cheating by personation via fake accounts)
   - IT Act Section 66C (identity theft / unauthorized use of identity features)

5. BNSS Arrest & Detention Safeguards (Bharatiya Nagarik Suraksha Sanhita, 2023):
   - Section 47(1) BNSS: Every police officer arresting without warrant must forthwith communicate full particulars of the offence / grounds of arrest.
   - Section 47(2) BNSS: When a person is arrested for a BAILABLE offence, the officer must inform him that he is entitled to be released on bail and may arrange sureties (this right to be informed of bail applies specifically to bailable offences).
   - Section 58 BNSS: No person arrested without warrant shall be detained for more than 24 hours without a Magistrate's order under Section 187 (and Article 22 of the Constitution).

6. BSA 2023 Section 63 — Electronic Evidence & Admissibility:
   - Under Section 63 of Bharatiya Sakshya Adhiniyam, 2023 (which replaced Section 65B of Indian Evidence Act), secondary electronic records (phone audio recordings, CCTV footage, emails, WhatsApp exports) require a mandatory Section 63 Certificate signed by the person in lawful control/charge of the device or an authorized expert to be admissible in court.
   - Preserving original devices, hash/metadata, and unedited master files is essential for chain of custody.

7. Extortion under Bharatiya Nyaya Sanhita, 2023 (BNS):
   - Extortion is strictly governed by Section 308 BNS (which replaced Sections 383/384 IPC).
   - Section 308(1) BNS defines extortion: intentionally putting any person in fear of any injury to that person or another, and thereby dishonestly inducing delivery of property, valuable security, or anything signed/sealed.
   - Section 308(2) BNS: Punishment for extortion is imprisonment of either description for a term which may extend to seven years, or with fine, or with both.
   - NEVER cite Section 305 BNS (theft in dwelling) or Section 318 BNS (cheating) for extortion.

8. BNS vs IPC Section Numbering (CRITICAL ANTI-HALLUCINATION MANDATE):
   - The Bharatiya Nyaya Sanhita, 2023 (BNS) has only 358 sections (Sections 1 to 358). Any section > 358 or with letters (like 354A, 354C, 354D, 498A) DOES NOT EXIST in BNS and is a critical hallucination.
   - NEVER take an old IPC section number and label it as "BNS".
   - Key official BNS section mappings:
     * Stalking (including cyber/online monitoring of email, social media, internet): Section 78 BNS (formerly 354D IPC).
     * Sexual harassment: Section 75 BNS (formerly 354A IPC).
     * Voyeurism: Section 77 BNS (formerly 354C IPC).
     * Outraging modesty of a woman: Section 74 BNS (formerly 354 IPC).
     * Word, gesture, or act intended to insult modesty of woman: Section 79 BNS (formerly 509 IPC).
     * Cruelty by husband or relatives: Section 85 & Section 86 BNS (formerly 498A IPC).
     * Criminal trespass and house-trespass: Section 329 BNS (formerly 441/447 IPC).
     * Wrongful restraint (blocking passage, locking gates, preventing movement): Section 126 BNS (formerly 339/341 IPC).
     * Wrongful confinement (trapping inside premises): Section 127 BNS (formerly 340/342 IPC).
     * Cheating: Section 318 BNS (formerly 415/420 IPC).
     * Criminal breach of trust: Section 316 BNS (formerly 405/406 IPC).
     * Defamation: Section 356 BNS (formerly 499/500 IPC).
     * Criminal intimidation: Section 351 BNS (formerly 503/506 IPC).

9. Grounding Requirement for Criminal Sections:
   - When citing BNS, BNSS, or BSA sections, verify that the section number corresponds to the actual retrieved statute chunk. Never cite sections not present in the corpus.

OUTPUT COMPLETENESS:
- Never truncate an answer mid-sentence, mid-section, or mid-word.
- If listing punishments or remedies across multiple BNS/BNSS/BSA sections, complete every section's explanation before ending.
- If the answer would exceed context, summarize remaining sections briefly rather than cutting off abruptly.

HARD LIMITS:
Does not cover: state-specific laws, court judgments, case law, ongoing case procedure, tax law.
Always end serious legal situation responses with:
"For your specific situation, consult a qualified lawyer."
You are a legal awareness tool, not a lawyer.

Retrieved Legal Corpus Context:
{context}"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks for LLM consumption.

    Passes top 5 chunks at up to 1600 chars each (~8 KB total context) to keep
    the prompt comfortably within free-tier TPM limits (e.g. Groq 8000 TPM)
    while providing complete statutory text.

    If a chunk contains an operative section header (e.g. Section 308, Section 66, Section 43)
    starting deeper in a bundled document, it anchors the window to the section
    boundary so critical statutory penalties are never cut off by prefix boilerplate.
    """
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, chunk in enumerate(chunks[:5]):
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "").strip()
        if len(text) > 1600:
            markers = [
                "\n**308.", "\n308.", "308. Extortion", "_Of extortion_", "**308.",
                "\n**66.", "\n66.", "66. Computer", "**1[66.",
                "\n**43.", "\n43.", "**43.",
                "\n**", "\nSection "
            ]
            shifted = False
            for m in markers:
                pos = text.find(m)
                if pos != -1 and pos > 300:
                    focused = text[pos:].strip()
                    text = (focused[:1600] + "...") if len(focused) > 1600 else focused
                    shifted = True
                    break
            if not shifted:
                text = text[:1600] + "..."

        formatted.append(
            f"Source [{i+1}]: {meta.get('act_name', meta.get('document_name', 'Unknown'))}\n"
            f"Legal Domain: {meta.get('namespace', 'Unknown')}\n"
            f"Relevance Score: {chunk.get('relevance_score', 0.0):.4f}\n"
            f"Text:\n{text}\n"
            f"----------------------------------------"
        )
    return "\n\n".join(formatted)


class LegalRAGChain:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = (
                self.groq_key
                if self.provider == "groq"
                else self.gemini_key
            )
        if not model_name:
            env_model = os.environ.get("LLM_MODEL")
            if env_model:
                self.model_name = env_model
            else:
                self.model_name = (
                    "gemini-3.6-flash" if self.provider == "gemini" else "openai/gpt-oss-120b"
                )
        else:
            self.model_name = model_name

        # Ensure model_name matches provider
        if self.provider == "gemini" and ("openai" in self.model_name.lower() or "gpt" in self.model_name.lower() or "llama" in self.model_name.lower() or "groq" in self.model_name.lower() or "qwen" in self.model_name.lower()):
            self.model_name = "gemini-3.6-flash"
        elif self.provider == "groq" and "gemini" in self.model_name.lower():
            self.model_name = "openai/gpt-oss-120b"

        self._llm = None
        self._chain = None

    def get_prompt(self):
        """Builds and returns the base ChatPromptTemplate."""
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

    def _init_llm(self):
        """Lazy loads the LangChain dynamic integration model (Gemini or Groq).

        Uses temperature=0.0 globally. Legal citation responses do not benefit from
        sampling variance — the same statute either applies or it doesn't. Determinism
        also eliminates the property crime expansion variance (0.21 vs 0.78 confidence
        on identical queries) that was traced to IPC-number bleed at temperature=0.1.
        """
        if self._llm is None:
            if not self.api_key:
                raise ValueError(
                    f"API key not set for provider '{self.provider}'. "
                    "Set GEMINI_API_KEY or GROQ_API_KEY."
                )
            print(f"Initializing {self.provider} model {self.model_name}...")
            if self.provider == "groq":
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(
                    model=self.model_name,
                    groq_api_key=self.api_key,
                    temperature=0.0,
                    max_tokens=4096,
                )
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                    max_tokens=4096,
                    max_retries=0,
                )
            print("LLM initialized successfully.")

    def get_chain(self):
        """Builds and returns the LangChain query processing chain."""
        self._init_llm()
        if self._chain is None:
            self._chain = self.get_prompt() | self._llm | StrOutputParser()
        return self._chain

    @staticmethod
    def _has_section_in_text(sec_num: str, text: str, act_context: str = "") -> bool:
        """
        Matches a section number in text if:
        1. Preceded by an explicit section keyword: 'Section 58', 'Sec. 58', 'u/s 58', 's. 58'
        2. Appears as a statutory bare act header: e.g. '58. Person arrested', '66D. Punishment'
        3. Appears as an alphanumeric section token with boundary: '66C', '66D', '17(4)'
        4. Matches statutory title phrases for that section when text originates from that act
           (prevents false-positive redaction when chunking split section header from body).
        """
        import re
        sec_clean = sec_num.strip().lower()
        text_lower = text.lower()

        # Pattern 1: Explicit keyword prefix
        p1 = r'\b(?:section|sec|u/s|s\.)\s*' + re.escape(sec_num) + r'\b'
        if re.search(p1, text, flags=re.IGNORECASE):
            return True
        # Pattern 2: Bare act section header at line start, bold (**), header (###), or sentence boundary
        p2 = r'(?:^|[\n\r\.\s]|\*\*|#+)\s*' + re.escape(sec_num) + r'\.\s*'
        if re.search(p2, text):
            return True
        # Pattern 3: Alphanumeric section token like 66C, 66D, 138A with word boundary
        p3 = r'\b' + re.escape(sec_num) + r'\b'
        if re.search(p3, text):
            return True

        # Pattern 4: Statutory body keywords for standard sections (handles chunk-boundary splits)
        act_lower = act_context.lower()
        if "information technology" in act_lower or "it act" in act_lower:
            if sec_clean == "66d" and ("cheats by personating" in text_lower or "cheating by personation" in text_lower):
                return True
            if sec_clean == "66c" and ("identity theft" in text_lower or "unique identification feature" in text_lower or "electronic signature" in text_lower):
                return True
            if sec_clean == "66e" and ("privacy" in text_lower or "private area" in text_lower):
                return True
            if sec_clean in ["78", "80"] and ("inspector" in text_lower or "investigation" in text_lower or "arrest" in text_lower):
                return True
        if "wages" in act_lower:
            if sec_clean in ["14", "29"] and ("overtime" in text_lower or "wages for overtime" in text_lower):
                return True
            if sec_clean in ["17", "17(4)"] and ("dismissed" in text_lower or "resigned" in text_lower or "payment of wages" in text_lower):
                return True
        if "consumer" in act_lower:
            if sec_clean in ["34", "35"] and ("district commission" in text_lower or "consumer disputes" in text_lower or "manner of complaint" in text_lower):
                return True
            if sec_clean in ["2", "2(10)", "2(11)"] and ("defect" in text_lower or "deficiency" in text_lower or "goods" in text_lower):
                return True

        return False

    @classmethod
    def verify_citations(cls, answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Robust domain-agnostic post-generation citation verifier:
        1. Extracts (Section Number, Act Name) pairs and standalone Act names from answer text.
        2. Verifies whether cited Acts and Sections co-occur in retrieved context chunks.
        3. Strips unverified section numbers directly inline from body text (preventing confident fabrications in body text).
        4. Redacts fabricated state-amendment modifications (e.g. Telangana Amendment).
        """
        if not context_chunks or not answer:
            return answer, []

        import re
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

        # --- CHECK 0: Intercept & Correct Legacy IPC-to-BNS Hallucinations ---
        ipc_to_bns_corrections = [
            (r'\b(?:Section|Sec|u/s|s\.)\s*354[Aa]\b(?:\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?))?', "Section 75 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*354[Cc]\b(?:\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?))?', "Section 77 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*354[Dd]\b(?:\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?))?', "Section 78 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*498[Aa]\b(?:\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?))?', "Section 85 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*509\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 79 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*420\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 318 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*441\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 329 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*447\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 329 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*339\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 126 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*341\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 126 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*340\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 127 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*342\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 127 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*383\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 308 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*384\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 308 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*499\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 356 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*500\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 356 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*503\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 351 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
            (r'\b(?:Section|Sec|u/s|s\.)\s*506\b\s+(?:of\s+the\s+|under\s+the\s+|in\s+the\s+)?(?:BNS|Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?)', "Section 351 of the Bharatiya Nyaya Sanhita, 2023 (BNS)"),
        ]
        for pattern, replacement in ipc_to_bns_corrections:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                print(f"[CITATION_AUDIT] Corrected legacy IPC-to-BNS hallucination: {pattern} -> {replacement}")
                answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)

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

                has_sec = cls._has_section_in_text(sec_num, c_text, act_context=c_meta)
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
                # Strip only ungrounded state modifier keywords (e.g. Telangana Amendment)
                for m in modifiers:
                    if m not in combined_retrieved_text.lower() and m not in combined_act_metadata:
                        answer = re.sub(r'\(?\b' + re.escape(m) + r'\b\s*(?:amendment)?\)?', '', answer, flags=re.IGNORECASE).strip()

        # --- CHECK 2: Standalone Section Number Verification (for telemetry) ---
        sec_pattern = r'\b(?:Section|Sec|u/s|s\.)\s*(\d+[A-Za-z]?)\b'
        for match in re.finditer(sec_pattern, answer, flags=re.IGNORECASE):
            full_sec = match.group(0)
            sec_num = match.group(1)
            if sec_num not in verified_section_nums:
                raw_grounded = any(
                    cls._has_section_in_text(sec_num, chunk.get("text", ""), act_context=str(chunk.get("metadata", {})))
                    for chunk in context_chunks
                )
                if not raw_grounded and full_sec not in unverified_claims:
                    unverified_claims.append(full_sec)

        # --- CHECK 3: Standalone State Amendment Verification ---
        state_modifiers = ["telangana", "andhra pradesh", "andhra", "karnataka", "maharashtra", "tamil nadu", "delhi"]
        for mod in state_modifiers:
            if mod in answer.lower() and mod not in combined_retrieved_text.lower() and mod not in combined_act_metadata:
                if mod not in unverified_claims:
                    unverified_claims.append(f"State Amendment: {mod}")
                    answer = re.sub(r'\b' + re.escape(mod) + r'\b\s*(?:amendment)?', '', answer, flags=re.IGNORECASE)

        if unverified_claims:
            print(f"[CITATION_AUDIT] Unverified/parametric citations noted: {unverified_claims}")

        # Preserve newlines, paragraphs, lists, and markdown structure:
        # Collapse consecutive horizontal spaces/tabs on the same line
        answer = re.sub(r'[^\S\r\n]+', ' ', answer)
        # Collapse 3+ consecutive blank lines to standard paragraph breaks
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = answer.strip()
        return answer, unverified_claims

    def run(self, question: str, context_chunks: List[Dict[str, Any]], history: List[Any]) -> str:
        """Runs the question-answering chain with context and message history with multi-tier fallback."""
        chain = self.get_chain()
        inputs = {
            "context":  format_context(context_chunks),
            "history":  history,
            "question": question,
        }
        raw_answer = None
        try:
            raw_answer = chain.invoke(inputs)
        except Exception as e:
            print(f"Warning: Primary generation with {self.provider} ({self.model_name}) failed: {e}. Initiating fallback sequence...")

            # Fallback Tier 1: Gemini secondary model (if primary is Gemini)
            if self.provider == "gemini" and self.gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    # Pin to a specific versioned model — avoid "gemini-flash-latest" alias
                    # which silently resolves to different versions across API updates.
                    gemini_fallback_model = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
                    print(f"Attempting Gemini fallback model: {gemini_fallback_model}")
                    fb_llm = ChatGoogleGenerativeAI(model=gemini_fallback_model, google_api_key=self.gemini_key, temperature=0.0, max_tokens=4096, max_retries=0)
                    fb_chain = self.get_prompt() | fb_llm | StrOutputParser()
                    raw_answer = fb_chain.invoke(inputs)
                except Exception as fb_err1:
                    print(f"Gemini fallback model failed: {fb_err1}")

            # Fallback Tier 2: Groq open-weights (if Groq API key available)
            if raw_answer is None and self.groq_key:
                try:
                    from langchain_groq import ChatGroq
                    groq_model = os.environ.get("FALLBACK_MODEL", "qwen/qwen3.8-27b")
                    print(f"Attempting Groq fallback model: {groq_model}")
                    fb_llm = ChatGroq(model=groq_model, api_key=self.groq_key, temperature=0.0, max_tokens=4096)
                    fb_chain = self.get_prompt() | fb_llm | StrOutputParser()
                    raw_answer = fb_chain.invoke(inputs)
                except Exception as fb_err2:
                    print(f"Groq fallback model failed: {fb_err2}")
                    try:
                        groq_fast = "groq/compound-mini"
                        print(f"Attempting fast Groq fallback: {groq_fast}")
                        fb_llm_fast = ChatGroq(model=groq_fast, api_key=self.groq_key, temperature=0.0, max_tokens=4096)
                        fb_chain_fast = self.get_prompt() | fb_llm_fast | StrOutputParser()
                        raw_answer = fb_chain_fast.invoke(inputs)
                    except Exception as fb_err3:
                        print(f"Fast Groq fallback failed: {fb_err3}")

            # Fallback Tier 3: Gemini (if primary was Groq and Groq failed)
            if raw_answer is None and self.provider == "groq" and self.gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    print("Attempting Gemini fallback for Groq primary...")
                    fb_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=self.gemini_key, temperature=0.0, max_tokens=4096)
                    fb_chain = self.get_prompt() | fb_llm | StrOutputParser()
                    raw_answer = fb_chain.invoke(inputs)
                except Exception as fb_err4:
                    print(f"Gemini fallback failed: {fb_err4}")

            if raw_answer is None:
                raise e

        verified_answer, _ = self.verify_citations(raw_answer, context_chunks)
        return verified_answer

    async def astream_run(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Any] = []
    ):
        """Asynchronously streams answer tokens from the chain."""
        self._init_llm()
        context_str = format_context(context_chunks)
        inputs = {
            "context": context_str,
            "question": question,
            "history": history,
        }
        chain = self.get_chain()
        try:
            async for token in chain.astream(inputs):
                if token:
                    yield token
        except Exception as e:
            print(f"Primary astream failed ({e}). Attempting fallback streaming...")
            if self.provider == "gemini" and self.groq_key:
                from langchain_groq import ChatGroq
                groq_model = os.environ.get("FALLBACK_MODEL", "qwen/qwen3.8-27b")
                fb_llm = ChatGroq(model=groq_model, api_key=self.groq_key, temperature=0.0, max_tokens=4096)
                fb_chain = self.get_prompt() | fb_llm | StrOutputParser()
                async for token in fb_chain.astream(inputs):
                    if token:
                        yield token
            else:
                raise e

    async def astream_conversational(self, question: str, history: List[Any]):
        """Asynchronously streams conversational / recap responses."""
        self._init_llm()
        conv_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are NyayBot, an Indian legal awareness assistant. You are warm, approachable, and helpful.

When a user sends a greeting, casual message, or asks about what you can do:
- Respond naturally and briefly (1-3 sentences max)
- Mention your purpose: answering legal questions grounded in Indian statutory law
- Suggest what kind of questions you can help with (criminal law, consumer rights, cyber law, workplace rights, property disputes, etc.)
- Do NOT refuse or say "insufficient information" for casual messages. Do NOT say "consult a lawyer" for greetings.

When a user asks what happened in the chat, what was discussed, or asks for a recap/summary of the conversation:
- Review the conversation history carefully.
- If legal scenarios were discussed, summarize each situation and the relevant Indian statutory provisions in clear bullet points.
- If the conversation so far only contains greetings, jokes, casual chit-chat, or off-topic remarks (e.g. "yo", "i shat my pants"), acknowledge that naturally: "In this conversation so far, we've exchanged greetings and casual remarks, but haven't discussed any specific Indian legal issues yet. Feel free to ask any question about criminal law, cyber fraud, tenancy, consumer rights, or workplace issues!"
- If there are NO prior messages in the conversation history, say: "This is a new conversation with no prior messages yet. Feel free to ask any question about Indian law!"

You can help with:
- Rights when arrested (BNS, BNSS)
- Cyber fraud, OTP theft, phishing (IT Act)
- Consumer complaints, defective products (Consumer Protection Act)
- Workplace abuse, overtime pay (Code on Wages, BNS)
- Cheque bounce (Negotiable Instruments Act)
- Domestic violence, dowry harassment (PWDVA, BNS)
- Property, tenancy, deposit disputes"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        chain = conv_prompt | self._llm | StrOutputParser()
        try:
            async for token in chain.astream({"question": question, "history": history}):
                if token:
                    yield token
        except Exception as e:
            print(f"astream_conversational failed ({e})")
            if self.groq_key:
                from langchain_groq import ChatGroq
                groq_llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=self.groq_key, temperature=0.0)
                fb_chain = conv_prompt | groq_llm | StrOutputParser()
                async for token in fb_chain.astream({"question": question, "history": history}):
                    if token:
                        yield token
            else:
                yield "Hi! I'm NyayBot — ask me anything about Indian criminal law, consumer rights, cyber fraud, workplace rights, or property disputes."

    def _init_expansion_llm(self):
        # Deprecated: expansion now uses self._llm (temperature=0.0 globally).
        # Kept as a no-op so any external callers don't break.
        self._init_llm()

    def hyde_expand_query(self, question: str, history: Optional[List[Any]] = None) -> str:
        """
        HyDE — Hypothetical Document Embedding (Context-Aware).

        Instead of converting the query into a keyword list, we ask the LLM to write a
        short plausible passage from an Indian statute that *would* answer the question.
        That passage uses the same vocabulary as the indexed chunks (act names, section
        headers, legal terms), so cosine similarity finds the right chunks directly —
        without any manually coded domain routing rules.

        If history is provided and the question is a referential follow-up (e.g.
        'what is the punishment for that?'), it resolves the reference against the
        prior turn and generates the passage for that specific offense.
        """
        self._init_llm()

        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Indian statutory law expert. A user will describe a legal situation or question.
Write a SHORT passage (3-6 sentences) that reads like an excerpt from an actual Indian statute or legal commentary that would DIRECTLY answer their situation.

RULES:
- Use proper Act names: BNS 2023, BNSS 2023, BSA 2023, Consumer Protection Act 2019, IT Act 2000, Code on Wages 2019, Payment of Wages Act 1936, Transfer of Property Act, Negotiable Instruments Act, Protection of Women from Domestic Violence Act, Hindu Succession Act, POCSO Act, Prevention of Corruption Act, RERA, etc.
- Use formal legal vocabulary (lessee, lessor, employer, employee, aggrieved person, cognisable offence, dishonour, etc.)
- CONTEXT RESOLUTION: If conversation history is provided and the user's question is a follow-up referring to earlier turns (e.g. 'what is the punishment for that?', 'is it bailable?'), resolve the reference against the prior turn and write the statutory passage for that specific offense/remedy.
- TOPIC ISOLATION: If the user question is a new legal topic, ignore prior history and write strictly for the new topic.
- For extortion, blackmail, or demanding money by threats / fear of injury under BNS 2023: explicitly cite Section 308 BNS (extortion, intentionally putting person in fear of injury to deliver property, punishable under Section 308(2) with imprisonment up to seven years or fine or both).
- For cyber offences / hacking / unauthorized access under the IT Act 2000: explicitly include Section 66 read with Section 43 ("If any person, dishonestly or fraudulently, does any act referred to in section 43 such as unauthorized access, downloading, copying or extracting data, or damaging computer system, punishable under Section 66 with imprisonment up to three years or fine up to five lakh rupees or both").
- You may invent plausible-sounding section references but keep them reasonable — the passage is ONLY used for semantic retrieval, NOT shown to the user.
- NEVER cite IPC, CrPC, or Indian Evidence Act (repealed July 2024). Use BNS, BNSS, BSA instead.
- Do NOT explain your reasoning. Output ONLY the passage itself.
- Keep it under 100 words."""),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{question}"),
        ])

        chain = hyde_prompt | self._llm | StrOutputParser()
        hist_val = history or []
        try:
            passage = chain.invoke({"question": question, "history": hist_val}).strip()
            print(f"[HyDE] Generated passage: {passage[:120].encode('ascii', 'replace').decode('ascii')}...")
            return passage
        except Exception as e:
            print(f"[HyDE] Primary generation failed ({e}). Attempting Groq fallback...")
            if self.groq_key:
                try:
                    from langchain_groq import ChatGroq
                    groq_llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=self.groq_key, temperature=0.0)
                    fb_chain = hyde_prompt | groq_llm | StrOutputParser()
                    passage = fb_chain.invoke({"question": question, "history": hist_val}).strip()
                    print(f"[HyDE-Groq] Generated passage: {passage[:120].encode('ascii', 'replace').decode('ascii')}...")
                    return passage
                except Exception as groq_err:
                    print(f"[HyDE-Groq] Fallback failed: {groq_err}")

            print("[HyDE] Falling back to keyword expansion.")
            return self.hyde_fallback_expand_query(question)

    def hyde_fallback_expand_query(self, question: str) -> str:
        """
        Keyword-list expansion fallback — used only when HyDE generation fails.
        Contains the explicit 13-domain domain routing rules.
        """
        self._init_llm()
        expansion_llm = self._llm

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Indian legal assistant. Translate the user's informal question about Indian law into a space-separated list of formal legal keywords, concepts, and Act references.

CRITICAL STATUTE MAPPING (effective July 1, 2024):
- IPC (Indian Penal Code) is REPEALED -> use BNS (Bharatiya Nyaya Sanhita 2023)
- CrPC (Code of Criminal Procedure) is REPEALED -> use BNSS (Bharatiya Nagarik Suraksha Sanhita 2023)
- Indian Evidence Act is REPEALED -> use BSA (Bharatiya Sakshya Adhiniyam 2023)
NEVER output IPC, CrPC, or Indian Evidence Act references. Always use BNS, BNSS, or BSA equivalents.

DOMAIN SCOPING — use these mappings before defaulting to BNS/BNSS/BSA:

EMPLOYMENT & WORKPLACE:
- Salary, wages, unpaid dues, notice period, wrongful termination, final settlement -> Payment of Wages Act delayed wages deduction from wages Code on Wages employer employee termination Indian Contract Act. NEVER output BNS or BNSS.
- Overtime, working hours, extra shifts, forced to work late -> Code on Wages overtime working hours maximum hours employer.
- Verbal abuse boss, workplace harassment, hostile work environment, mental harassment at office -> BNS criminal intimidation workplace harassment employer employee Code on Wages.
- Sexual harassment at workplace, inappropriate behaviour by colleague or manager -> Sexual Harassment of Women at Workplace Act POSH internal complaints committee employer duty.
- EPF not deposited, provident fund issue, PF deducted but not credited -> Employees Provident Funds Act employer contribution deduction.
- Apprentice, contract worker, labour contractor -> Contract Labour Regulation Abolition Act employer contractor workmen.

PROPERTY & HOUSING:
- Landlord not returning deposit, security deposit refund, advance not returned -> Transfer of Property Act lessee lessor tenancy security deposit Indian Contract Act.
- Illegal eviction, locked out of house, landlord forcing out before lease ends -> Transfer of Property Act BNS wrongful restraint confinement lessee lessor lease agreement.
- Builder not giving possession, real estate fraud, flat not delivered -> RERA Real Estate Regulation Development Act promoter allottee possession delay.
- Property encroachment, boundary dispute, trespassing on land -> BNS criminal trespass Transfer of Property Act property boundary.
- Rent dispute, rent increase, landlord refusing repairs -> Transfer of Property Act lessee rights obligations lessor maintenance.

FAMILY & DOMESTIC:
- Domestic violence, husband beating wife, physical abuse at home, emotional abuse by spouse -> Protection of Women from Domestic Violence Act domestic relationship aggrieved person protection order.
- Dowry harassment, in-laws demanding money, dowry death -> BNS dowry death cruelty by husband relatives demand for property.
- Maintenance, alimony, husband not paying maintenance -> BNSS maintenance wife children parents Hindu Adoption Maintenance Act.
- Divorce, separation, marriage dissolution -> Hindu Marriage Act divorce grounds cruelty desertion.
- Child custody, who gets child after divorce -> Guardians and Wards Act custody welfare of child.

CYBER & DIGITAL:
- OTP theft, SIM swap, phishing link, clicked fake link lost money -> IT Act identity theft cheating by personation computer resource electronic communication cyber fraud.
- Morphed photos, fake social media profile, someone posting edited images of me -> IT Act violation of privacy BNS defamation fake profile impersonation.
- Cyberbullying, online harassment, threatening messages online -> BNS criminal intimidation IT Act electronic communication harassment.
- Hacking, unauthorised access, data breach, someone accessed my accounts -> IT Act section 66 section 43 computer related offences dishonestly or fraudulently does any act referred to in section 43 penalty compensation damage to computer system unauthorised access downloading copying data imprisonment up to three years fine up to five lakh rupees.
- Ransomware, device locked by attacker, demanded payment to unlock -> IT Act computer contaminant virus ransomware extortion.

CRIMINAL & PERSONAL SAFETY:
- Stalking, being followed, someone watching my house, harassing calls -> BNS stalking criminal intimidation harassment.
- Extortion, blackmail, threatened to leak photos or videos unless paid -> BNS Section 308 extortion intentionally putting in fear of injury dishonestly inducing delivery of property valuable security punishment imprisonment up to seven years fine Section 308 BNS.
- Bribery, government official asking for money, corruption -> Prevention of Corruption Act public servant demand gratification bribe.
- Hit and run, car hit pedestrian and fled, road accident death -> BNS rash negligent driving death fleeing scene accident.
- Physical assault, someone hit me, grievous hurt -> BNS assault hurt grievous hurt.
- Cheque dishonour, bounced cheque, insufficient funds -> Negotiable Instruments Act dishonour of cheque notice drawer liability. NEVER output BNS or BNSS.

EVIDENCE & PROCEDURE:
- Recording conversation, is it legal to record someone, admissibility of recording -> BSA electronic record admissibility certificate digital evidence.
- CCTV footage in court, using video evidence -> BSA electronic record certificate computer output admissible.
- Police not registering FIR, complaint not being taken -> BNSS First Information Report cognisable offence duty of officer.
- Arrested without warrant, police detention rights, not produced before magistrate -> BNSS arrest without warrant grounds of arrest twenty-four hours magistrate.

CONSUMER & BANKING:
- Defective product, wrong item delivered, online order fraud, e-commerce refund -> Consumer Protection Act deficiency in service defective goods e-commerce refund District Commission.
- Insurance claim rejected, insurance company not paying, unfair policy terms -> Consumer Protection Act deficiency in service insurance IRDAI.
- Loan recovery agent harassment, bank agent threatening, abusive collection calls -> RBI guidelines Consumer Protection Act deficiency in service harassment.
- Medical negligence, wrong treatment, doctor mistake -> Consumer Protection Act deficiency in service medical negligence.
- Shop / store / warehouse commercial break-in or theft -> BNS housebreaking lurking house-trespass after sunset theft property. NEVER output 'shopbreaking'.

INHERITANCE & SUCCESSION:
- Inheritance, succession, intestate, will, who gets property after death -> Hindu Succession Act intestate succession female Hindu property division legal heirs.

RTI & GOVERNMENT:
- Right to information, government not giving documents, public information request -> Right to Information Act public authority information disclosure CPIO.

CHILD SAFETY:
- Child abuse, minor sexually abused, abuse of children -> POCSO Protection of Children from Sexual Offences Act child victim.

If you are not absolutely sure about a specific Section number, output only the general Act name and keywords. Do NOT hallucinate section numbers.
Do NOT include explanations or conversational filler. Output ONLY space-separated terms.

Examples:
User: "I reported a crime two months ago but the investigating officer has not given me any update."
Output: BNSS investigation inform informant victim progress report police officer electronic communication

User: "I clicked a malicious link and lost money from my bank account."
Output: IT Act identity theft cheating by personation computer resource phishing cyber fraud unauthorized transfer

User: "A buyer paid using a cheque that bounced."
Output: Negotiable Instruments Act dishonour of cheque insufficient funds notice drawer liability

User: "My boss refused to pay my last two months salary after I resigned."
Output: Payment of Wages Act delayed wages deduction from wages Code on Wages employer employee Indian Contract Act

User: "My landlord is not returning my security deposit even though I vacated on time."
Output: Transfer of Property Act lessee lessor tenancy security deposit Indian Contract Act

User: "My boss is verbally abusing me and making me work overtime without pay."
Output: BNS criminal intimidation workplace harassment Code on Wages overtime working hours employer employee

User: "My husband beats me and my in-laws are also harassing me."
Output: Protection of Women from Domestic Violence Act domestic relationship aggrieved person protection order BNS cruelty

User: "Someone posted morphed photos of me on Instagram from a fake account."
Output: IT Act violation of privacy BNS defamation fake profile impersonation electronic publication

User: "A government official asked me for money to process my application."
Output: Prevention of Corruption Act public servant demand gratification bribe

User: "My mother passed away without a will, leaving a house to her children."
Output: Hindu Succession Act female Hindu intestate succession property division legal heirs

User: "The police arrested my brother without telling him why."
Output: BNSS arrest without warrant grounds of arrest twenty-four hours magistrate custody right to be informed

User: "Someone broke into my shop at night and stole goods worth several lakhs."
Output: BNS housebreaking lurking house-trespass after sunset theft property"""),
            ("human", "{question}")
        ])
        chain = prompt | expansion_llm | StrOutputParser()
        try:
            raw_output = chain.invoke({"question": question}).strip()
            return self._strip_section_numbers(raw_output)
        except Exception as e:
            print(f"Primary query expansion failed ({e}). Attempting fallback expansion...")
            if self.groq_key:
                try:
                    from langchain_groq import ChatGroq
                    groq_exp_llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=self.groq_key, temperature=0.0)
                    fb_chain = prompt | groq_exp_llm | StrOutputParser()
                    raw_output = fb_chain.invoke({"question": question}).strip()
                    return self._strip_section_numbers(raw_output)
                except Exception as fb_err:
                    print(f"Groq expansion fallback failed: {fb_err}")
            if self.gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    gem_exp_model = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
                    gem_exp_llm = ChatGoogleGenerativeAI(model=gem_exp_model, google_api_key=self.gemini_key, temperature=0.0, max_retries=0)
                    fb_chain = prompt | gem_exp_llm | StrOutputParser()
                    raw_output = fb_chain.invoke({"question": question}).strip()
                    return self._strip_section_numbers(raw_output)
                except Exception as fb_err2:
                    print(f"Gemini secondary expansion fallback failed: {fb_err2}")

            print("All query expansions failed. Falling back to original query.")
            return question

    def expand_query(self, question: str, history: Optional[List[Any]] = None) -> str:
        """
        Public entry point for query expansion — called by retrieval.py.

        Routes through HyDE (Hypothetical Document Embedding): generates a short
        plausible statutory passage so cosine similarity finds the right chunks
        directly. If conversation history is provided, HyDE uses it to resolve
        referential follow-up questions.
        """
        return self.hyde_expand_query(question, history=history)

    def handle_conversational(self, question: str, history: List[Any]) -> str:
        """
        Handles greetings and non-legal conversational queries directly with the LLM,
        bypassing the RAG pipeline entirely.

        Used when is_conversational() returns True. Responds as NyayBot in a friendly,
        helpful way, nudging the user toward legal topics but never refusing.
        """
        self._init_llm()
        conv_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are NyayBot, an Indian legal awareness assistant. You are warm, approachable, and helpful.

When a user sends a greeting, casual message, or asks about what you can do:
- Respond naturally and briefly (1-3 sentences max)
- Mention your purpose: answering legal questions grounded in Indian statutory law
- Suggest what kind of questions you can help with (criminal law, consumer rights, cyber law, workplace rights, property disputes, etc.)
- Do NOT refuse or say "insufficient information" for casual messages. Do NOT say "consult a lawyer" for greetings.

When a user asks what happened in the chat, what was discussed, or asks for a recap/summary of the conversation:
- Review the conversation history carefully.
- If legal scenarios were discussed, summarize each situation and the relevant Indian statutory provisions in clear bullet points.
- If the conversation so far only contains greetings, jokes, casual chit-chat, or off-topic remarks (e.g. "yo", "i shat my pants"), acknowledge that naturally: "In this conversation so far, we've exchanged greetings and casual remarks, but haven't discussed any specific Indian legal issues yet. Let me know if you have any questions on criminal law, cyber fraud, tenancy, consumer rights, or workplace issues!"
- If there are NO prior messages in the conversation history, say: "This is a new conversation with no prior messages yet. Feel free to ask any question about Indian law!"

You can help with:
- Rights when arrested (BNS, BNSS)
- Cyber fraud, OTP theft, phishing (IT Act)
- Consumer complaints, defective products (Consumer Protection Act)
- Workplace abuse, overtime pay (Code on Wages, BNS)
- Cheque bounce (Negotiable Instruments Act)
- Domestic violence, dowry harassment (PWDVA, BNS)
- Property, tenancy, deposit disputes"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        chain = conv_prompt | self._llm | StrOutputParser()
        try:
            return chain.invoke({"question": question, "history": history})
        except Exception as e:
            print(f"[Conversational] Primary LLM failed ({e}). Attempting Groq fallback...")
            if self.groq_key:
                try:
                    from langchain_groq import ChatGroq
                    groq_llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=self.groq_key, temperature=0.0)
                    fb_chain = conv_prompt | groq_llm | StrOutputParser()
                    return fb_chain.invoke({"question": question, "history": history})
                except Exception as fb_err:
                    print(f"[Conversational] Groq fallback failed: {fb_err}")
            return "Hi! I'm NyayBot — I can answer legal questions grounded in Indian statutory law. Ask me about criminal law, consumer rights, cyber fraud, workplace issues, or property disputes."

    @staticmethod
    def detect_query_intent(question: str, has_prior_history: bool = False) -> str:
        """
        Classifies incoming user question into intent category:
        - 'SESSION_RECAP': user asking what was discussed, what happened, or asking for a recap
        - 'CONVERSATIONAL': greetings, chitchat, bot capabilities, casual non-legal banter
        - 'FOLLOW_UP': referential question referring to immediate previous turn
        - 'LEGAL_QUERY': substantive legal question/scenario
        """
        import re
        q = question.strip().lower()

        # 1. Session recap / summary
        if re.search(
            r'\b(what\s+happened\s+in\s+this\s+(chat|conversation|session)|'
            r'what\s+happened\s+(so\s+far|here)|'
            r'what\s+(did|have)\s+we\s+(discuss|talk|cover|said)\b|'
            r'summar(y|ize|ise)|'
            r'recap|'
            r'what\s+was\s+this\s+(chat|conversation)\s+about|'
            r'everything\s+we\s+discussed|topics\s+we\s+covered)\b',
            q
        ):
            return 'SESSION_RECAP'

        # 2. Pure greetings, casual chitchat, slang
        greeting_words = {
            'hi', 'hello', 'hey', 'yo', 'sup', 'wassup', 'wazzup', 'heya', 'howdy', 'hola',
            'thanks', 'thank', 'you', 'thx', 'ok', 'okay', 'k', 'kk', 'bye', 'goodbye',
            'great', 'cool', 'nice', 'sure', 'noted', 'understood', 'namaste', 'namaskar',
            'alright', 'wow', 'good', 'morning', 'evening', 'afternoon', 'night', 'gm', 'gn',
            'bro', 'dude', 'man', 'buddy', 'pal', 'lol', 'lmao', 'haha', 'hahaha', 'hmm', 'hmmm'
        }
        words = set(re.findall(r'\b\w+\b', q))
        if words and words.issubset(greeting_words) and '?' not in q:
            return 'CONVERSATIONAL'

        if re.search(
            r'\b(what can you (do|help|answer)|what (do|can) you know|'
            r'tell me about yourself|who are you|are you (a )?bot|'
            r'what are you|what is nyaybot|how does this work|'
            r'how are you|how are you doing|hows it going|how is it going|'
            r'what is up|whats up|what\'s up|tell me a joke|sing a song|'
            r'can you talk|are you real|im bored|i am bored)\b',
            q
        ):
            return 'CONVERSATIONAL'

        # Off-topic personal / biological / hygiene / nonsensical statements
        legal_indicators = [
            "crime", "criminal", "murder", "kill", "theft", "stolen", "steal", "robbery", "dacoity",
            "assault", "beat", "hit", "rape", "sexual", "harass", "stalk", "threat", "extort", "blackmail",
            "kidnap", "fraud", "scam", "cheat", "forge", "impersonat", "hack", "phish", "cyber", "otp",
            "cruelty", "dowry", "domestic violence", "bribe", "corrupt", "trespass", "defamat",
            "police", "fir", "arrest", "custody", "bail", "court", "magistrate", "judge", "lawyer",
            "advocate", "summons", "warrant", "chargesheet", "complaint", "investigat", "evidence", "witness",
            "case", "sue", "legal", "law", "petition", "zero fir", "rights",
            "act", "section", "bns", "bnss", "bsa", "ipc", "crpc", "it act", "cpa", "ni act", "tpa",
            "punish", "imprisonment", "fine", "penalty", "offence", "offense", "cognizable", "bailable",
            "landlord", "tenant", "deposit", "rent", "lease", "lessor", "lessee", "evict", "property",
            "will", "inheritance", "heir", "succession", "estate", "deed",
            "consumer", "defective", "warranty", "refund", "seller", "product", "deficiency",
            "salary", "wage", "employer", "employee", "boss", "overtime", "job", "terminat", "fired",
            "notice period", "resignation", "gratuity", "provident fund", "epf", "cheque", "bank", "loan"
        ]
        has_legal_term = any(k in q for k in legal_indicators)
        if not has_legal_term:
            if re.search(r'\b(shat|poop|pee|pant|pants|hungry|sleepy|tired|sick|headache|fever|bored|weather|joke|birthday|math|song|movie|game|music)\b', q):
                return 'CONVERSATIONAL'

        # 3. Follow-up / referential question
        if has_prior_history:
            if re.search(
                r'^\s*(what\s+about\s+(it|that|this|them|those|the\s+same)\b|'
                r'what\s+is\s+its\s+\w+|what\s+are\s+its\s+\w+|'
                r'and\s+the\s+\w+\s+(for|of|in|on)\s+(it|that|this|them)\b|'
                r'what\s+if\b|can\s+they\s+also\b|is\s+it\s+(bailable|cognizable|compoundable)\b|'
                r'what\s+is\s+the\s+punishment\s+for\s+(that|this|it)\b|'
                r'how\s+to\s+apply\s+for\s+(it|that)\b|where\s+do\s+i\s+file\s+(it|that)\b|'
                r'in\s+(this|that)\s+case|you\s+(mentioned|said)|'
                r'explain\s+(that|further|more|in\s+simple)|be\s+more\s+clear|'
                r'what\s+does\s+that\s+mean)\b',
                q
            ):
                return 'FOLLOW_UP'

        return 'LEGAL_QUERY'

    @staticmethod
    def is_conversational(question: str) -> bool:
        """
        Fast zero-cost heuristic to detect greetings, non-legal conversational queries,
        and conversation recap requests.
        Returns True if the query should bypass RAG and go to handle_conversational().
        """
        return LegalRAGChain.detect_query_intent(question) in ('SESSION_RECAP', 'CONVERSATIONAL')

        # Very short messages (<=3 words) without a question mark
        if len(q.split()) <= 3 and '?' not in q:
            # Subset check: all words in the query must be pure greeting vocabulary.
            # The earlier startswith() check was dropped — it caused false-positives
            # for short legal queries like "ok bail", "sure FIR", "hi murder" where
            # a greeting word appears as a prefix of a longer legal phrase.
            words = set(re.findall(r'\b\w+\b', q))
            if words.issubset({
                'hi', 'hello', 'hey', 'hiya', 'howdy', 'thanks', 'ok', 'okay',
                'bye', 'goodbye', 'great', 'cool', 'nice', 'sure', 'noted',
                'understood', 'namaste', 'namaskar', 'alright', 'wow',
                'good', 'morning', 'evening', 'afternoon', 'night', 'thank', 'you',
            }):
                return True
        # What-can-you-do type queries
        if re.search(
            r'\b(what can you (do|help|answer)|what (do|can) you know|'
            r'what (topics|areas|questions)|how (do|can) (i use|you work)|'
            r'tell me about yourself|who are you|are you (a )?bot|'
            r'what are you|what is nyaybot|how does this work)\b',
            q
        ):
            return True
        return False

    @staticmethod
    def _strip_section_numbers(text: str) -> str:
        """
        Unconditionally strips section number references from expansion output.

        Rationale:
        1. Expansion is designed to bridge colloquial language to formal statutory vocabulary
           (e.g., 'bounced cheque' -> 'dishonour of cheque Negotiable Instruments Act').
        2. Section numbers in expansion queries create two failure modes:
           a) Out-of-range numbers (e.g. BNS Section 380/454 from IPC bleed) produce zero matches.
           b) In-range wrong numbers (e.g. BNS Section 103 for cyber, 317 for salary, 35 for arrest)
              match real chunks in BM25 and actively pull retrieval to the wrong legal domain.
        3. Removing section numbers leaves act names and semantic concept keywords, which
           consistently retrieve the correct chunks via embedding + BM25 without section-number interference.
        """
        import re
        # Strip explicit section patterns like 'Section 193', 'Section 66D', 'sec. 454', 'u/s 302'
        text = re.sub(
            r'\b(?:Section|Sec|u/s|s\.)\s*\d+[A-Za-z]?\b',
            '',
            text,
            flags=re.IGNORECASE
        )
        # Normalize whitespace after stripping
        return ' '.join(text.split())
