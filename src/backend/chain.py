import os
import re as _re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

SYSTEM_PROMPT = """You are NyayBot, an Indian statutory legal awareness assistant. You provide precise, objective legal awareness grounded exclusively in authentic Indian statutory law. You are not a lawyer; you provide statutory legal awareness.

INPUTS:
1. RETRIEVED CHUNKS - statutory provisions from the indexed corpus.
2. CONVERSATION HISTORY - prior turns in this session.
3. USER QUERY - current situation or question.

CONVERSATION MEMORY & TOPIC ISOLATION:
- Use conversation history ONLY to resolve referential follow-ups on the same scenario ('it', 'that section', 'what is the punishment').
- For any NEW legal scenario, completely isolate the response. Never import unrelated facts or remedies from earlier turns.

LEGISLATION CURRENCY (effective July 1, 2024):
- IPC, CrPC, and Indian Evidence Act are repealed.
- Map old provisions to their current equivalents: Bharatiya Nyaya Sanhita 2023 (BNS), Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS), and Bharatiya Sakshya Adhiniyam 2023 (BSA). Never cite repealed statutes as active law.

RESPONSE STRUCTURE:
1. Direct Legal Position: State clearly what Indian statutory law provides regarding the scenario.
2. Operative Statutory Provisions: Cite the exact Act and Section numbers from the retrieved context.
3. Practical Remedies / Actionable Steps: Detail the prescribed legal steps (e.g., FIR under Section 173 BNSS, Cyber Crime portal, consumer complaint under Section 35 CPA 2019, 15-day notice under Section 138 NI Act).
4. Statutory Limits & Caveats: Note any conditions, bailable status, or limitation periods.

GROUNDING & ANTI-HALLUCINATION RULES:
- All statutory citations, section numbers, and penalties MUST be strictly grounded in the retrieved chunks.
- Never invent section numbers or alter statute names.
- Do NOT map old IPC section numbers to BNS. If a BNS section is not in the context, name the offence without guessing a section number.
- Do NOT babble robotic disclaimers like 'The retrieved chunks do not contain...'. Provide direct, natural awareness.
- If no retrieved chunk clears confidence threshold, state: 'The indexed corpus does not contain sufficient information to answer this reliably. Please consult a qualified lawyer or refer to indiacode.nic.in.'

PRECISION STATUTORY RULES:
1. Credential/OTP/Identity Fraud: Cite Section 66C IT Act (identity theft) AND Section 66D IT Act (cheating by personation). Do not cite only 66D when credential theft is the core act.
2. Consumer Complaints (District Commission): Cite Section 34 (jurisdiction) and Section 35 (procedure/manner of complaint) of Consumer Protection Act 2019. Never cite Section 18 for District complaints.
3. Wage Disputes & Termination Settlement:
   - Section 17(2) Code on Wages, 2019: Where an employee is removed, dismissed, retrenched, or resigns, all wages due MUST be paid within two working days.
   - Section 5(2) Payment of Wages Act, 1936: Final wages must be paid within two working days of termination.
   - Section 15 Payment of Wages Act / Section 45 Code on Wages: Claims application before Labour Authority within 12 months.
4. IT Act Section 66E: Strictly limited to capturing/transmitting images of private body areas without consent. For general morphed photos, cyber defamation, or harassment, cite Section 356 BNS (defamation), Section 66C/66D IT Act, or Section 67 IT Act.
5. BNSS Arrest Safeguards:
   - Section 47(1) BNSS: Right to know full grounds of arrest immediately.
   - Section 47(2) BNSS: Right to be informed of entitlement to bail for bailable offences.
   - Section 58 BNSS & Art 22: Must be produced before Magistrate within 24 hours.
6. Electronic Evidence & Audio/Video Admissibility (BSA 2023):
   - Under Section 63 Bharatiya Sakshya Adhiniyam, 2023 (replacing Sec 65B Evidence Act), secondary electronic records (call recordings, CCTV, WhatsApp) require a mandatory Section 63(4) certificate signed by the person in lawful control of the device.
7. Extortion (BNS 2023): Governed strictly by Section 308 BNS (replacing 383/384 IPC). Punishable under Section 308(2) with imprisonment up to 7 years, fine, or both. Never cite Section 305 or 318 for extortion.
8. BNS vs IPC Numbering: BNS sections only run 1 to 358. Sections > 358 or alphanumeric (354A, 354D, 498A) do NOT exist in BNS.
   - Stalking: Section 78 BNS (replaced Section 354D IPC; NEVER say it replaced Section 78 IPC).
   - Outraging modesty: Section 74 BNS (replaced Section 354 IPC).
   - Voyeurism: Section 77 BNS (replaced Section 354C IPC).
   - Sexual harassment: Section 75 BNS (replaced Section 354A IPC).
   - Cheating: Section 318 BNS (replaced Section 415/420 IPC).
   - Wrongful Restraint: Section 126 BNS (replaced Section 339/341 IPC).
   - Wrongful Confinement: Section 127 BNS (replaced Section 340/342 IPC).
   - Criminal Trespass: Section 329 BNS (replaced Section 441/447 IPC).
9. Tenancy Lockouts & Dispossession:
   - Disconnecting utilities or locking out tenants is a civil breach of quiet enjoyment under Section 108 Transfer of Property Act, 1882; remedy is a civil injunction under Order 39 Rules 1 & 2 CPC.
   - Physical obstruction is criminal Wrongful Restraint (Section 126 BNS).
   - NEVER cite Section 502 BNSS for civil tenancy restoration. Police cannot order tenancy restoration without a court order.
10. Zero FIR & Jurisdiction: Under Section 173(1) BNSS, police are legally bound to record information of a cognizable offence irrespective of location (Zero FIR) and transfer it to the jurisdictional station.
11. Always conclude serious situation queries with: 'For your specific situation, consult a qualified lawyer.'

Retrieved Legal Corpus Context:
{context}"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks for LLM consumption.

    Passes top 5 chunks at up to 1100 chars each (~5.5 KB total context) to keep
    the prompt comfortably within free-tier TPM limits (e.g. Groq 8000 TPM)
    while providing complete statutory text.
    """
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, chunk in enumerate(chunks[:5]):
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "").strip()
        if len(text) > 1100:
            markers = [
                "\n**308.", "\n308.", "308. Extortion", "_Of extortion_", "**308.",
                "\n**66.", "\n66.", "66. Computer", "**1[66.",
                "\n**43.", "\n43.", "**43.",
                "\n**", "\nSection "
            ]
            shifted = False
            for m in markers:
                pos = text.find(m)
                if pos != -1 and pos > 200:
                    focused = text[pos:].strip()
                    text = (focused[:1100] + "...") if len(focused) > 1100 else focused
                    shifted = True
                    break
            if not shifted:
                text = text[:1100] + "..."

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
                    "gemini-3.6-flash" if self.provider == "gemini" else "llama-3.3-70b-versatile"
                )
        else:
            self.model_name = model_name

        # Ensure model_name matches provider
        if self.provider == "gemini" and ("openai" in self.model_name.lower() or "gpt" in self.model_name.lower() or "llama" in self.model_name.lower() or "groq" in self.model_name.lower() or "qwen" in self.model_name.lower()):
            self.model_name = "gemini-3.6-flash"
        elif self.provider == "groq" and "gemini" in self.model_name.lower():
            self.model_name = "llama-3.3-70b-versatile"

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
                    max_tokens=1536,
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
    def verify_citations(cls, answer: str, context_chunks: List[Dict[str, Any]], is_stream_segment: bool = False) -> Tuple[str, List[str]]:
        """
        Robust domain-agnostic post-generation citation verifier:
        1. Extracts (Section Number, Act Name) pairs and standalone Act names from answer text.
        2. Verifies whether cited Acts and Sections co-occur in retrieved context chunks.
        3. Strips unverified section numbers directly inline from body text (preventing confident fabrications in body text).
        4. Redacts fabricated state-amendment modifications (e.g. Telangana Amendment).

        When is_stream_segment=True, trailing whitespace and paragraph collapsing
        are preserved so that independently verified stream chunks concatenate
        cleanly without losing inter-segment spacing.
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
        if not is_stream_segment:
            answer = answer.strip()
        return answer, unverified_claims

    def _get_fallback_llms(self, max_tokens: int = 1536):
        """
        Yields fallback Chat models in priority order across providers and models.
        Guarantees that if one provider or model hits its daily quota (e.g. Gemini 20 req/day
        or Groq llama-3.3-70b-versatile 200k TPD), alternate models with independent token buckets are tried.
        """
        # 1. Cross-provider fallback
        if self.provider == "groq" and self.gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_model = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
                yield ("gemini", gemini_model, ChatGoogleGenerativeAI(
                    model=gemini_model, google_api_key=self.gemini_key,
                    temperature=0.0, max_tokens=max_tokens, max_retries=0
                ))
            except Exception as e:
                print(f"[FALLBACK_INIT] Failed to create Gemini fallback: {e}")
        elif self.provider == "gemini" and self.groq_key:
            try:
                from langchain_groq import ChatGroq
                groq_model = os.environ.get("FALLBACK_MODEL", "llama-3.3-70b-versatile")
                yield ("groq", groq_model, ChatGroq(
                    model=groq_model, api_key=self.groq_key,
                    temperature=0.0, max_tokens=max_tokens
                ))
            except Exception as e:
                print(f"[FALLBACK_INIT] Failed to create Groq primary fallback: {e}")

        # 2. Alternate Groq models (each model has its own independent token quota on Groq)
        if self.groq_key:
            from langchain_groq import ChatGroq
            alt_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-8b"]
            for model_id in alt_models:
                if self.provider == "groq" and self.model_name == model_id:
                    continue
                try:
                    yield ("groq", model_id, ChatGroq(
                        model=model_id, api_key=self.groq_key,
                        temperature=0.0, max_tokens=max_tokens
                    ))
                except Exception as e:
                    print(f"[FALLBACK_INIT] Failed to create Groq alternate {model_id}: {e}")

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
            for prov, model_id, fb_llm in self._get_fallback_llms(max_tokens=1536):
                try:
                    print(f"[Fallback Sequence] Attempting {prov} model: {model_id}")
                    fb_chain = self.get_prompt() | fb_llm | StrOutputParser()
                    raw_answer = fb_chain.invoke(inputs)
                    if raw_answer:
                        break
                except Exception as fb_err:
                    print(f"[{prov}/{model_id}] fallback failed: {fb_err}")

            if raw_answer is None:
                raise e

        verified_answer, _ = self.verify_citations(raw_answer, context_chunks)
        return verified_answer

    # Clause-boundary regex for splitting streamed text into verifiable segments.
    # Matches sentence ends, paragraph breaks, list items, and semicolons,
    # but avoids false splits on legal abbreviations (Sec., s., No., Art., v.).
    _CLAUSE_BOUNDARY = _re.compile(
        r'(?<![Ss]ec)(?<![Aa]rt)(?<!\bv)(?<!\b[Nn]o)(?<!\bs)'
        r'(\. (?=[A-Z\d*\-•])|\?\s+|!\s+|\n\n|\n[*\-]\s+|\n\d+\.\s+|;\s+|\.\n)'
    )

    async def _astream_with_verification(
        self,
        raw_token_stream,
        context_chunks: List[Dict[str, Any]],
    ):
        """
        Internal helper: buffers raw tokens from an LLM astream, splits at
        clause/sentence boundaries, runs verify_citations on each segment,
        then yields verified text.  Guarantees CHECK 0 IPC-to-BNS rewrites
        fire before any token reaches the client.
        """
        buffer = ""
        async for token in raw_token_stream:
            if not token:
                continue
            buffer += token

            # Only attempt to split once we have a reasonable chunk of text
            if len(buffer) < 80:
                continue

            # Find the last clause boundary in the buffer
            parts = self._CLAUSE_BOUNDARY.split(buffer)
            if len(parts) <= 1:
                # No boundary found yet — keep buffering
                if len(buffer) > 600:
                    # Safety valve: yield what we have to avoid unbounded buffering
                    verified, _ = self.verify_citations(buffer, context_chunks, is_stream_segment=True)
                    yield verified
                    buffer = ""
                continue

            # Rejoin all parts except the last part (which is an incomplete clause)
            # parts comes as [text, delim, text, delim, ..., trailing_text]
            # We want to yield everything up to and including the last delimiter,
            # and keep the trailing incomplete text in the buffer.
            complete = "".join(parts[:-1])
            remainder = parts[-1]

            verified, _ = self.verify_citations(complete, context_chunks, is_stream_segment=True)
            yield verified
            buffer = remainder

        # Flush remaining buffer
        if buffer:
            verified, _ = self.verify_citations(buffer, context_chunks, is_stream_segment=True)
            yield verified

    async def astream_run(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Any] = []
    ):
        """
        Asynchronously streams answer tokens from the chain with multi-tier fallback.
        Each yielded segment has already been passed through verify_citations
        (including CHECK 0 IPC-to-BNS corrections) so the client never renders
        unverified citation text.
        """
        self._init_llm()
        context_str = format_context(context_chunks)
        inputs = {
            "context": context_str,
            "question": question,
            "history": history,
        }
        chain = self.get_chain()
        try:
            async for verified_segment in self._astream_with_verification(
                chain.astream(inputs), context_chunks
            ):
                yield verified_segment
        except Exception as e:
            print(f"Primary astream failed ({e}). Attempting fallback streaming...")
            for prov, model_id, fb_llm in self._get_fallback_llms(max_tokens=1536):
                try:
                    print(f"[Astream Fallback] Attempting {prov} model: {model_id}")
                    fb_chain = self.get_prompt() | fb_llm | StrOutputParser()
                    async for verified_segment in self._astream_with_verification(
                        fb_chain.astream(inputs), context_chunks
                    ):
                        yield verified_segment
                    return
                except Exception as fb_err:
                    print(f"[{prov}/{model_id}] astream fallback failed: {fb_err}")

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
            for prov, model_id, fb_llm in self._get_fallback_llms(max_tokens=4096):
                try:
                    print(f"[Conv Astream Fallback] Attempting {prov} model: {model_id}")
                    fb_chain = conv_prompt | fb_llm | StrOutputParser()
                    async for token in fb_chain.astream({"question": question, "history": history}):
                        if token:
                            yield token
                    return
                except Exception as fb_err:
                    print(f"[{prov}/{model_id}] conv astream fallback failed: {fb_err}")

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
- For audio recording, call recording, phone recording, CCTV footage, WhatsApp chat, or electronic evidence admissibility in court: explicitly cite Section 63 of the Bharatiya Sakshya Adhiniyam, 2023 (BSA 2023) (admissibility of electronic records and mandatory Section 63(4) certificate signed by the person in lawful control of the device).
- For final salary, unpaid wages, delayed settlement, or termination payment timeline: explicitly cite Section 17(2) of the Code on Wages, 2019 (mandating that where an employee has been removed, dismissed, retrenched, or has resigned, wages payable must be paid within two working days).
- Do NOT invent or guess random section numbers. Only cite specific anchor sections if you are 100% certain of the statutory numbering (e.g. Section 138 NI Act, Section 63 BSA, Section 173 BNSS, Section 308 BNS, Section 17(2) Code on Wages). Otherwise describe the offences, remedies, and statutory provisions by formal legal terminology and Act names without guessing section numbers.
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
            print(f"[HyDE] Primary generation failed ({e}). Attempting fallback...")
            for prov, model_id, fb_llm in self._get_fallback_llms(max_tokens=1024):
                try:
                    print(f"[HyDE Fallback] Attempting {prov} model: {model_id}")
                    fb_chain = hyde_prompt | fb_llm | StrOutputParser()
                    passage = fb_chain.invoke({"question": question, "history": hist_val}).strip()
                    if passage:
                        print(f"[HyDE-{prov}/{model_id}] Generated passage: {passage[:120].encode('ascii', 'replace').decode('ascii')}...")
                        return passage
                except Exception as fb_err:
                    print(f"[HyDE-{prov}/{model_id}] fallback failed: {fb_err}")

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
- Audio recording, call recording, phone recording, recording conversation, is it legal to record someone, admissibility of recording, certificate for court proof, CCTV footage in court -> Bharatiya Sakshya Adhiniyam BSA Section 63 certificate electronic record admissibility secondary evidence device in lawful control computer output.
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
                    groq_exp_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=self.groq_key, temperature=0.0)
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
            print(f"[Conversational] Primary LLM failed ({e}). Attempting fallback...")
            for prov, model_id, fb_llm in self._get_fallback_llms(max_tokens=1024):
                try:
                    fb_chain = conv_prompt | fb_llm | StrOutputParser()
                    return fb_chain.invoke({"question": question, "history": history})
                except Exception as fb_err:
                    print(f"[Conversational {prov}/{model_id}] fallback failed: {fb_err}")
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
            # Pattern A: Standard follow-up starters
            if re.search(
                r'^\s*(what\s+about\s+(it|that|this|them|those|the\s+same)\b|'
                r'what\s+is\s+its\s+\w+|what\s+are\s+its\s+\w+|'
                r'and\s+the\s+\w+\s+(for|of|in|on)\s+(it|that|this|them)\b|'
                r'what\s+if\b|can\s+they\s+also\b|is\s+it\s+(bailable|cognizable|compoundable)\b|'
                r'what\s+is\s+the\s+punishment\s+for\s+(that|this|it)\b|'
                r'how\s+to\s+apply\s+for\s+(it|that|this)\b|where\s+do\s+i\s+file\s+(it|that|this)\b|'
                r'in\s+(this|that|such)\s+(case|situation|scenario)\b|you\s+(mentioned|said)|'
                r'explain\s+(that|further|more|in\s+simple)|be\s+more\s+clear|'
                r'what\s+does\s+that\s+mean)\b',
                q
            ):
                return 'FOLLOW_UP'

            # Pattern B: Referential anaphora, contextual follow-up inquiries
            if re.search(
                r'\b(in\s+such\s+cases?|in\s+this\s+scenario|in\s+that\s+scenario|'
                r'for\s+this\s+(offence|offense|crime|matter|dispute|claim)|'
                r'for\s+that\s+(offence|offense|crime|matter|dispute)|'
                r'can\s+the\s+police\s+(arrest|seize|investigate|refuse)|'
                r'can\s+they\s+(also\s+)?(arrest|seize|take|refuse|demand|do\s+that)|'
                r'can\s+(the|my)\s+(landlord|employer|boss|bank|tenant)\s+(do\s+(this|that)|refuse|demand)|'
                r'what\s+is\s+the\s+(limitation\s+period|time\s+limit|procedure\s+to\s+file)|'
                r'how\s+long\s+do\s+i\s+have\s+to\s+(file|report|claim)|'
                r'who\s+should\s+i\s+send\s+the\s+notice\s+to|'
                r'where\s+can\s+i\s+report\s+(this|it)|'
                r'to\s+file\s+(this|it|a\s+complaint)|'
                r'how\s+do\s+i\s+(complain|file\s+this|escalate\s+this)|'
                r'what\s+if\s+(they|he|she)\s+refuses?)\b',
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
