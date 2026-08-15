import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

CONVERSATION MEMORY:
Use conversation history to resolve pronouns and topic carryover — "it", "that section", "what about its punishment", "the same offence". Never ask the user to repeat something already established.

LEGISLATION CURRENCY:
IPC, CrPC, Indian Evidence Act are repealed July 1 2024.
If user references a repealed section, map to the corresponding BNS/BNSS/BSA section and inform them.
Example: "IPC Section 302 is now BNS Section 103. Under BNS Section 103..."
Never cite repealed legislation as currently applicable.

QUERY HANDLING:
Conceptual queries — user asks what a law means:
- Retrieve relevant chunk
- Explain in plain language
- Cite exact act and section number

Situation queries — user describes a real scenario:
- Identify which act and section applies
- Explain what the law says about their situation
- State what legal remedy exists under that section

Cross-namespace queries — situation spans multiple domains (e.g. online fraud → IT Act + BNS both apply):
- Retrieve from all relevant namespaces
- Synthesize into one coherent answer

MISCONCEPTION CORRECTION:
If user's question contains a legally incorrect premise, correct it directly before answering.
"That is incorrect. [Correct statement]. Here is what the law actually says..."

GROUNDING AND REFUSAL:
- Every answer must come strictly from retrieved chunks
- Always cite: act name + section number as present in the retrieved chunks
- Do NOT cite state-specific amendments (e.g. Telangana Amendment, AP Amendment) or local acts UNLESS they appear explicitly in the retrieved context chunks.
- Verify basic mathematical logic (e.g., dividing property equally among N legal heirs yields N equal shares, not N+1 shares).
- If no chunk clears confidence threshold:
  "The indexed corpus does not contain sufficient information to answer this reliably. Please consult a qualified lawyer or refer to indiacode.nic.in."
- Never answer from training memory alone
- Never hallucinate section numbers

HARD LIMITS:
Does not cover: state-specific laws, court judgments, case law, ongoing case procedure, tax law.
Always end serious legal situation responses with:
"For your specific situation, consult a qualified lawyer."
You are a legal awareness tool, not a lawyer.

Retrieved Legal Corpus Context:
{context}"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks for LLM consumption."""
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        formatted.append(
            f"Source [{i+1}]: {meta.get('act_name', meta.get('document_name', 'Unknown'))}\n"
            f"Legal Domain: {meta.get('namespace', 'Unknown')}\n"
            f"Relevance Score: {chunk.get('relevance_score', 0.0):.4f}\n"
            f"Text:\n{chunk.get('text', '')}\n"
            f"----------------------------------------"
        )
    return "\n\n".join(formatted)


class LegalRAGChain:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = (
                os.environ.get("GROQ_API_KEY")
                if self.provider == "groq"
                else os.environ.get("GEMINI_API_KEY")
            )
        if not model_name:
            env_model = os.environ.get("LLM_MODEL")
            if env_model:
                self.model_name = env_model
            else:
                self.model_name = (
                    "llama-3.3-70b-versatile" if self.provider == "groq" else "gemini-3.6-flash"
                )
        else:
            self.model_name = model_name

        # Ensure model_name matches provider
        if self.provider == "gemini" and ("llama" in self.model_name.lower() or "groq" in self.model_name.lower()):
            self.model_name = "gemini-3.6-flash"
        elif self.provider == "groq" and "gemini" in self.model_name.lower():
            self.model_name = "llama-3.3-70b-versatile"

        self._llm = None
        self._chain = None

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
                    max_tokens=2048,
                )
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                    max_tokens=2048,
                )
            print("LLM initialized successfully.")

    def get_chain(self):
        """Builds and returns the LangChain query processing chain."""
        self._init_llm()
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            self._chain = prompt | self._llm | StrOutputParser()
        return self._chain

    @staticmethod
    def verify_citations(answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
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
            
            if sec_num in verified_sections:
                continue
            
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

        if unverified_claims:
            print(f"WARNING: Citation Verification Guard detected and redacted uncited claim(s): {unverified_claims}")

        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer, unverified_claims

    def run(self, question: str, context_chunks: List[Dict[str, Any]], history: List[Any]) -> str:
        """Runs the question-answering chain with context and message history."""
        chain = self.get_chain()
        raw_answer = chain.invoke({
            "context":  format_context(context_chunks),
            "history":  history,
            "question": question,
        })
        verified_answer, _ = self.verify_citations(raw_answer, context_chunks)
        return verified_answer

    def _init_expansion_llm(self):
        # Deprecated: expansion now uses self._llm (temperature=0.0 globally).
        # Kept as a no-op so any external callers don't break.
        self._init_llm()

    def expand_query(self, question: str) -> str:
        """
        Translates colloquial query into legal keywords for embedding retrieval.

        Uses self._llm at temperature=0.0 (set globally in _init_llm).
        Post-processes output to strip section numbers that cannot exist in the
        indexed corpus (e.g. 'BNS Section 380' — BNS has 358 sections; 380 is an
        IPC number relabeled BNS without updating the number).
        """
        self._init_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Indian legal assistant. Translate the user's informal question about Indian law into a space-separated list of formal legal keywords, concepts, and Act references.

CRITICAL STATUTE MAPPING (effective July 1, 2024):
- IPC (Indian Penal Code) is REPEALED -> use BNS (Bharatiya Nyaya Sanhita 2023)
- CrPC (Code of Criminal Procedure) is REPEALED -> use BNSS (Bharatiya Nagarik Suraksha Sanhita 2023)
- Indian Evidence Act is REPEALED -> use BSA (Bharatiya Sakshya Adhiniyam 2023)
NEVER output IPC, CrPC, or Indian Evidence Act references. Always use BNS, BNSS, or BSA equivalents.

DOMAIN SCOPING — override BNS/BNSS/BSA default for these topics:
- Employment, salary, wages, notice period, employer-employee disputes -> Indian Contract Act, Payment of Wages Act, Industrial Disputes Act. NEVER output BNS or BNSS for these.
- Cheque dishonour, negotiable instrument, bounced cheque -> Negotiable Instruments Act only. NEVER output BNS or BNSS.
- Inheritance, succession, will -> Hindu Succession Act or Indian Succession Act only.
- Cyber fraud, phishing, hacking, online fraud -> IT Act (Information Technology Act 2000). BNS is secondary only.
- Shop / store / warehouse / commercial break-in or theft -> BNS theft dwelling house building housebreaking lurking house-trespass property. NEVER output 'shopbreaking' or 'shop breaking' (not BNS statutory terms).

If you are not absolutely sure about a specific Section number, output only the general Act name and keywords. Do NOT hallucinate section numbers.
Do NOT include any explanations or conversational filler. Output ONLY the space-separated terms.

Examples:
User: "I reported a crime two months ago but the investigating officer has not given me any update on the case."
Output: BNSS investigation inform informant victim ninety days progress report police officer electronic communication

User: "A fraudster pretended to be a bank employee and convinced me to share my OTP, then transferred funds from my account."
Output: cheating by personation IT Act computer resource impersonation bank fraud online unauthorized transfer BNS

User: "A customer paid with a bill that bounced due to lack of balance in their account."
Output: dishonour of cheque Negotiable Instruments Act insufficient funds notice drawer liability

User: "An employer did not pay the agreed salary to a worker who completed the notice period and resigned."
Output: Indian Contract Act breach of contract compensation unpaid salary Payment of Wages Act employee dues notice period

User: "A company refused to honor our signed agreement and pay for services rendered."
Output: breach of contract Indian Contract Act compensation damages

User: "My mother passed away without a written testament, how is her house inherited?"
Output: Hindu Succession Act intestate succession self acquired property female Hindu legal heirs

User: "Someone broke into my house at night and stole electronics and cash."
Output: BNS theft dwelling house building housebreaking lurking house-trespass property"""),
            ("human", "{question}")
        ])
        chain = prompt | self._llm | StrOutputParser()
        try:
            raw_output = chain.invoke({"question": question}).strip()
            return self._strip_section_numbers(raw_output)
        except Exception as e:
            print(f"Query expansion failed: {e}. Falling back to original query.")
            return question

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
