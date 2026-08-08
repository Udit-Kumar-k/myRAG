import os
from typing import List, Dict, Any, Optional
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
        if model_name:
            self.model_name = model_name
        else:
            env_model = os.environ.get("LLM_MODEL")
            if env_model:
                self.model_name = env_model
            else:
                self.model_name = (
                    "llama-3.3-70b-versatile" if self.provider == "groq" else "gemini-2.5-flash"
                )
        self._llm = None
        self._chain = None

    def _init_llm(self):
        """Lazy loads the LangChain dynamic integration model (Gemini or Groq)."""
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
                    temperature=0.1,
                    max_tokens=2048,
                )
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.1,
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

    def run(self, question: str, context_chunks: List[Dict[str, Any]], history: List[Any]) -> str:
        """Runs the question-answering chain with context and message history."""
        chain = self.get_chain()
        return chain.invoke({
            "context":  format_context(context_chunks),
            "history":  history,
            "question": question,
        })

    def expand_query(self, question: str) -> str:
        """Translates colloquial query into legal keywords/concepts."""
        self._init_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Indian legal assistant. Translate the user's informal question about Indian law into a space-separated list of formal legal keywords, concepts, and Act references.

CRITICAL STATUTE MAPPING (effective July 1, 2024):
- IPC (Indian Penal Code) is REPEALED → use BNS (Bharatiya Nyaya Sanhita 2023)
- CrPC (Code of Criminal Procedure) is REPEALED → use BNSS (Bharatiya Nagarik Suraksha Sanhita 2023)
- Indian Evidence Act is REPEALED → use BSA (Bharatiya Sakshya Adhiniyam 2023)
NEVER output IPC, CrPC, or Indian Evidence Act references. Always use BNS, BNSS, or BSA equivalents.

If you are not absolutely sure about a specific Section number, output only the general Act name and keywords. Do NOT hallucinate section numbers.
Do NOT include any explanations or conversational filler. Output ONLY the space-separated terms.

Examples:
User: "Someone broke into my shop at night and stole goods."
Output: housebreaking theft BNS Section 305 Section 303 commercial burglary stolen property

User: "The police detained my brother without explaining why or taking him to a judge."
Output: arrest detention BNSS Section 35 grounds of arrest production before magistrate constitutional rights

User: "A customer paid with a bill that bounced due to lack of balance in their account."
Output: dishonour of cheque Negotiable Instruments Act Section 138 insufficient funds notice drawer liability

User: "A company refused to honor our signed agreement and pay for services rendered."
Output: breach of contract Indian Contract Act Section 73 compensation damages non payment

User: "My mother passed away without a written testament, how is her house inherited?"
Output: Hindu Succession Act intestate succession self acquired property legal heirs Class I heirs Section 8 Section 15"""),
            ("human", "{question}")
        ])
        chain = prompt | self._llm | StrOutputParser()
        try:
            return chain.invoke({"question": question}).strip()
        except Exception as e:
            print(f"Query expansion failed: {e}. Falling back to original query.")
            return question
