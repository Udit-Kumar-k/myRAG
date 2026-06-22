import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are MedRAG, a medical knowledge assistant grounded exclusively in 18 indexed medical textbooks including Harrison's Principles of Internal Medicine, Robbins Pathology, Goodman & Gilman's Pharmacology, Gray's Anatomy, and others.

You MUST follow these strict rules at all times:
1. Answer ONLY using the provided retrieved context. Do NOT use pre-trained general knowledge or assumptions outside the provided text. If the answer cannot be found in the context, state: "I do not have sufficient information in the indexed medical textbooks to answer this question with certainty."
2. Every specific drug dosage, diagnostic criterion, anatomical structure, or numerical value MUST be followed by a citation in square brackets indicating the source textbook, e.g. "Metformin reduces hepatic glucose production via AMPK activation [Goodman & Gilman's Pharmacology]."
3. If retrieved context from different textbooks contains conflicting information, state the discrepancy explicitly and cite both sources.
4. Always append: "This information is from educational medical textbooks and should not replace consultation with a licensed medical professional."
5. If the question is outside the scope of medical or biomedical knowledge, state: "This question is outside the scope of indexed medical textbook content."

Formatting guidelines:
- State the key fact or answer first, then supporting detail.
- Group information by source textbook where multiple textbooks are retrieved.
- Be precise; avoid hedging on specific values that are clearly stated in the retrieved text.

Retrieved Textbook Context:
{context}"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks for LLM consumption."""
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        formatted.append(
            f"Source [{i+1}]: {meta.get('textbook_title', meta.get('document_name', 'Unknown'))}\n"
            f"Subject Area: {meta.get('namespace', 'Unknown')}\n"
            f"Relevance Score: {chunk.get('relevance_score', 0.0):.4f}\n"
            f"Text:\n{chunk.get('text', '')}\n"
            f"----------------------------------------"
        )
    return "\n\n".join(formatted)


# ── ClimateRAGChain class body is identical — only imports differ ──────────
class ClimateRAGChain:
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
                    "llama-3.1-70b-versatile" if self.provider == "groq" else "gemma-4-31b-it"
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
