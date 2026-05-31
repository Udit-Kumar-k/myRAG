import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are the G20 Climate Commitments Analyst, an AI assistant dedicated to answering questions about what the world's 20 largest economies have legally committed to on climate change.

You MUST follow these strict rules at all times:
1. Answer ONLY using the provided retrieved context. Do NOT use any pre-trained general knowledge or assumptions. If the answer cannot be found in the context, you must state: "I do not have sufficient information in the indexed G20 climate policy documents to answer this question."
2. Every specific figure, statistic, percentage, or target year you state MUST be followed by the exact verbatim quote containing that figure in square brackets, e.g. "India committed to reducing emissions intensity by 45% [emissions intensity of its GDP by 45 percent by 2030 from 2005 level] by 2030."
3. If the context contains conflicting commitments/information (e.g. from different years, such as a 2015 NDC vs a 2022 NDC), you MUST state the conflict and differences explicitly.
4. If any cited source was published before 2018, you MUST add a note at the end of your response stating that guidelines/targets may have been updated since the 2018 IPCC Special Report on 1.5°C.
5. If the user's question is outside the scope of G20 climate policy, you must state: "This question is outside the scope of G20 climate policy analysis."

Formatting guidelines:
- Group facts by country or by document namespace (national laws vs NDC commitments) where applicable.
- Make responses concise, structured, and easy to read.

Retrieved Context:
{context}"""

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with metadata for presentation to the LLM."""
    if not chunks:
        return "No relevant context found."
        
    formatted = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        formatted.append(
            f"Source [{i+1}]: {meta.get('document_name', 'Unknown')}\n"
            f"Jurisdiction/Country: {meta.get('geography_iso', 'Unknown')}\n"
            f"Publication Year: {meta.get('pub_year', 'Unknown')}\n"
            f"Namespace: {meta.get('namespace', 'Unknown')}\n"
            f"Source URL: {meta.get('source_url', '')}\n"
            f"Relevance Score: {chunk.get('relevance_score', 0.0):.4f}\n"
            f"Text:\n{chunk.get('text', '')}\n"
            f"----------------------------------------"
        )
    return "\n\n".join(formatted)

class ClimateRAGChain:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemma-4-31b-it"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self._llm = None
        self._chain = None

    def _init_llm(self):
        """Lazy loads the LangChain Gemini integration model."""
        if self._llm is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set. Please provide it or set the environment variable.")
            
            print(f"Initializing model {self.model_name}...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1, # Enforce factual output
                max_tokens=2048
            )
            print("LLM initialized successfully.")
            
    def get_chain(self):
        """Builds and returns the LangChain query processing chain."""
        self._init_llm()
        
        if self._chain is None:
            # We construct a prompt template supporting history and context
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            self._chain = (
                prompt
                | self._llm
                | StrOutputParser()
            )
        return self._chain

    def run(self, question: str, context_chunks: List[Dict[str, Any]], history: List[Any]) -> str:
        """Runs the question-answering chain with context and message history."""
        chain = self.get_chain()
        formatted_context = format_context(context_chunks)
        
        response = chain.invoke({
            "context": formatted_context,
            "history": history,
            "question": question
        })
        return response
