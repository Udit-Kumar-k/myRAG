import os
import sys
import json
import re
from typing import List, Dict, Any

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.eval import EVAL_QUERIES

def run_real_pipeline(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs queries through the actual RAG pipeline and Chain."""
    print("Running queries through the actual MedRAG pipeline...")
    from src.backend.indexing import MedicalIndexManager
    from src.backend.retrieval import MedicalRAGPipeline
    from src.backend.chain import MedicalRAGChain
    
    index_manager = MedicalIndexManager()
    if not index_manager.load_indexes():
        raise ValueError("Failed to load indexes. Build them first.")
        
    pipeline = MedicalRAGPipeline(index_manager)
    chain = MedicalRAGChain()
    
    dataset_records = []
    for item in queries:
        q = item["question"]
        res = pipeline.query(q)
        
        refused = res["refused"]
        retrieved_chunks = res["retrieved_chunks"]
        contexts = [c["text"] for c in retrieved_chunks]
        
        if refused:
            answer = "Insufficient evidence found in indexed medical textbooks to confidently answer this question. Consider consulting a licensed medical professional or PubMed."
        else:
            try:
                answer = chain.run(q, retrieved_chunks, history=[])
            except Exception as e:
                print(f"Error running chain for query '{q}': {e}")
                answer = "Error generating response."
                
        dataset_records.append({
            "question": q,
            "answer": answer,
            "contexts": contexts
        })
        
    return dataset_records

def run_ragas_evaluation(dataset_records: List[Dict[str, Any]]) -> float:
    """Computes RAGAS faithfulness score using the configured LLM provider."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    
    # We require the configured provider's API key
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set to run evaluation.")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set to run evaluation.")
            
    print(f"Initializing RAGAS evaluator using {provider}...")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        
        # Prepare datasets
        data = {
            "question": [r["question"] for r in dataset_records],
            "answer": [r["answer"] for r in dataset_records],
            "contexts": [r["contexts"] for r in dataset_records]
        }
        dataset = Dataset.from_dict(data)
        
        # Instantiate LLM evaluator
        if provider == "groq":
            from langchain_groq import ChatGroq
            eval_model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
            evaluator_llm = ChatGroq(
                model=eval_model,
                groq_api_key=api_key,
                temperature=0.0
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            eval_model = os.environ.get("LLM_MODEL", "gemini-1.5-flash")
            evaluator_llm = ChatGoogleGenerativeAI(
                model=eval_model,
                google_api_key=api_key,
                temperature=0.0
            )
            
        # Instantiate Embeddings evaluator
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            evaluator_embeds = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_key
            )
        else:
            print("GEMINI_API_KEY not found. Using local HuggingFace embeddings for RAGAS evaluation...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            evaluator_embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
        print("Computing RAGAS faithfulness...")
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=evaluator_llm,
            embeddings=evaluator_embeds
        )
        
        score = results.get("faithfulness", 0.0)
        print(f"RAGAS Faithfulness: {score:.4f}")
        return score
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        raise e

def main():
    print("=== MedRAG CI/CD RAGAS Eval Gate ===")
    
    # Pick 20 queries from the eval queries (first 20 single-country queries)
    test_queries = EVAL_QUERIES[:20]
    
    # Check if index files exist to decide between real and simulation mode
    index_exists = os.path.exists("data/indexes/basic_sciences_chunks.pkl")
    if not index_exists:
        raise ValueError("Failed to load indexes. Build them first using: python -m src.backend.indexing")
        
    try:
        records = run_real_pipeline(test_queries)
        faithfulness_score = run_ragas_evaluation(records)
        
        threshold = 0.75
        if faithfulness_score < threshold:
            print(f"FAIL: RAGAS Faithfulness score {faithfulness_score:.4f} is below threshold {threshold:.2f}")
            sys.exit(1)
        else:
            print(f"PASS: RAGAS Faithfulness score {faithfulness_score:.4f} satisfies threshold {threshold:.2f}")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error executing eval gate: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
