import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Import project modules
from src.backend.indexing import LegalIndexManager
from src.backend.retrieval import LegalRAGPipeline
from src.backend.chain import LegalRAGChain

# -------------------------------------------------------------
# PIPELINE GLOBALS (declared here so lifespan can reference them)
# -------------------------------------------------------------
index_manager = LegalIndexManager()
rag_pipeline = None
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler — replaces deprecated @app.on_event."""
    global rag_pipeline, rag_chain
    success = index_manager.load_indexes()
    if not success:
        print("WARNING: NyayBot indexes could not be loaded on startup.")
    threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.65))
    rag_pipeline = LegalRAGPipeline(index_manager, confidence_threshold=threshold)
    
    # Pre-load the embedding model on startup to prevent slow first-query timeouts.
    # NOTE: The reranker is NOT pre-loaded here — it loads lazily on first query.
    # Loading both BGE-M3 and BGE-Reranker simultaneously on CPU exceeds 8 GB RAM.
    try:
        print("Pre-loading embedding model...")
        index_manager.load_embedding_model()
    except Exception as e:
        print(f"WARNING: Error pre-loading embedding model: {e}")

    rag_chain = LegalRAGChain()
    yield  # application runs here
    # Shutdown logic can go here if needed

# Initialize FastAPI Web Application
app = FastAPI(title="NyayBot API", version="2.0.0", lifespan=lifespan)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# SUPABASE CLIENT INITIALIZATION
# -------------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# Validate that the URL looks like a real Supabase URL before initialising client
_supabase_configured = (
    SUPABASE_URL.startswith("https://") and
    ".supabase.co" in SUPABASE_URL and
    SUPABASE_SERVICE_KEY and
    SUPABASE_SERVICE_KEY != "your_supabase_service_role_key_here"
)

supabase: Optional[Client] = None
if _supabase_configured:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
else:
    print("Supabase not configured. Running in local mock mode only.")

# -------------------------------------------------------------
# DATABASE AND AUTH MANAGER (SUPABASE POSTGRESQL + LOCAL FALLBACK)
# -------------------------------------------------------------

class DatabaseManager:
    """Manages chat history, persisting to Supabase PostgreSQL, falling back to local file if offline."""
    def __init__(self):
        self.local_db_path = "data/local_db.json"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.local_db_path):
            with open(self.local_db_path, "w") as f:
                json.dump({}, f)

    def save_message(self, uid: str, conv_id: str, message: Dict[str, Any]):
        """Saves a message to Supabase PostgreSQL or local DB."""
        # 1. Try Supabase write
        if supabase and os.environ.get("MOCK_AUTH", "true").lower() != "true":
            try:
                # First ensure conversation exists (upsert)
                supabase.table("conversations").upsert({
                    "id": conv_id,
                    "user_id": uid,
                    "title": f"Session {conv_id[:4].upper()}"
                }).execute()
                
                # Insert message
                supabase.table("messages").insert({
                    "conversation_id": conv_id,
                    "user_id": uid,
                    "role": message["role"],
                    "content": message["content"],
                    "sources": message.get("sources", []),
                    "confidence": message.get("confidence_score"),
                    "refused": message.get("refused", False),
                    "namespace_searched": message.get("namespace_searched")
                }).execute()
                return
            except Exception as e:
                print(f"Supabase Postgres save error: {e}. Falling back to local file.")
                
        # 2. Local JSON Fallback (useful for mock development and testing)
        try:
            with open(self.local_db_path, "r") as f:
                data = json.load(f)
            
            user_key = f"user_{uid}"
            if user_key not in data:
                data[user_key] = {}
            if conv_id not in data[user_key]:
                data[user_key][conv_id] = []
                
            data[user_key][conv_id].append({
                "message_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                **message
            })
            
            with open(self.local_db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save message to local DB: {e}")

    def get_history(self, uid: str, conv_id: str) -> List[Dict[str, Any]]:
        """Retrieves conversation history, sorted by created_at/timestamp."""
        # 1. Try Supabase read
        if supabase and os.environ.get("MOCK_AUTH", "true").lower() != "true":
            try:
                res = supabase.table("messages")\
                              .select("*")\
                              .eq("conversation_id", conv_id)\
                              .eq("user_id", uid)\
                              .order("created_at", desc=False)\
                              .execute()
                
                # Standardize database columns to match application field naming
                formatted_history = []
                for row in res.data:
                    formatted_history.append({
                        "question": row.get("content") if row.get("role") == "user" else "",
                        "answer": row.get("content") if row.get("role") == "assistant" else "",
                        "role": row.get("role"),
                        "content": row.get("content"),
                        "sources": row.get("sources", []),
                        "confidence_score": row.get("confidence"),
                        "refused": row.get("refused", False),
                        "namespace_searched": row.get("namespace_searched")
                    })
                return formatted_history
            except Exception as e:
                print(f"Supabase Postgres read error: {e}. Falling back to local file.")

        # 2. Local JSON Fallback
        try:
            with open(self.local_db_path, "r") as f:
                data = json.load(f)
            
            user_key = f"user_{uid}"
            history = data.get(user_key, {}).get(conv_id, [])
            history.sort(key=lambda x: x.get("timestamp", 0))
            return history
        except Exception as e:
            print(f"Failed to read from local DB: {e}")
            return []

# Initialize database manager
db_manager = DatabaseManager()

def authenticate_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Validates token against Supabase Auth.
    If Supabase is offline or MOCK_AUTH=true, returns a standard mock UUID string.
    """
    mock_uuid = "00000000-0000-0000-0000-000000000000"
    
    if not authorization or not authorization.startswith("Bearer "):
        if os.environ.get("MOCK_AUTH", "true").lower() == "true":
            return mock_uuid
        raise HTTPException(status_code=401, detail="Invalid Authorization header format. Must be 'Bearer <token>'.")
        
    token = authorization.split("Bearer ")[1]
    
    if token == "mock-token" or os.environ.get("MOCK_AUTH", "true").lower() == "true":
        return mock_uuid
        
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase authentication client not initialized.")
        
    try:
        # Validate Supabase access token (JWT)
        user_res = supabase.auth.get_user(token)
        return user_res.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired Supabase token: {str(e)}")

# (Pipeline globals and lifespan handler moved above app initialization)

# -------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# -------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    conversation_id: str

class SourceMetadata(BaseModel):
    document_name: str
    legal_domain: str
    pub_year: int
    namespace: str
    source_url: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    confidence_score: float
    refused: bool

# -------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest, uid: str = Depends(authenticate_user)):
    """Runs a question through the RAG pipeline."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Check logs.")

    # 1. Run Retrieval Pipeline
    pipeline_res = rag_pipeline.query(req.question, conversation_id=req.conversation_id)
    
    refused = pipeline_res["refused"]
    confidence = pipeline_res["confidence_score"]
    namespace_searched = pipeline_res["namespace_searched"]
    retrieved_chunks = pipeline_res["retrieved_chunks"]
    
    sources = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        sources.append({
            "document_name": meta["document_name"],
            "legal_domain":  meta["legal_domain"],
            "pub_year":      meta["pub_year"],
            "namespace":     meta["namespace"],
            "source_url":    meta["source_url"],
            "relevance_score": chunk["relevance_score"]
        })
        
    if refused:
        answer = "The indexed corpus does not contain sufficient information to answer this reliably. Please consult a qualified lawyer or refer to indiacode.nic.in."
    else:
        # 2. Fetch past conversation history for LangChain context window
        database_url = os.environ.get("DATABASE_URL")
        is_placeholder = not database_url or "[project-ref]" in database_url or "placeholder" in database_url
        history_msgs = []
        
        # If DATABASE_URL is active and not mock, load from SQLChatMessageHistory exclusively
        if database_url and not is_placeholder and os.environ.get("MOCK_AUTH", "true").lower() != "true":
            try:
                from langchain_community.chat_message_histories import SQLChatMessageHistory
                chat_history = SQLChatMessageHistory(
                    session_id=f"{uid}:{req.conversation_id}",
                    connection_string=database_url
                )
                # Keep last 10 messages to avoid token bloating
                history_msgs = chat_history.messages[-10:]
            except Exception as e:
                print(f"SQLChatMessageHistory connection failed: {e}")
                raise HTTPException(status_code=500, detail="Database history retrieval failed.")
        else:
            # Otherwise use local DB/JSON file exclusively
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                history_records = db_manager.get_history(uid, req.conversation_id)
                for rec in history_records[-10:]:
                    if rec.get("refused", False):
                        continue
                    if rec.get("role") == "user" or rec.get("question"):
                        history_msgs.append(HumanMessage(content=rec.get("content") or rec.get("question")))
                    elif rec.get("role") == "assistant" or rec.get("answer"):
                        history_msgs.append(AIMessage(content=rec.get("content") or rec.get("answer")))
            except Exception as e:
                print(f"Error parsing local history messages: {e}")

        # 3. Generate Answer using LangChain Gemma 4
        try:
            answer = rag_chain.run(req.question, retrieved_chunks, history_msgs)
        except Exception as e:
            print(f"LLM chain execution error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

        # 4. Save exchange to context history store
        if database_url and not is_placeholder and os.environ.get("MOCK_AUTH", "true").lower() != "true":
            try:
                from langchain_community.chat_message_histories import SQLChatMessageHistory
                chat_history = SQLChatMessageHistory(
                    session_id=f"{uid}:{req.conversation_id}",
                    connection_string=database_url
                )
                chat_history.add_user_message(req.question)
                chat_history.add_ai_message(answer)
            except Exception as e:
                print(f"Failed to append to SQLChatMessageHistory: {e}")
                raise HTTPException(status_code=500, detail="Database history save failed.")

    # 5. Save exchange to UI Database tables (messages & conversations)
    # User message
    user_record = {
        "role": "user",
        "content": req.question,
        "sources": [],
        "confidence_score": None,
        "refused": False,
        "namespace_searched": None
    }
    db_manager.save_message(uid, req.conversation_id, user_record)
    
    # Assistant response
    assistant_record = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence_score": confidence,
        "refused": refused,
        "namespace_searched": namespace_searched
    }
    db_manager.save_message(uid, req.conversation_id, assistant_record)

    return {
        "answer": answer,
        "sources": sources,
        "confidence_score": confidence,
        "refused": refused
    }

@app.get("/history/{conversation_id}")
def get_conversation_history(conversation_id: str, uid: str = Depends(authenticate_user)):
    """Retrieves full conversation history for the authenticated user."""
    history = db_manager.get_history(uid, conversation_id)
    return {"history": history}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    loaded_ns = list(index_manager.faiss_indexes.keys())
    # Healthy if at least one namespace is loaded
    indexes_ready = len(loaded_ns) > 0
    return {
        "status": "healthy" if indexes_ready else "uninitialized",
        "indexes_loaded": indexes_ready,
        "loaded_namespaces": loaded_ns,
        "missing_namespaces": [ns for ns in index_manager.namespaces if ns not in index_manager.faiss_indexes],
        "embedding_model_loaded": index_manager.model is not None,
        "supabase_connected": supabase is not None,
        "timestamp": time.time()
    }
