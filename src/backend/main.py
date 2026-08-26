import os
import traceback
import json
import time
import uuid
import shutil
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple, Literal
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
        # Indexes not on disk — attempt auto-build from corpus.
        # In production (HF Spaces), indexes are baked into the image layer so
        # this path should not be hit. It fires on first local run before
        # `python -m src.backend.indexing` has been executed.
        print("No indexes found on disk — attempting auto-build from corpus (this takes 15–60 min on CPU)...")
        try:
            from src.backend.ingestion import process_corpus
            hf_token = os.environ.get("HF_TOKEN")
            chunks = process_corpus(hf_token=hf_token if hf_token != "your_huggingface_token_here" else None)
            if chunks:
                print(f"Auto-build: {len(chunks)} chunks generated. Building indexes...")
                index_manager.build_indexes(chunks, batch_size=16)
                index_manager.load_indexes()
                print("Auto-build complete.")
            else:
                print("WARNING: Auto-build produced no chunks. Check HF_TOKEN and corpus access.")
        except Exception as e:
            print(f"WARNING: Auto-build failed: {e}. Server will start without indexes.")

    env_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.65))
    threshold = max(0.65, env_threshold)
    print(f"Confidence threshold configured: {env_threshold} (enforced safety floor -> resolved: {threshold})")
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

# Rate limiter — keyed on remote IP address
_query_rate_limit = os.environ.get("QUERY_RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Setup
# Allow credentials with explicit origin list — wildcard + credentials is a
# security anti-pattern (echoes Origin header on every credentialed request).
_ALLOWED_ORIGINS = [
    o.strip() for o in
    os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:8001,http://127.0.0.1:8001"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
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
        if supabase and os.environ.get("MOCK_AUTH", "false").lower() != "true":
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

            # Atomic write: write to .tmp then rename so a mid-write crash
            # never leaves local_db.json truncated to 0 bytes.
            tmp_path = self.local_db_path + '.tmp'
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, self.local_db_path)
        except Exception as e:
            print(f"Failed to save message to local DB: {e}")

    def get_history(self, uid: str, conv_id: str) -> List[Dict[str, Any]]:
        """Retrieves conversation history, sorted by created_at/timestamp."""
        # 1. Try Supabase read
        if supabase and os.environ.get("MOCK_AUTH", "false").lower() != "true":
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
    def save_feedback(self, uid: str, feedback: Dict[str, Any]):
        """Saves user feedback on an answer to Supabase or local DB."""
        if supabase and os.environ.get("MOCK_AUTH", "false").lower() != "true":
            try:
                supabase.table("feedback").insert({
                    "conversation_id": feedback.get("conversation_id"),
                    "user_id": uid,
                    "message_id": feedback.get("message_id"),
                    "query": feedback.get("query"),
                    "rating": feedback.get("rating"),
                    "category": feedback.get("category", "other"),
                    "comment": feedback.get("comment", "")
                }).execute()
                return
            except Exception as e:
                print(f"Supabase feedback save error: {e}. Falling back to local file.")

        try:
            with open(self.local_db_path, "r") as f:
                data = json.load(f)
            if "feedback" not in data:
                data["feedback"] = []
            data["feedback"].append({
                "id": str(uuid.uuid4()),
                "user_id": uid,
                "timestamp": time.time(),
                **feedback
            })
            # Atomic write
            tmp_path = self.local_db_path + '.tmp'
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, self.local_db_path)
        except Exception as e:
            print(f"Failed to save feedback to local DB: {e}")

    def get_telemetry(self) -> Dict[str, Any]:
        """Calculates query drift, refusal rates, confidence, and namespace metrics."""
        total_queries = 0
        refused_queries = 0
        conf_scores = []
        ns_counts: Dict[str, int] = {}
        provider_counts: Dict[str, int] = {}
        feedback_counts = {"thumbs_up": 0, "thumbs_down": 0, "flag": 0}

        try:
            with open(self.local_db_path, "r") as f:
                data = json.load(f)

            for key, val in data.items():
                if key == "feedback":
                    for fb in val:
                        r = fb.get("rating", "other")
                        feedback_counts[r] = feedback_counts.get(r, 0) + 1
                    continue
                if isinstance(val, dict):
                    for conv_id, msgs in val.items():
                        for m in msgs:
                            if m.get("role") == "assistant":
                                total_queries += 1
                                if m.get("refused", False):
                                    refused_queries += 1
                                c = m.get("confidence_score")
                                if c is not None:
                                    conf_scores.append(float(c))
                                ns = m.get("namespace_searched", "unknown")
                                ns_counts[ns] = ns_counts.get(ns, 0) + 1
                                prov = m.get("provider", "unknown")
                                provider_counts[prov] = provider_counts.get(prov, 0) + 1
        except Exception as e:
            print(f"Error calculating telemetry: {e}")

        avg_conf = float(sum(conf_scores) / len(conf_scores)) if conf_scores else 0.0
        refusal_rate = float(refused_queries / total_queries) if total_queries else 0.0

        return {
            "total_queries": total_queries,
            "refused_queries": refused_queries,
            "refusal_rate": round(refusal_rate, 4),
            "average_confidence": round(avg_conf, 4),
            "namespaces": ns_counts,
            "providers": provider_counts,
            "feedback": feedback_counts,
            "timestamp": time.time()
        }

# Initialize database manager
db_manager = DatabaseManager()

def authenticate_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Validates token against Supabase Auth.
    If Supabase is offline or MOCK_AUTH=true (opt-in), returns a standard mock UUID string.
    Defaults to real auth — set MOCK_AUTH=true explicitly in .env for local dev.
    """
    mock_uuid = "00000000-0000-0000-0000-000000000000"
    
    if not authorization or not authorization.startswith("Bearer "):
        # MOCK_AUTH defaults to "false" — opt-in to mock mode, not opt-out.
        if os.environ.get("MOCK_AUTH", "false").lower() == "true":
            return mock_uuid
        print("[AUTH_FAILURE] missing or malformed Authorization header")
        raise HTTPException(status_code=401, detail="Invalid Authorization header format. Must be 'Bearer <token>'.")

    token = authorization.split("Bearer ")[1]

    # SECURITY: Do NOT check `token == "mock-token"` here unconditionally.
    # Gate this exclusively behind MOCK_AUTH.
    if os.environ.get("MOCK_AUTH", "false").lower() == "true":
        return mock_uuid

    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase authentication client not initialized.")

    try:
        # Validate Supabase access token (JWT)
        user_res = supabase.auth.get_user(token)
        return user_res.user.id
    except Exception as e:
        token_prefix = token[:8] if len(token) >= 8 else token
        print(f"[AUTH_FAILURE] token_prefix={token_prefix}... error={str(e)[:120]}")
        raise HTTPException(status_code=401, detail=f"Invalid or expired Supabase token: {str(e)}")

# (Pipeline globals and lifespan handler moved above app initialization)

# -------------------------------------------------------------
# SESSION CHUNK CACHE (in-process, per conversation)
# Stores last retrieved chunks per conversation so follow-up
# clarification queries can reuse them without re-retrieval.
# Capped at 200 entries to avoid unbounded memory growth.
# TTL: 30 minutes — stale cache entries are discarded on access.
# -------------------------------------------------------------
_MAX_CACHE = 200
_CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 1800))  # 30 min default
# Values: (chunks_list, timestamp_float)
_session_chunk_cache: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}


# --- Follow-up query detection ---
_FOLLOWUP_PATTERNS = [
    # Clarification / elaboration / simplification
    r"\b(be\s+more\s+clear|more\s+clear|clarif|elaborate|explain\s+(further|more|that|again|in\s+simple|this)|simpl(y|ify)|in\s+plain|in\s+simple\s+(terms|language|words)|what\s+does\s+that\s+mean|what\s+do\s+you\s+mean)\b",
    # Summary / recap of previous turns
    r"\b(summar(y|ize|ise)|recap|what\s+(did|have)\s+we\s+(discuss|talk|cover|said)|everything\s+we\s+discussed|what\s+all\s+we\s+talked|list\s+everything|tell\s+me\s+everything)\b",
    # Referential pronouns / anaphora pointing to previous answer
    r"^\s*(what\s+about\s+(it|that|this|them|those|the\s+same)\b|what\s+is\s+its\s+\w+|what\s+are\s+its\s+\w+|and\s+the\s+\w+\s+(for|of|in|on)\s+(it|that|this|them)\b)",
    r"\b(in\s+(this|that)\s+case|for\s+(this|that)|you\s+mentioned|you\s+said|previously\s+mentioned)\b",
    # Short conversational responses / continuations
    r"^\s*(ok|okay|thanks|thank\s+you|got\s+it|understood|yes|no|sure|cool|alright|fine|great)\.?\s*$",
    r"^\s*(and\s+then\??|what\s+next\??|what\s+now\??|why\??|how\s+so\??|what\s+else\??|are\s+you\s+sure\??|tell\s+me\s+more\.?)\s*$",
]
import re as _re
_FOLLOWUP_RE = [_re.compile(p, _re.IGNORECASE) for p in _FOLLOWUP_PATTERNS]

def _is_followup_query(question: str, has_prior_history: bool) -> bool:
    """Returns True if the query is a conversational follow-up that should not
    be routed through the RAG retrieval pipeline independently."""
    if not has_prior_history:
        return False
    q = question.strip()
    return any(pat.search(q) for pat in _FOLLOWUP_RE)

# -------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# -------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Legal question to answer")
    conversation_id: str = Field(..., min_length=1, max_length=64, pattern=r'^conv_[a-z0-9]{7}$', description="Session identifier")

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
    provider: Optional[str] = "gemini"
    model: Optional[str] = "gemini-3.6-flash"
    latency_ms: Optional[int] = 0

class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: Optional[str] = None
    query: Optional[str] = None
    rating: Literal['thumbs_up', 'thumbs_down', 'flag']
    category: Optional[Literal['wrong_section', 'outdated_law', 'hallucination', 'incorrect_advice', 'other']] = "other"
    comment: Optional[str] = ""

# -------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
@limiter.limit(_query_rate_limit)
def run_query(request: Request, req: QueryRequest, uid: str = Depends(authenticate_user)):
    """Runs a question through the RAG pipeline."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Check logs.")

    try:
        return _run_query_inner(req, uid)
    except HTTPException:
        raise  # re-raise FastAPI HTTP exceptions as-is
    except Exception as e:
        tb = traceback.format_exc()
        print(f"UNHANDLED QUERY ERROR:\n{tb}")
        raise HTTPException(status_code=500, detail="Internal server error. Check backend logs.")

def _run_query_inner(req: QueryRequest, uid: str):
    start_t = time.time()

    # 0. Load conversation history first — needed for follow-up detection
    history_msgs = []
    history_records = []
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
        print(f"WARNING: Error loading conversation history: {e}")

    has_prior = len(history_msgs) > 0

    # 1a. Follow-up / clarification shortcut — bypass retrieval gate
    if _is_followup_query(req.question, has_prior):
        cached_entry = _session_chunk_cache.get(req.conversation_id)
        if cached_entry:
            cached_chunks_raw, cache_ts = cached_entry
            cached_chunks = cached_chunks_raw if (time.time() - cache_ts) < _CACHE_TTL_SECONDS else []
        else:
            cached_chunks = []
        print(f"[FOLLOWUP] Detected conversational follow-up. Reusing {len(cached_chunks)} cached chunks.")
        refused = False
        # Use mean relevance of cached chunks rather than hardcoding 1.0
        if cached_chunks:
            confidence = sum(c.get('relevance_score', 0.0) for c in cached_chunks) / len(cached_chunks)
        else:
            confidence = 0.5  # unknown quality, not zero
        namespace_searched = "memory"
        retrieved_chunks = cached_chunks

        raw_provider = getattr(rag_chain, "provider", "gemini")
        raw_model = getattr(rag_chain, "model_name", "gemini-3.6-flash")
        provider = raw_provider if isinstance(raw_provider, str) else "gemini"
        model_name = raw_model if isinstance(raw_model, str) else "gemini-3.6-flash"

        try:
            answer = rag_chain.run(req.question, retrieved_chunks, history_msgs)
        except Exception as e:
            print(f"LLM chain execution error (follow-up): {e}")
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

        latency_ms = int((time.time() - start_t) * 1000)
        sources = []
        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            sources.append({
                "document_name": meta["document_name"],
                "legal_domain":  meta.get("legal_domain", "general"),
                "pub_year":      meta["pub_year"],
                "namespace":     meta["namespace"],
                "source_url":    meta["source_url"],
                "relevance_score": chunk.get("relevance_score", 0.0)
            })

        print(f"[QUERY_TELEMETRY] uid={uid[:8]}... conv={req.conversation_id[:8]}... conf={confidence:.4f} ns={namespace_searched} refused={refused} provider={provider} model={model_name} latency_ms={latency_ms}")

        for record, role_key, content_val in [
            ({"role": "user", "content": req.question, "sources": [], "confidence_score": None, "refused": False, "namespace_searched": None}, None, None),
            ({"role": "assistant", "content": answer, "sources": sources, "confidence_score": confidence, "refused": False, "namespace_searched": namespace_searched, "provider": provider, "model": model_name, "latency_ms": latency_ms}, None, None),
        ]:
            db_manager.save_message(uid, req.conversation_id, record)

        return {
            "answer": answer,
            "sources": sources,
            "confidence_score": confidence,
            "refused": refused,
            "provider": provider,
            "model": model_name,
            "latency_ms": latency_ms
        }

    # 1b. Normal path — full retrieval pipeline
    pipeline_res = rag_pipeline.query(req.question, conversation_id=req.conversation_id)

    refused = pipeline_res["refused"]
    confidence = pipeline_res["confidence_score"]
    namespace_searched = pipeline_res["namespace_searched"]
    retrieved_chunks = pipeline_res["retrieved_chunks"]

    # Cache chunks for future follow-up reuse (with timestamp for TTL)
    if not refused and retrieved_chunks:
        if len(_session_chunk_cache) >= _MAX_CACHE:
            # Evict oldest entry
            oldest_key = next(iter(_session_chunk_cache))
            del _session_chunk_cache[oldest_key]
        _session_chunk_cache[req.conversation_id] = (retrieved_chunks, time.time())

    sources = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        sources.append({
            "document_name": meta["document_name"],
            "legal_domain":  meta.get("legal_domain", "general"),
            "pub_year":      meta["pub_year"],
            "namespace":     meta["namespace"],
            "source_url":    meta["source_url"],
            "relevance_score": chunk["relevance_score"]
        })

    raw_provider = getattr(rag_chain, "provider", "gemini")
    raw_model = getattr(rag_chain, "model_name", "gemini-3.6-flash")
    provider = raw_provider if isinstance(raw_provider, str) else "gemini"
    model_name = raw_model if isinstance(raw_model, str) else "gemini-3.6-flash"

    if refused:
        answer = "The indexed corpus does not contain sufficient information to answer this reliably. Please consult a qualified lawyer or refer to indiacode.nic.in."
    else:
        # 2 & 3. Generate Answer using LangChain with pre-loaded history
        try:
            answer = rag_chain.run(req.question, retrieved_chunks, history_msgs)
        except Exception as e:
            print(f"LLM chain execution error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    latency_ms = int((time.time() - start_t) * 1000)

    # Structured telemetry logging for drift / audit tracking
    print(f"[QUERY_TELEMETRY] uid={uid[:8]}... conv={req.conversation_id[:8]}... conf={confidence:.4f} ns={namespace_searched} refused={refused} provider={provider} model={model_name} latency_ms={latency_ms}")

    # 4. Persist exchange
    user_record = {
        "role": "user",
        "content": req.question,
        "sources": [],
        "confidence_score": None,
        "refused": False,
        "namespace_searched": None
    }
    db_manager.save_message(uid, req.conversation_id, user_record)

    assistant_record = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence_score": confidence,
        "refused": refused,
        "namespace_searched": namespace_searched,
        "provider": provider,
        "model": model_name,
        "latency_ms": latency_ms
    }
    db_manager.save_message(uid, req.conversation_id, assistant_record)

    return {
        "answer": answer,
        "sources": sources,
        "confidence_score": confidence,
        "refused": refused,
        "provider": provider,
        "model": model_name,
        "latency_ms": latency_ms
    }

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, uid: str = Depends(authenticate_user)):
    """Allows users to flag bad answers or submit satisfaction ratings."""
    db_manager.save_feedback(uid, req.model_dump())
    print(f"[USER_FEEDBACK] uid={uid[:8]}... conv={req.conversation_id[:8]}... rating={req.rating} category={req.category} comment={req.comment}")
    return {"status": "success", "message": "Feedback recorded successfully."}

@app.get("/telemetry")
def get_telemetry_metrics(uid: str = Depends(authenticate_user)):
    """Returns query drift metrics, refusal rates, confidence, and namespace stats.
    Requires authentication to prevent system fingerprinting by external parties.
    """
    return db_manager.get_telemetry()

@app.get("/history/{conversation_id}")
def get_conversation_history(conversation_id: str, uid: str = Depends(authenticate_user)):
    """Retrieves full conversation history for the authenticated user."""
    history = db_manager.get_history(uid, conversation_id)
    return {"history": history if isinstance(history, list) else []}

@app.get("/health")
def health_check():
    """Public health check — used by HF Spaces to determine liveness.
    Only exposes fields the UI needs (index status, namespace gaps).
    Sensitive internals (model name, provider, supabase state) are stripped
    to prevent system fingerprinting by unauthenticated callers.
    """
    loaded_ns = list(index_manager.faiss_indexes.keys())
    indexes_ready = len(loaded_ns) > 0
    missing_ns = [ns for ns in index_manager.namespaces if ns not in index_manager.faiss_indexes]
    status = "healthy" if indexes_ready else ("partial" if missing_ns else "uninitialized")
    return {
        "status": status,
        "indexes_loaded": indexes_ready,
        "loaded_namespaces": loaded_ns,
        "missing_namespaces": missing_ns,
        "timestamp": time.time()
    }

# -------------------------------------------------------------
# STATIC FILES — frontend SPA (Vite build output)
# -------------------------------------------------------------
# Mount last so API routes take priority. The existence check means
# running the backend locally without `npm run build` still works —
# the SPA simply won't be served, but /query, /health etc. are fine.
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
