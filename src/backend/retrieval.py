import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
from src.backend.indexing import LegalIndexManager, tokenize_for_bm25

load_dotenv()

import re

# ── Indian legal namespace router ────────────────────────────────────────

CRIMINAL_KEYWORDS = [
    "murder", "homicide", "theft", "robbery", "dacoity", "assault",
    "kidnapping", "abduction", "cheating", "forgery", "rape", "dowry",
    "bail", "fir", "arrest", "chargesheet", "cognizable", "non-cognizable",
    "bailable", "non-bailable", "imprisonment", "fine", "punishment",
    "bns", "bnss", "bsa", "nyaya sanhita", "nagarik suraksha", "sakshya",
    "criminal", "offence", "offense", "accused", "complainant",
    "magistrate", "sessions court", "evidence", "witness", "confession",
    "investigation", "prosecution", "summons", "warrant", "remand",
    "anticipatory bail", "custody", "sentence", "death penalty",
    "life imprisonment", "hurt", "grievous hurt", "mischief",
    "criminal conspiracy", "attempt", "abetment", "police",
    "police station", "police complaint",
    # Colloquial property crime vocabulary
    "stole", "stolen", "robbed", "rob", "looted", "snatched",
    "burglary", "housebreaking", "break-in", "broke into", "breaking into",
    "shoplifting", "extortion", "trespass", "criminal trespass",
    # Verbal abuse / intimidation vocabulary
    "intimidation", "criminal intimidation", "verbal abuse", "verbally abusing",
    "threat", "threatened", "threatening", "blackmail", "blackmailed",
    # Electronic evidence / court proof vocabulary (Bharatiya Sakshya Adhiniyam, 2023)
    "audio recording", "call recording", "phone recording", "voice recording",
    "cctv", "cctv footage", "electronic evidence", "electronic record",
    "digital evidence", "section 63", "section 65b", "certificate",
    "admissibility", "admissible in court", "court proof",
]

CYBER_KEYWORDS = [
    "cybercrime", "cyber crime", "hacking", "phishing", "ransomware",
    "information technology", "it act", "data protection", "privacy",
    "identity theft", "cyber fraud", "online fraud", "social media",
    "intermediary", "digital", "electronic", "computer", "network",
    "data breach", "unauthorized access", "cyber terrorism",
    "obscene content", "defamation online", "email", "website",
    # Colloquial attack-vector phrasings & online fraud
    "malicious link", "otp", "otp fraud", "account hacked", "lost money online",
    "bank account hacked", "suspicious link", "suspicious message", "sms",
    "clicked", "scam", "scammed", "scammers", "fraud", "fraudulent",
    "electricity bill", "bill scam", "fake link", "phishing link",
    "entered otp", "entering otp", "unauthorized transaction",
    "unauthorised transaction", "unauthorized debit", "unauthorised debit",
    # Fake profile / impersonation / social-platform crimes
    "fake account", "fake profile", "fake id", "fake identity",
    "impersonation", "impersonate", "impersonating", "impersonated",
    "instagram", "facebook", "whatsapp", "twitter", "telegram", "snapchat",
    "youtube", "linkedin", "threads", "social media profile",
    "morphed photos", "morphed images", "morphed video", "morphed picture",
    "edited photos", "edited images", "fake photos", "fake images",
    "deepfake", "revenge porn", "non-consensual",
    # Loan recovery harassment via electronic communications
    "loan recovery", "recovery agent", "recovery agents",
    "harassment calls", "threatening calls", "abusive calls",
    "collection calls", "loan agent",
]

CONSUMER_KEYWORDS = [
    "consumer", "consumer protection", "defective product", "deficiency",
    "service deficiency", "warranty", "guarantee", "refund", "compensation",
    "unfair trade", "misleading advertisement", "consumer forum",
    "consumer commission", "product liability", "e-commerce",
    "defective goods", "consumer complaint", "consumer rights",
]

BANKING_KEYWORDS = [
    "rbi", "reserve bank", "banking", "bank", "loan", "interest rate",
    "npa", "non-performing asset", "upi", "neft", "rtgs", "imps",
    "cheque", "cheques", "cheque bounce", "bounced cheque", "cheque dishonour",
    "dishonour of cheque", "dishonoured", "negotiable instrument", "section 138",
    "mortgage", "insurance", "nbfc", "microfinance",
]

CIVIL_KEYWORDS = [
    # Tenancy, landlord, lease, security deposit, and property disputes (Transfer of Property Act, 1882)
    "landlord", "tenant", "deposit", "security deposit", "rent", "lease", "lessor", "lessee",
    "eviction", "tenancy", "transfer of property", "property dispute", "vacate",
    # Employment and salary disputes (route to general — ICA/Payment of Wages Act/Code on Wages)
    "salary", "wages", "employment", "employer", "employee", "boss", "overtime",
    "working hours", "workplace", "workplace harassment", "office", "hostile work environment",
    "notice period", "termination", "wrongful termination",
    "labour", "labor", "payment of wages", "provident fund", "epf", "gratuity", "bonus",
    "resigned", "resignation", "dismissed", "dismissal", "posh",
    "contract of employment", "appointment letter", "minimum wage",
    # Inheritance and succession (route to general — Hindu Succession Act)
    "inheritance", "succession", "intestate", "testament", "probate",
    "last will", "without a will", "making a will", "leave a will", "will and testament",
    "heir", "legal heir", "property division", "estate",
    # Civil contract disputes (route to general — Indian Contract Act)
    "breach of contract", "specific performance", "injunction",
    "civil suit", "damages", "compensation for breach",
]


def route_query(query: str) -> str:
    """
    Classifies a legal query to a subject namespace.
    Returns: 'criminal', 'cyber', 'consumer', 'banking', 'general', or 'all'.

    Most statutory law questions hit 'criminal' (BNS/BNSS/BSA);
    IT Act / online questions hit 'cyber';
    product / service complaints hit 'consumer';
    financial / banking questions hit 'banking';
    salary / employment / succession / civil contract hit 'general'
      (isolates ICA, Payment of Wages, Hindu Succession from BNS chunk volume).
    Ties and ambiguous queries fall back to 'all' (searches every namespace).
    """
    q_lower = query.lower()
    words = set(re.findall(r"\b\w+\b", q_lower))

    def get_score(keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if " " in kw:
                if kw in q_lower:
                    score += 1
            else:
                if kw in words:
                    score += 1
        return score

    criminal_score = get_score(CRIMINAL_KEYWORDS)
    cyber_score    = get_score(CYBER_KEYWORDS)
    consumer_score = get_score(CONSUMER_KEYWORDS)
    banking_score  = get_score(BANKING_KEYWORDS)
    civil_score    = get_score(CIVIL_KEYWORDS)

    # Special handling for hybrid Tenancy & Physical Force / Lockout / Criminal Trespass:
    # When a query involves both civil property/tenancy and physical force/lockout,
    # configure query router to retrieve from both general and criminal indexes simultaneously.
    has_tenancy = any(k in q_lower for k in ["landlord", "tenant", "lease", "rent", "deposit", "evict", "eviction", "premises", "lessor", "lessee"])
    has_physical_force = any(k in q_lower for k in [
        "lock", "locked", "lockout", "gate", "cut off", "cutoff", "electricity", "water",
        "force", "forced", "forcibly", "threat", "threatened", "assault", "beat", "trespass", "throw out", "threw out", "harass"
    ])
    if has_tenancy and has_physical_force:
        return "general,criminal"

    # Special handling for hybrid Cyber & Extortion / Blackmail (Sec 308 BNS + Sec 66E/67 IT Act):
    is_extortion = any(k in q_lower for k in ["extort", "extortion", "blackmail", "blackmailed", "demanding money", "threat to leak", "threatened to post"])
    is_digital_media = any(k in q_lower for k in ["photos", "images", "video", "whatsapp", "instagram", "facebook", "telegram", "online", "morphed"])
    if is_extortion and is_digital_media:
        return "cyber,criminal"

    # Special handling for online/OTP/phishing banking fraud:
    # If query involves digital attack vectors (OTP, SMS, link, phishing, hacked),
    # the governing penal statute is IT Act (cyber namespace), not banking regulations.
    if cyber_score > 0 and ("otp" in q_lower or "sms" in q_lower or "link" in q_lower or "phishing" in q_lower or "hacked" in q_lower):
        return "cyber"

    scores = {
        "criminal": criminal_score,
        "cyber":    cyber_score,
        "consumer": consumer_score,
        "banking":  banking_score,
        "general":  civil_score,
    }

    max_score = max(scores.values())
    if max_score == 0:
        return "all"

    tied = [ns for ns, s in scores.items() if s == max_score]
    if len(tied) > 1:
        return "all"

    return tied[0]


def is_direct_statutory_query(query: str, history: Optional[List[Any]] = None) -> bool:
    """
    Determines if a query already contains explicit statutory anchors (e.g. section numbers,
    exact act names) such that LLM HyDE expansion can be safely bypassed for near-instant retrieval.
    Follow-up queries with history NEVER skip HyDE because they require referential resolution.
    """
    if history:
        return False
    q = query.strip().lower()

    # Explicit section reference (e.g. "Section 138", "Sec. 308", "u/s 66D", "s. 17(2)")
    has_explicit_sec = bool(re.search(r'\b(?:section|sec|u/s|s\.)\s*\d+[a-z]?(?:\(\d+\))?\b', q))
    if has_explicit_sec:
        return True

    # Explicit recognized statutory enactments
    statute_indicators = [
        "negotiable instruments act", "consumer protection act",
        "bharatiya nyaya sanhita", "bharatiya nagarik suraksha",
        "bharatiya sakshya adhiniyam", "information technology act",
        "payment of wages act", "code on wages", "transfer of property act",
        "domestic violence act", "pwdva", "hindu succession act", "pocso act",
        "prevention of corruption act", "industrial disputes act", "rera",
        "it act 2000", "cpa 2019", "ni act", "bns 2023", "bnss 2023", "bsa 2023"
    ]
    if any(act in q for act in statute_indicators):
        return True

    return False


class LegalRAGPipeline:
    def __init__(self, index_manager: LegalIndexManager, confidence_threshold: float = 0.55):
        self.index_manager = index_manager
        self.confidence_threshold = confidence_threshold

    def rerank_with_cohere(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate chunks using Cohere's Cross-Encoder API (model: rerank-v3.5).
        Gracefully falls back to existing FAISS/RRF score ordering if:
        - COHERE_API_KEY is not set or empty
        - USE_COHERE_RERANK is set to false
        - Monthly quota is reached or network fails
        """
        if not candidates:
            return []

        # Calibrate fallback relevance_score for all candidates from FAISS/RRF:
        # If Cohere fails or is disabled, the threshold gate (0.55) evaluates this calibrated score.
        for cand in candidates:
            raw_faiss = float(cand.get("faiss_score", 0.0))
            if raw_faiss <= 0.0 and "rrf_score" in cand:
                # Top RRF hit is ~0.03. Scale to comparable cosine range [0.45, 0.75]
                raw_faiss = min(0.75, cand["rrf_score"] * 25.0)
            
            # Map raw cosine score into calibrated confidence:
            # Raw BGE-M3 cosine for relevant legal matches is typically 0.52 - 0.85.
            if raw_faiss >= 0.50:
                calibrated = min(0.95, 0.55 + (raw_faiss - 0.50) * 0.80)
            else:
                calibrated = max(0.0, raw_faiss * 0.88)
            cand["relevance_score"] = float(calibrated)

        api_key = os.environ.get("COHERE_API_KEY", "").strip()
        use_cohere = os.environ.get("USE_COHERE_RERANK", "true").lower() == "true"

        if not api_key or not use_cohere:
            candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            return candidates[:top_n]

        try:
            import cohere
            co = cohere.Client(api_key=api_key)
            docs = [c.get("text", "")[:2048] for c in candidates]
            response = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=min(top_n, len(candidates))
            )

            reranked = []
            for item in response.results:
                chunk = dict(candidates[item.index])
                co_score = float(item.relevance_score)
                chunk["cohere_score"] = co_score
                # Conservative calibration of Cohere cross-encoder scores:
                #   co_score >= 0.15 → smooth linear mapping into [0.58, 0.95]
                #                      (reaches 0.70+ green threshold at raw co_score ~0.43)
                #   co_score  < 0.15 → capped at <= 0.44 (strictly below 0.55 gate, low/red)
                if co_score >= 0.15:
                    scaled_score = min(0.95, 0.52 + co_score * 0.42)
                else:
                    scaled_score = min(0.44, co_score * 2.5)
                chunk["relevance_score"] = float(scaled_score)
                reranked.append(chunk)

            if reranked:
                print(f"[COHERE RERANK] Reranked {len(candidates)} candidates down to {len(reranked)} (top Cohere score: {reranked[0]['cohere_score']:.4f}, calibrated score: {reranked[0]['relevance_score']:.4f})")
                return reranked
        except Exception as e:
            print(f"[COHERE RERANK FALLBACK] Cohere rerank unavailable or failed ({e}). Falling back to FAISS/RRF scores.")

        candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return candidates[:top_n]

    @staticmethod
    def _filter_irrelevant_chunks(
        chunks: List[Dict[str, Any]],
        namespace: str,
        query_lower: str,
    ) -> List[Dict[str, Any]]:
        """
        Post-rerank filter that drops chunks from paramilitary / armed-forces
        acts when the query is routed to the 'general' namespace (employment,
        succession, contract) and the query itself has no military vocabulary.

        Problem: Acts like CRPF Act 1949, BSF Act, CISF Act contain 'service',
        'employment', 'pay', 'duty' keywords that match civilian employment
        queries in the general namespace, polluting the sources list.

        The filter is a narrow blocklist — it only fires when namespace='general'
        and the query does not contain explicit police/military terminology.
        It never modifies results for criminal, cyber, consumer, or banking queries.
        Safety net: if filtering removes everything, the original list is returned.
        """
        if namespace != "general":
            return chunks  # only relevant for civilian employment / succession

        # If user explicitly mentions a military/police context, allow all chunks
        military_terms = {
            "police", "military", "army", "navy", "airforce", "air force",
            "crpf", "bsf", "cisf", "itbp", "ssb", "paramilitary",
            "defence", "defense", "armed forces", "constable", "inspector",
            "sipahi", "havildar", "jawan", "trooper", "battalion",
        }
        if any(t in query_lower for t in military_terms):
            return chunks

        # Act name fragments that are irrelevant to civilian employment / succession
        blocklist = [
            "central reserve police", "crpf",
            "border security force", "bsf act",
            "central industrial security", "cisf",
            "indo-tibetan border", "itbp",
            "sashastra seema bal", "ssb act",
            "coast guard",
            "armed forces", "army act", "navy act", "air force act",
            "national security guard",
            # Obsolete pre-independence colonial revenue / toll / customs statutes that dilute labor/civil law
            "indian tolls act", "tobacco duty", "bonded warehouse", "shore nuisances",
            "embankment act", "ghatwali", "straits settlement", "forfeited deposits",
            "board of revenue", "calcutta land-revenue", "bills of lading act 1856",
            "improvements in towns act", "rent recovery act, 1853", "fatal accidents act, 1855",
            "coasting vessels", "boundary-marks", "legal representatives suits act",
        ]

        filtered = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            act_key = " ".join([
                str(meta.get("act_name", "")),
                str(meta.get("document_name", "")),
            ]).lower()
            if any(b in act_key for b in blocklist):
                print(f"[CHUNK FILTER] Dropped paramilitary chunk: {act_key[:80]}")
                continue
            filtered.append(chunk)

        # Safety net: never return empty — fall back to unfiltered if all matched blocklist
        return filtered if filtered else chunks

    def rrf_merge(self, dense_results: List[Tuple[Dict[str, Any], int]], sparse_results: List[Tuple[Dict[str, Any], int]], k: int = 60, temporal_boost: float = 0.1) -> List[Dict[str, Any]]:
        """
        Merges dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).
        Applies publication year temporal weighting.
        """
        rrf_scores: Dict[str, float] = {}
        faiss_best: Dict[str, float] = {}  # best FAISS cosine score per chunk
        # Keep map of chunk ID/unique key to the chunk object itself
        chunk_map: Dict[str, Dict[str, Any]] = {}

        def get_chunk_key(chunk: Dict[str, Any]) -> str:
            # Create a unique key for the chunk based on text hash or document name + text subset
            meta = chunk["metadata"]
            return f"{meta['document_name']}_{meta.get('legal_domain', 'unknown')}_{hash(chunk['text'])}"

        # Process dense results.
        # Use the rank *r* stored in the tuple by retrieve() — this is the
        # per-namespace rank assigned during FAISS/BM25 search and carries
        # the correct ordering within each namespace.  Do NOT re-derive rank
        # from enumerate() over the flat concatenated list: that would give
        # correct ranks only to the first namespace's results and push every
        # subsequent namespace's top hits to artificially bad positions.
        for _, (chunk, r) in enumerate(dense_results):
            key = get_chunk_key(chunk)
            chunk_map[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + r + 1)
            # Keep the best (max) faiss_score seen for this chunk across namespaces
            if "faiss_score" in chunk:
                existing = faiss_best.get(key, -1.0)
                faiss_best[key] = max(existing, chunk["faiss_score"])

        # Process sparse results (same rank-preservation rationale as above).
        for _, (chunk, r) in enumerate(sparse_results):
            key = get_chunk_key(chunk)
            chunk_map[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + r + 1)

        # Apply temporal boost
        # Normalize pub_year in range [1990, 2026]
        if temporal_boost > 0:
            for key, chunk in chunk_map.items():
                pub_year = chunk.get("metadata", {}).get("pub_year", 2000)
                norm_year = max(0.0, min(1.0, (pub_year - 1990) / (2026 - 1990)))
                rrf_scores[key] = rrf_scores[key] * (1.0 + temporal_boost * norm_year)

        # Sort and build result list with scores attached
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        merged_results = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            chunk_copy = dict(chunk)
            chunk_copy["rrf_score"] = rrf_scores[key]
            # Attach the best FAISS score seen for this chunk (0.0 if not in dense results)
            chunk_copy["faiss_score"] = faiss_best.get(key, chunk_copy.get("faiss_score", 0.0))
            merged_results.append(chunk_copy)

        # Sort descending by final score
        merged_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged_results

    def retrieve(self, query: str, target_namespace: str = "all", top_n: int = 20, temporal_boost: float = 0.1) -> List[Dict[str, Any]]:
        """
        Runs dual retrieval (dense + sparse) on the chosen namespace(s)
        and merges results via RRF.
        """
        # Determine which namespaces to search
        namespaces_to_search = []
        if isinstance(target_namespace, list):
            namespaces_to_search = target_namespace
        elif target_namespace == "all":
            namespaces_to_search = self.index_manager.namespaces
        elif "," in target_namespace:
            namespaces_to_search = [ns.strip() for ns in target_namespace.split(",") if ns.strip()]
        else:
            namespaces_to_search = [target_namespace]

        all_dense_results = []
        all_sparse_results = []

        # Lazy load embeddings model if we are performing search
        model = self.index_manager.load_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True).astype('float32').reshape(1, -1)
        query_tokens = tokenize_for_bm25(query)

        for ns in namespaces_to_search:
            if ns not in self.index_manager.faiss_indexes or ns not in self.index_manager.bm25_indexes:
                continue

            # 1. Dense retrieval (FAISS)
            faiss_idx = self.index_manager.faiss_indexes[ns]
            chunks = self.index_manager.chunks[ns]
            
            # Retrieve top_n
            scores, indices = faiss_idx.search(query_vector, top_n)
            ns_dense = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(chunks):
                    # Store the raw cosine similarity score on the chunk dict
                    # so it can be used as a confidence signal after RRF merge.
                    chunk_with_score = dict(chunks[idx])
                    chunk_with_score["faiss_score"] = float(score)
                    ns_dense.append((chunk_with_score, rank))
            all_dense_results.extend(ns_dense)

            # 2. Sparse retrieval (BM25)
            bm25_idx = self.index_manager.bm25_indexes[ns]
            bm25_scores = bm25_idx.get_scores(query_tokens)
            
            # Sort indices by BM25 score
            sorted_indices = np.argsort(bm25_scores)[::-1][:top_n]
            ns_sparse = []
            for rank, idx in enumerate(sorted_indices):
                if bm25_scores[idx] > 0: # only include terms with positive matching scores
                    ns_sparse.append((chunks[idx], rank))
            all_sparse_results.extend(ns_sparse)

        # Merge results using RRF and sort
        merged = self.rrf_merge(all_dense_results, all_sparse_results, k=60, temporal_boost=temporal_boost)
        return merged[:top_n]

    def smart_truncate(self, text: str, query: str, max_chars: int = 2048) -> str:
        """
        Extracts a window of text of length max_chars centered around the first
        occurrence of query keywords, to avoid truncation issues on CPU.
        """
        if len(text) <= max_chars:
            return text
            
        # Extract clean alphanumeric keywords from query (excluding short words)
        query_words = re.findall(r"\b\w{3,}\b", query.lower())
        if not query_words:
            return text[:max_chars]
            
        # Find the positions of the matches
        text_lower = text.lower()
        match_positions = []
        for word in query_words:
            for m in re.finditer(r"\b" + re.escape(word) + r"\b", text_lower):
                match_positions.append(m.start())
                
        if not match_positions:
            # Fallback to simple substring search if word boundaries don't match
            for word in query_words:
                start = 0
                while True:
                    pos = text_lower.find(word, start)
                    if pos == -1:
                        break
                    match_positions.append(pos)
                    start = pos + len(word)
                    
        if not match_positions:
            return text[:max_chars]
            
        # Find the first match and center the window around it
        first_match = min(match_positions)
        start_idx = max(0, first_match - max_chars // 2)
        end_idx = start_idx + max_chars
        
        # If the window goes past the end of the text, shift it back
        if end_idx > len(text):
            end_idx = len(text)
            start_idx = max(0, end_idx - max_chars)
            
        # Align to word boundaries if possible
        if start_idx > 0:
            space_idx = text.rfind(" ", max(0, start_idx - 50), start_idx)
            if space_idx != -1:
                start_idx = space_idx + 1
                
        return text[start_idx : start_idx + max_chars]

    def query(self, query_text: str, conversation_id: str = "", history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Runs the full retrieval pipeline including routing, dual search,
        RRF merge, cross-encoder reranking, and confidence threshold gate.

        If initial confidence is below the threshold and the query contains
        cyber-domain keywords, a second search is attempted against the
        isolated 'cyber' namespace (IT Act / cyber crime chunks only).
        The higher of the two scores is used.

        Returns a dict:
        {
            "query": query_text,
            "refused": bool,
            "confidence_score": float,
            "namespace_searched": str,
            "retrieved_chunks": List[Dict[str, Any]]
        }
        """
        # Step 1: Query Expansion / HyDE decision
        # Direct statutory queries skip HyDE initially (Fast-Path) to cut latency by ~2-3s and save 50% API tokens.
        is_fast_path = is_direct_statutory_query(query_text, history=history)
        expanded_query = query_text

        if is_fast_path:
            print(f"[FAST-PATH] Direct statutory query detected — skipping HyDE expansion: '{query_text[:60]}'")
        else:
            try:
                from src.backend.chain import LegalRAGChain
                provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
                api_key = os.environ.get("GROQ_API_KEY") if provider == "groq" else os.environ.get("GEMINI_API_KEY")
                if api_key:
                    chain = LegalRAGChain()
                    expanded_query = chain.expand_query(query_text, history=history)
                    print(f"Original query: {query_text.encode('ascii', 'replace').decode('ascii')}")
                    print(f"Expanded query: {expanded_query.encode('ascii', 'replace').decode('ascii')}")
            except Exception as e:
                print(f"Query expansion failed or skipped: {e}")

        # Step 2: Domain Routing
        # The user's explicit query has highest priority.
        # If the user query is multi-domain (e.g. mentions BNS from criminal AND cybercrime from cyber),
        # keep 'all' so chunks from both acts are retrieved and synthesized.
        user_namespace = route_query(query_text)
        if user_namespace != "all":
            namespace = user_namespace
        else:
            # Check if user query has explicit keywords in multiple legal domains
            q_lower = query_text.lower()
            q_words = set(re.findall(r"\b\w+\b", q_lower))
            has_criminal = any(k in q_words for k in ["bns", "bnss", "bsa", "crime", "criminal", "murder", "theft", "fir", "police", "bail"])
            has_cyber = any((k in q_lower if " " in k else k in q_words) for k in ["cyber", "cybercrime", "cyber crime", "hack", "hacking", "phishing", "online", "otp", "link"])
            has_consumer = any(k in q_words for k in ["consumer", "refund", "warranty", "defective", "landlord", "deposit"])
            has_banking = any(k in q_words for k in ["bank", "banking", "rbi", "cheque", "loan", "upi"])
            has_civil = any(k in q_words for k in ["salary", "wages", "employer", "employee", "termination", "will", "inheritance"])
            
            multi_domain_hit = sum([has_criminal, has_cyber, has_consumer, has_banking, has_civil]) >= 2
            if multi_domain_hit:
                namespace = "all"
            else:
                # Ambiguous / colloquial phrasing with 0 keyword hits (e.g. "someone took my stuff"):
                # Use HyDE expansion to classify the domain.
                combined_routing_query = f"{query_text} {expanded_query}"
                namespace = route_query(combined_routing_query)
        print(f"[ROUTER] Routed query to namespace: '{namespace}'")

        # Step 3: Dual Retrieval and RRF Merge (top 20 candidate set for reranking)
        candidates = self.retrieve(expanded_query, target_namespace=namespace, top_n=20)
        
        if not candidates:
            return {
                "query": query_text,
                "refused": True,
                "confidence_score": 0.0,
                "namespace_searched": namespace,
                "retrieved_chunks": []
            }

        # Step 4: Reranking & Score Assignment
        # Uses Cohere rerank-v3.5 API if COHERE_API_KEY is configured,
        # otherwise gracefully falls back to BGE-M3 FAISS cosine similarity.
        top_chunks = self.rerank_with_cohere(expanded_query, candidates, top_n=5)

        # Step 4a: Post-rerank relevance filter
        # Drops domain-irrelevant chunks (e.g. CRPF Act in civilian salary queries)
        # that score well on surface keywords but don't belong to the query domain.
        top_chunks = self._filter_irrelevant_chunks(top_chunks, namespace, query_text.lower())

        # Top-1 score is the confidence score
        confidence = top_chunks[0].get("relevance_score", 0.0) if top_chunks else 0.0
        print(f"[CONFIDENCE] Top relevance score: {confidence:.4f} (threshold: {self.confidence_threshold})")

        # Step 4b: Cyber-namespace fallback
        # If confidence is below the gate AND the query contains at least one
        # cyber keyword, retry search on the isolated 'cyber' namespace.
        # This handles queries like "clicked a malicious link" where the 'all'
        # namespace mixes in off-topic criminal/banking chunks that dilute scores.
        q_lower_fb = query_text.lower()
        cyber_keyword_hit = any(
            (kw in q_lower_fb) for kw in CYBER_KEYWORDS
        )
        if confidence < self.confidence_threshold and cyber_keyword_hit and namespace != "cyber":
            print(f"[Cyber fallback] Initial confidence {confidence:.4f} < gate; retrying on isolated cyber namespace.")
            fb_candidates = self.retrieve(expanded_query, target_namespace="cyber", top_n=15)
            if fb_candidates:
                fb_top_chunks = self.rerank_with_cohere(expanded_query, fb_candidates, top_n=5)
                fb_top_chunks = self._filter_irrelevant_chunks(fb_top_chunks, "cyber", query_text.lower())
                fb_confidence = fb_top_chunks[0].get("relevance_score", 0.0) if fb_top_chunks else 0.0
                print(f"[Cyber fallback] Cyber-namespace confidence: {fb_confidence:.4f}")
                if fb_confidence > confidence:
                    print(f"[Cyber fallback] Adopting cyber-namespace result (better score).")
                    top_chunks = fb_top_chunks
                    confidence = fb_confidence
                    namespace = "cyber (fallback)"
                else:
                    print(f"[Cyber fallback] Original result retained.")

        # Step 4c: Fast-Path Fallback to HyDE
        # If fast-path was used but confidence remained below threshold, invoke HyDE
        # as a safety net before refusing.
        if confidence < self.confidence_threshold and is_fast_path:
            print(f"[FAST-PATH FALLBACK] Fast-path confidence {confidence:.4f} < gate; triggering HyDE expansion fallback.")
            try:
                from src.backend.chain import LegalRAGChain
                provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
                api_key = os.environ.get("GROQ_API_KEY") if provider == "groq" else os.environ.get("GEMINI_API_KEY")
                if api_key:
                    chain = LegalRAGChain()
                    hyde_query = chain.expand_query(query_text, history=history)
                    fb_candidates = self.retrieve(hyde_query, target_namespace=namespace, top_n=20)
                    if fb_candidates:
                        fb_top_chunks = self.rerank_with_cohere(hyde_query, fb_candidates, top_n=5)
                        fb_top_chunks = self._filter_irrelevant_chunks(fb_top_chunks, namespace, query_text.lower())
                        fb_conf = fb_top_chunks[0].get("relevance_score", 0.0) if fb_top_chunks else 0.0
                        if fb_conf > confidence:
                            print(f"[FAST-PATH FALLBACK] HyDE fallback improved confidence: {confidence:.4f} -> {fb_conf:.4f}")
                            top_chunks = fb_top_chunks
                            confidence = fb_conf
            except Exception as e:
                print(f"[FAST-PATH FALLBACK] HyDE fallback failed: {e}")

        # Step 5: Confidence Gate
        refused = confidence < self.confidence_threshold
        
        return {
            "query": query_text,
            "refused": refused,
            "confidence_score": confidence,
            "namespace_searched": namespace,
            "retrieved_chunks": [] if refused else top_chunks
        }
