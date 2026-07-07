import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from src.backend.indexing import LegalIndexManager, tokenize_for_bm25

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
    "criminal conspiracy", "attempt", "abetment",
]

CYBER_KEYWORDS = [
    "cybercrime", "cyber crime", "hacking", "phishing", "ransomware",
    "information technology", "it act", "data protection", "privacy",
    "identity theft", "cyber fraud", "online fraud", "social media",
    "intermediary", "digital", "electronic", "computer", "network",
    "data breach", "unauthorized access", "cyber terrorism",
    "obscene content", "defamation online", "email", "website",
]

CONSUMER_KEYWORDS = [
    "consumer", "consumer protection", "defective product", "deficiency",
    "service", "warranty", "guarantee", "refund", "compensation",
    "unfair trade", "misleading advertisement", "consumer forum",
    "consumer commission", "product liability", "e-commerce",
    "goods", "services", "consumer complaint", "consumer rights",
    "landlord", "tenant", "deposit", "rent",
]

BANKING_KEYWORDS = [
    "rbi", "reserve bank", "banking", "bank", "loan", "interest rate",
    "npa", "non-performing asset", "upi", "neft", "rtgs", "imps",
    "payment", "cheque bounce", "negotiable instrument",
    "financial fraud", "credit", "debit", "mortgage", "insurance",
    "nbfc", "microfinance", "digital payment",
]


def route_query(query: str) -> str:
    """
    Classifies a legal query to a subject namespace.
    Returns: 'criminal', 'cyber', 'consumer', 'banking', or 'all'.

    Most statutory law questions hit 'criminal' (BNS/BNSS/BSA);
    IT Act / online questions hit 'cyber';
    product / service complaints hit 'consumer';
    financial / banking questions hit 'banking'.
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

    scores = {
        "criminal": criminal_score,
        "cyber":    cyber_score,
        "consumer": consumer_score,
        "banking":  banking_score,
    }

    max_score = max(scores.values())
    if max_score == 0:
        return "all"

    tied = [ns for ns, s in scores.items() if s == max_score]
    if len(tied) > 1:
        return "all"

    return tied[0]

class LegalRAGPipeline:
    def __init__(self, index_manager: LegalIndexManager, reranker_name: str = "BAAI/bge-reranker-v2-m3", confidence_threshold: float = 0.65):
        self.index_manager = index_manager
        self.reranker_name = reranker_name
        self.confidence_threshold = confidence_threshold
        self.reranker = None

    def load_reranker(self):
        """Lazy loads the cross-encoder reranking model."""
        if self.reranker is None:
            print(f"Loading reranker model {self.reranker_name}...")
            from sentence_transformers import CrossEncoder
            import torch
            device = os.environ.get("RERANKER_DEVICE", "cpu")
            self.reranker = CrossEncoder(self.reranker_name, device=device)
            print(f"Reranker loaded successfully on device={device}.")
        return self.reranker

    def rrf_merge(self, dense_results: List[Tuple[Dict[str, Any], int]], sparse_results: List[Tuple[Dict[str, Any], int]], k: int = 60, temporal_boost: float = 0.1) -> List[Dict[str, Any]]:
        """
        Merges dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).
        Applies publication year temporal weighting.
        """
        rrf_scores: Dict[str, float] = {}
        # Keep map of chunk ID/unique key to the chunk object itself
        chunk_map: Dict[str, Dict[str, Any]] = {}

        def get_chunk_key(chunk: Dict[str, Any]) -> str:
            # Create a unique key for the chunk based on text hash or document name + text subset
            meta = chunk["metadata"]
            return f"{meta['document_name']}_{meta.get('legal_domain', meta.get('geography_iso', 'unknown'))}_{hash(chunk['text'])}"

        # Process dense results
        for rank, (chunk, _) in enumerate(dense_results):
            key = get_chunk_key(chunk)
            chunk_map[key] = chunk
            # RRF Score formula: 1 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

        # Process sparse results
        for rank, (chunk, _) in enumerate(sparse_results):
            key = get_chunk_key(chunk)
            chunk_map[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

        # Apply temporal boost
        # Normalize pub_year in range [1990, 2026]
        min_year, max_year = 1990, 2026
        
        merged_results = []
        for key, rrf_score in rrf_scores.items():
            chunk = chunk_map[key]
            pub_year = chunk["metadata"].get("pub_year", 2000)
            
            # Normalize year to [0, 1] range, clipping if outside
            normalized_year = (pub_year - min_year) / (max_year - min_year)
            normalized_year = max(0.0, min(1.0, normalized_year))
            
            # Combine RRF score with temporal weight
            final_score = rrf_score + (temporal_boost * normalized_year)
            
            # Add final score to metadata for transparency/debugging
            chunk_copy = dict(chunk)
            chunk_copy["rrf_score"] = final_score
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
        if target_namespace == "all":
            namespaces_to_search = self.index_manager.namespaces
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
                    ns_dense.append((chunks[idx], rank))
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

    def query(self, query_text: str, conversation_id: str = "") -> Dict[str, Any]:
        """
        Runs the full retrieval pipeline including routing, dual search,
        RRF merge, cross-encoder reranking, and confidence threshold gate.
        
        Returns a dict:
        {
            "query": query_text,
            "refused": bool,
            "confidence_score": float,
            "namespace_searched": str,
            "retrieved_chunks": List[Dict[str, Any]]
        }
        """
        # Step 1: Routing
        namespace = route_query(query_text)
        
        # Step 2 & 3: Dual Retrieval and RRF Merge (top 20 candidate set)
        candidates = self.retrieve(query_text, target_namespace=namespace, top_n=20)
        
        if not candidates:
            return {
                "query": query_text,
                "refused": True,
                "confidence_score": 0.0,
                "namespace_searched": namespace,
                "retrieved_chunks": []
            }

        # Step 4: Cross-Encoder Reranking
        # Truncate text to 512 chars for reranking — the reranker only needs
        # enough context to judge relevance, not full 16k-char statute sections.
        # Without truncation, the attention mask (~batch × seq² floats) exceeds
        # available RAM on CPU and crashes with DefaultCPUAllocator OOM.
        MAX_RERANK_CHARS = 512
        reranker = self.load_reranker()
        pairs = [[query_text, cand["text"][:MAX_RERANK_CHARS]] for cand in candidates]
        
        # Run cross-encoder scoring
        rerank_scores = reranker.predict(pairs, batch_size=4)
        
        # Add reranker scores and sort
        for cand, score in zip(candidates, rerank_scores):
            # Sigmoid normalization if scores are raw logits
            # bge-reranker-v2-m3 outputs logits. We convert to 0-1 range via sigmoid:
            prob = 1.0 / (1.0 + np.exp(-score))
            cand["relevance_score"] = float(prob)

        # Sort descending by relevance score
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_chunks = candidates[:5]
        
        # Top-1 score is the confidence score
        confidence = top_chunks[0]["relevance_score"] if top_chunks else 0.0
        
        # Step 5: Confidence Gate
        refused = confidence < self.confidence_threshold
        
        return {
            "query": query_text,
            "refused": refused,
            "confidence_score": confidence,
            "namespace_searched": namespace,
            "retrieved_chunks": [] if refused else top_chunks
        }
