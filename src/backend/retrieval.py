import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from src.backend.indexing import MedicalIndexManager, tokenize_for_bm25

import re

# ── Medical subject namespace router ─────────────────────────────────────

BASIC_SCIENCE_KEYWORDS = [
    "anatomy", "anatomical", "physiology", "physiological",
    "biochemistry", "biochemical", "genetics", "genetic",
    "enzyme", "cellular", "membrane", "action potential", "axon",
    "neurotransmitter", "dna", "rna", "protein synthesis",
    "mitosis", "meiosis", "embryology", "histology",
    "krebs cycle", "glycolysis", "oxidative phosphorylation",
    "ligament", "tendon", "cartilage", "muscle fiber", "myocyte",
]

PHARMACOLOGY_KEYWORDS = [
    "drug", "medication", "pharmacology", "pharmacokinetics",
    "pharmacodynamics", "antibiotic", "antiviral", "antimicrobial",
    "antifungal", "chemotherapy", "mechanism of action",
    "drug interaction", "toxicity", "adverse effect", "side effect",
    "penicillin", "amoxicillin", "statin", "beta blocker",
    "ace inhibitor", "diuretic", "receptor agonist", "receptor antagonist",
    "bacteria", "bacterium", "virus", "pathogen", "gram positive",
    "gram negative", "pathology", "neoplasm", "tumor", "carcinoma",
    "metastasis", "microbiology", "immunology", "antibody", "antigen",
]

CLINICAL_KEYWORDS = [
    "patient", "presents", "presentation", "diagnosis",
    "differential diagnosis", "treatment", "management", "prognosis",
    "complication", "emergency", "physician", "clinical", "symptom",
    "sign", "examination", "laboratory findings", "imaging", "surgery",
    "therapy", "first-line", "year-old", "comes to the",
]


def route_query(query: str) -> str:
    """
    Classifies the medical query to a subject namespace.
    Returns: 'basic_sciences', 'pharmacology', 'clinical_medicine', or 'all'.

    Most USMLE-style patient-scenario questions hit 'clinical_medicine';
    pure mechanism / drug questions hit 'pharmacology';
    structure / process questions hit 'basic_sciences'.
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

    basic_score    = get_score(BASIC_SCIENCE_KEYWORDS)
    pharma_score   = get_score(PHARMACOLOGY_KEYWORDS)
    clinical_score = get_score(CLINICAL_KEYWORDS)

    scores = {
        "basic_sciences":  basic_score,
        "pharmacology":    pharma_score,
        "clinical_medicine": clinical_score,
    }

    max_score = max(scores.values())
    if max_score == 0:
        return "all"

    tied = [ns for ns, s in scores.items() if s == max_score]
    if len(tied) > 1:
        return "all"

    return tied[0]

class MedicalRAGPipeline:
    def __init__(self, index_manager: MedicalIndexManager, reranker_name: str = "BAAI/bge-reranker-v2-m3", confidence_threshold: float = 0.65):
        self.index_manager = index_manager
        self.reranker_name = reranker_name
        self.confidence_threshold = confidence_threshold
        self.reranker = None

    def load_reranker(self):
        """Lazy loads the cross-encoder reranking model."""
        if self.reranker is None:
            print(f"Loading reranker model {self.reranker_name}...")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.reranker_name)
            print("Reranker loaded successfully.")
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
            return f"{meta['document_name']}_{meta['geography_iso']}_{hash(chunk['text'])}"

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
        reranker = self.load_reranker()
        pairs = [[query_text, cand["text"]] for cand in candidates]
        
        # Run cross-encoder scoring
        rerank_scores = reranker.predict(pairs)
        
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
