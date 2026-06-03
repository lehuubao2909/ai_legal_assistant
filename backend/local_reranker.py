"""Local cross-encoder reranker (highest-ROI retrieval stage).

Wraps AITeamVN/Vietnamese_Reranker. Given a query and candidate docs (from
hybrid RRF), scores each (query, doc) pair and returns the top-k re-sorted.
"""
from typing import List, Dict, Any

import torch
from sentence_transformers import CrossEncoder

import local_models_config as cfg


class LocalReranker:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading reranker '{cfg.RERANKER_MODEL}' on {self.device} ...")
        # max_length 2048 fits long legal articles; trust_remote_code off (BGE-based).
        self.model = CrossEncoder(cfg.RERANKER_MODEL, max_length=2048, device=self.device)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k docs re-sorted by cross-encoder relevance score."""
        if not docs:
            return []
        pairs = [[query, f"{d.get('title','')} {d.get('text','')}"] for d in docs]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_k]
