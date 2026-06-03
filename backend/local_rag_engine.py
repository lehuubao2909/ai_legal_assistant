"""Offline hybrid retrieval engine for the competition pipeline.

Pipeline:  dense (Vietnamese_Embedding, raw query) ‖ sparse (BM25)
        →  RRF fusion  →  cross-encoder rerank  →  top-K Điều.

Temporal filtering / legislative-hierarchy re-sorting / slang expansion were
removed: they hurt F2 recall and add no scoring value for this task (KISS).
"""
import re
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import local_models_config as cfg
from local_reranker import LocalReranker

# Optional Vietnamese word segmentation improves BM25 (dense model uses raw text).
try:
    from pyvi import ViTokenizer
    _HAS_PYVI = True
except ImportError:
    _HAS_PYVI = False


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    if _HAS_PYVI:
        text = ViTokenizer.tokenize(text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


class LocalLegalRAGEngine:
    def __init__(self, use_reranker: bool = True, candidate_k: int = 50):
        self.candidate_k = candidate_k
        self.client = chromadb.PersistentClient(path=cfg.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model '{cfg.EMBEDDING_MODEL}' on {self.device} ...")
        self.model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=self.device)

        self.reranker = LocalReranker() if use_reranker else None

        self.bm25 = None
        self.corpus_ids: List[str] = []
        self.corpus_meta: List[Dict[str, Any]] = []
        self._build_bm25()

    def _build_bm25(self):
        count = self.collection.count()
        if count == 0:
            print("⚠ ChromaDB empty — run build_corpus.py + local_ingestion.py first.")
            return
        print(f"Building BM25 index over {count} articles ...")
        rec = self.collection.get(include=["metadatas", "documents"])
        self.corpus_ids = rec["ids"]
        self.corpus_meta = rec["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(d) for d in rec["documents"]])
        print(f"BM25 ready (pyvi={'on' if _HAS_PYVI else 'off'}).")

    def _dense(self, query: str) -> List[str]:
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        n = min(self.candidate_k, self.collection.count())   # avoid n_results > corpus size
        res = self.collection.query(query_embeddings=[vec], n_results=n)
        return res.get("ids", [[]])[0]

    def _sparse(self, query: str) -> List[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:self.candidate_k]
        return [self.corpus_ids[i] for i in order if scores[i] > 0.0]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k article metadata dicts for the query."""
        if self.collection.count() == 0:
            return []
        if not self.bm25 or len(self.corpus_ids) != self.collection.count():
            self._build_bm25()

        dense_ids = self._dense(query)
        sparse_ids = self._sparse(query)

        # Reciprocal Rank Fusion (k=60).
        rrf: Dict[str, float] = {}
        for ids in (dense_ids, sparse_ids):
            for rank, _id in enumerate(ids):
                rrf[_id] = rrf.get(_id, 0.0) + 1.0 / (60.0 + rank + 1)
        if not rrf:
            return []

        fused = sorted(rrf, key=rrf.get, reverse=True)[:self.candidate_k]
        id2meta = dict(zip(self.corpus_ids, self.corpus_meta))
        candidates = [dict(id2meta[i]) for i in fused if i in id2meta]

        # Cross-encoder rerank (or RRF order if disabled).
        ranked = self.reranker.rerank(query, candidates, top_k=top_k) if self.reranker \
            else candidates[:top_k]

        # Dedup by (doc_number, article), preserve order.
        seen, out = set(), []
        for d in ranked:
            key = (d.get("doc_number", ""), d.get("article", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out


if __name__ == "__main__":
    eng = LocalLegalRAGEngine()
    q = "Người lao động tự ý bỏ việc bao nhiêu ngày thì bị sa thải?"
    print(f"\nQuery: {q}\n")
    for i, r in enumerate(eng.retrieve(q), 1):
        score = r.get("rerank_score")
        print(f"[{i}] {r.get('doc_number')} | {r.get('article')} | {r.get('clean_name')}")
        print(f"    score={score:.3f} " if score is not None else "    ", end="")
        print(r.get("title", "")[:80])
