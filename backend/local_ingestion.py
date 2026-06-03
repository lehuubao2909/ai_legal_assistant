"""Embed the article-level corpus into ChromaDB using the local VN embedding model.

Reads `data/corpus_articles.jsonl` (from build_corpus.py), embeds each article
with AITeamVN/Vietnamese_Embedding, upserts into a fresh ChromaDB collection.

Run:  python local_ingestion.py
"""
import json
import os
import sys

import chromadb
import torch
from sentence_transformers import SentenceTransformer

import local_models_config as cfg

BATCH = 64


class LocalLegalIngestion:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=cfg.CHROMA_PATH)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model '{cfg.EMBEDDING_MODEL}' on {self.device} ...")
        self.model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=self.device)
        self.collection = None

    def reset_collection(self):
        try:
            self.client.delete_collection(cfg.CHROMA_COLLECTION)
            print("Purged old collection.")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=cfg.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts):
        """Encode a batch; fail loud (no silent zero-vectors)."""
        vecs = self.model.encode(
            texts, batch_size=BATCH, show_progress_bar=False, normalize_embeddings=True
        )
        return [v.tolist() for v in vecs]

    def ingest(self):
        if not os.path.exists(cfg.CORPUS_JSONL):
            sys.exit(f"Corpus not found: {cfg.CORPUS_JSONL}\nRun build_corpus.py first.")

        rows = []
        missing_code = 0
        with open(cfg.CORPUS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("doc_number"):          # validate join key
                    missing_code += 1
                    continue
                rows.append(r)

        if not rows:
            sys.exit("No valid articles to ingest (all missing doc_number?).")
        if missing_code:
            print(f"⚠ Skipped {missing_code} articles missing doc_number (unscorable).")

        self.reset_collection()
        print(f"Ingesting {len(rows)} articles in batches of {BATCH} ...")

        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            docs = [f"{r['title']}\n{r['text']}" for r in chunk]
            embeddings = self._embed(docs)
            self.collection.upsert(
                ids=[r["id"] for r in chunk],
                embeddings=embeddings,
                documents=docs,
                metadatas=[{
                    "id": r["id"],
                    "doc_number": r["doc_number"],
                    "clean_name": r.get("clean_name", ""),
                    "legal_type": r.get("legal_type", ""),
                    "article": r.get("article", ""),
                    "title": r.get("title", ""),
                    "text": r["text"],
                    "source_url": r.get("source_url", ""),
                } for r in chunk],
            )
            done = min(i + BATCH, len(rows))
            if done % (BATCH * 10) == 0 or done == len(rows):
                print(f"  embedded {done}/{len(rows)}")

        print(f"\nDone. Collection '{cfg.CHROMA_COLLECTION}' now has {self.collection.count()} articles.")


if __name__ == "__main__":
    LocalLegalIngestion().ingest()
