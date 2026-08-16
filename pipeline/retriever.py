"""
pipeline/retriever.py — FAISS Vector DB Retrieval

Builds a flat L2 / inner-product index over chunk embeddings.
Supports:
  - build()   : encode chunks → add to FAISS index → persist to disk
  - load()    : load index + metadata from disk
  - search()  : top-k nearest neighbours for a query string
"""

import json
import time
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

import config


class FAISSRetriever:
    """
    Embedding + FAISS index for fast chunk retrieval.

    Usage (index already built):
        retriever = FAISSRetriever.load()
        results, latency_ms = retriever.search("What is MSMARCO?")
    """

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None   = None   # inner-product (cosine after normalise)
        self.chunks: list[dict]                = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict], verbose: bool = True) -> None:
        """Encode all chunks and build FAISS index."""
        self.chunks = chunks
        self._ensure_model()

        texts = [c["text"] for c in chunks]
        if verbose:
            print(f"🔢  Encoding {len(texts):,} chunks …")

        embeddings = self.model.encode(
            texts,
            batch_size=512,
            normalize_embeddings=True,   # cosine via inner product
            show_progress_bar=verbose,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(config.EMBED_DIM)
        self.index.add(embeddings)

        if verbose:
            print(f"[+] FAISS index built: {self.index.ntotal:,} vectors")

    def save(self, index_path: Path = config.INDEX_PATH) -> None:
        """Persist FAISS index + chunk metadata."""
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path / "index.faiss"))

        meta_path = index_path / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"[+] Saved index -> {index_path}")

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, index_path: Path = config.INDEX_PATH) -> "FAISSRetriever":
        """Load a previously saved index."""
        index_path = Path(index_path)
        instance   = cls()
        instance._ensure_model()

        instance.index = faiss.read_index(str(index_path / "index.faiss"))

        with open(index_path / "metadata.jsonl", encoding="utf-8") as f:
            instance.chunks = [json.loads(l) for l in f if l.strip()]

        print(f"[+] Loaded FAISS index: {instance.index.ntotal:,} vectors")
        return instance

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K,
    ) -> tuple[list[dict], float]:
        """
        Return top_k most relevant chunks + retrieval latency in ms.

        Returns:
            (results, latency_ms)
            results: list of chunk dicts, each with an added 'score' field
        """
        t0 = time.perf_counter()
        self._ensure_model()

        q_embed = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_embed, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results, latency_ms

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(config.EMBED_MODEL)
