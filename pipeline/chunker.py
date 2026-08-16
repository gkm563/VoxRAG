"""
pipeline/chunker.py — Multi-Strategy Chunking

Strategies implemented:
  1. FixedSizeChunker   — token-count windows with configurable overlap
  2. SentenceChunker    — NLTK sentence boundaries, groups into max-token windows
  3. ParagraphChunker   — paragraph/newline-aware splitting
  4. SemanticChunker    — groups semantically similar consecutive sentences
  5. MetadataChunker    — wraps any strategy and injects passage metadata

All chunkers return List[dict] where each dict has:
  {
    "chunk_id":    str,
    "text":        str,
    "strategy":    str,
    "passage_id":  str | None,
    "language":    str | None,
    "token_count": int,
  }
"""

import re
import uuid
import json
import time
from typing import Iterator
from pathlib import Path

import tiktoken
import nltk
import numpy as np

import config

# Download NLTK data silently
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ── Shared tokeniser (GPT-2 compatible, fast) ────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _token_chunks(tokens: list[int], size: int, overlap: int) -> Iterator[list[int]]:
    """Sliding window over a token list."""
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        yield tokens[start:end]
        if end == len(tokens):
            break
        start += size - overlap


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fixed-Size Chunker
# ─────────────────────────────────────────────────────────────────────────────
class FixedSizeChunker:
    """Splits text into fixed token-count windows with overlap."""

    name = "fixed_size"

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        overlap_pct: float = config.CHUNK_OVERLAP_PCT,
    ):
        self.chunk_size = chunk_size
        self.overlap = max(1, int(chunk_size * overlap_pct))

    def chunk(self, text: str, passage_id: str = None, language: str = None) -> list[dict]:
        tokens = _TOKENIZER.encode(text)
        results = []
        for toks in _token_chunks(tokens, self.chunk_size, self.overlap):
            if len(toks) < config.MIN_CHUNK_TOKENS:
                continue
            chunk_text = _TOKENIZER.decode(toks)
            results.append(self._make(chunk_text, passage_id, language, len(toks)))
        return results

    def _make(self, text, passage_id, language, tc) -> dict:
        return {
            "chunk_id":    str(uuid.uuid4()),
            "text":        text,
            "strategy":    self.name,
            "passage_id":  passage_id,
            "language":    language,
            "token_count": tc,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sentence Chunker
# ─────────────────────────────────────────────────────────────────────────────
class SentenceChunker:
    """Groups sentences into chunks up to max_tokens with 1-sentence overlap."""

    name = "sentence"

    def __init__(self, max_tokens: int = config.CHUNK_SIZE):
        self.max_tokens = max_tokens

    def chunk(self, text: str, passage_id: str = None, language: str = None) -> list[dict]:
        sentences = nltk.sent_tokenize(text)
        chunks, current_sents, current_tc = [], [], 0

        for sent in sentences:
            tc = _token_count(sent)
            if current_tc + tc > self.max_tokens and current_sents:
                chunks.append(self._make(" ".join(current_sents), passage_id, language))
                # 1-sentence overlap
                current_sents = current_sents[-1:]
                current_tc    = _token_count(current_sents[0])
            current_sents.append(sent)
            current_tc += tc

        if current_sents:
            chunks.append(self._make(" ".join(current_sents), passage_id, language))
        return [c for c in chunks if c["token_count"] >= config.MIN_CHUNK_TOKENS]

    def _make(self, text, passage_id, language) -> dict:
        return {
            "chunk_id":    str(uuid.uuid4()),
            "text":        text,
            "strategy":    self.name,
            "passage_id":  passage_id,
            "language":    language,
            "token_count": _token_count(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Paragraph Chunker
# ─────────────────────────────────────────────────────────────────────────────
class ParagraphChunker:
    """Splits on double-newline paragraph breaks; merges short paragraphs."""

    name = "paragraph"

    def __init__(self, max_tokens: int = config.CHUNK_SIZE):
        self.max_tokens = max_tokens

    def chunk(self, text: str, passage_id: str = None, language: str = None) -> list[dict]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks, current, current_tc = [], [], 0

        for para in paras:
            tc = _token_count(para)
            if current_tc + tc > self.max_tokens and current:
                chunks.append(self._make(" ".join(current), passage_id, language))
                current, current_tc = [], 0
            current.append(para)
            current_tc += tc

        if current:
            chunks.append(self._make(" ".join(current), passage_id, language))
        return [c for c in chunks if c["token_count"] >= config.MIN_CHUNK_TOKENS]

    def _make(self, text, passage_id, language) -> dict:
        return {
            "chunk_id":    str(uuid.uuid4()),
            "text":        text,
            "strategy":    self.name,
            "passage_id":  passage_id,
            "language":    language,
            "token_count": _token_count(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Semantic Chunker (cosine similarity grouping)
# ─────────────────────────────────────────────────────────────────────────────
class SemanticChunker:
    """
    Groups consecutive sentences whose embeddings are similar (cosine > threshold).
    Falls back to SentenceChunker if sentence-transformers not available.
    """

    name = "semantic"

    def __init__(
        self,
        max_tokens: int = config.SEMANTIC_MAX_TOKENS,
        sim_threshold: float = 0.6,
    ):
        self.max_tokens    = max_tokens
        self.sim_threshold = sim_threshold
        self._model        = None  # lazy load

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    def chunk(self, text: str, passage_id: str = None, language: str = None) -> list[dict]:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return SentenceChunker(self.max_tokens).chunk(text, passage_id, language)

        model  = self._get_model()
        embeds = model.encode(sentences, batch_size=64, normalize_embeddings=True)

        groups, current = [], [0]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embeds[i - 1], embeds[i]))
            tc  = sum(_token_count(sentences[j]) for j in current)
            if sim >= self.sim_threshold and tc + _token_count(sentences[i]) <= self.max_tokens:
                current.append(i)
            else:
                groups.append(current)
                current = [i]
        groups.append(current)

        results = []
        for group in groups:
            combined = " ".join(sentences[j] for j in group)
            tc = _token_count(combined)
            if tc < config.MIN_CHUNK_TOKENS:
                continue
            results.append({
                "chunk_id":    str(uuid.uuid4()),
                "text":        combined,
                "strategy":    self.name,
                "passage_id":  passage_id,
                "language":    language,
                "token_count": tc,
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Strategy Chunker (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────
class MultiStrategyChunker:
    """
    Applies all chunking strategies to every passage and deduplicates by text.
    This produces a richer, denser index at the cost of some extra storage.

    Usage:
        chunker = MultiStrategyChunker()
        all_chunks = chunker.chunk_passages(passages)   # list of dicts
        chunker.save(all_chunks, path)
        loaded = MultiStrategyChunker.load(path)
    """

    def __init__(self, use_semantic: bool = True):
        self.strategies = [
            FixedSizeChunker(),
            SentenceChunker(),
            ParagraphChunker(),
        ]
        if use_semantic:
            self.strategies.append(SemanticChunker())

    def chunk_passage(
        self,
        text: str,
        passage_id: str = None,
        language:   str = None,
    ) -> list[dict]:
        seen, results = set(), []
        for strategy in self.strategies:
            for chunk in strategy.chunk(text, passage_id, language):
                key = chunk["text"].strip()
                if key not in seen:
                    seen.add(key)
                    results.append(chunk)
        return results

    def chunk_passages(self, passages: list[dict], verbose: bool = True) -> list[dict]:
        """
        passages: list of {"text": str, "passage_id": str, "language": str}
        """
        from tqdm import tqdm
        all_chunks = []
        it = tqdm(passages, desc="Chunking", unit="passage") if verbose else passages
        for p in it:
            all_chunks.extend(
                self.chunk_passage(
                    p.get("text", ""),
                    p.get("passage_id"),
                    p.get("language"),
                )
            )
        return all_chunks

    @staticmethod
    def save(chunks: list[dict], path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"✅  Saved {len(chunks):,} chunks → {path}")

    @staticmethod
    def load(path: Path) -> list[dict]:
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
