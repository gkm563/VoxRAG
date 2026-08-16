"""
build_index.py — One-time script to:
  1. Download MSMARCO-XI dataset from HuggingFace (FREE)
  2. Apply multi-strategy chunking
  3. Build and persist FAISS index

Run once before using main.py:
    python build_index.py
"""

import sys
import time

# Fix Windows console encoding for progress output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
from tqdm import tqdm

import config
from pipeline.chunker   import MultiStrategyChunker
from pipeline.retriever import FAISSRetriever


def load_passages() -> list[dict]:
    """Stream MSMARCO-XI passages and return as list of dicts."""
    print(f"[*] Loading dataset: {config.DATASET_NAME} (lang={config.DATASET_LANGUAGE}) ...")
    t0 = time.perf_counter()

    ds = load_dataset(
        config.DATASET_NAME,
        config.DATASET_LANGUAGE,
        split=config.DATASET_SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    passages = []
    cap = config.MAX_PASSAGES or float("inf")

    for row in tqdm(ds, total=config.MAX_PASSAGES, desc="Loading passages", unit="row"):
        # MSMARCO-XI schema: {"id", "passage", "query", "answers", ...}
        text = row.get("passage") or row.get("text") or ""
        if not text.strip():
            continue

        passages.append({
            "passage_id": str(row.get("id", len(passages))),
            "text":       text.strip(),
            "language":   config.DATASET_LANGUAGE,
        })

        if len(passages) >= cap:
            break

    elapsed = time.perf_counter() - t0
    print(f"[+] Loaded {len(passages):,} passages in {elapsed:.1f}s")
    return passages


def main():
    # ── 1. Load dataset ───────────────────────────────────────────────────────
    passages = load_passages()

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    print("\n[*] Chunking with multi-strategy chunker ...")
    chunker    = MultiStrategyChunker(use_semantic=True)
    all_chunks = chunker.chunk_passages(passages, verbose=True)
    print(f"[+] Total chunks produced: {len(all_chunks):,}")

    # Save chunks to disk
    MultiStrategyChunker.save(all_chunks, config.CHUNKS_PATH)

    # ── 3. Build FAISS index ──────────────────────────────────────────────────
    print("\n[*] Building FAISS index ...")
    retriever = FAISSRetriever()
    retriever.build(all_chunks, verbose=True)
    retriever.save(config.INDEX_PATH)

    print("\n[DONE] Index build complete!")
    print(f"   Chunks : {len(all_chunks):,}")
    print(f"   Vectors: {retriever.index.ntotal:,}")
    print(f"   Index  : {config.INDEX_PATH}")
    print(f"   Chunks : {config.CHUNKS_PATH}")
    print("\nYou can now run:  python main.py --mode demo")


if __name__ == "__main__":
    main()
