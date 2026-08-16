"""
build_index.py — Downloads MSMARCO-XI, chunks passages, builds FAISS index.
Dataset schema: passages.English_passages (list of strings per row)
Run once: python build_index.py
"""

import sys, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
from tqdm import tqdm

import config
from pipeline.chunker   import MultiStrategyChunker
from pipeline.retriever import FAISSRetriever


def load_passages() -> list[dict]:
    print(f"[*] Loading dataset: {config.DATASET_NAME} (config=default, split=train) ...")
    t0 = time.perf_counter()

    ds = load_dataset(
        config.DATASET_NAME,
        "default",
        split=config.DATASET_SPLIT,
        streaming=True,
    )

    passages = []
    cap = config.MAX_PASSAGES or float("inf")

    for row in tqdm(ds, total=config.MAX_PASSAGES, desc="Loading", unit="row"):
        # Schema: passages = {English_passages: [...], Translated_passages: [...], is_selected: [...]}
        eng_passages = row.get("passages", {}).get("English_passages", [])
        is_selected  = row.get("passages", {}).get("is_selected", [])
        query_id     = str(row.get("query_id", len(passages)))

        for i, (text, selected) in enumerate(zip(eng_passages, is_selected)):
            text = text.strip()
            if not text:
                continue
            passages.append({
                "passage_id": f"{query_id}_{i}",
                "text":       text,
                "language":   "en",
                "selected":   bool(selected),
            })
            if len(passages) >= cap:
                break

        if len(passages) >= cap:
            break

    elapsed = time.perf_counter() - t0
    print(f"[+] Loaded {len(passages):,} passages in {elapsed:.1f}s")
    return passages


def main():
    passages = load_passages()

    print("\n[*] Chunking with multi-strategy chunker ...")
    chunker    = MultiStrategyChunker(use_semantic=True)
    all_chunks = chunker.chunk_passages(passages, verbose=True)
    print(f"[+] Total chunks: {len(all_chunks):,}")

    MultiStrategyChunker.save(all_chunks, config.CHUNKS_PATH)

    print("\n[*] Building FAISS index ...")
    retriever = FAISSRetriever()
    retriever.build(all_chunks, verbose=True)
    retriever.save(config.INDEX_PATH)

    print(f"\n[DONE] Chunks={len(all_chunks):,} | Vectors={retriever.index.ntotal:,}")
    print("Run: python server.py")


if __name__ == "__main__":
    main()
