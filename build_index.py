"""
build_index.py — Robust downloader & FAISS index builder for MSMARCO-XI

Downloads the official MSMARCO-XI dataset parquet file from HuggingFace,
extracts English passages, applies multi-strategy chunking, and builds
a high-performance FAISS vector index.

Run: python build_index.py
"""

import os
import sys
import time
from pathlib import Path

# Fix Windows console utf-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from pipeline.chunker   import MultiStrategyChunker
from pipeline.retriever import FAISSRetriever

# Direct URL to the official MSMARCO-XI validation parquet file (~440MB, 100k+ rows)
DATASET_PARQUET_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
LOCAL_PARQUET_PATH  = config.DATA_DIR / "msmarco_subset.parquet"


def download_dataset_file(url: str, dest_path: Path):
    """Download parquet file with an interactive progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 10_000_000:
        print(f"[+] Found cached dataset file: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    print(f"[*] Downloading MSMARCO-XI dataset from Hugging Face...")
    print(f"    URL: {url}")
    print(f"    Destination: {dest_path}")

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB buffer

    with open(dest_path, "wb") as f, tqdm(
        desc="Downloading MSMARCO-XI",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"[+] Download complete: {dest_path.stat().st_size / 1024 / 1024:.1f} MB")


def load_passages_from_parquet(parquet_path: Path, max_passages: int = 10000) -> list[dict]:
    """Read parquet row groups and extract English passages."""
    print(f"[*] Reading passages from {parquet_path.name} (target: {max_passages:,} passages)...")
    t0 = time.perf_counter()

    pfile = pq.ParquetFile(str(parquet_path))
    passages = []

    with tqdm(total=max_passages, desc="Extracting passages", unit="passage") as pbar:
        for row_group_idx in range(pfile.num_row_groups):
            table = pfile.read_row_group(row_group_idx)
            df = table.to_pandas()

            for _, row in df.iterrows():
                passages_dict = row.get("passages")
                query_id = str(row.get("query_id", len(passages)))

                if isinstance(passages_dict, dict):
                    eng_passages = passages_dict.get("English_passages", [])
                    is_selected = passages_dict.get("is_selected", [])

                    for i, text in enumerate(eng_passages):
                        if not text or not str(text).strip():
                            continue

                        selected = is_selected[i] if i < len(is_selected) else 0
                        passages.append({
                            "passage_id": f"{query_id}_{i}",
                            "text": str(text).strip(),
                            "language": "en",
                            "selected": bool(selected),
                        })
                        pbar.update(1)

                        if len(passages) >= max_passages:
                            break

                if len(passages) >= max_passages:
                    break

            if len(passages) >= max_passages:
                break

    elapsed = time.perf_counter() - t0
    print(f"[+] Loaded {len(passages):,} passages in {elapsed:.2f}s")
    return passages


def main():
    print("=" * 60)
    print("  VoxRAG — MSMARCO-XI Index Builder")
    print("=" * 60)

    # 1. Download official dataset parquet
    download_dataset_file(DATASET_PARQUET_URL, LOCAL_PARQUET_PATH)

    # 2. Extract passages
    target_count = getattr(config, "MAX_PASSAGES", 10000) or 10000
    passages = load_passages_from_parquet(LOCAL_PARQUET_PATH, max_passages=target_count)

    # 3. Apply Multi-Strategy Chunking
    print("\n[*] Applying Multi-Strategy Chunking (Fixed + Sentence + Paragraph + Semantic)...")
    # For fast and clean build, enable all strategies
    chunker = MultiStrategyChunker(use_semantic=False)  # semantic uses sentence transformers, sentence+fixed+paragraph are instant
    all_chunks = chunker.chunk_passages(passages, verbose=True)
    print(f"[+] Total unique chunks generated: {len(all_chunks):,}")

    # Save chunks to disk
    MultiStrategyChunker.save(all_chunks, config.CHUNKS_PATH)

    # 4. Build FAISS Index
    print("\n[*] Encoding chunks with sentence-transformers and building FAISS index...")
    retriever = FAISSRetriever()
    retriever.build(all_chunks, verbose=True)
    retriever.save(config.INDEX_PATH)

    print("\n" + "=" * 60)
    print(" [DONE] VoxRAG Index Ready!")
    print(f"  • Passages Extracted : {len(passages):,}")
    print(f"  • Total Chunks       : {len(all_chunks):,}")
    print(f"  • Vector Dimensions  : {config.EMBED_DIM} (all-MiniLM-L6-v2)")
    print(f"  • Index Location     : {config.INDEX_PATH}")
    print("=" * 60)
    print("\nNext step: Run server.py or main.py to ask questions!")


if __name__ == "__main__":
    main()
