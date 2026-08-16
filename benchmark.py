"""
benchmark.py — Latency Analytics for VoxRAG

Runs the pipeline over N test queries and reports:
  - P50 (median) latency
  - P70 latency
  - P100 (max) latency
  - Per-stage breakdown

Results saved to benchmark_results.json

Usage:
    python benchmark.py
    python benchmark.py --n 100
    python benchmark.py --output my_results.json
"""

import argparse
import json
import time
import statistics
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from pipeline.retriever  import FAISSRetriever
from pipeline.generator  import AnswerGenerator
from pipeline.guardrails import Guardrails
from pipeline.harness    import RAGHarness, PipelineInput


# ── Extended benchmark query set ──────────────────────────────────────────────
BENCHMARK_QUERIES = [
    # Factual
    "What is the MSMARCO dataset?",
    "How does BM25 work in information retrieval?",
    "What is a vector database?",
    "Explain dense passage retrieval.",
    "What is the difference between TF-IDF and BM25?",
    "How does FAISS work?",
    "What is RAG in machine learning?",
    "What is sentence-transformers?",
    "How does cosine similarity work?",
    "What is the purpose of chunking in RAG?",
    # Reasoning
    "Why is retrieval augmented generation better than pure LLM?",
    "What are the limitations of fixed-size chunking?",
    "How do semantic embeddings improve search?",
    "Why is overlap important in text chunking?",
    "What happens when an LLM hallucinates?",
    # Edge cases (should be blocked/flagged)
    "Tell me a joke",
    "What is the capital of Mars?",
    "What is 2 + 2?",
    # More QA
    "What languages are in MSMARCO-XI?",
    "How is passage retrieval evaluated?",
    "What is MRR in information retrieval?",
    "Explain the concept of inverted index.",
    "What is NDCG?",
    "How does cross-encoder reranking work?",
    "What is bi-encoder architecture?",
    "How do you handle multilingual queries in RAG?",
    "What is semantic search?",
    "How does chunking affect retrieval quality?",
    "What is sentence boundary detection?",
    "How is a FAISS index built?",
    # More varied
    "What are common evaluation metrics for QA systems?",
    "How does retrieval work in open-domain QA?",
    "What is the role of context in language models?",
    "How do you prevent hallucinations in RAG?",
    "What are guardrails in AI systems?",
    "How does re-ranking improve RAG pipelines?",
    "What is hybrid search?",
    "What is approximate nearest neighbour search?",
    "How does HNSW differ from flat FAISS?",
    "What is quantization in vector databases?",
    "What is a passage vs a document?",
    "How does query expansion work?",
    "What is HyDE in RAG?",
    "What is the purpose of a system prompt?",
    "How does temperature affect LLM output?",
    "What is prompt engineering?",
    "How does fine-tuning differ from RAG?",
    "What is knowledge grounding?",
    "What is the difference between P50 and P99 latency?",
    "How is latency measured in ML pipelines?",
]


def run_benchmark(harness: RAGHarness, n: int) -> list[dict]:
    queries = (BENCHMARK_QUERIES * ((n // len(BENCHMARK_QUERIES)) + 1))[:n]
    records = []

    print(f"\n🏁  Running benchmark over {n} queries …\n")
    for query in tqdm(queries, desc="Benchmarking", unit="query"):
        inp = PipelineInput(query=query)
        out = harness.run(inp)
        records.append({
            "query":            query,
            "blocked":          out.blocked,
            "total_latency_ms": out.total_latency_ms,
            "latency":          out.latency,
            "confidence":       out.confidence,
            "grounded":         out.grounded,
        })

    return records


def compute_percentiles(values: list[float]) -> dict:
    arr = sorted(values)
    n   = len(arr)
    def pct(p):
        idx = int(np.ceil(p / 100 * n)) - 1
        return arr[max(0, idx)]
    return {
        "p50":  pct(50),
        "p70":  pct(70),
        "p90":  pct(90),
        "p100": max(arr),
        "mean": statistics.mean(arr),
        "min":  min(arr),
    }


def print_report(records: list[dict]):
    totals = [r["total_latency_ms"] for r in records]
    pcts   = compute_percentiles(totals)

    print("\n" + "═" * 55)
    print("  📊  VoxRAG Latency Benchmark Report")
    print("═" * 55)
    print(f"  Queries run   : {len(records)}")
    print(f"  Blocked       : {sum(r['blocked'] for r in records)}")
    print(f"  Grounded      : {sum(r.get('grounded', False) for r in records)}")
    print()
    print(f"  P50  (median) : {pcts['p50']:7.1f} ms")
    print(f"  P70           : {pcts['p70']:7.1f} ms")
    print(f"  P90           : {pcts['p90']:7.1f} ms")
    print(f"  P100 (max)    : {pcts['p100']:7.1f} ms")
    print(f"  Mean          : {pcts['mean']:7.1f} ms")
    print(f"  Min           : {pcts['min']:7.1f} ms")
    print()

    # Per-stage breakdown (average across non-blocked runs)
    valid = [r for r in records if not r["blocked"]]
    if valid:
        print("  Per-stage averages:")
        all_stages = set()
        for r in valid:
            all_stages.update(r["latency"].keys())
        all_stages.discard("total")

        for stage in sorted(all_stages):
            vals = [r["latency"][stage] for r in valid if stage in r["latency"]]
            if vals:
                avg = statistics.mean(vals)
                bar = "█" * max(1, int(avg / 5))
                print(f"    {stage:<22} {avg:6.1f} ms  {bar}")

    target_ok = pcts["p100"] < config.PIPELINE_TIMEOUT_MS
    status    = "✅  UNDER 200ms target" if target_ok else "⚠️  EXCEEDS 200ms target"
    print(f"\n  {status}")
    print("═" * 55 + "\n")
    return pcts


def main():
    parser = argparse.ArgumentParser(description="VoxRAG Latency Benchmark")
    parser.add_argument("--n",      type=int,  default=50,                       help="Number of queries (default: 50)")
    parser.add_argument("--output", type=str,  default="benchmark_results.json", help="Output file")
    args = parser.parse_args()

    print("⚙️   Loading pipeline …")
    retriever  = FAISSRetriever.load(config.INDEX_PATH)
    generator  = AnswerGenerator()
    guardrails = Guardrails()
    harness    = RAGHarness(retriever, generator, guardrails)
    print("✅  Pipeline loaded.\n")

    records = run_benchmark(harness, args.n)
    pcts    = print_report(records)

    # Save results
    output = {
        "num_queries": len(records),
        "percentiles": pcts,
        "records":     records,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾  Results saved → {args.output}")


if __name__ == "__main__":
    main()
