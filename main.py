"""
main.py — VoxRAG Entry Point

Modes:
  --mode demo    : Run a set of benchmark demo queries
  --mode text    : Accept typed query from stdin
  --mode voice   : Record from mic -> transcribe -> RAG pipeline -> print answer
  --query "..."  : Run a single query directly (text mode)

Usage:
  python main.py --mode demo
  python main.py --mode text
  python main.py --query "What is MSMARCO?"
"""

import sys
import argparse
import json

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from pipeline.retriever  import FAISSRetriever
from pipeline.generator  import AnswerGenerator
from pipeline.guardrails import Guardrails
from pipeline.harness    import RAGHarness, PipelineInput


DEMO_QUERIES = [
    "What is the MSMARCO dataset used for?",
    "How does information retrieval work?",
    "What is the difference between BM25 and dense retrieval?",
    "What languages does MSMARCO-XI cover?",
    "How do transformer models generate text?",
]


def load_harness() -> RAGHarness:
    """Load all pipeline components and return a ready harness."""
    print("[*] Loading VoxRAG pipeline ...")

    retriever  = FAISSRetriever.load(config.INDEX_PATH)
    generator  = AnswerGenerator()
    guardrails = Guardrails(embed_model=retriever.model)
    harness    = RAGHarness(retriever, generator, guardrails)

    print(f"[+] Pipeline ready! Indexed vectors: {retriever.index.ntotal:,}\n")
    return harness


def run_query(harness: RAGHarness, query: str, stt_latency: float = 0.0):
    """Run a single query through the pipeline and pretty-print results."""
    inp = PipelineInput(query=query, stt_latency=stt_latency)
    out = harness.run(inp)

    print("\n" + "=" * 60)
    print(f" Query     : {out.query}")
    print("=" * 60)

    if out.blocked:
        print(f"[BLOCKED] Reason: {out.block_reason}")
    else:
        print(f"[ANSWER]  : {out.answer}")
        print(f"[CONF]    : {out.confidence:.0%}")
        print(f"[GROUNDED]: {'Yes' if out.grounded else 'No'}")
        if out.sources:
            print(f"[SOURCES] : {', '.join(str(s) for s in out.sources[:3])}")

    print("\n[LATENCY BREAKDOWN]")
    for stage, ms in out.latency.items():
        bar = "#" * max(1, int(ms / 5))
        print(f"   {stage:<20} {ms:7.1f} ms  {bar}")

    total = out.total_latency_ms
    target_ok = "PASS (< 200ms)" if total < config.PIPELINE_TIMEOUT_MS else "OVER TARGET"
    print(f"\n   {'TOTAL':<20} {total:7.1f} ms  [{target_ok}]")
    print("=" * 60 + "\n")
    return out


def mode_demo(harness: RAGHarness):
    print("[*] Running demo queries ...\n")
    for query in DEMO_QUERIES:
        run_query(harness, query)


def mode_text(harness: RAGHarness):
    print("[*] Text mode — type your query (Ctrl+C to exit)\n")
    while True:
        try:
            query = input("Query: ").strip()
            if query:
                run_query(harness, query)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def mode_voice(harness: RAGHarness):
    from pipeline.stt import SpeechToText
    stt = SpeechToText()

    while True:
        input("Press ENTER to start recording (Ctrl+C to exit) ...")
        try:
            query, stt_ms = stt.from_mic()
            print(f"\n[TRANSCRIPT]: \"{query}\" ({stt_ms:.0f}ms)")
            run_query(harness, query, stt_latency=stt_ms)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"[ERROR]: {e}")


def main():
    parser = argparse.ArgumentParser(description="VoxRAG — Voice-Enabled RAG Pipeline")
    parser.add_argument(
        "--mode", choices=["voice", "demo", "text"], default="demo",
        help="Input mode (default: demo)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single query and exit"
    )
    args = parser.parse_args()

    harness = load_harness()

    if args.query:
        run_query(harness, args.query)
    elif args.mode == "voice":
        mode_voice(harness)
    elif args.mode == "demo":
        mode_demo(harness)
    elif args.mode == "text":
        mode_text(harness)


if __name__ == "__main__":
    main()
