"""
server.py — VoxRAG FastAPI Backend
Serves the HTML dashboard and handles all API calls.

Run: python server.py
Then open: http://localhost:8000
"""

import json, time, os, tempfile, traceback
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# ── Pipeline globals (loaded once at startup) ─────────────────────────────────
_harness   = None
_stt       = None
_stats     = {"chunks": 0, "vectors": 0, "ready": False}
_latencies = []   # rolling list of total_latency_ms values


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _harness, _stt, _stats
    try:
        from pipeline.retriever  import FAISSRetriever
        from pipeline.generator  import AnswerGenerator
        from pipeline.guardrails import Guardrails
        from pipeline.harness    import RAGHarness
        from pipeline.stt        import SpeechToText

        print("[*] Loading VoxRAG pipeline ...")
        retriever = FAISSRetriever.load(config.INDEX_PATH)
        generator = AnswerGenerator()
        guardrails= Guardrails()
        _harness  = RAGHarness(retriever, generator, guardrails)
        _stt      = SpeechToText(mode="sarvam")
        _stats    = {
            "chunks":  len(retriever.chunks),
            "vectors": retriever.index.ntotal,
            "ready":   True,
        }
        print(f"[+] Pipeline ready | vectors={_stats['vectors']:,}")
    except Exception as e:
        print(f"[!] Pipeline not loaded: {e} — run build_index.py first")
    yield


app = FastAPI(title="VoxRAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Serve static files (index.html, etc.)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_ready": _stats["ready"]}


@app.get("/api/stats")
async def stats():
    return {
        "dataset":  "MSMARCO-XI",
        "chunks":   f"{_stats['chunks']:,}" if _stats["ready"] else "Building...",
        "vectors":  f"{_stats['vectors']:,}" if _stats["ready"] else "Building...",
        "model":    config.GROQ_MODEL,
        "embed":    config.EMBED_MODEL.split("/")[-1],
        "ready":    _stats["ready"],
    }


class TextQuery(BaseModel):
    query: str
    top_k: int = config.TOP_K


@app.post("/api/query/text")
async def query_text(body: TextQuery):
    if not _harness:
        raise HTTPException(503, "Pipeline not ready. Run build_index.py first.")

    from pipeline.harness import PipelineInput
    inp = PipelineInput(query=body.query, top_k=body.top_k)
    out = _harness.run(inp)
    _latencies.append(out.total_latency_ms)

    # Also retrieve chunks for display
    chunks_display = []
    if not out.blocked and _harness:
        try:
            retrieved, _ = _harness.retriever.search(body.query, body.top_k)
            for c in retrieved:
                chunks_display.append({
                    "id":       c["chunk_id"][:16] + "...",
                    "text":     c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                    "score":    round(c.get("score", 0), 3),
                    "strategy": c.get("strategy", "unknown"),
                    "passage":  c.get("passage_id", "—"),
                })
        except Exception:
            pass

    return {
        "query":      out.query,
        "answer":     out.answer,
        "blocked":    out.blocked,
        "reason":     out.block_reason,
        "confidence": round(out.confidence, 2),
        "grounded":   out.grounded,
        "sources":    out.sources[:5],
        "chunks":     chunks_display,
        "latency":    {k: round(v, 1) for k, v in out.latency.items()},
        "total_ms":   round(out.total_latency_ms, 1),
        "guardrails": {
            "off_topic":     not out.blocked,
            "safety":        not out.blocked,
            "hallucination": out.grounded,
            "grounded":      out.grounded,
        },
    }


@app.post("/api/query/voice")
async def query_voice(audio: UploadFile = File(...)):
    if not _harness or not _stt:
        raise HTTPException(503, "Pipeline not ready.")

    # Save uploaded audio
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        tmp_path = f.name

    try:
        transcript, stt_ms = _stt.from_file(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(400, f"STT failed: {e}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

    from pipeline.harness import PipelineInput
    inp = PipelineInput(query=transcript, stt_latency=stt_ms)
    out = _harness.run(inp)
    _latencies.append(out.total_latency_ms)

    chunks_display = []
    if not out.blocked:
        try:
            retrieved, _ = _harness.retriever.search(transcript, config.TOP_K)
            for c in retrieved:
                chunks_display.append({
                    "id":       c["chunk_id"][:16] + "...",
                    "text":     c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                    "score":    round(c.get("score", 0), 3),
                    "strategy": c.get("strategy", "unknown"),
                    "passage":  c.get("passage_id", "—"),
                })
        except Exception:
            pass

    return {
        "transcript": transcript,
        "stt_ms":     round(stt_ms, 1),
        "query":      out.query,
        "answer":     out.answer,
        "blocked":    out.blocked,
        "reason":     out.block_reason,
        "confidence": round(out.confidence, 2),
        "grounded":   out.grounded,
        "sources":    out.sources[:5],
        "chunks":     chunks_display,
        "latency":    {k: round(v, 1) for k, v in out.latency.items()},
        "total_ms":   round(out.total_latency_ms, 1),
        "guardrails": {
            "off_topic":     not out.blocked,
            "safety":        not out.blocked,
            "hallucination": out.grounded,
            "grounded":      out.grounded,
        },
    }


@app.get("/api/analytics")
async def analytics():
    import numpy as np
    lats = _latencies[-100:] if _latencies else []

    def pct(arr, p):
        if not arr: return 0
        return round(float(np.percentile(arr, p)), 1)

    return {
        "p50":     pct(lats, 50),
        "p70":     pct(lats, 70),
        "p100":    pct(lats, 100),
        "mean":    pct(lats, 50),
        "count":   len(lats),
        "history": lats[-30:],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
