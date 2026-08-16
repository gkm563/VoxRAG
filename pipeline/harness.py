"""
pipeline/harness.py — RAG Orchestration Harness

Responsibilities:
  - Structured input/output (Pydantic models)
  - Retry logic with exponential back-off (tenacity)
  - Per-stage latency tracking
  - Error recovery (partial results, graceful degradation)
  - Pipeline timeout enforcement
  - Tool-call style stage dispatch
"""

import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel

import config
from pipeline.retriever  import FAISSRetriever
from pipeline.generator  import AnswerGenerator, RAGAnswer
from pipeline.guardrails import Guardrails, GuardrailResult


# ── Pipeline I/O schemas ──────────────────────────────────────────────────────
class PipelineInput(BaseModel):
    query:        str
    top_k:        int  = config.TOP_K
    stt_latency:  float = 0.0   # pre-filled by caller when using voice


class PipelineOutput(BaseModel):
    query:          str
    answer:         str
    confidence:     float
    sources:        list[str]
    grounded:       bool
    blocked:        bool   = False
    block_reason:   str    = ""
    latency: dict[str, float] = {}   # stage → ms
    total_latency_ms: float  = 0.0


# ── Stage result helper ────────────────────────────────────────────────────────
@dataclass
class StageResult:
    data:       object
    latency_ms: float
    ok:         bool    = True
    error:      str     = ""


# ── Harness ───────────────────────────────────────────────────────────────────
class RAGHarness:
    """
    Orchestrates the full RAG pipeline with retries, structured I/O, and
    per-stage latency tracking.

    Usage:
        harness = RAGHarness(retriever, generator, guardrails)
        output  = harness.run(PipelineInput(query="What is MSMARCO?"))
        print(output.answer)
    """

    def __init__(
        self,
        retriever:  FAISSRetriever,
        generator:  AnswerGenerator,
        guardrails: Guardrails,
    ):
        self.retriever  = retriever
        self.generator  = generator
        self.guardrails = guardrails

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, inp: PipelineInput) -> PipelineOutput:
        """Execute the full pipeline for a single query."""
        wall_start = time.perf_counter()
        latency: dict[str, float] = {}

        if inp.stt_latency > 0:
            latency["stt"] = inp.stt_latency

        # ── Stage 1: Input guardrail ──────────────────────────────────────────
        guard_result: GuardrailResult = self._stage(
            "input_guardrail",
            lambda: self.guardrails.check_input(inp.query),
            latency,
        ).data

        if not guard_result.allowed:
            return self._blocked(inp.query, guard_result.reason, latency, wall_start)

        # ── Stage 2: Retrieval (with retry) ──────────────────────────────────
        retrieval_stage = self._stage_with_retry(
            "retrieval",
            lambda: self.retriever.search(inp.query, inp.top_k),
            latency,
        )
        if not retrieval_stage.ok:
            return self._error_output(inp.query, retrieval_stage.error, latency, wall_start)

        chunks, _ = retrieval_stage.data   # (chunks, inner_latency) from retriever

        # ── Stage 3: Answer generation (with retry) ───────────────────────────
        gen_stage = self._stage_with_retry(
            "generation",
            lambda: self.generator.generate(inp.query, chunks),
            latency,
        )
        if not gen_stage.ok:
            return self._error_output(inp.query, gen_stage.error, latency, wall_start)

        answer_obj: RAGAnswer
        answer_obj, _ = gen_stage.data

        # ── Stage 4: Output guardrail ─────────────────────────────────────────
        out_guard: GuardrailResult = self._stage(
            "output_guardrail",
            lambda: self.guardrails.check_output(answer_obj, chunks),
            latency,
        ).data

        if not out_guard.allowed:
            return self._blocked(inp.query, out_guard.reason, latency, wall_start)

        # ── Assemble output ───────────────────────────────────────────────────
        total_ms = (time.perf_counter() - wall_start) * 1000
        latency["total"] = total_ms

        return PipelineOutput(
            query             = inp.query,
            answer            = answer_obj.answer,
            confidence        = answer_obj.confidence,
            sources           = answer_obj.sources,
            grounded          = answer_obj.grounded,
            latency           = latency,
            total_latency_ms  = total_ms,
        )

    # ── Stage helpers ─────────────────────────────────────────────────────────

    def _stage(self, name: str, fn: Callable, latency: dict) -> StageResult:
        t0 = time.perf_counter()
        try:
            result = fn()
            ms = (time.perf_counter() - t0) * 1000
            latency[name] = ms
            return StageResult(data=result, latency_ms=ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            latency[name] = ms
            return StageResult(data=None, latency_ms=ms, ok=False, error=str(e))

    def _stage_with_retry(self, name: str, fn: Callable, latency: dict) -> StageResult:
        """Wraps fn with tenacity retry logic."""
        @retry(
            stop=stop_after_attempt(config.MAX_RETRIES),
            wait=wait_random_exponential(
                min=config.RETRY_WAIT_MIN,
                max=config.RETRY_WAIT_MAX,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _inner():
            return fn()

        t0 = time.perf_counter()
        try:
            result = _inner()
            ms = (time.perf_counter() - t0) * 1000
            latency[name] = ms
            return StageResult(data=result, latency_ms=ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            latency[name] = ms
            err = f"{name} failed after {config.MAX_RETRIES} retries: {e}"
            print(f"❌  {err}")
            traceback.print_exc()
            return StageResult(data=None, latency_ms=ms, ok=False, error=err)

    # ── Output factories ──────────────────────────────────────────────────────

    def _blocked(self, query, reason, latency, wall_start) -> PipelineOutput:
        total_ms = (time.perf_counter() - wall_start) * 1000
        latency["total"] = total_ms
        return PipelineOutput(
            query            = query,
            answer           = "",
            confidence       = 0.0,
            sources          = [],
            grounded         = False,
            blocked          = True,
            block_reason     = reason,
            latency          = latency,
            total_latency_ms = total_ms,
        )

    def _error_output(self, query, error, latency, wall_start) -> PipelineOutput:
        total_ms = (time.perf_counter() - wall_start) * 1000
        latency["total"] = total_ms
        return PipelineOutput(
            query            = query,
            answer           = "An internal error occurred. Please try again.",
            confidence       = 0.0,
            sources          = [],
            grounded         = False,
            blocked          = True,
            block_reason     = error,
            latency          = latency,
            total_latency_ms = total_ms,
        )
