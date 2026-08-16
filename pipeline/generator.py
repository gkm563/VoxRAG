"""
pipeline/generator.py — LLM Answer Generation

Primary  : Groq (llama3-8b-8192) — ultra-low latency
Fallback : Returns a safe "could not generate" response if Groq fails
"""

import re
import time
import json
from typing import Optional
from pydantic import BaseModel, Field

import config


# ── Structured output schema ──────────────────────────────────────────────────
class RAGAnswer(BaseModel):
    answer:     str   = Field(description="The answer grounded in retrieved context")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's self-reported confidence")
    sources:    list[str] = Field(description="chunk_ids used to form the answer")
    grounded:   bool  = Field(description="True if answer is supported by context")


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are VoxRAG, a precise, context-grounded question-answering assistant.

Rules:
1. Answer ONLY using the provided context passages. Do NOT use outside knowledge.
2. If the context does not contain enough information, say exactly: "I don't have enough information to answer that."
3. Be concise — 1-3 sentences unless more detail is explicitly needed.
4. Always return valid JSON matching the schema provided.
"""

_USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Respond with a JSON object ONLY (no markdown, no extra text):
{{
  "answer":     "<answer text>",
  "confidence": <0.0-1.0>,
  "sources":    ["<chunk_id1>", "<chunk_id2>"],
  "grounded":   true
}}"""


class AnswerGenerator:
    """
    Calls Groq to generate a grounded answer from retrieved chunks.

    Usage:
        gen = AnswerGenerator()
        result, latency_ms = gen.generate(query, chunks)
    """

    def __init__(self):
        self._groq_client = None

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        chunks:   list[dict],
    ) -> tuple[RAGAnswer, float]:
        """
        Generate an answer given retrieved chunks.
        Returns: (RAGAnswer, latency_ms)
        """
        context = self._format_context(chunks)
        prompt  = _USER_TEMPLATE.format(context=context, question=question)

        t0  = time.perf_counter()
        raw = self._call_groq(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = self._parse(raw, chunks)
        return answer, latency_ms

    # ── Internal ──────────────────────────────────────────────────────────────

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            score = c.get('score', 0)
            parts.append(
                f"[{i}] (chunk_id={c['chunk_id']}, score={score:.3f})\n{c['text']}"
            )
        return "\n\n".join(parts)

    def _call_groq(self, prompt: str) -> Optional[str]:
        try:
            if self._groq_client is None:
                from groq import Groq
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)

            resp = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Groq error: {e}")
            return None

    def _parse(self, raw: Optional[str], chunks: list[dict]) -> RAGAnswer:
        if raw is None:
            return RAGAnswer(
                answer     = "Unable to generate an answer at this time. Please try again.",
                confidence = 0.0,
                sources    = [],
                grounded   = False,
            )

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            data = json.loads(raw)
            return RAGAnswer(
                answer     = data.get("answer", ""),
                confidence = float(data.get("confidence", 0.5)),
                sources    = data.get("sources", []),
                grounded   = bool(data.get("grounded", True)),
            )
        except Exception:
            # Graceful degradation — return raw text as answer
            return RAGAnswer(
                answer     = raw,
                confidence = 0.3,
                sources    = [c["chunk_id"] for c in chunks[:1]],
                grounded   = False,
            )
