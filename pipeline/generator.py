"""
pipeline/generator.py — LLM Answer Generation with Conversational Memory

Supports:
  - Multi-turn conversation history (context-aware follow-up queries)
  - Groq LPU acceleration (groq/compound-mini)
  - Strict grounding in retrieved context
  - Pydantic structured output validation
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
_SYSTEM_PROMPT = """You are VoxRAG, a precise, conversational, context-grounded question-answering assistant.

Rules:
1. Answer using the provided context passages and conversation history.
2. If the user asks a follow-up question (e.g. "What are its types?", "Who owns it?"), use the conversation history to understand the subject.
3. If the context does not contain enough information, say politely: "Based on the provided context, I don't have enough specific information to answer that."
4. Be concise and natural (2-4 sentences).
5. Always return a valid JSON object matching the requested schema.
"""

_USER_TEMPLATE = """Context passages:
{context}

Current Question: {question}

Respond with a valid JSON object ONLY:
{{
  "answer":     "<conversational grounded answer>",
  "confidence": <0.0-1.0>,
  "sources":    ["<chunk_id1>", "<chunk_id2>"],
  "grounded":   true
}}"""


class AnswerGenerator:
    """
    Calls Groq to generate a grounded answer from retrieved chunks and conversational history.
    """

    def __init__(self):
        self._groq_client = None

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        chunks:   list[dict],
        history:  Optional[list[dict]] = None,
    ) -> tuple[RAGAnswer, float]:
        """
        Generate an answer given retrieved chunks and conversation history.
        Returns: (RAGAnswer, latency_ms)
        """
        context = self._format_context(chunks)
        user_prompt = _USER_TEMPLATE.format(context=context, question=question)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Include recent conversation turns (up to last 6 turns)
        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})

        t0  = time.perf_counter()
        raw = self._call_groq(messages)
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

    def _call_groq(self, messages: list[dict]) -> Optional[str]:
        try:
            if self._groq_client is None:
                from groq import Groq
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)

            resp = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Groq generation error: {e}")
            return None

    def _parse(self, raw: Optional[str], chunks: list[dict]) -> RAGAnswer:
        if raw is None:
            return RAGAnswer(
                answer     = "I am unable to generate an answer at this moment. Please try again.",
                confidence = 0.0,
                sources    = [],
                grounded   = False,
            )

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            data = json.loads(raw)
            return RAGAnswer(
                answer     = data.get("answer", "").strip(),
                confidence = float(data.get("confidence", 0.85)),
                sources    = data.get("sources", [c["chunk_id"] for c in chunks[:2]]),
                grounded   = bool(data.get("grounded", True)),
            )
        except Exception:
            # Graceful fallback
            return RAGAnswer(
                answer     = raw,
                confidence = 0.5,
                sources    = [c["chunk_id"] for c in chunks[:1]],
                grounded   = True,
            )
