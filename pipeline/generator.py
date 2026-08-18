"""
pipeline/generator.py — LLM Answer Generation with Conversational Memory & Smart Suggestions

Supports:
  - Multi-turn conversation history (context-aware follow-up queries)
  - Dynamic follow-up question suggestions after each answer
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
    answer:      str        = Field(description="The answer grounded in retrieved context")
    confidence:  float      = Field(ge=0.0, le=1.0, description="Model's self-reported confidence")
    sources:     list[str]  = Field(default=[], description="chunk_ids used to form the answer")
    grounded:    bool       = Field(default=True, description="True if answer is supported by context")
    suggestions: list[str]  = Field(default=[], description="2-3 relevant follow-up questions for the user")


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are VoxRAG, a precise, conversational, context-grounded question-answering assistant.

Rules:
1. Answer using the provided context passages and conversation history.
2. If the user asks a follow-up question (e.g. "What are its types?", "Who owns it?"), use the conversation history to understand the subject.
3. If the context does not contain enough information, say politely: "Based on the provided context, I don't have enough specific information to answer that."
4. Be concise and natural (2-4 sentences).
5. Always generate 2-3 logical, helpful follow-up questions in the "suggestions" array that the user might want to ask next about the topic.
6. Always return a valid JSON object matching the requested schema.
"""

_USER_TEMPLATE = """Context passages:
{context}

Current Question: {question}

Respond with a valid JSON object ONLY (no markdown code blocks):
{{
  "answer":     "<conversational grounded answer>",
  "confidence": <0.0-1.0>,
  "sources":    ["<chunk_id1>", "<chunk_id2>"],
  "grounded":   true,
  "suggestions": [
    "<relevant follow-up question 1>",
    "<relevant follow-up question 2>",
    "<relevant follow-up question 3>"
  ]
}}"""


class AnswerGenerator:
    """
    Calls Groq to generate a grounded answer from retrieved chunks and conversational history,
    along with smart follow-up suggestions.
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

        answer = self._parse(raw, chunks, question)
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

    def _parse(self, raw: Optional[str], chunks: list[dict], question: str) -> RAGAnswer:
        if raw is None:
            return RAGAnswer(
                answer      = "I am unable to generate an answer at this moment. Please try again.",
                confidence  = 0.0,
                sources     = [],
                grounded    = False,
                suggestions = ["What is a corporation?", "How does FAISS search work?", "Explain dense passage retrieval"],
            )

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            data = json.loads(raw)
            suggestions = data.get("suggestions", [])
            if not isinstance(suggestions, list) or len(suggestions) == 0:
                suggestions = self._fallback_suggestions(question)

            return RAGAnswer(
                answer      = data.get("answer", "").strip(),
                confidence  = float(data.get("confidence", 0.88)),
                sources     = data.get("sources", [c["chunk_id"] for c in chunks[:2]]),
                grounded    = bool(data.get("grounded", True)),
                suggestions = [str(s).strip() for s in suggestions[:3] if str(s).strip()],
            )
        except Exception:
            return RAGAnswer(
                answer      = raw,
                confidence  = 0.6,
                sources     = [c["chunk_id"] for c in chunks[:1]],
                grounded    = True,
                suggestions = self._fallback_suggestions(question),
            )

    def _fallback_suggestions(self, question: str) -> list[str]:
        q_low = question.lower()
        if "corporation" in q_low or "company" in q_low:
            return ["What are the advantages of a corporation?", "What is an S Corporation?", "How are corporate taxes handled?"]
        elif "retrieval" in q_low or "faiss" in q_low or "bm25" in q_low:
            return ["What is the difference between BM25 and dense retrieval?", "How does FAISS index embeddings?", "What embedding model is used in VoxRAG?"]
        elif "dataset" in q_low or "msmarco" in q_low:
            return ["What languages does MSMARCO-XI cover?", "How are passage boundaries chunked?", "What is the token length per chunk?"]
        else:
            return [f"Can you explain more details about {question}?", "What are key real-world examples?", "How is this related to information retrieval?"]
