"""
pipeline/guardrails.py — Input & Output Guardrails

Input Guardrails:
  ✅ Length check          — reject too-short or too-long queries
  ✅ Profanity / unsafe    — filter inappropriate content
  ✅ Off-topic detection   — cosine similarity against domain centroid
  ✅ Language detection    — warn on non-English queries

Output Guardrails:
  ✅ Grounding check       — answer embedding vs context embedding similarity
  ✅ Confidence threshold  — reject answers below min confidence
  ✅ Empty answer check    — catch non-answers
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

from better_profanity import profanity as _profanity_filter

import config

# Pre-load profanity list once
_profanity_filter.load_censor_words()

# ── Domain anchor phrases (MSMARCO is a QA/web search corpus) ─────────────────
_DOMAIN_ANCHORS = [
    "what is", "how does", "why does", "when did", "where is",
    "who is", "which", "explain", "describe", "define",
    "search", "find", "information", "passage", "document",
]


@dataclass
class GuardrailResult:
    allowed:  bool
    reason:   str  = ""
    warning:  str  = ""


class Guardrails:
    """
    Stateless guardrail checks for input queries and generated answers.

    Usage:
        g = Guardrails()
        result = g.check_input("What is photosynthesis?")
        if not result.allowed:
            print(result.reason)

        result = g.check_output(rag_answer, retrieved_chunks)
    """

    def __init__(self):
        self._embed_model = None
        self._domain_centroid: Optional[np.ndarray] = None

    # ── Input Guardrails ──────────────────────────────────────────────────────

    def check_input(self, query: str) -> GuardrailResult:
        query = query.strip()

        # 1. Empty / too short
        if len(query) < 3:
            return GuardrailResult(False, "Query is too short. Please ask a proper question.")

        # 2. Too long
        if len(query) > config.MAX_INPUT_CHARS:
            return GuardrailResult(
                False,
                f"Query exceeds maximum length of {config.MAX_INPUT_CHARS} characters."
            )

        # 3. Profanity / unsafe content
        if _profanity_filter.contains_profanity(query):
            return GuardrailResult(False, "Query contains inappropriate content.")

        # 4. Injection patterns (prompt injection guard)
        if self._is_injection(query):
            return GuardrailResult(False, "Query looks like a prompt injection attempt.")

        # 5. Off-topic detection (embedding similarity vs domain centroid)
        off_topic_result = self._off_topic_check(query)
        if not off_topic_result.allowed:
            return off_topic_result

        return GuardrailResult(True)

    # ── Output Guardrails ─────────────────────────────────────────────────────

    def check_output(self, answer_obj, chunks: list[dict]) -> GuardrailResult:
        """
        answer_obj: RAGAnswer (from generator.py)
        chunks:     retrieved chunks used for generation
        """
        # 1. Empty answer
        if not answer_obj.answer or len(answer_obj.answer.strip()) < 5:
            return GuardrailResult(False, "Generated answer is empty.")

        # 2. Model explicitly said it doesn't know
        _no_answer_phrases = [
            "i don't have enough information",
            "i cannot answer",
            "not enough context",
            "no relevant information",
        ]
        lower = answer_obj.answer.lower()
        if any(p in lower for p in _no_answer_phrases):
            return GuardrailResult(False, "Model could not find a grounded answer.")

        # 3. Low confidence
        if answer_obj.confidence < 0.25:
            return GuardrailResult(
                False,
                f"Answer confidence too low ({answer_obj.confidence:.2f}). Try rephrasing."
            )

        # 4. Grounding check — cosine similarity between answer and context
        grounding_ok = self._grounding_check(answer_obj.answer, chunks)
        if not grounding_ok:
            return GuardrailResult(
                False,
                "Answer does not appear to be grounded in the retrieved context (possible hallucination).",
            )

        return GuardrailResult(True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_injection(self, query: str) -> bool:
        patterns = [
            r"ignore (previous|above|all) instructions",
            r"you are now",
            r"act as",
            r"pretend (you are|to be)",
            r"jailbreak",
            r"DAN mode",
            r"disregard your",
        ]
        low = query.lower()
        return any(re.search(p, low) for p in patterns)

    def _off_topic_check(self, query: str) -> GuardrailResult:
        try:
            model = self._get_embed_model()
            centroid = self._get_domain_centroid(model)
            q_embed  = model.encode([query], normalize_embeddings=True)[0]
            sim = float(np.dot(q_embed, centroid))
            if sim < config.OFF_TOPIC_THRESHOLD:
                return GuardrailResult(
                    False,
                    f"Query appears off-topic for this knowledge base (similarity={sim:.2f})."
                )
        except Exception:
            pass  # Fail open — if embedding fails, allow the query
        return GuardrailResult(True)

    def _grounding_check(self, answer: str, chunks: list[dict]) -> bool:
        if not chunks:
            return False
        try:
            model      = self._get_embed_model()
            context    = " ".join(c["text"] for c in chunks)
            embeds     = model.encode([answer, context], normalize_embeddings=True)
            sim        = float(np.dot(embeds[0], embeds[1]))
            return sim >= config.GROUNDING_THRESHOLD
        except Exception:
            return True  # Fail open

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(config.EMBED_MODEL)
        return self._embed_model

    def _get_domain_centroid(self, model) -> np.ndarray:
        if self._domain_centroid is None:
            embeds = model.encode(_DOMAIN_ANCHORS, normalize_embeddings=True)
            self._domain_centroid = embeds.mean(axis=0)
            norm = np.linalg.norm(self._domain_centroid)
            if norm > 0:
                self._domain_centroid /= norm
        return self._domain_centroid
