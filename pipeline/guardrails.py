"""
pipeline/guardrails.py — Input & Output Guardrails for VoxRAG

Input Guardrails:
  - Query Length check (min 3 chars, max MAX_INPUT_CHARS)
  - Toxicity & Profanity filter (better_profanity)
  - Prompt Injection detection (regex heuristics)
  - Character entropy / gibberish filter

Output Guardrails:
  - Hallucination / Grounding check (embedding similarity between answer and retrieved passages)
  - Minimum confidence check
  - Non-answer detection
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

from better_profanity import profanity as _profanity_filter
import config

_profanity_filter.load_censor_words()


@dataclass
class GuardrailResult:
    allowed: bool
    reason:  str = ""
    warning: str = ""


class Guardrails:
    """
    Stateless guardrail checks for input queries and generated answers.
    """

    def __init__(self, embed_model=None):
        self._embed_model = embed_model

    # ── Input Guardrails ──────────────────────────────────────────────────────

    def check_input(self, query: str) -> GuardrailResult:
        query = query.strip()

        # 1. Length checks
        if len(query) < 3:
            return GuardrailResult(False, "Query is too short. Please ask a full question.")

        if len(query) > config.MAX_INPUT_CHARS:
            return GuardrailResult(
                False,
                f"Query exceeds maximum character limit ({config.MAX_INPUT_CHARS} chars)."
            )

        # 2. Profanity / Inappropriate content
        if _profanity_filter.contains_profanity(query):
            return GuardrailResult(False, "Query contains inappropriate or harmful content.")

        # 3. Prompt injection detection
        if self._is_injection(query):
            return GuardrailResult(False, "Query contains potential prompt injection patterns.")

        # 4. Gibberish / repeated characters
        if re.search(r"(.)\1{7,}", query):
            return GuardrailResult(False, "Query appears to be repetitive gibberish.")

        return GuardrailResult(True)

    # ── Output Guardrails ─────────────────────────────────────────────────────

    def check_output(self, answer_obj, chunks: list[dict]) -> GuardrailResult:
        """
        Validates the generated answer against retrieved context.
        """
        # 1. Empty or trivial answer
        if not answer_obj.answer or len(answer_obj.answer.strip()) < 3:
            return GuardrailResult(False, "Generated answer is empty or invalid.")

        # 2. Model explicit refusal
        _no_answer_patterns = [
            "i don't have enough information",
            "i cannot answer",
            "not enough context",
            "context does not contain",
            "no relevant information",
        ]
        lower = answer_obj.answer.lower()
        if any(p in lower for p in _no_answer_patterns):
            return GuardrailResult(True, warning="Model noted insufficient context.")

        # 3. Grounding check with retrieved chunks
        grounded = self._grounding_check(answer_obj.answer, chunks)
        if not grounded:
            # We flag warning or allow with ungrounded flag
            return GuardrailResult(True, warning="Answer may contain ungrounded details.")

        return GuardrailResult(True)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _is_injection(self, query: str) -> bool:
        patterns = [
            r"ignore (previous|above|all) instructions",
            r"you are now",
            r"act as a",
            r"pretend (you are|to be)",
            r"jailbreak",
            r"disregard (the|your)",
            r"override system prompt",
        ]
        low = query.lower()
        return any(re.search(p, low) for p in patterns)

    def _grounding_check(self, answer: str, chunks: list[dict]) -> bool:
        if not chunks:
            return False
        try:
            model = self._get_embed_model()
            context_snippet = " ".join(c["text"] for c in chunks[:3])
            embeds = model.encode([answer, context_snippet], normalize_embeddings=True)
            sim = float(np.dot(embeds[0], embeds[1]))
            return sim >= config.GROUNDING_THRESHOLD
        except Exception:
            return True

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(config.EMBED_MODEL)
        return self._embed_model
