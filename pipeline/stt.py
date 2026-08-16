"""
pipeline/stt.py — Resilient Speech-to-Text for VoxRAG

Supports:
  1. Sarvam AI API (saarika:v1) — primary fast cloud STT
  2. OpenAI Whisper (local) — automatic offline fallback

Automatically ensures audio is converted/handled properly.
"""

import io
import os
import time
import tempfile
import requests
import numpy as np

import config

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False


class SpeechToText:
    """
    Speech-to-Text engine with Sarvam AI primary + local Whisper fallback.
    """

    def __init__(self, mode: str = "sarvam"):
        self.mode = mode
        self._whisper_model = None

    def from_file(self, audio_path: str) -> tuple[str, float]:
        """
        Transcribe audio file.
        Returns: (transcript_text, latency_ms)
        """
        t0 = time.perf_counter()
        transcript = ""

        # Try Sarvam AI first if configured
        if self.mode == "sarvam" and config.SARVAM_API_KEY:
            try:
                transcript = self._sarvam_transcribe(audio_path)
            except Exception as e:
                print(f"[!] Sarvam STT failed ({e}) — falling back to Whisper")
                transcript = self._whisper_transcribe(audio_path)
        else:
            transcript = self._whisper_transcribe(audio_path)

        latency_ms = (time.perf_counter() - t0) * 1000
        return transcript, latency_ms

    def _sarvam_transcribe(self, audio_path: str) -> str:
        """POST audio to Sarvam AI STT API."""
        headers = {"api-subscription-key": config.SARVAM_API_KEY}

        # Ensure we send appropriate content type
        filename = os.path.basename(audio_path)
        content_type = "audio/wav" if audio_path.endswith(".wav") else "audio/webm"

        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, content_type)}
            data = {
                "model":           config.SARVAM_STT_MODEL,
                "language_code":   "en-IN",
                "with_timestamps": False,
            }
            resp = requests.post(
                config.SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=12,
            )

        resp.raise_for_status()
        payload = resp.json()
        transcript = payload.get("transcript", "").strip()

        if not transcript:
            raise ValueError(f"Empty transcript received from Sarvam API: {payload}")

        return transcript

    def _whisper_transcribe(self, audio_path: str) -> str:
        """Local Whisper transcription."""
        if self._whisper_model is None:
            import whisper
            print(f"[*] Loading Whisper model '{config.WHISPER_MODEL}'...")
            self._whisper_model = whisper.load_model(config.WHISPER_MODEL)

        result = self._whisper_model.transcribe(
            audio_path,
            language="en",
            fp16=False,
        )
        text = result.get("text", "").strip()
        if not text:
            raise ValueError("Whisper could not detect speech in audio.")
        return text
