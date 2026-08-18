"""
pipeline/stt.py — Resilient Multi-Provider Speech-to-Text for VoxRAG

Supports:
  1. Sarvam AI API (saarika:v1) — primary fast Indian speech-to-text
  2. Groq Cloud Whisper (whisper-large-v3-turbo) — ultra-fast instant fallback (< 150ms)
  3. Local OpenAI Whisper — offline fallback
"""

import io
import os
import time
import tempfile
import requests
import config

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False


class SpeechToText:
    """
    Speech-to-Text engine with Sarvam AI primary + Groq Whisper + local Whisper.
    """

    def __init__(self, mode: str = "sarvam"):
        self.mode = mode
        self._whisper_model = None
        self._groq_client = None

    def from_file(self, audio_path: str) -> tuple[str, float]:
        """
        Transcribe audio file.
        Returns: (transcript_text, latency_ms)
        """
        t0 = time.perf_counter()
        transcript = ""
        last_err = None

        # 1. Try Sarvam AI first if configured
        if self.mode == "sarvam" and config.SARVAM_API_KEY:
            try:
                transcript = self._sarvam_transcribe(audio_path)
            except Exception as e:
                print(f"[!] Sarvam STT note ({e}) — trying Groq Whisper")
                last_err = e

        # 2. Try Groq Whisper API (super fast, robust to all audio formats)
        if not transcript and config.GROQ_API_KEY:
            try:
                transcript = self._groq_transcribe(audio_path)
            except Exception as e:
                print(f"[!] Groq Whisper note ({e}) — trying local Whisper")
                last_err = e

        # 3. Try Local Whisper
        if not transcript:
            try:
                transcript = self._local_whisper_transcribe(audio_path)
            except Exception as e:
                last_err = e

        if not transcript:
            raise ValueError(f"Could not transcribe audio: {last_err}")

        latency_ms = (time.perf_counter() - t0) * 1000
        return transcript, latency_ms

    def _sarvam_transcribe(self, audio_path: str) -> str:
        """POST audio to Sarvam AI STT API."""
        headers = {"api-subscription-key": config.SARVAM_API_KEY}
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
                timeout=4,
            )

        resp.raise_for_status()
        payload = resp.json()
        transcript = payload.get("transcript", "").strip()
        if not transcript:
            raise ValueError(f"Empty transcript from Sarvam: {payload}")
        return transcript

    def _groq_transcribe(self, audio_path: str) -> str:
        """Transcribe with Groq's high-speed Whisper-large-v3-turbo."""
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=config.GROQ_API_KEY)

        with open(audio_path, "rb") as f:
            transcription = self._groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="json",
                temperature=0.0,
            )
        text = transcription.text.strip() if hasattr(transcription, "text") else str(transcription).strip()
        if not text:
            raise ValueError("Groq Whisper returned empty transcript")
        return text

    def _local_whisper_transcribe(self, audio_path: str) -> str:
        """Local Whisper transcription."""
        if self._whisper_model is None:
            import whisper
            print(f"[*] Loading local Whisper model '{config.WHISPER_MODEL}'...")
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
