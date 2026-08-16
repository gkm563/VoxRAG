"""
pipeline/stt.py — Speech-to-Text

Two modes (both FREE):
  1. Whisper (PRIMARY) — OpenAI Whisper, runs 100% locally, no API key needed
  2. Sarvam AI (SECONDARY) — Required by task spec, free API key

Usage:
    stt = SpeechToText(mode="whisper")   # local, free
    stt = SpeechToText(mode="sarvam")    # Sarvam AI, free tier
    text, latency_ms = stt.from_mic()
    text, latency_ms = stt.from_file("audio.wav")
"""

import io
import os
import time
import wave
import tempfile
import requests
import numpy as np

import config

try:
    import sounddevice as sd
    import soundfile as sf
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False


class SpeechToText:
    """
    STT wrapper supporting Whisper (local/free) and Sarvam AI (free tier).

    Args:
        mode: "whisper" (default, fully local & free) or "sarvam" (free API)
    """

    def __init__(self, mode: str = "whisper"):
        self.mode = mode
        self._whisper_model = None   # lazy load

        if mode == "sarvam" and not config.SARVAM_API_KEY:
            raise EnvironmentError("SARVAM_API_KEY not set in .env")

    # ── Public API ────────────────────────────────────────────────────────────

    def from_mic(self) -> tuple[str, float]:
        """Record from microphone, transcribe, return (text, latency_ms)."""
        if not _AUDIO_AVAILABLE:
            raise ImportError("sounddevice / soundfile not installed. Run: pip install sounddevice soundfile")

        print(f"🎙️  Recording for {config.AUDIO_RECORD_SECS}s … speak now!")
        audio = sd.rec(
            int(config.AUDIO_RECORD_SECS * config.AUDIO_SAMPLE_RATE),
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
        )
        sd.wait()
        print("✅  Recording done.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(config.AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(config.AUDIO_SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        result = self.from_file(tmp_path)
        os.unlink(tmp_path)
        return result

    def from_file(self, audio_path: str) -> tuple[str, float]:
        """Transcribe an audio file. Returns (text, latency_ms)."""
        t0 = time.perf_counter()
        if self.mode == "whisper":
            text = self._whisper_transcribe(audio_path)
        else:
            text = self._sarvam_transcribe(audio_path)
        latency_ms = (time.perf_counter() - t0) * 1000
        return text, latency_ms

    # ── Whisper (FREE, local, open-source) ───────────────────────────────────

    def _whisper_transcribe(self, audio_path: str) -> str:
        """
        Uses OpenAI's Whisper model locally — 100% free, no internet needed.
        Model is downloaded once (~150MB for 'base', ~1.5GB for 'large').
        """
        if self._whisper_model is None:
            try:
                import whisper
            except ImportError:
                raise ImportError(
                    "Whisper not installed. Run: pip install openai-whisper"
                )
            print(f"📥  Loading Whisper model '{config.WHISPER_MODEL}' (first run only) …")
            self._whisper_model = whisper.load_model(config.WHISPER_MODEL)

        result = self._whisper_model.transcribe(
            audio_path,
            language="en",
            fp16=False,          # safe for CPU
        )
        transcript = result.get("text", "").strip()
        if not transcript:
            raise ValueError("Whisper returned empty transcript.")
        return transcript

    # ── Sarvam AI (FREE tier, required by task spec) ──────────────────────────

    def _sarvam_transcribe(self, audio_path: str) -> str:
        """POST audio to Sarvam AI STT endpoint (free tier)."""
        headers = {"api-subscription-key": config.SARVAM_API_KEY}
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data  = {
                "model":           config.SARVAM_STT_MODEL,
                "language_code":   "en-IN",
                "with_timestamps": False,
            }
            resp = requests.post(
                config.SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=10,
            )

        resp.raise_for_status()
        payload    = resp.json()
        transcript = payload.get("transcript", "").strip()
        if not transcript:
            raise ValueError(f"Empty transcript from Sarvam AI: {payload}")
        return transcript
