"""
pipeline/stt.py — Speech-to-Text via Sarvam AI (saarika:v1)

Supports two input modes:
  1. Microphone recording (live voice)
  2. Pre-recorded audio file path
"""

import io
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
    Wraps Sarvam AI speech-to-text REST API.

    Usage:
        stt = SpeechToText()
        text, latency_ms = stt.from_mic()          # live recording
        text, latency_ms = stt.from_file("q.wav")  # from file
    """

    def __init__(self):
        if not config.SARVAM_API_KEY:
            raise EnvironmentError(
                "SARVAM_API_KEY is not set. Add it to your .env file."
            )
        self.api_key = config.SARVAM_API_KEY
        self.url     = config.SARVAM_STT_URL
        self.model   = config.SARVAM_STT_MODEL

    # ── Public API ────────────────────────────────────────────────────────────

    def from_mic(self) -> tuple[str, float]:
        """Record from microphone, transcribe, return (text, latency_ms)."""
        if not _AUDIO_AVAILABLE:
            raise ImportError("sounddevice / soundfile not installed.")

        print(f"🎙️  Recording for {config.AUDIO_RECORD_SECS}s … speak now!")
        audio = sd.rec(
            int(config.AUDIO_RECORD_SECS * config.AUDIO_SAMPLE_RATE),
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
        )
        sd.wait()
        print("✅  Recording done.")

        # Save to a temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(config.AUDIO_CHANNELS)
            wf.setsampwidth(2)  # int16 → 2 bytes
            wf.setframerate(config.AUDIO_SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        return self.from_file(tmp_path)

    def from_file(self, audio_path: str) -> tuple[str, float]:
        """Transcribe an existing audio file. Returns (text, latency_ms)."""
        t0 = time.perf_counter()
        text = self._call_sarvam(audio_path)
        latency_ms = (time.perf_counter() - t0) * 1000
        return text, latency_ms

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_sarvam(self, audio_path: str) -> str:
        """POST audio to Sarvam AI STT endpoint and return transcript."""
        headers = {"api-subscription-key": self.api_key}
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data  = {
                "model":            self.model,
                "language_code":    "en-IN",
                "with_timestamps":  False,
            }
            resp = requests.post(
                self.url, headers=headers, files=files, data=data, timeout=10
            )

        resp.raise_for_status()
        payload = resp.json()

        # Sarvam response: {"transcript": "...", ...}
        transcript = payload.get("transcript", "").strip()
        if not transcript:
            raise ValueError(f"Empty transcript from Sarvam AI: {payload}")
        return transcript
