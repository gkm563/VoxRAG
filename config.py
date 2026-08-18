"""
config.py — Central configuration for the RAG pipeline.
All tunable parameters live here; environment-variable overrides via .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INDEX_PATH = Path(os.getenv("INDEX_PATH", str(DATA_DIR / "faiss_index")))
CHUNKS_PATH= Path(os.getenv("CHUNKS_PATH", str(DATA_DIR / "chunks.jsonl")))
DATA_DIR.mkdir(exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────────
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "ai4bharat/MSMARCO-XI"
DATASET_SPLIT   = "train"
DATASET_LANGUAGE= "en"          # start with English passages
MAX_PASSAGES    = 50_000        # cap for fast demo; set to None for full

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE          = 256       # tokens for fixed-size strategy
CHUNK_OVERLAP_PCT   = 0.20      # 20% overlap
MIN_CHUNK_TOKENS    = 32        # discard tiny shards
SEMANTIC_MAX_TOKENS = 512

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # fast, 384-dim
EMBED_DIM   = 384

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_MODEL      = "openai/gpt-oss-20b"    # high TPM, ultra-fast generation
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE     = 0.1

# ── Harness ──────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_WAIT_MIN  = 0.5   # seconds
RETRY_WAIT_MAX  = 2.0

# ── Latency ──────────────────────────────────────────────────────────────────
PIPELINE_TIMEOUT_MS = 200   # hard target

# ── Speech-to-Text ───────────────────────────────────────────────────────────
# Whisper (local, free, open-source) — "tiny"/"base"/"small"/"medium"/"large"
WHISPER_MODEL     = "base"          # ~150MB, good accuracy, runs on CPU
SARVAM_STT_URL    = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL  = "saarika:v1"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS    = 1
AUDIO_RECORD_SECS = 6       # max recording length for demo

# ── Guardrails ───────────────────────────────────────────────────────────────
OFF_TOPIC_THRESHOLD    = 0.25  # cosine sim below this → off-topic
GROUNDING_THRESHOLD    = 0.30  # answer–context sim below this → hallucination
MAX_INPUT_CHARS        = 500   # reject absurdly long inputs
