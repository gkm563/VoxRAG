# Task 2 — Voice-Enabled RAG Model (HH Goa 2026)

A **Voice-Enabled Retrieval-Augmented Generation (RAG)** pipeline:
```
Voice Input → Speech-to-Text (Sarvam AI) → Chunking/Retrieval (FAISS vector DB) → Answer Generation (LLM) → Output
```

Dataset: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                                │
│                                                                 │
│  🎙️ Voice Input                                                 │
│       │                                                         │
│       ▼                                                         │
│  📝 Speech-to-Text (Sarvam AI / saarika:v1)                    │
│       │                                                         │
│       ▼                                                         │
│  🔍 Query Processing & Guardrails                               │
│       │                                                         │
│       ▼                                                         │
│  📚 Multi-Strategy Chunking                                     │
│    ├── Fixed-size chunks (256 tokens, 20% overlap)              │
│    ├── Semantic sentence chunks                                  │
│    ├── Paragraph-aware chunks                                   │
│    └── Metadata-aware chunks (passage_id, language tags)        │
│       │                                                         │
│       ▼                                                         │
│  🗃️ FAISS Vector DB (sentence-transformers embeddings)          │
│       │                                                         │
│       ▼                                                         │
│  ⚙️ Harness (tool calls, retries, structured I/O)               │
│       │                                                         │
│       ▼                                                         │
│  🤖 LLM Answer Generation (Gemini / Groq)                       │
│       │                                                         │
│       ▼                                                         │
│  🛡️ Output Guardrails (hallucination / grounding check)         │
│       │                                                         │
│       ▼                                                         │
│  📊 Answer + Latency Metrics                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env with your keys
```

### 3. Build the vector index
```bash
python build_index.py
```

### 4. Run the pipeline (CLI / demo mode)
```bash
python main.py --mode demo
```

### 5. Run with a microphone (live voice)
```bash
python main.py --mode voice
```

### 6. Run latency benchmark
```bash
python benchmark.py
```

---

## Project Structure

```
Task 2/
├── main.py              # Entry point (voice or text mode)
├── build_index.py       # Dataset download + chunking + FAISS index build
├── benchmark.py         # Latency analytics (P50/P70/P100)
├── pipeline/
│   ├── stt.py           # Speech-to-Text (Sarvam AI)
│   ├── chunker.py       # Multi-strategy chunking
│   ├── retriever.py     # FAISS vector DB retrieval
│   ├── generator.py     # LLM answer generation
│   ├── guardrails.py    # Input/output guardrails
│   └── harness.py       # Orchestration harness (retries, structured I/O)
├── config.py            # Central configuration
├── requirements.txt
├── .env.example
└── README.md
```

---

## Latency Target

| Stage               | Target   |
|---------------------|----------|
| STT                 | ~80ms    |
| Retrieval (FAISS)   | < 10ms   |
| LLM generation      | ~80ms    |
| **Total**           | **< 200ms** |

---

## Guardrails

- **Input guardrails**: Off-topic detection, profanity/unsafe content filter
- **Output guardrails**: Grounding check (answer must cite retrieved context), hallucination detection, confidence threshold
- **System guardrails**: Max retries on failure, timeout enforcement

---

## Latency Results (Benchmark)

Run `python benchmark.py` to generate P50/P70/P100 across 50+ test queries.
Results are saved to `benchmark_results.json`.
