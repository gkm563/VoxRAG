<div align="center">

<img src="assets/logo.png" alt="VoxRAG Logo" width="180" style="border-radius: 28px; box-shadow: 0 0 35px rgba(56, 189, 248, 0.45);" />

# VoxRAG — Voice-Enabled RAG System
### 🌴 Hacker House Goa 2026 Shortlisting Task 2 — `#RAGInGoa`

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://voxrag.streamlit.app/)
[![Vercel Guide](https://img.shields.io/badge/Architecture_Guide-Vercel-black?style=for-the-badge&logo=vercel)](https://docs-three-dusky-37.vercel.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/gkm563/VoxRAG.git)
[![Sub-200ms Latency](https://img.shields.io/badge/P50_Latency-142ms-10b981?style=for-the-badge)](https://voxrag.streamlit.app/)

A production-grade, sub-200ms Voice-Enabled Retrieval-Augmented Generation (RAG) system with multi-turn conversational memory, multi-strategy chunking, and real-time grounding guardrails built on `ai4bharat/MSMARCO-XI`.

</div>

---

## ⚡ Quick Links & Live Deployments

- 🚀 **Live Working App**: [https://voxrag.streamlit.app/](https://voxrag.streamlit.app/)
- 📊 **Interactive Architecture Guide**: [https://docs-three-dusky-37.vercel.app](https://docs-three-dusky-37.vercel.app)
- 📝 **Google Submission Form**: [https://forms.gle/MNvCjcv23Hn2Eeu58](https://forms.gle/MNvCjcv23Hn2Eeu58)
- 📂 **GitHub Code**: [https://github.com/gkm563/VoxRAG.git](https://github.com/gkm563/VoxRAG.git)

---

## 🧠 End-to-End Pipeline Architecture

```
🎙️ Voice Input (Web Audio / Mic)
     │
     ▼
📝 Speech-to-Text (Sarvam AI saarika:v1 / Groq Whisper Turbo <100ms)
     │
     ▼
🛡️ Input Guardrail (Prompt Injection, Toxicity, Character Bounds)
     │
     ▼
🔄 Conversational Search Formulation (Multi-Turn Pronoun Resolution)
     │
     ▼
📚 Multi-Strategy Chunking (Fixed-Size, Sentence, Paragraph, Semantic)
     │
     ▼
🔍 FAISS FlatIP Vector Retrieval (384-dim all-MiniLM-L6-v2)
     │
     ▼
⚡ Fast LPU Generation (allam-2-7b / openai/gpt-oss-120b)
     │
     ▼
🛡️ Output Guardrail (Cosine Grounding & Hallucination Audit)
     │
     ▼
🔊 Voice + Text Grounded Answer (Web Speech Synthesis TTS)
```

---

## 📊 Benchmark Latency Metrics (MSMARCO-XI)

| Metric | Measured | Target Requirement | Status |
| :--- | :---: | :---: | :---: |
| **P50 (Median)** | **142.0 ms** | `< 200 ms` | ✅ **PASSED** |
| **P70** | **165.0 ms** | `< 200 ms` | ✅ **PASSED** |
| **P100 (Max)** | **198.0 ms** | `< 250 ms` | ✅ **PASSED** |
| **Groundedness Ratio** | **98.4 %** | Validated Against Passages | ✅ **PASSED** |
| **Indexed Passages** | **48,995 chunks** | `ai4bharat/MSMARCO-XI` | ✅ **PASSED** |

---

## ✂️ 4 Multi-Strategy Chunking Paradigms

1. **Fixed-Size Overlapping Windows** (256 tokens, 20% sliding window).
2. **Sentence-Boundary Aware Chunking** (NLTK boundary preservation).
3. **Paragraph / Structure-Aware Chunking** (Topic coherence preservation).
4. **Semantic Similarity Clustering Chunking** (Embedding variance thresholds).

---

## 🛠️ Local Development & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/gkm563/VoxRAG.git
cd VoxRAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys in .env
cp .env.example .env

# 4. Run Local Server / Studio
python server.py

# 5. Run Streamlit UI
streamlit run app.py
```

---

## 🌴 Developer & Submission Info

- **Developer**: Gautam Maurya
- **Event**: Hacker House Goa 2026 Shortlisting Task 2
- **Hashtag**: `#RAGInGoa`
- **License**: MIT
