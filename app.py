"""
app.py — VoxRAG Streamlit Web UI
Live demo interface for the voice-enabled RAG pipeline.
"""

import time
import tempfile
import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np

import config
from pipeline.retriever  import FAISSRetriever
from pipeline.generator  import AnswerGenerator
from pipeline.guardrails import Guardrails
from pipeline.harness    import RAGHarness, PipelineInput
from pipeline.stt        import SpeechToText

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoxRAG",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.8rem; font-weight: 700; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem; color: #64748b; margin-top: 0.2rem; margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #86efac; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .blocked-box {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border: 1px solid #fdba74; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .latency-bar {
        background: #f1f5f9; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.3rem 0;
        font-family: monospace; font-size: 0.85rem;
    }
    .metric-card {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #6366f1; }
    .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important; font-size: 1rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    .chunk-card {
        background: #fafafa; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; font-size: 0.85rem;
    }
    .badge-green { background:#dcfce7; color:#166534; padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; }
    .badge-red   { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; }
    .badge-blue  { background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading VoxRAG pipeline …")
def load_pipeline():
    retriever  = FAISSRetriever.load(config.INDEX_PATH)
    generator  = AnswerGenerator()
    guardrails = Guardrails()
    harness    = RAGHarness(retriever, generator, guardrails)
    stt        = SpeechToText()
    return harness, stt


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎙️ VoxRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Voice-Enabled Retrieval-Augmented Generation · MSMARCO-XI Dataset · Built for HH Goa 2026</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k       = st.slider("Retrieved chunks (Top-K)", 1, 10, config.TOP_K)
    record_secs = st.slider("Recording duration (sec)", 2, 10, config.AUDIO_RECORD_SECS)
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_latency= st.toggle("Show latency breakdown", value=True)

    st.markdown("---")
    st.markdown("## 📊 About")
    st.markdown("""
    **VoxRAG Pipeline:**
    - 🎙️ STT: Sarvam AI (saarika:v1)
    - ✂️ Chunking: Fixed + Sentence + Paragraph + Semantic
    - 🗃️ Vector DB: FAISS (cosine similarity)
    - 🤖 LLM: Groq (llama3-8b-8192)
    - 🛡️ Guardrails: Input + Output
    - 🏗️ Harness: Pydantic + Tenacity
    """)
    st.markdown("---")
    st.markdown("**Target latency:** `< 200ms`")
    st.markdown("[GitHub Repo](https://github.com/gkm563/VoxRAG) · #RAGInGoa")

# ── Load pipeline ─────────────────────────────────────────────────────────────
try:
    harness, stt = load_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"❌ Pipeline not ready: {e}\n\nRun `python build_index.py` first to build the FAISS index.")
    pipeline_ready = False

if pipeline_ready:
    # ── Input tabs ────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🎙️ Voice Input", "⌨️ Text Input"])

    query      = None
    stt_latency= 0.0

    with tab1:
        st.markdown("### Record your question")
        col1, col2 = st.columns([1, 2])
        with col1:
            record_btn = st.button("🔴 Record & Ask", use_container_width=True)
        with col2:
            st.caption(f"Will record for {record_secs} seconds. Speak clearly after clicking.")

        if record_btn:
            with st.spinner(f"🎙️ Recording for {record_secs}s … speak now!"):
                audio = sd.rec(
                    int(record_secs * config.AUDIO_SAMPLE_RATE),
                    samplerate=config.AUDIO_SAMPLE_RATE,
                    channels=config.AUDIO_CHANNELS,
                    dtype="int16",
                )
                sd.wait()

            with st.spinner("📝 Transcribing with Sarvam AI …"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                sf.write(tmp_path, audio, config.AUDIO_SAMPLE_RATE)
                try:
                    query, stt_latency = stt.from_file(tmp_path)
                    st.success(f"📝 Transcribed: **\"{query}\"** *(STT: {stt_latency:.0f}ms)*")
                except Exception as e:
                    st.error(f"STT Error: {e}")
                finally:
                    os.unlink(tmp_path)

        # Upload audio file
        st.markdown("---")
        st.markdown("**Or upload an audio file:**")
        audio_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
        if audio_file:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_file.read())
                tmp_path = f.name
            with st.spinner("📝 Transcribing …"):
                try:
                    query, stt_latency = stt.from_file(tmp_path)
                    st.success(f"📝 Transcribed: **\"{query}\"** *(STT: {stt_latency:.0f}ms)*")
                except Exception as e:
                    st.error(f"STT Error: {e}")
                finally:
                    os.unlink(tmp_path)

    with tab2:
        st.markdown("### Type your question")
        text_query = st.text_input(
            "Query", placeholder="e.g. What is the MSMARCO dataset used for?", label_visibility="collapsed"
        )
        ask_btn = st.button("🔍 Ask", use_container_width=False)
        if ask_btn and text_query.strip():
            query = text_query.strip()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if query:
        st.markdown("---")
        with st.spinner("🔍 Retrieving and generating answer …"):
            inp = PipelineInput(query=query, top_k=top_k, stt_latency=stt_latency)
            out = harness.run(inp)

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("## 💬 Result")

        if out.blocked:
            st.markdown(f"""
            <div class="blocked-box">
                <b>🚫 Query Blocked</b><br>
                {out.block_reason}
            </div>
            """, unsafe_allow_html=True)
        else:
            grounded_badge = '<span class="badge-green">✓ Grounded</span>' if out.grounded else '<span class="badge-red">⚠ Ungrounded</span>'
            conf_pct = int(out.confidence * 100)

            st.markdown(f"""
            <div class="answer-box">
                <b>Answer</b> &nbsp; {grounded_badge}<br><br>
                {out.answer}
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Confidence", f"{conf_pct}%")
            with m2:
                total = out.total_latency_ms
                st.metric("Total Latency", f"{total:.0f}ms", delta="✅ Under 200ms" if total < 200 else "⚠️ Over 200ms")
            with m3:
                st.metric("Chunks Retrieved", str(top_k))
            with m4:
                st.metric("Sources Cited", str(len(out.sources)))

        # ── Latency breakdown ─────────────────────────────────────────────────
        if show_latency and out.latency:
            st.markdown("### ⏱️ Latency Breakdown")
            max_ms = max(out.latency.values()) or 1
            for stage, ms in out.latency.items():
                if stage == "total":
                    continue
                bar_pct = int((ms / 200) * 100)  # out of 200ms target
                bar_pct = min(bar_pct, 100)
                color = "#6366f1" if ms < 100 else "#f59e0b" if ms < 180 else "#ef4444"
                st.markdown(f"""
                <div class="latency-bar">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span>{stage}</span><span style="font-weight:600">{ms:.1f}ms</span>
                    </div>
                    <div style="background:#e2e8f0; border-radius:4px; height:6px;">
                        <div style="background:{color}; width:{bar_pct}%; height:6px; border-radius:4px; transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            total = out.total_latency_ms
            target_color = "#16a34a" if total < 200 else "#dc2626"
            st.markdown(f"""
            <div style="margin-top:0.5rem; padding:0.5rem 1rem; background:{target_color}15;
                        border:1px solid {target_color}40; border-radius:8px;
                        color:{target_color}; font-weight:600; font-size:0.9rem;">
                Total: {total:.1f}ms {'✅ Under 200ms target' if total < 200 else '⚠️ Exceeds 200ms target'}
            </div>
            """, unsafe_allow_html=True)

        # ── Retrieved chunks ──────────────────────────────────────────────────
        if show_chunks and not out.blocked:
            st.markdown("### 📚 Retrieved Context Chunks")
            # Re-run retrieval to show chunks (already done in harness, display only)
            chunks, _ = harness.retriever.search(query, top_k)
            for i, chunk in enumerate(chunks, 1):
                strategy = chunk.get("strategy", "unknown")
                score    = chunk.get("score", 0)
                pid      = chunk.get("passage_id", "—")
                with st.expander(f"Chunk {i} · score={score:.3f} · strategy={strategy} · passage={pid}"):
                    st.write(chunk["text"])
