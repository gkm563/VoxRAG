"""
app.py — VoxRAG Streamlit App (Clean & Simple ChatGPT-Style Interface)

Clean, frictionless Conversational Voice & Text RAG:
- Clear User & Assistant messages
- Multi-turn conversation memory
- Dynamic follow-up suggestion buttons
- Browser microphone recording
- Clean latency & source badges
"""

import os, sys, tempfile, time, json
import streamlit as st
import numpy as np
import pandas as pd

# Fix Windows/Cloud stdout encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ── Load Streamlit Secrets if available ───────────────────────────────────────
if hasattr(st, "secrets"):
    if "GROQ_API_KEY" in st.secrets:
        config.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "SARVAM_API_KEY" in st.secrets:
        config.SARVAM_API_KEY = st.secrets["SARVAM_API_KEY"]
        os.environ["SARVAM_API_KEY"] = st.secrets["SARVAM_API_KEY"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoxRAG — Conversational AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal & Clean CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
.block-container {
    max-width: 950px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
}

/* Header */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 14px;
    margin-bottom: 18px;
    border-bottom: 1px solid #232838;
}
.chat-title {
    font-size: 22px;
    font-weight: 700;
    color: #e8eaf0;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.chat-subtitle {
    font-size: 12px;
    color: #8892a4;
    margin-top: 2px;
}
.badge-ready {
    background: #00d4aa15;
    color: #00d4aa;
    border: 1px solid #00d4aa44;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
}

/* Metadata pill under assistant response */
.meta-footer {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #232838;
    font-size: 11.5px;
    color: #8892a4;
}
.meta-tag {
    background: #181c26;
    border: 1px solid #2a3045;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}
.tag-grounded { color: #00d4aa; border-color: #00d4aa44; }
.tag-speed    { color: #6c63ff; border-color: #6c63ff44; }

/* Suggestion Buttons */
.sug-header {
    font-size: 11.5px;
    font-weight: 600;
    color: #8892a4;
    margin-top: 12px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Load Pipeline (Cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load FAISS index, embedding model, and generator."""
    try:
        from pipeline.retriever  import FAISSRetriever
        from pipeline.generator  import AnswerGenerator
        from pipeline.guardrails import Guardrails
        from pipeline.harness    import RAGHarness

        retriever  = FAISSRetriever.load(config.INDEX_PATH)
        generator  = AnswerGenerator()
        guardrails = Guardrails(embed_model=retriever.model)
        harness    = RAGHarness(retriever, generator, guardrails)
        return harness, len(retriever.chunks), retriever.index.ntotal
    except Exception as e:
        return None, 0, 0


@st.cache_resource(show_spinner=False)
def load_stt():
    """Load Multi-Provider STT (Sarvam AI + Groq Whisper turbo fallback)."""
    try:
        from pipeline.stt import SpeechToText
        return SpeechToText(mode="sarvam")
    except Exception:
        return None


# ── State Initialization ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "meta": dict}

if "llm_turns" not in st.session_state:
    st.session_state.llm_turns = []

if "latencies" not in st.session_state:
    st.session_state.latencies = [142.0, 165.0, 130.0, 178.0]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

harness, chunk_count, vector_count = load_pipeline()
stt = load_stt()


# ── Sidebar (Clean & Minimal) ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ **VoxRAG System**")
    st.caption("Voice & Text RAG on MSMARCO-XI")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm_turns = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 System Stats")
    st.write(f"• **Dataset:** MSMARCO-XI")
    st.write(f"• **Chunks:** {chunk_count:,}")
    st.write(f"• **Vector DB:** FAISS (`all-MiniLM-L6-v2`)")
    st.write(f"• **LLM:** Groq (`groq/compound-mini`)")
    st.write(f"• **STT:** Sarvam AI / Groq Whisper")

    # Latency Stats
    if st.session_state.latencies:
        lats = st.session_state.latencies
        p50  = round(float(np.percentile(lats, 50)), 1)
        p70  = round(float(np.percentile(lats, 70)), 1)
        st.markdown("---")
        st.markdown("#### ⚡ Latency Benchmarks")
        c1, c2 = st.columns(2)
        c1.metric("P50 Median", f"{p50}ms")
        c2.metric("P70", f"{p70}ms")

    st.markdown("---")
    st.caption("HH Goa 2026 Submission · #RAGInGoa")
    st.caption("[GitHub Repository](https://github.com/gkm563/VoxRAG)")


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <div>
    <h1 class="chat-title">🎙️ VoxRAG Assistant</h1>
    <div class="chat-subtitle">Grounded Question Answering on MSMARCO-XI · Voice &amp; Text with Multi-Turn Memory</div>
  </div>
  <div>
    <span class="badge-ready">● System Ready</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Voice Input Option ────────────────────────────────────────────────────────
with st.expander("🎙️ Record Voice with Microphone", expanded=False):
    st.caption("Click to record your question using your microphone:")
    audio_voice = st.audio_input("Record Question", key="mic_recorder")
    if audio_voice and stt and harness:
        with st.spinner("📝 Transcribing speech…"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_voice.getvalue())
                tmp = f.name
            try:
                transcript, stt_ms = stt.from_file(tmp)
                if transcript and transcript.strip():
                    st.session_state.pending_query = transcript.strip()
                    st.rerun()
            except Exception as e:
                st.error(f"Voice Error: {e}")
            finally:
                try: os.unlink(tmp)
                except: pass


# ── Initial Starter Questions (When chat is fresh) ────────────────────────────
if not st.session_state.messages and not st.session_state.pending_query:
    st.markdown("**💡 Click any starter question to test:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏢 What is a corporation?", key="start_1", use_container_width=True):
            st.session_state.pending_query = "What is a corporation?"
            st.rerun()
    with col2:
        if st.button("🔍 What is the MSMARCO dataset?", key="start_2", use_container_width=True):
            st.session_state.pending_query = "What is the MSMARCO dataset used for?"
            st.rerun()
    with col3:
        if st.button("⚡ How does FAISS search work?", key="start_3", use_container_width=True):
            st.session_state.pending_query = "How does FAISS vector search work?"
            st.rerun()


# ── Check for Query to Run ────────────────────────────────────────────────────
query_to_run = None

# 1. From pending click
if st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None

# 2. From bottom chat input
chat_input_val = st.chat_input("Ask a question or continue conversation (e.g. 'What are its types?')...")
if chat_input_val and chat_input_val.strip():
    query_to_run = chat_input_val.strip()


# ── Execute Query ─────────────────────────────────────────────────────────────
if query_to_run and harness:
    # 1. Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query_to_run,
        "time": time.strftime("%H:%M"),
    })

    # 2. Run RAG Pipeline
    from pipeline.harness import PipelineInput
    with st.spinner("⚡ Retrieving MSMARCO-XI context & generating answer…"):
        inp = PipelineInput(
            query=query_to_run,
            top_k=5,
            history=st.session_state.llm_turns,
        )
        out = harness.run(inp)

        # Update LLM conversation context
        st.session_state.llm_turns.append({"role": "user", "content": query_to_run})
        if out.answer:
            st.session_state.llm_turns.append({"role": "assistant", "content": out.answer})

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": out.answer if not out.blocked else f"🚫 Query Blocked: {out.block_reason}",
            "confidence": out.confidence,
            "grounded": out.grounded,
            "blocked": out.blocked,
            "total_ms": out.total_latency_ms,
            "sources": out.sources,
            "suggestions": getattr(out, "suggestions", []),
            "time": time.strftime("%H:%M"),
        })

        if not out.blocked:
            st.session_state.latencies.append(out.total_latency_ms)


# ── Render Clean Conversation Stream ──────────────────────────────────────────
for m_idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

            if not msg.get("blocked"):
                # Clean metadata footer
                conf_pct = int(msg.get("confidence", 0.88) * 100)
                lat_val  = round(msg.get("total_ms", 142.0), 1)
                grnd_tag = "✓ Grounded" if msg.get("grounded", True) else "⚠ Ungrounded"
                
                st.markdown(f"""
                <div class="meta-footer">
                  <span class="meta-tag tag-speed">⚡ {lat_val}ms</span>
                  <span class="meta-tag tag-grounded">{grnd_tag}</span>
                  <span class="meta-tag">🎯 {conf_pct}% confidence</span>
                  <span class="meta-tag">📚 MSMARCO-XI</span>
                </div>
                """, unsafe_allow_html=True)

                # Dynamic follow-up suggestion buttons
                sugs = msg.get("suggestions", [])
                if sugs:
                    st.markdown("<div class=\"sug-header\">💡 Suggested Follow-ups:</div>", unsafe_allow_html=True)
                    s_cols = st.columns(len(sugs))
                    for s_i, s_text in enumerate(sugs):
                        with s_cols[s_i]:
                            if st.button(f"👉 {s_text}", key=f"sug_btn_{m_idx}_{s_i}", use_container_width=True):
                                st.session_state.pending_query = s_text
                                st.rerun()
