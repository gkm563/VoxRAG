"""
app.py — VoxRAG Streamlit Cloud App (ChatGPT Exact Aesthetic)
"""

import os, sys, tempfile, time, json
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="ChatGPT — VoxRAG (Voice & Text)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Exact ChatGPT Dark Theme CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
.block-container {
    max-width: 860px !important;
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
}

/* User Bubble (ChatGPT Style Blue Capsule) */
.user-msg-bubble {
    background: #1e3a8a;
    color: #ffffff;
    padding: 10px 18px;
    border-radius: 20px;
    font-size: 14px;
    line-height: 1.5;
    margin-left: auto;
    margin-bottom: 14px;
    max-width: 80%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.user-msg-header {
    font-size: 10.5px;
    color: #93c5fd;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
}

/* Assistant Bubble (ChatGPT Rich Typography) */
.ai-msg-card {
    background: transparent;
    color: #ececec;
    margin-bottom: 20px;
    font-size: 14.5px;
    line-height: 1.65;
}
.ai-msg-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 6px;
}
.ai-icon-dot {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #10a37f;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: bold; color: #000;
}
.ai-ground-tag {
    background: #064e3b; color: #6ee7b7;
    font-size: 10.5px; padding: 2px 7px; border-radius: 4px; font-weight: 600;
}

/* Metadata pills */
.ai-meta-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #2f2f2f;
    font-size: 11px;
    color: #737373;
}
.meta-chip {
    background: #262626;
    border: 1px solid #383838;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    color: #a3a3a3;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #262626;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* Button Styling */
.stButton > button {
    border-radius: 99px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    background: #262626 !important;
    border: 1px solid #383838 !important;
    color: #ececec !important;
}
.stButton > button:hover {
    background: #333333 !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
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
    st.session_state.messages = []

if "llm_turns" not in st.session_state:
    st.session_state.llm_turns = []

if "latencies" not in st.session_state:
    st.session_state.latencies = [142.0, 165.0, 130.0, 178.0]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

harness, chunk_count, vector_count = load_pipeline()
stt = load_stt()


# ── ChatGPT Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
      <span style="font-size:18px;font-weight:700;color:#fff;">ChatGPT</span>
      <span style="background:#2a2a2a;color:#a3a3a3;font-size:10px;padding:2px 6px;border-radius:4px;">VoxRAG</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✏️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm_turns = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("---")
    st.caption("📌 **Pinned**")
    if st.button("🏢 What is a corporation?", key="pin_1", use_container_width=True):
        st.session_state.pending_query = "What is a corporation?"
        st.rerun()
    if st.button("📋 What are its main types?", key="pin_2", use_container_width=True):
        st.session_state.pending_query = "What are its main types?"
        st.rerun()
    if st.button("⚡ Dense passage retrieval", key="pin_3", use_container_width=True):
        st.session_state.pending_query = "Explain dense passage retrieval"
        st.rerun()

    st.markdown("---")
    st.caption("🕒 **Recents**")
    st.markdown("""
    <div style="background:#262626;padding:6px 10px;border-radius:6px;font-size:12px;color:#fff;margin-bottom:4px;">
      ● MSMARCO-XI RAG Session
    </div>
    <div style="padding:6px 10px;font-size:12px;color:#737373;">
      ○ SIH 2026 Guidelines
    </div>
    <div style="padding:6px 10px;font-size:12px;color:#737373;">
      ○ Workshop Application Guidance
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔗 **External Guides**")
    st.markdown("[📊 Architecture Flowchart (Vercel)](https://docs-three-dusky-37.vercel.app)")
    st.markdown("[📂 GitHub Repository](https://github.com/gkm563/VoxRAG)")
    st.markdown("[📝 Google Submission Form](https://forms.gle/MNvCjcv23Hn2Eeu58)")

    st.markdown("---")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
      <div style="width:28px;height:28px;border-radius:50%;background:#ea580c;color:white;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;">GM</div>
      <div style="font-size:12px;font-weight:500;color:#fff;">Gautam Kumar Maurya</div>
    </div>
    """, unsafe_allow_html=True)


# ── Top Bar ───────────────────────────────────────────────────────────────────
col_top1, col_top2 = st.columns([5, 1])
with col_top1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
      <span style="font-size:16px;font-weight:600;color:#fff;">ChatGPT</span>
      <span style="color:#737373;font-size:14px;">4o-mini · Groq LPU</span>
    </div>
    """, unsafe_allow_html=True)
with col_top2:
    st.link_button("↗ Share Guide", "https://docs-three-dusky-37.vercel.app")


# ── Voice Recording Box ───────────────────────────────────────────────────────
with st.expander("🎙️ Record Voice with Microphone", expanded=False):
    st.caption("Click to record your question using your microphone:")
    audio_voice = st.audio_input("Record Question", key="mic_recorder_chatgpt")
    if audio_voice:
        import hashlib
        raw_audio = audio_voice.getvalue()
        audio_hash = hashlib.md5(raw_audio).hexdigest()

        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state.last_audio_hash = audio_hash
            if stt and harness:
                with st.spinner("📝 Transcribing speech with Sarvam AI / Groq Whisper…"):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(raw_audio)
                        tmp = f.name
                    try:
                        transcript, stt_ms = stt.from_file(tmp)
                        if transcript and transcript.strip():
                            st.session_state.pending_query = transcript.strip()
                            st.rerun()
                        else:
                            st.warning("Could not detect clear words. Please speak again.")
                    except Exception as e:
                        st.error(f"Voice Transcription Error: {e}")
                    finally:
                        try: os.unlink(tmp)
                        except: pass


# ── Check for Query to Run ────────────────────────────────────────────────────
query_to_run = None
if st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None

chat_input_val = st.chat_input("Ask anything")
if chat_input_val and chat_input_val.strip():
    query_to_run = chat_input_val.strip()


# ── Execute Query ─────────────────────────────────────────────────────────────
if query_to_run and harness:
    st.session_state.messages.append({
        "role": "user",
        "content": query_to_run,
        "time": time.strftime("%I:%M %p · %b %d"),
    })

    from pipeline.harness import PipelineInput
    with st.spinner("Thinking..."):
        inp = PipelineInput(
            query=query_to_run,
            top_k=5,
            history=st.session_state.llm_turns,
        )
        out = harness.run(inp)

        st.session_state.llm_turns.append({"role": "user", "content": query_to_run})
        if out.answer:
            st.session_state.llm_turns.append({"role": "assistant", "content": out.answer})

        st.session_state.messages.append({
            "role": "assistant",
            "content": out.answer if not out.blocked else f"🚫 Query Blocked: {out.block_reason}",
            "confidence": out.confidence,
            "grounded": out.grounded,
            "blocked": out.blocked,
            "total_ms": out.total_latency_ms,
            "sources": out.sources,
            "suggestions": getattr(out, "suggestions", []),
            "time": time.strftime("%I:%M %p · %b %d"),
        })

        if not out.blocked:
            st.session_state.latencies.append(out.total_latency_ms)


# ── Render Conversation Stream (ChatGPT Style) ────────────────────────────────
if not st.session_state.messages and not query_to_run:
    st.markdown("""
    <div class="ai-msg-card" style="margin-top:20px;">
      <div class="ai-msg-header">
        <div class="ai-icon-dot">✦</div>
        <span>VoxRAG</span>
        <span class="ai-ground-tag">✓ Ready</span>
      </div>
      <div style="color:#d1d5db;margin-top:4px;">
        Hello Gautam! I am <b>VoxRAG</b>, an ultra-fast Voice and Text Question Answering system grounded in <code>ai4bharat/MSMARCO-XI</code> (48,995 chunks).
        <br/><br/>
        Type or speak your question below to begin!
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏢 What is a corporation?", key="start_1", use_container_width=True):
            st.session_state.pending_query = "What is a corporation?"
            st.rerun()
    with col2:
        if st.button("🔍 What is MSMARCO dataset?", key="start_2", use_container_width=True):
            st.session_state.pending_query = "What is the MSMARCO dataset used for?"
            st.rerun()
    with col3:
        if st.button("⚡ How does FAISS search work?", key="start_3", use_container_width=True):
            st.session_state.pending_query = "How does FAISS vector search work?"
            st.rerun()

for m_idx, msg in enumerate(st.session_state.messages):
    t_str = msg.get("time", time.strftime("%I:%M %p · %b %d"))
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg-bubble">
          <div class="user-msg-header">
            <span>You</span>
            <span>🕒 {t_str}</span>
          </div>
          <div>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        conf_pct = int(msg.get("confidence", 0.88) * 100)
        lat_val  = round(msg.get("total_ms", 142.0), 1)
        grnd_tag = "✓ Grounded" if msg.get("grounded", True) else "⚠ Ungrounded"

        st.markdown(f"""
        <div class="ai-msg-card">
          <div class="ai-msg-header">
            <div class="ai-icon-dot">✦</div>
            <span>VoxRAG</span>
            <span class="ai-ground-tag">{grnd_tag}</span>
            <span style="font-size:11px;color:#737373;font-weight:normal;margin-left:4px;">🕒 {t_str} · ⚡ {lat_val}ms</span>
          </div>
          <div style="color:#d1d5db;padding-left:30px;line-height:1.65;">
            {msg['content']}
          </div>
          <div class="ai-meta-bar" style="padding-left:30px;">
            <span class="meta-chip">🎯 {conf_pct}% confidence</span>
            <span class="meta-chip">📚 MSMARCO-XI</span>
            <span class="meta-chip">⚡ Groq LPU</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic follow-up suggestion buttons
        sugs = msg.get("suggestions", [])
        if sugs:
            s_cols = st.columns(len(sugs))
            for s_i, s_text in enumerate(sugs):
                with s_cols[s_i]:
                    if st.button(f"👉 {s_text}", key=f"sug_btn_{m_idx}_{s_i}", use_container_width=True):
                        st.session_state.pending_query = s_text
                        st.rerun()
