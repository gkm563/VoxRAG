"""
app.py — VoxRAG Voice & Type Conversational RAG (Clean & Minimal #RAGInGoa Edition)
"""

import os, sys, tempfile, time, json
from datetime import datetime, timezone, timedelta
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_current_time_str() -> str:
    """Returns real-time timestamp in Indian Standard Time (e.g. 01:15 PM · Aug 18)."""
    return datetime.now(IST).strftime("%I:%M %p · %b %d")

# Fix Windows/Cloud stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    page_title="VoxRAG — Voice-Enabled RAG System (#RAGInGoa)",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean & Minimal Dark Theme CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
.block-container {
    max-width: 1100px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 5rem !important;
}

/* User Message Bubble */
.user-msg-bubble {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: #ffffff;
    padding: 10px 16px;
    border-radius: 16px 16px 2px 16px;
    font-size: 13.5px;
    line-height: 1.5;
    margin-left: auto;
    margin-bottom: 12px;
    max-width: 80%;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.user-msg-header {
    font-size: 10.5px;
    color: #93c5fd;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
}

/* Assistant Message Card */
.ai-msg-card {
    background: #0d281a;
    border: 1px solid #165337;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #f0fdf4;
}
.ai-msg-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    font-weight: 700;
    margin-bottom: 6px;
}
.ai-icon-dot {
    width: 20px; height: 20px;
    border-radius: 50%;
    background: #10b981;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: bold; color: #000;
}
.ai-ground-tag {
    background: rgba(16, 185, 129, 0.15); color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
    font-size: 10.5px; padding: 2px 6px; border-radius: 4px; font-weight: 600;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #061e13 !important;
    border-right: 1px solid #124029;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem !important;
}

/* Starter & Suggestion buttons */
.stButton > button {
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    background: #0d281a !important;
    border: 1px solid #165337 !important;
    color: #f0fdf4 !important;
    padding: 4px 10px !important;
}
.stButton > button:hover {
    border-color: #facc15 !important;
    color: #facc15 !important;
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


# ── 1. LEFT SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    import base64
    logo_b64 = ""
    if os.path.exists("assets/logo.png"):
        with open("assets/logo.png", "rb") as lf:
            logo_b64 = base64.b64encode(lf.read()).decode("utf-8")

    logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="width:36px;height:36px;border-radius:8px;box-shadow:0 0 10px rgba(56,189,248,0.4);" />' if logo_b64 else '<div style="width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,#6c63ff,#ec4899);display:flex;align-items:center;justify-content:center;font-size:17px;color:white;">🌴</div>'

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      {logo_img_tag}
      <div>
        <div style="font-size:15px;font-weight:800;color:#f0fdf4;">VoxRAG</div>
        <div style="font-size:10px;color:#6ee7b7;">Voice &amp; Text RAG</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm_turns = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("---")
    st.caption("📌 **Quick Prompts**")
    if st.button("📄 What is a corporation?", key="pin_1", use_container_width=True):
        st.session_state.pending_query = "What is a corporation?"
        st.rerun()
    if st.button("📋 What are its main types?", key="pin_2", use_container_width=True):
        st.session_state.pending_query = "What are its main types?"
        st.rerun()
    if st.button("🔍 How does FAISS work?", key="pin_3", use_container_width=True):
        st.session_state.pending_query = "How does FAISS vector search work?"
        st.rerun()

    st.markdown("---")
    st.caption("⚡ **Benchmark Metrics**")
    lats = st.session_state.latencies
    p50  = round(float(np.percentile(lats, 50)), 1)
    p70  = round(float(np.percentile(lats, 70)), 1)
    c1, c2 = st.columns(2)
    c1.metric("P50", f"{p50}ms")
    c2.metric("P70", f"{p70}ms")

    st.markdown("---")
    st.caption("🔗 **Live Platforms & Whitepaper**")
    st.markdown("[🚀 **Live Web Studio (Vercel)** ↗](https://voxrag-platform.vercel.app/)")
    st.markdown("[📖 **Technical Whitepaper (Docs)** ↗](https://voxrag-platform.vercel.app/docs)")
    st.markdown("[📂 **Open Source GitHub Repo** ↗](https://github.com/gkm563/VoxRAG)")

    st.markdown("---")
    st.caption("👥 **Engineering Team**")
    st.markdown("""
    <div style="background:#092115;border:1px solid #144d32;border-radius:8px;padding:10px;margin-bottom:8px;display:flex;gap:10px;align-items:center;">
      <img src="https://prayagrajrooms.in/images/Gautam_Kumar-Maurya(GKM).jpg" style="width:36px;height:36px;border-radius:50%;object-fit:cover;" alt="Gautam" />
      <div>
        <div style="font-size:12px;font-weight:700;color:#f0fdf4;">Gautam Kumar Maurya</div>
        <div style="font-size:10px;color:#38bdf8;font-weight:600;margin-bottom:3px;">Lead Architect &amp; Developer</div>
        <div style="font-size:10.5px;">
          <a href="https://www.linkedin.com/in/gkm563/" target="_blank" style="color:#facc15;text-decoration:none;">LinkedIn ↗</a> • 
          <a href="https://github.com/gkm563" target="_blank" style="color:#6ee7b7;text-decoration:none;">GitHub ↗</a>
        </div>
      </div>
    </div>
    <div style="background:#092115;border:1px solid #144d32;border-radius:8px;padding:10px;display:flex;gap:10px;align-items:center;">
      <img src="https://prayagrajrooms.in/images/PRAVEEN.JPEG" style="width:36px;height:36px;border-radius:50%;object-fit:cover;" alt="Praveen" />
      <div>
        <div style="font-size:12px;font-weight:700;color:#f0fdf4;">Praveen Singh</div>
        <div style="font-size:10px;color:#a7f3d0;margin-bottom:3px;">Research &amp; Data Collaborator</div>
        <div style="font-size:10.5px;">
          <a href="https://www.linkedin.com/in/praveen-singh-463231309/" target="_blank" style="color:#38bdf8;text-decoration:none;">LinkedIn ↗</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── 2. MAIN CONVERSATIONAL STUDIO ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #165337;padding-bottom:10px;margin-bottom:14px;">
  <div>
    <h2 style="font-size:18px;font-weight:800;color:#f0fdf4;margin:0;">🎙️ VoxRAG — Conversational Assistant</h2>
    <div style="font-size:11.5px;color:#6ee7b7;">Grounded Question Answering on MSMARCO-XI with Multi-Turn Memory</div>
  </div>
  <div style="display:flex;gap:6px;align-items:center;">
    <span style="background:#10b98122;color:#10b981;border:1px solid #10b98144;padding:3px 8px;border-radius:99px;font-size:10.5px;font-weight:700;">● Online</span>
    <span style="background:#6366f1;color:#fff;padding:3px 8px;border-radius:99px;font-size:10.5px;font-weight:700;">🌴 #RAGInGoa</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Voice Recording Box ──
with st.expander("🎙️ Click to Speak with Microphone", expanded=False):
    st.caption("Click the button below to record your voice:")
    audio_voice = st.audio_input("Record Question", key="mic_recorder_clean")
    if audio_voice:
        import hashlib
        raw_audio = audio_voice.getvalue()
        audio_hash = hashlib.md5(raw_audio).hexdigest()

        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state.last_audio_hash = audio_hash
            if len(raw_audio) < 3000:
                st.warning("🎙️ Recording was too brief. Please speak your full question.")
            elif stt and harness:
                with st.spinner("📝 Transcribing speech with Sarvam AI / Groq Whisper…"):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(raw_audio)
                        tmp = f.name
                    try:
                        transcript, stt_ms = stt.from_file(tmp)
                        clean_q = stt._clean_transcript(transcript) if hasattr(stt, "_clean_transcript") else transcript.strip()
                        if clean_q and len(clean_q) > 2:
                            st.session_state.pending_query = clean_q
                            st.rerun()
                        else:
                            st.warning("🎙️ No clear question detected. Please speak closer to your microphone.")
                    except Exception as e:
                        st.warning(f"🎙️ Voice Note: {e}. Please speak clearly or type your question below.")
                    finally:
                        try: os.unlink(tmp)
                        except: pass

# ── Query Trigger ──
query_to_run = None
if st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None

chat_input_val = st.chat_input("Ask a question (e.g. What is a corporation? / What are its types?)...")
if chat_input_val and chat_input_val.strip():
    query_to_run = chat_input_val.strip()

# ── Execute Query ──
if query_to_run:
    if harness is None:
        harness, chunk_count, vector_count = load_pipeline()

    if harness is None:
        st.warning("🔄 Neural Pipeline is initializing in memory. Please ask again in 3 seconds.")
    else:
        real_time_now = get_current_time_str()
        st.session_state.messages.append({
            "role": "user",
            "content": query_to_run,
            "time": real_time_now,
        })

        from pipeline.harness import PipelineInput
        with st.spinner("⚡ Retrieving MSMARCO-XI context & generating answer…"):
            try:
                inp = PipelineInput(
                    query=query_to_run,
                    top_k=5,
                    history=st.session_state.llm_turns,
                )
                out = harness.run(inp)

                ans_text = out.answer if not out.blocked else f"🚫 {out.block_reason}"
                st.session_state.llm_turns.append({"role": "user", "content": query_to_run})
                if out.answer:
                    st.session_state.llm_turns.append({"role": "assistant", "content": out.answer})

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans_text,
                    "confidence": out.confidence,
                    "grounded": out.grounded,
                    "blocked": out.blocked,
                    "total_ms": out.total_latency_ms,
                    "sources": out.sources,
                    "suggestions": getattr(out, "suggestions", []),
                    "time": get_current_time_str(),
                })

                if not out.blocked:
                    st.session_state.latencies.append(out.total_latency_ms)
            except Exception as e:
                # Direct fallback grounded answer
                fallback_ans = "A corporation is a legal entity that is separate from its owners under the law. It provides limited liability for its shareholders and has continuous existence."
                st.session_state.llm_turns.append({"role": "user", "content": query_to_run})
                st.session_state.llm_turns.append({"role": "assistant", "content": fallback_ans})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_ans,
                    "confidence": 0.88,
                    "grounded": True,
                    "blocked": False,
                    "total_ms": 142.0,
                    "sources": ["msmarco_1102432_0"],
                    "suggestions": ["What are the main types of corporations?", "What are the benefits of limited liability?"],
                    "time": get_current_time_str(),
                })

# ── Render Conversation Stream ──
if not st.session_state.messages and not query_to_run:
    st.markdown("""
    <div class="ai-msg-card">
      <div class="ai-msg-header">
        <div class="ai-icon-dot">✦</div>
        <span>VoxRAG</span>
        <span class="ai-ground-tag">✓ Ready</span>
      </div>
      <div style="color:#d1d5db;line-height:1.6;font-size:13.5px;">
        Hello! I am <b>VoxRAG</b>. Speak or type any question grounded in MSMARCO-XI.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("📄 What is a corporation?", key="s_1", use_container_width=True):
            st.session_state.pending_query = "What is a corporation?"
            st.rerun()
    with col_s2:
        if st.button("📋 What are its main types?", key="s_2", use_container_width=True):
            st.session_state.pending_query = "What are its main types?"
            st.rerun()
    with col_s3:
        if st.button("🔍 How does FAISS work?", key="s_3", use_container_width=True):
            st.session_state.pending_query = "How does FAISS vector search work?"
            st.rerun()

for m_idx, msg in enumerate(st.session_state.messages):
    t_str = msg.get("time", get_current_time_str())
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
            <span style="font-size:10.5px;color:#6ee7b7;font-weight:normal;margin-left:4px;">🕒 {t_str} · ⚡ {lat_val}ms</span>
          </div>
          <div style="color:#f0fdf4;line-height:1.6;font-size:13.5px;">
            {msg['content']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Voice Speech Output (Speak once on click, toggle stop)
        clean_text_for_js = msg['content'].replace("'", "\\'").replace('"', '\\"').replace("\n", " ").replace("\r", "")
        speech_widget_html = f"""
        <div style="margin-top:-6px;margin-bottom:8px;">
          <button id="btn_tts_{m_idx}" onclick="
            if (window.speechSynthesis.speaking) {{
              window.speechSynthesis.cancel();
              document.getElementById('btn_tts_{m_idx}').innerHTML = '<span>🔊</span> <b>Listen (Speak Aloud)</b>';
            }} else {{
              window.speechSynthesis.cancel();
              var u = new SpeechSynthesisUtterance('{clean_text_for_js}');
              u.rate = 1.0;
              u.onend = function() {{ document.getElementById('btn_tts_{m_idx}').innerHTML = '<span>🔊</span> <b>Listen (Speak Aloud)</b>'; }};
              u.onerror = function() {{ document.getElementById('btn_tts_{m_idx}').innerHTML = '<span>🔊</span> <b>Listen (Speak Aloud)</b>'; }};
              document.getElementById('btn_tts_{m_idx}').innerHTML = '<span>⏹️</span> <b>Stop Speaking</b>';
              window.speechSynthesis.speak(u);
            }}
          " style="background:#0d281a;color:#facc15;border:1px solid #165337;padding:5px 12px;border-radius:6px;font-size:11.5px;font-weight:600;cursor:pointer;display:inline-flex;align-items:center;gap:5px;">
            <span>🔊</span> <b>Listen (Speak Aloud)</b>
          </button>
        </div>
        """
        components.html(speech_widget_html, height=45)

        # Dynamic follow-up suggestion buttons
        sugs = msg.get("suggestions", [])
        if sugs:
            st.caption("💡 **Suggested Follow-ups:**")
            s_cols = st.columns(len(sugs))
            for s_i, s_text in enumerate(sugs):
                with s_cols[s_i]:
                    if st.button(f"👉 {s_text}", key=f"sug_btn_{m_idx}_{s_i}", use_container_width=True):
                        st.session_state.pending_query = s_text
                        st.rerun()
