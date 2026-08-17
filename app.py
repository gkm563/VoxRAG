"""
app.py — VoxRAG Streamlit Cloud App
Uses st.audio_input() for browser-native recording — no PortAudio needed.
"""

import os, sys, tempfile, time
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoxRAG — Voice-Enabled RAG",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hide Streamlit default chrome */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }

.main-header {
    background: linear-gradient(135deg, #0d0f14 0%, #12151c 100%);
    border: 1px solid #232838;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo-area { display: flex; align-items: center; gap: 14px; }
.logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.logo-title { font-size: 22px; font-weight: 700; letter-spacing: -0.03em; color: #e8eaf0; margin: 0; }
.logo-sub { font-size: 12px; color: #8892a4; margin: 2px 0 0; }
.status-area { display: flex; align-items: center; gap: 10px; }
.status-chip {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 99px;
    background: #12151c; border: 1px solid #232838;
    font-size: 12px; font-weight: 500; color: #8892a4;
}
.dot-green { width: 7px; height: 7px; border-radius: 50%; background: #00d4aa;
    box-shadow: 0 0 6px #00d4aa; display: inline-block; }
.live-chip {
    padding: 6px 16px; border-radius: 99px;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    font-size: 12px; font-weight: 700; color: white;
}

/* Cards */
.vox-card {
    background: #12151c;
    border: 1px solid #232838;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title { font-size: 14px; font-weight: 600; color: #e8eaf0; margin-bottom: 4px; }
.card-sub   { font-size: 11px; color: #8892a4; margin-bottom: 16px; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #00d4aa0d, #00d4aa06);
    border: 1px solid #00d4aa33;
    border-radius: 12px; padding: 16px 20px; margin: 12px 0;
}
.answer-box h4 { color: #00d4aa; font-size: 13px; margin: 0 0 8px; }
.answer-text { color: #e8eaf0; font-size: 14px; line-height: 1.7; }

/* Blocked box */
.blocked-box {
    background: #ff6b6b0d; border: 1px solid #ff6b6b33;
    border-radius: 12px; padding: 16px 20px; margin: 12px 0;
}
.blocked-box h4 { color: #ff6b6b; font-size: 13px; margin: 0 0 8px; }

/* Source chips */
.source-chip {
    display: flex; justify-content: space-between;
    background: #181c26; border: 1px solid #2a3045;
    border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;
    font-size: 12px;
}
.source-name { color: #6c63ff; font-family: 'JetBrains Mono', monospace; }
.source-score { color: #8892a4; }

/* Metrics */
.metric-row { display: flex; gap: 10px; margin: 12px 0; }
.met-card {
    flex: 1; background: #181c26; border: 1px solid #232838;
    border-radius: 10px; padding: 12px; text-align: center;
}
.met-val { font-size: 22px; font-weight: 700; color: #6c63ff; font-family: 'JetBrains Mono', monospace; }
.met-lbl { font-size: 10px; color: #8892a4; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }

/* Latency bars */
.lat-bar-row { margin: 6px 0; }
.lat-label { display: flex; justify-content: space-between; font-size: 11px; color: #8892a4; margin-bottom: 3px; }
.lat-track { background: #232838; border-radius: 4px; height: 6px; }
.lat-fill  { height: 6px; border-radius: 4px; transition: width 0.4s; }

/* Guardrail rows */
.guard-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid #232838; font-size: 12.5px;
}
.guard-row:last-child { border-bottom: none; }
.guard-pass { color: #00d4aa; font-weight: 600; font-size: 11px; }
.guard-fail { color: #ff6b6b; font-weight: 600; font-size: 11px; }

/* Pipeline flow */
.pipe-flow { display: flex; align-items: center; gap: 4px; }
.pipe-node { flex: 1; text-align: center; }
.pipe-ico  { font-size: 22px; margin-bottom: 4px; }
.pipe-name { font-size: 10px; font-weight: 600; color: #8892a4; }
.pipe-ms   { font-size: 10px; color: #4a5568; margin-top: 2px; }
.pipe-arr  { color: #4a5568; font-size: 12px; padding-bottom: 12px; }

/* Status bar */
.stat-bar {
    background: #12151c; border: 1px solid #232838; border-radius: 10px;
    padding: 10px 20px; display: flex; gap: 28px; align-items: center; margin-top: 8px;
}
.stat-item label { font-size: 10px; color: #4a5568; display: block; text-transform: uppercase; letter-spacing: 0.06em; }
.stat-item span  { font-size: 12px; font-weight: 600; color: #00d4aa; }

/* Override Streamlit button */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 8px 20px !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
div[data-testid="stAudioInput"] { background: #181c26 !important; border: 1px solid #2a3045 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load all pipeline components. Returns None if index not built yet."""
    try:
        import config
        from pipeline.retriever  import FAISSRetriever
        from pipeline.generator  import AnswerGenerator
        from pipeline.guardrails import Guardrails
        from pipeline.harness    import RAGHarness

        retriever  = FAISSRetriever.load(config.INDEX_PATH)
        generator  = AnswerGenerator()
        guardrails = Guardrails(embed_model=retriever.model)
        harness    = RAGHarness(retriever, generator, guardrails)
        return harness, {
            "chunks":  len(retriever.chunks),
            "vectors": retriever.index.ntotal,
        }
    except Exception as e:
        return None, {"error": str(e)}


@st.cache_resource(show_spinner=False)
def load_stt():
    """Load STT (Sarvam API — no system deps needed on cloud)."""
    try:
        from pipeline.stt import SpeechToText
        return SpeechToText(mode="sarvam")
    except Exception:
        return None


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading VoxRAG pipeline…"):
    harness, pipe_info = load_pipeline()
    stt = load_stt()

pipeline_ready = harness is not None

# ── Header ────────────────────────────────────────────────────────────────────
status_label = "All Systems Operational" if pipeline_ready else "Building Index…"
st.markdown(f"""
<div class="main-header">
  <div class="logo-area">
    <div class="logo-icon">🎙️</div>
    <div>
      <p class="logo-title">VoxRAG</p>
      <p class="logo-sub">Voice · Retrieve · Generate · #RAGInGoa</p>
    </div>
  </div>
  <div class="status-area">
    <div class="status-chip"><span class="dot-green"></span>{status_label}</div>
    <div class="live-chip">● Live</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── If pipeline not ready ─────────────────────────────────────────────────────
if not pipeline_ready:
    st.warning(f"""
    **⏳ Pipeline index not built yet.**

    Run this command locally to build the FAISS index, then redeploy:
    ```bash
    python build_index.py
    ```
    Or add `data/` folder to your repo after building.

    Error: `{pipe_info.get('error','Unknown')}`
    """)

# ── Main layout ───────────────────────────────────────────────────────────────
col_main, col_right = st.columns([1.8, 1], gap="large")

with col_main:
    # ── Voice / Text input ────────────────────────────────────────────────────
    st.markdown("""<div class="vox-card">
      <div class="card-title">🎙️ Ask your question</div>
      <div class="card-sub">Record voice or type below — powered by Sarvam AI STT + Groq LLM</div>
    </div>""", unsafe_allow_html=True)

    tab_voice, tab_text = st.tabs(["🎙️ Voice Input", "⌨️ Text Input"])

    query      = None
    stt_ms     = 0.0
    transcript = None

    # ── Voice tab ─────────────────────────────────────────────────────────────
    with tab_voice:
        st.markdown("<br>", unsafe_allow_html=True)
        audio_bytes = st.audio_input(
            "Click the mic button to record your question",
            key="voice_recorder",
        )
        if audio_bytes and pipeline_ready:
            if stt is None:
                st.error("STT not loaded. Check SARVAM_API_KEY in secrets.")
            else:
                with st.spinner("📝 Transcribing with Sarvam AI…"):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_bytes.getvalue())
                        tmp = f.name
                    try:
                        transcript, stt_ms = stt.from_file(tmp)
                        st.success(f"📝 Transcribed: **\"{transcript}\"** *(STT: {stt_ms:.0f}ms)*")
                        query = transcript
                    except Exception as e:
                        st.error(f"STT Error: {e}")
                    finally:
                        try: os.unlink(tmp)
                        except: pass

    # ── Text tab ──────────────────────────────────────────────────────────────
    with tab_text:
        st.markdown("<br>", unsafe_allow_html=True)
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            text_query = st.text_input(
                "Query",
                placeholder="e.g. What is the MSMARCO dataset used for?",
                label_visibility="collapsed",
            )
        with col_btn:
            ask_btn = st.button("Ask ➤", use_container_width=True)
        if ask_btn and text_query.strip():
            query = text_query.strip()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if query and pipeline_ready:
        with st.spinner("🔍 Retrieving context and generating answer…"):
            from pipeline.harness import PipelineInput
            inp = PipelineInput(query=query, top_k=5, stt_latency=stt_ms)
            t0  = time.perf_counter()
            out = harness.run(inp)
            total_ms = out.total_latency_ms

        st.markdown("---")

        # ── Answer ────────────────────────────────────────────────────────────
        if out.blocked:
            st.markdown(f"""<div class="blocked-box">
              <h4>🚫 Query Blocked by Guardrails</h4>
              <div>{out.block_reason}</div>
            </div>""", unsafe_allow_html=True)
        else:
            grounded_icon = "✅" if out.grounded else "⚠️"
            st.markdown(f"""<div class="answer-box">
              <h4>✦ VoxRAG Answer &nbsp; {grounded_icon} {'Grounded' if out.grounded else 'Check sources'}</h4>
              <div class="answer-text">{out.answer}</div>
            </div>""", unsafe_allow_html=True)

            # Metrics
            conf_pct = int(out.confidence * 100)
            under    = total_ms < 200
            st.markdown(f"""<div class="metric-row">
              <div class="met-card"><div class="met-val">{conf_pct}%</div><div class="met-lbl">Confidence</div></div>
              <div class="met-card"><div class="met-val" style="color:{'#00d4aa' if under else '#ff6b6b'}">{total_ms:.0f}ms</div><div class="met-lbl">{'✅ Under 200ms' if under else '⚠️ Over 200ms'}</div></div>
              <div class="met-card"><div class="met-val">{len(out.sources)}</div><div class="met-lbl">Sources Cited</div></div>
            </div>""", unsafe_allow_html=True)

            # Sources
            retrieved, _ = harness.retriever.search(query, 5)
            if retrieved:
                st.markdown("**📚 Retrieved Context Chunks**")
                for i, c in enumerate(retrieved, 1):
                    with st.expander(f"Chunk {i} · score={c.get('score',0):.3f} · {c.get('strategy','—')} · passage {c.get('passage_id','—')}"):
                        st.write(c["text"])

        # ── Latency breakdown ─────────────────────────────────────────────────
        st.markdown("**⏱️ Latency Breakdown**")
        for stage, ms in out.latency.items():
            if stage == "total": continue
            pct   = min(int((ms / 200) * 100), 100)
            color = "#6c63ff" if ms < 80 else "#ffd93d" if ms < 160 else "#ff6b6b"
            st.markdown(f"""<div class="lat-bar-row">
              <div class="lat-label"><span>{stage}</span><span>{ms:.1f}ms</span></div>
              <div class="lat-track"><div class="lat-fill" style="width:{pct}%;background:{color};"></div></div>
            </div>""", unsafe_allow_html=True)

        # Copy answer button
        if not out.blocked:
            st.code(out.answer, language=None)

    elif query and not pipeline_ready:
        st.info("⏳ Pipeline index not built yet. Run `python build_index.py` locally first.")

    # ── Status bar ────────────────────────────────────────────────────────────
    chunks_label  = f"{pipe_info.get('chunks', 0):,}"  if pipeline_ready else "Building…"
    vectors_label = f"{pipe_info.get('vectors', 0):,}" if pipeline_ready else "Building…"
    st.markdown(f"""<div class="stat-bar">
      <div class="stat-item"><label>Dataset</label><span>MSMARCO-XI</span></div>
      <div class="stat-item"><label>Chunks</label><span>{chunks_label}</span></div>
      <div class="stat-item"><label>Vectors</label><span>{vectors_label}</span></div>
      <div class="stat-item"><label>LLM</label><span>Groq Llama 3</span></div>
      <div class="stat-item"><label>STT</label><span style="color:#4dabf7">Sarvam AI</span></div>
      <div class="stat-item"><label>Status</label><span>{'Ready' if pipeline_ready else 'Indexing'}</span></div>
    </div>""", unsafe_allow_html=True)


# ── Right panel ───────────────────────────────────────────────────────────────
with col_right:

    # Pipeline overview
    st.markdown("""<div class="vox-card">
      <div class="card-title">Pipeline Overview</div>
      <div class="card-sub">End-to-end flow</div>
      <div class="pipe-flow">
        <div class="pipe-node"><div class="pipe-ico">🎙️</div><div class="pipe-name">Voice Input</div><div class="pipe-ms">Sarvam STT</div></div>
        <div class="pipe-arr">›</div>
        <div class="pipe-node"><div class="pipe-ico">✂️</div><div class="pipe-name">Chunking</div><div class="pipe-ms">4 Strategies</div></div>
        <div class="pipe-arr">›</div>
        <div class="pipe-node"><div class="pipe-ico">🔍</div><div class="pipe-name">Retrieval</div><div class="pipe-ms">FAISS</div></div>
        <div class="pipe-arr">›</div>
        <div class="pipe-node"><div class="pipe-ico">⚡</div><div class="pipe-name">Generate</div><div class="pipe-ms">Groq LLM</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Latency analytics (session-based)
    if "lat_history" not in st.session_state:
        st.session_state.lat_history = []

    lats = st.session_state.lat_history
    if "out" in dir() and 'out' in locals() and not out.blocked:
        lats.append(out.total_latency_ms)
        st.session_state.lat_history = lats[-50:]

    import numpy as np
    p50  = round(float(np.percentile(lats, 50)), 1)  if lats else 0
    p70  = round(float(np.percentile(lats, 70)), 1)  if lats else 0
    p100 = round(float(np.percentile(lats, 100)), 1) if lats else 0

    st.markdown(f"""<div class="vox-card">
      <div class="card-title">Latency Analytics</div>
      <div class="card-sub">{len(lats)} queries measured</div>
      <div class="metric-row">
        <div class="met-card">
          <div class="met-val" style="color:#00d4aa">{p50 or '—'}</div>
          <div class="met-lbl">P50</div>
        </div>
        <div class="met-card">
          <div class="met-val" style="color:#ffd93d">{p70 or '—'}</div>
          <div class="met-lbl">P70</div>
        </div>
        <div class="met-card">
          <div class="met-val" style="color:#ff6b6b">{p100 or '—'}</div>
          <div class="met-lbl">P100</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if lats:
        import pandas as pd
        chart_data = pd.DataFrame({"Latency (ms)": lats})
        st.line_chart(chart_data, color="#6c63ff", height=100, use_container_width=True)
        st.caption("200ms target line not shown — aim for all bars below 200ms")

    # Guardrails
    guards = {"off_topic": True, "safety": True, "hallucination": True, "grounded": True}
    if "out" in dir() and 'out' in locals() and query:
        if out.blocked:
            guards = {"off_topic": False, "safety": False, "hallucination": True, "grounded": True}
        else:
            guards["hallucination"] = out.grounded
            guards["grounded"]      = out.grounded

    all_pass     = all(guards.values())
    guard_labels = {
        "off_topic":     "Off-topic Detection",
        "safety":        "Safety & Toxicity",
        "hallucination": "Hallucination Check",
        "grounded":      "Grounded in Context",
    }
    rows = "".join([
        f"""<div class="guard-row">
          <span>{'✅' if v else '❌'} {guard_labels[k]}</span>
          <span class="{'guard-pass' if v else 'guard-fail'}">{'Passed' if v else 'Failed'}</span>
        </div>"""
        for k, v in guards.items()
    ])
    st.markdown(f"""<div class="vox-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div class="card-title" style="margin:0">Guardrails</div>
        <span style="font-size:11px;font-weight:600;color:{'#00d4aa' if all_pass else '#ff6b6b'}">
          {'All checks passed' if all_pass else 'Issues detected'}
        </span>
      </div>
      {rows}
    </div>""", unsafe_allow_html=True)
