"""
Enterprise Policy Compliance Auditor — Phase 5A redesign.
All backend logic (RAG, agent, Snowflake) is unchanged.
Only the presentation layer has been redesigned.
"""

import gc
import json
import os
import time

from conflict_renderer import parse_conflict_item, conflict_card_html

import streamlit as st
from dotenv import load_dotenv

from database import (
    CHROMA_DB_DIR,
    _COLLECTION_NAME,
    _detect_legacy_collection_name,
    _is_legacy_index,
    _pointer_file,
    _read_pointer,
    _slot_dir,
    get_embeddings,
    index_documents,
)
from main import ask_question, MODEL_NAME
from snowflake_logger import (
    create_logs_table_if_not_exists,
    is_snowflake_configured,
    log_query_to_snowflake,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Policy Compliance Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — config.toml forces the light base theme.
# CSS here handles sidebar colours, header, badges, conflict cards.
# We do NOT wrap Streamlit widgets in custom div tags (they don't nest).
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] {
    background: #102A4D !important;
}
[data-testid="stSidebar"] *:not(button) {
    color: #C8D8E8 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1E3A5A !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #C8D8E8 !important;
    text-align: left !important;
    width: 100% !important;
    padding: 0.55rem 0.75rem !important;
    border-radius: 6px !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    margin: 1px 0 !important;
    box-shadow: none !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.09) !important;
    color: #FFFFFF !important;
}

/* ── Main layout ──────────────────────────────────────────────────────────── */
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1300px !important;
}

/* ── App header banner ────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #102A4D 0%, #1E4080 100%);
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.25rem;
    border-left: 5px solid #2563EB;
}
.app-header h1 {
    margin: 0 0 0.4rem;
    font-size: 1.65rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
}
.app-header p {
    margin: 0;
    font-size: 0.9rem;
    color: #9BB8D4;
    line-height: 1.55;
    max-width: 820px;
}

/* ── Section heading inside containers ────────────────────────────────────── */
.sec-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    margin: 0 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #EEF2F9;
}

/* ── Status pills ─────────────────────────────────────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
    margin: 2px 3px;
}
.pill-ok    { background:#DCFCE7; color:#14532D; }
.pill-warn  { background:#FEF9C3; color:#713F12; }
.pill-error { background:#FEE2E2; color:#7F1D1D; }
.pill-info  { background:#DBEAFE; color:#1E3A5C; }
.pill-muted { background:#F1F5F9; color:#475569; }

/* ── Answer badges ────────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    border: 1px solid transparent;
}
.badge-success { background:#D1FAE5; color:#065F46; border-color:#6EE7B7; }
.badge-warning { background:#FEF3C7; color:#78350F; border-color:#FCD34D; }
.badge-error   { background:#FEE2E2; color:#7F1D1D; border-color:#FCA5A5; }
.badge-info    { background:#DBEAFE; color:#1E3A5C; border-color:#93C5FD; }

/* ── Conflict card ────────────────────────────────────────────────────────── */
.conflict-card {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 4px solid #EA580C;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin: 0.6rem 0;
}
.conflict-card h4 {
    margin: 0 0 0.65rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: #9A3412;
}
/* Two-column policy comparison grid */
.conflict-compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}
@media (max-width: 640px) {
    .conflict-compare { grid-template-columns: 1fr; }
}
.conflict-policy-box {
    background: #FFFBF0;
    border: 1px solid #FCD34D;
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
}
.conflict-policy-label {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #92400E;
    margin-bottom: 0.25rem;
}
.conflict-policy-name {
    font-size: 0.87rem;
    font-weight: 600;
    color: #7C2D12;
    margin-bottom: 0.35rem;
    word-break: break-word;
}
.conflict-excerpt {
    font-size: 0.82rem;
    color: #431407;
    font-style: italic;
    line-height: 1.55;
}
.conflict-explanation {
    font-size: 0.88rem;
    color: #431407;
    line-height: 1.65;
    margin-top: 0.6rem;
    padding-top: 0.55rem;
    border-top: 1px solid #FED7AA;
}
.conflict-explanation-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #92400E;
    margin-bottom: 0.25rem;
}
.conflict-footer {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    color: #7F1D1D;
    background: #FEE2E2;
    padding: 3px 9px;
    border-radius: 4px;
    margin-top: 0.6rem;
}

/* ── Document row ─────────────────────────────────────────────────────────── */
.doc-list { margin: 0.25rem 0; }
.doc-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.55rem 0;
    border-bottom: 1px solid #EEF2F9;
    font-size: 0.9rem;
    color: #172B4D;
}
.doc-row:last-child { border-bottom: none; }
.doc-name  { flex: 1; font-weight: 500; }
.doc-meta  { font-size: 0.78rem; color: #64748B; }

/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #94A3B8;
}
.empty-state .es-icon { font-size: 2.4rem; margin-bottom: 0.6rem; }
.empty-state h3 { color: #64748B; font-size: 1rem; margin: 0 0 0.3rem; font-weight: 600; }
.empty-state p  { font-size: 0.85rem; margin: 0; }

/* border containers: match our card style */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #D7E1EF !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(16,42,77,0.06) !important;
    background: #FFFFFF !important;
}

/* ── Suppress Streamlit default chrome ────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_indexed_docs() -> list[dict]:
    """Read source_filename + chunk counts from the active ChromaDB slot.
    Never modifies any index. Returns [] on any error.
    """
    try:
        base = CHROMA_DB_DIR
        from langchain_community.vectorstores import Chroma as ChromaVS

        if _is_legacy_index(base):
            coll_name = _detect_legacy_collection_name(base)
            vs = ChromaVS(
                persist_directory=base,
                embedding_function=get_embeddings(),
                collection_name=coll_name,
            )
        else:
            ptr = _pointer_file(base)
            if not os.path.exists(ptr):
                return []
            active_slot, _ = _read_pointer(base)
            slot_dir = _slot_dir(active_slot, base)
            if not os.path.exists(slot_dir):
                return []
            vs = ChromaVS(
                persist_directory=slot_dir,
                embedding_function=get_embeddings(),
                collection_name=_COLLECTION_NAME,
            )

        result = vs.get(include=["metadatas"])
        vs._client.close()
        del vs
        gc.collect()

        counts: dict[str, int] = {}
        for m in (result.get("metadatas") or []):
            name = m.get("source_filename") or "Unknown"
            counts[name] = counts.get(name, 0) + 1
        return [{"filename": k, "chunks": v} for k, v in sorted(counts.items())]
    except Exception:
        return []


def _badge(label: str, level: str) -> str:
    return f'<span class="badge badge-{level}">{label}</span>'


def _pill(label: str, level: str = "info") -> str:
    return f'<span class="pill pill-{level}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Snowflake bootstrap
# ─────────────────────────────────────────────────────────────────────────────
snowflake_enabled = is_snowflake_configured()
if snowflake_enabled and "snowflake_table_ready" not in st.session_state:
    ok, msg = create_logs_table_if_not_exists()
    st.session_state["snowflake_table_ready"] = ok
    if not ok:
        st.session_state["snowflake_table_error"] = msg

# ─────────────────────────────────────────────────────────────────────────────
# Navigation state
# ─────────────────────────────────────────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "analysis"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.1rem 0.5rem 0.9rem;
                border-bottom:1px solid #1E3A5A;margin-bottom:0.9rem">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
                  text-transform:uppercase;color:#60A5FA;margin-bottom:0.2rem">
        Enterprise Platform
      </div>
      <div style="font-size:1.1rem;font-weight:700;color:#FFFFFF;line-height:1.3">
        Policy Compliance<br>Auditor
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#7A9AB8;margin:0 0 0.4rem 0.1rem">Navigation</p>',
        unsafe_allow_html=True,
    )

    # Nav buttons — use plain Streamlit buttons; active state via CSS
    if st.button("🔍  Policy Analysis", key="nav_analysis", use_container_width=True):
        st.session_state["nav_page"] = "analysis"
        st.rerun()

    if st.button("📂  Documents", key="nav_documents", use_container_width=True):
        st.session_state["nav_page"] = "documents"
        st.rerun()

    # Mark active page with an arrow indicator
    active = st.session_state["nav_page"]
    st.markdown(
        f'<p style="font-size:0.72rem;color:#60A5FA;margin:-0.3rem 0 0.5rem 0.8rem">'
        f'{"▶ Policy Analysis" if active == "analysis" else "▶ Documents"}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#7A9AB8;margin:0.75rem 0 0.4rem 0.1rem">Coming Soon</p>',
        unsafe_allow_html=True,
    )
    st.button("📋  Investigation History", disabled=True, key="nav_h", use_container_width=True)
    st.button("📊  Reports",               disabled=True, key="nav_r", use_container_width=True)
    st.button("⚙️   Settings",              disabled=True, key="nav_s", use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    api_key = os.getenv("NVIDIA_API_KEY")
    api_pill = (
        '<span class="pill pill-ok" style="font-size:0.72rem">● Active</span>'
        if api_key
        else '<span class="pill pill-error" style="font-size:0.72rem">● Missing</span>'
    )
    sf_pill = (
        '<span class="pill pill-ok" style="font-size:0.72rem">● On</span>'
        if snowflake_enabled
        else '<span class="pill pill-muted" style="font-size:0.72rem">● Off</span>'
    )
    st.markdown(f"""
    <table style="width:100%;font-size:0.82rem;border-collapse:collapse">
      <tr>
        <td style="padding:3px 0;color:#9BB8D4">NVIDIA API</td>
        <td style="text-align:right">{api_pill}</td>
      </tr>
      <tr>
        <td style="padding:3px 0;color:#9BB8D4">Audit Log</td>
        <td style="text-align:right">{sf_pill}</td>
      </tr>
      <tr>
        <td style="padding:3px 0;color:#9BB8D4">LLM</td>
        <td style="text-align:right;font-size:0.72rem;color:#60A5FA">Nemotron</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>Enterprise Policy Compliance Auditor</h1>
  <p>
    Upload your organisation's policy documents, then ask compliance questions across all indexed
    sources. The system retrieves relevant policy text, assesses supporting evidence, and flags
    cross-document conflicts requiring human review.
  </p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Documents
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state["nav_page"] == "documents":

    col_up, col_idx = st.columns([1, 1], gap="large")

    # ── Upload panel ──────────────────────────────────────────────────────
    with col_up:
        with st.container(border=True):
            st.markdown('<div class="sec-title">Upload Documents</div>',
                        unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Select PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            if uploaded_files:
                rows_html = "".join(
                    f'<div class="doc-row">'
                    f'<span>📄</span>'
                    f'<span class="doc-name">{f.name}</span>'
                    f'</div>'
                    for f in uploaded_files
                )
                st.markdown(
                    f'<p style="font-weight:600;margin:0 0 0.5rem">'
                    f'{len(uploaded_files)} file(s) selected</p>'
                    f'<div class="doc-list">{rows_html}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                if st.button("⬆️  Index Documents", type="primary",
                             use_container_width=True):
                    for key in ("answer", "sources", "last_question", "elapsed",
                                "log_status", "agent_result", "_cached_docs"):
                        st.session_state.pop(key, None)
                    st.session_state["indexed_filenames"] = [f.name for f in uploaded_files]
                    with st.spinner("Processing and indexing PDFs…"):
                        ok, msg, doc_count, chunk_count, filenames = index_documents(
                            uploaded_files
                        )
                    if ok:
                        st.success(
                            f"Indexed **{doc_count} document(s)** "
                            f"into **{chunk_count} chunks**."
                        )
                        st.session_state.pop("_cached_docs", None)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.markdown("""
                <div class="empty-state">
                  <div class="es-icon">📂</div>
                  <h3>No files selected</h3>
                  <p>Use the file picker above to choose one or more policy PDF documents.</p>
                </div>
                """, unsafe_allow_html=True)

    # ── Indexed document inventory ─────────────────────────────────────────
    with col_idx:
        with st.container(border=True):
            st.markdown('<div class="sec-title">Indexed Document Inventory</div>',
                        unsafe_allow_html=True)

            if "_cached_docs" not in st.session_state:
                with st.spinner("Reading index…"):
                    st.session_state["_cached_docs"] = _get_indexed_docs()

            indexed_docs = st.session_state["_cached_docs"]

            if indexed_docs:
                total_chunks = sum(d["chunks"] for d in indexed_docs)
                m1, m2 = st.columns(2)
                m1.metric("Documents", len(indexed_docs))
                m2.metric("Total chunks", total_chunks)

                rows_html = "".join(
                    f'<div class="doc-row">'
                    f'<span>📄</span>'
                    f'<span class="doc-name">{d["filename"]}</span>'
                    f'<span class="doc-meta">{d["chunks"]} chunks</span>'
                    f'</div>'
                    for d in indexed_docs
                )
                st.markdown(f'<div class="doc-list">{rows_html}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<p style="margin-top:0.75rem;font-size:0.78rem;color:#64748B">'
                    f'Active database: <code>{CHROMA_DB_DIR}</code></p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("""
                <div class="empty-state">
                  <div class="es-icon">🗄️</div>
                  <h3>No documents indexed</h3>
                  <p>Upload policy PDFs on the left and click Index Documents to begin.</p>
                </div>
                """, unsafe_allow_html=True)

    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Policy Analysis
# ═════════════════════════════════════════════════════════════════════════════

# Pre-load document inventory (fast after first call — cached in session state)
if "_cached_docs" not in st.session_state:
    st.session_state["_cached_docs"] = _get_indexed_docs()
indexed_docs  = st.session_state["_cached_docs"]
total_chunks  = sum(d["chunks"] for d in indexed_docs)

# ── Demo mode — populate mock conflict result for screenshots ─────────────
_demo = st.query_params.get("demo", "")
if _demo == "conflict" and not st.session_state.get("agent_result"):
    st.session_state["nav_page"] = "analysis"
    st.session_state["last_question"] = (
        "Does the organisation require IT Security approval before installing software?"
    )
    st.session_state["elapsed"] = 12.4
    st.session_state["agent_result"] = {
        "answer": (
            "The IT Security Policy (Section 3) states that all software must be "
            "approved by the IT Security team before installation. However, this "
            "requirement is directly contradicted by the Software Fast-Track Approval "
            "Policy (Section 3), which states that Finance approval alone is sufficient "
            "and no IT Security sign-off is required. These two policies cannot both "
            "be correct; the conflict must be resolved by authorised stakeholders."
        ),
        "answer_type": "partial_grounded",
        "evidence_status": "CONFLICTING",
        "conflict_status": "CONFLICT_DETECTED",
        "conflicts": [
            # Structured dict — tests the two-column policy comparison layout
            {
                "topic": "IT Security sign-off for software installation",
                "policy_a": "it_security_policy.pdf",
                "page_a": 1,
                "excerpt_a": (
                    "All software must be approved by the IT Security team before "
                    "installation. Requests must be submitted via the IT Security portal "
                    "with a business justification."
                ),
                "policy_b": "software_fast_track_policy.pdf",
                "page_b": 1,
                "excerpt_b": (
                    "Finance approval alone is sufficient to authorise the installation of "
                    "business software subscriptions. No additional approval or sign-off "
                    "from IT Security is required before installation."
                ),
                "explanation": (
                    "The IT Security Policy makes IT Security sign-off mandatory for all "
                    "software installations. The Software Fast-Track Approval Policy "
                    "explicitly removes this requirement, stating Finance approval alone is "
                    "sufficient. These two policies cannot both apply simultaneously; the "
                    "conflict must be resolved by authorised stakeholders."
                ),
                "review_flag": True,
                "structured": True,
            },
            # Plain-text string — tests the plain-text fallback path
            (
                "Finance Expense Policy vs Software Fast-Track Approval Policy: "
                "The Finance Expense Policy (Section 6) states that software "
                "subscriptions still require IT Security sign-off, which directly "
                "contradicts the Fast-Track Policy's explicit removal of that requirement."
            ),
        ],
        "sources": [],
        "supporting_passages": [
            {
                "source": "it_security_policy.pdf",
                "page": 1,
                "text": (
                    "Section 3 — Software Approval: All software must be approved by "
                    "the IT Security team before installation. Requests must be submitted "
                    "via the IT Security portal with a business justification."
                ),
            },
            {
                "source": "software_fast_track_policy.pdf",
                "page": 1,
                "text": (
                    "Section 3 — Fast-Track Authorisation Rule: Finance approval alone "
                    "is sufficient to authorise the installation of business software "
                    "subscriptions. No additional approval or sign-off from IT Security "
                    "is required before installation."
                ),
            },
        ],
        "retry_count": 1,
        "retrieved_docs": [],
    }

# ── Active index summary ───────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="sec-title">Active Index</div>', unsafe_allow_html=True)
    if indexed_docs:
        c1, c2, c3 = st.columns([1, 1, 4])
        c1.metric("Documents", len(indexed_docs))
        c2.metric("Chunks", total_chunks)
        with c3:
            pills_html = " ".join(
                _pill(f"📄 {d['filename']}", "info") for d in indexed_docs
            )
            st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.markdown(
            _pill("⚠  No documents indexed — go to Documents to upload PDFs", "warn"),
            unsafe_allow_html=True,
        )

# ── Investigation workspace ────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="sec-title">Investigation Workspace</div>',
                unsafe_allow_html=True)

    mode_col, help_col = st.columns([1, 2])
    with mode_col:
        analysis_mode = st.radio(
            "Analysis mode",
            ["Standard RAG", "Agentic Compliance Analysis"],
            label_visibility="collapsed",
        )

    mode_captions = {
        "Standard RAG":
            "⚡ **Standard RAG** — fast single-pass retrieval and generation · source citations",
        "Agentic Compliance Analysis":
            "🧠 **Agentic** — multi-step LangGraph · evidence verification · conflict detection",
    }
    with help_col:
        st.markdown(
            f'<div style="padding:0.5rem 0;font-size:0.88rem;color:#475569">'
            f'{mode_captions[analysis_mode]}</div>',
            unsafe_allow_html=True,
        )

    user_question = st.text_area(
        "Compliance question",
        placeholder=(
            "Ask a compliance question across all indexed policy documents…\n\n"
            "Example: Does the IT Security Policy require approval before installing software?"
        ),
        height=110,
        label_visibility="collapsed",
    )

    run_btn = st.button(
        "🔍  Analyse",
        key="btn_analyse",
        type="primary",
        disabled=not (user_question or "").strip(),
    )

# ── Empty state ────────────────────────────────────────────────────────────
if (not st.session_state.get("answer")
        and not st.session_state.get("agent_result")
        and not run_btn):
    with st.container(border=True):
        st.markdown("""
        <div class="empty-state">
          <div class="es-icon">🔍</div>
          <h3>No investigation yet</h3>
          <p>Enter a compliance question above and click Analyse to begin.</p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Query execution — Standard RAG
# ═════════════════════════════════════════════════════════════════════════════
if run_btn and (user_question or "").strip() and analysis_mode == "Standard RAG":
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("Cannot query without an NVIDIA API Key. Add it to your .env file.")
    else:
        with st.spinner("Retrieving relevant policy sections and generating answer…"):
            try:
                answer, sources, elapsed = ask_question(user_question.strip())
                st.session_state["answer"] = answer
                st.session_state["sources"] = sources
                st.session_state["last_question"] = user_question.strip()
                st.session_state["elapsed"] = elapsed
                st.session_state.pop("agent_result", None)

                if snowflake_enabled and sources is not None:
                    src_filenames = ", ".join(
                        d.metadata.get("source_filename", "Unknown") for d in sources
                    )
                    src_pages = ", ".join(
                        str(d.metadata.get("page", 0) + 1) for d in sources
                    )
                    doc_names = ", ".join(st.session_state.get("indexed_filenames", []))
                    evidence_found = (
                        len(sources) > 0
                        and "cannot find enough evidence" not in answer.lower()
                    )
                    log_ok, log_msg = log_query_to_snowflake(
                        document_names=doc_names,
                        user_question=user_question.strip(),
                        ai_answer=answer,
                        source_filenames=src_filenames,
                        source_pages=src_pages,
                        evidence_found=evidence_found,
                        model_name=MODEL_NAME,
                        response_time_seconds=elapsed,
                    )
                    st.session_state["log_status"] = (log_ok, log_msg)
                elif not snowflake_enabled:
                    st.session_state["log_status"] = (
                        None,
                        "Snowflake logging skipped — credentials not configured.",
                    )
            except ValueError as e:
                st.error(str(e))
            except TimeoutError:
                st.error("The request timed out. Please try again.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Query execution — Agentic
# ═════════════════════════════════════════════════════════════════════════════
if run_btn and (user_question or "").strip() and analysis_mode == "Agentic Compliance Analysis":
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("Cannot query without an NVIDIA API Key. Add it to your .env file.")
    else:
        with st.spinner("Running agentic compliance analysis…"):
            try:
                from agent import run_agent
                start = time.perf_counter()
                result = run_agent(user_question.strip())
                elapsed = time.perf_counter() - start

                st.session_state["agent_result"] = result
                st.session_state["last_question"] = user_question.strip()
                st.session_state["elapsed"] = elapsed
                st.session_state.pop("answer", None)
                st.session_state.pop("sources", None)

                if snowflake_enabled:
                    sources = result.get("sources") or []
                    src_filenames = ", ".join(
                        d.metadata.get("source_filename", "Unknown") for d in sources
                    )
                    src_pages = ", ".join(
                        str(d.metadata.get("page", 0) + 1) for d in sources
                    )
                    doc_names = ", ".join(st.session_state.get("indexed_filenames", []))
                    answer_type = result.get("answer_type", "")
                    evidence_found = answer_type in ("grounded", "partial_grounded")
                    log_ok, log_msg = log_query_to_snowflake(
                        document_names=doc_names,
                        user_question=user_question.strip(),
                        ai_answer=result.get("answer", ""),
                        source_filenames=src_filenames,
                        source_pages=src_pages,
                        evidence_found=evidence_found,
                        model_name=MODEL_NAME,
                        response_time_seconds=elapsed,
                    )
                    st.session_state["log_status"] = (log_ok, log_msg)
                elif not snowflake_enabled:
                    st.session_state["log_status"] = (
                        None,
                        "Snowflake logging skipped — credentials not configured.",
                    )
            except ValueError as e:
                st.error(str(e))
            except TimeoutError:
                st.error("The request timed out. Please try again.")
            except Exception as e:
                st.error(f"Agentic analysis encountered an error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Label maps
# ═════════════════════════════════════════════════════════════════════════════
_ANSWER_TYPE_LABELS = {
    "grounded":            ("✅ Evidence-supported (AI-assessed)", "success"),
    "partial_grounded":    ("⚠ Partially supported (AI-assessed)", "warning"),
    "unverified_grounded": ("⚠ Verification Incomplete", "warning"),
    "excerpt_fallback":    ("⚠ From Policy Excerpts (unverified)", "warning"),
    "refusal":             ("✗ Insufficient Evidence", "error"),
    "error":               ("✗ Analysis Error", "error"),
}

_EVIDENCE_STATUS_LABELS = {
    "SUPPORTED":           ("✅ Evidence Supported", "success"),
    "PARTIALLY_SUPPORTED": ("⚠ Partially Supported", "warning"),
    "INSUFFICIENT":        ("✗ Insufficient Evidence", "error"),
    "CONFLICTING":         ("⚠ Conflicting Evidence", "warning"),
    "UNAVAILABLE":         ("— Verification Unavailable", "info"),
    "":                    ("—", "info"),
}


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY — Standard RAG results
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.get("answer") and "agent_result" not in st.session_state:
    answer   = st.session_state["answer"]
    last_q   = st.session_state.get("last_question", "")
    elapsed  = st.session_state.get("elapsed")
    sources  = st.session_state.get("sources", [])
    has_ev   = bool(sources) and "cannot find enough evidence" not in answer.lower()

    with st.container(border=True):
        st.markdown('<div class="sec-title">Analysis Result — Standard RAG</div>',
                    unsafe_allow_html=True)
        if last_q:
            st.markdown(f"**Question:** {last_q}")

        badge_lbl = "✅ Evidence-supported (AI-assessed)" if has_ev else "✗ Insufficient Evidence"
        badge_lvl = "success" if has_ev else "error"
        st.markdown(_badge(badge_lbl, badge_lvl), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Answer**")
        st.write(answer)

        if elapsed:
            ct, cs = st.columns(2)
            ct.metric("Response time", f"{elapsed:.1f}s")
            cs.metric("Sources retrieved", len(sources))

        if has_ev:
            st.info(
                "AI evidence assessment confirms retrieved policy text addresses this question. "
                "**AI assessment does not independently verify factual accuracy** — "
                "all compliance conclusions require human review."
            )
        else:
            st.warning(
                "Insufficient evidence found. Consider uploading additional policy documents."
            )

        log_status = st.session_state.get("log_status")
        if log_status:
            lok, lmsg = log_status
            if lok is True:
                st.caption("Audit log saved to Snowflake.")
            elif lok is False:
                st.warning(lmsg)
            else:
                st.caption(lmsg)

    if sources:
        with st.container(border=True):
            st.markdown('<div class="sec-title">Source Evidence</div>',
                        unsafe_allow_html=True)
            for i, doc in enumerate(sources, 1):
                fn  = doc.metadata.get("source_filename", "Unknown")
                pg  = doc.metadata.get("page", None)
                lbl = f"Page {pg + 1}" if pg is not None else "Page unknown"
                with st.expander(f"Source {i}  ·  {fn}  ·  {lbl}"):
                    st.write(doc.page_content)


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY — Agentic results
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.get("agent_result"):
    result       = st.session_state["agent_result"]
    last_q       = st.session_state.get("last_question", "")
    elapsed      = st.session_state.get("elapsed")
    answer_type  = result.get("answer_type", "")
    ev_status    = result.get("evidence_status", "")
    at_lbl, at_lvl = _ANSWER_TYPE_LABELS.get(answer_type, (answer_type or "Unknown", "info"))
    ev_lbl, ev_lvl = _EVIDENCE_STATUS_LABELS.get(ev_status, (ev_status or "—", "info"))

    # ── Demo mode banner — shown only when ?demo=conflict is active ────────
    if _demo == "conflict":
        st.info(
            "⚠️  **Fictional demonstration only.** "
            "The results below — including all response times, retrieval counts, "
            "source citations and conflict findings — are simulated with fictional "
            "policy documents. They do not represent a real investigation and must "
            "not be relied upon as verified compliance analysis.",
            icon=None,
        )

    # ── Answer card ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<div class="sec-title">Analysis Result — Agentic Compliance Analysis</div>',
                    unsafe_allow_html=True)
        if last_q:
            st.markdown(f"**Question:** {last_q}")

        b1, b2 = st.columns(2)
        b1.markdown(
            f"**Finding status:** {_badge(at_lbl, at_lvl)}",
            unsafe_allow_html=True,
        )
        b2.markdown(
            f"**Evidence status:** {_badge(ev_lbl, ev_lvl)}",
            unsafe_allow_html=True,
        )
        st.markdown("")

        ans = result.get("answer", "")
        if ans:
            st.markdown("**Answer**")
            st.write(ans)
        else:
            st.info("No answer was generated.")

        retry_count  = result.get("retry_count", 0)
        sources_list = result.get("sources") or []
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Retrieval attempts", retry_count + 1)
        if elapsed:
            mc2.metric("Response time", f"{elapsed:.1f}s")
        mc3.metric("Sources retrieved", len(sources_list))

        if answer_type == "grounded":
            st.info(
                "AI evidence assessment confirms retrieved policy text addresses this question. "
                "Citation metadata (filename and page) has been validated. "
                "**AI assessment does not independently verify factual accuracy** — "
                "all compliance conclusions require human review."
            )
        elif answer_type in ("partial_grounded", "unverified_grounded", "excerpt_fallback"):
            st.warning(
                "This answer is not fully verified against retrieved policy evidence. "
                "Human review is required before acting on this conclusion."
            )
        elif answer_type == "refusal":
            st.error(
                "Insufficient evidence found in the indexed documents. "
                "Consider uploading additional policy documents."
            )

        log_status = st.session_state.get("log_status")
        if log_status:
            lok, lmsg = log_status
            if lok is True:
                st.caption("Audit log saved to Snowflake.")
            elif lok is False:
                st.warning(f"Audit logging: {lmsg}")
            else:
                st.caption(lmsg)

    # ── Conflict card ──────────────────────────────────────────────────────
    conflict_status = result.get("conflict_status", "")
    conflicts       = result.get("conflicts") or []

    if conflict_status == "CONFLICT_DETECTED" or conflicts:
        with st.container(border=True):
            st.markdown('<div class="sec-title">Policy Conflicts Detected</div>',
                        unsafe_allow_html=True)
            st.markdown(
                _badge(f"⚠  {len(conflicts)} conflict(s) — human review required", "error"),
                unsafe_allow_html=True,
            )
            st.markdown("")

            cards_html = "".join(
                conflict_card_html(parse_conflict_item(c), i)
                for i, c in enumerate(conflicts, 1)
            )
            st.markdown(cards_html, unsafe_allow_html=True)

    elif conflict_status == "UNKNOWN":
        st.warning(
            "Cross-document conflict analysis could not be completed. "
            "Manual review of retrieved sources is recommended."
        )

    # ── Supporting passages ────────────────────────────────────────────────
    passages = result.get("supporting_passages") or []
    if passages:
        with st.container(border=True):
            st.markdown('<div class="sec-title">Supporting Evidence</div>',
                        unsafe_allow_html=True)
            for i, p in enumerate(passages, 1):
                src = p.get("source", "Unknown")
                pg  = p.get("page")
                lbl = f"Page {pg}" if pg else "Page unknown"
                with st.expander(f"Passage {i}  ·  {src}  ·  {lbl}"):
                    st.write(p.get("text", ""))

    # ── Retrieved policy chunks ────────────────────────────────────────────
    retrieved = sources_list if sources_list else (result.get("retrieved_docs") or [])
    if retrieved:
        with st.container(border=True):
            st.markdown('<div class="sec-title">Retrieved Policy Chunks</div>',
                        unsafe_allow_html=True)
            for i, doc in enumerate(retrieved, 1):
                fn  = doc.metadata.get("source_filename", "Unknown")
                pg  = doc.metadata.get("page", None)
                lbl = f"Page {pg + 1}" if pg is not None else "Page unknown"
                with st.expander(f"Chunk {i}  ·  {fn}  ·  {lbl}"):
                    st.write(doc.page_content)

    st.markdown(
        "<small style='color:#94A3B8'>All compliance conclusions are subject to human "
        "review. This tool does not constitute legal advice.</small>",
        unsafe_allow_html=True,
    )
