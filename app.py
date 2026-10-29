import os
import time
import streamlit as st
from dotenv import load_dotenv
from database import index_documents
from main import ask_question, MODEL_NAME
from snowflake_logger import (
    is_snowflake_configured,
    create_logs_table_if_not_exists,
    log_query_to_snowflake,
)

load_dotenv()

st.set_page_config(page_title="Enterprise Policy Compliance Auditor", layout="wide")
st.title("Enterprise Policy Compliance Auditor")

# ── Snowflake bootstrap ──────────────────────────────────────────────────────
snowflake_enabled = is_snowflake_configured()
if snowflake_enabled and "snowflake_table_ready" not in st.session_state:
    success, msg = create_logs_table_if_not_exists()
    st.session_state["snowflake_table_ready"] = success
    if not success:
        st.session_state["snowflake_table_error"] = msg

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        st.success("NVIDIA API Key detected")
    else:
        st.error("NVIDIA API Key not found. Add it to your .env file.")

    if snowflake_enabled:
        st.success("Snowflake logging: enabled")
    else:
        st.info("Snowflake logging: disabled")

    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload one or more policy PDFs")
    st.markdown("2. Click 'Index Documents'")
    st.markdown("3. Select an analysis mode")
    st.markdown("4. Ask compliance questions across all indexed documents")

# ── Document upload ──────────────────────────────────────────────────────────
st.header("Upload Policy Documents")
uploaded_files = st.file_uploader(
    "Upload one or more policy PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} file(s) selected:**")
    for f in uploaded_files:
        st.write(f"- {f.name}")

    if st.button("Index Documents"):
        for key in ("answer", "sources", "last_question", "elapsed", "log_status",
                    "agent_result"):
            st.session_state.pop(key, None)
        st.session_state["indexed_filenames"] = [f.name for f in uploaded_files]
        with st.spinner("Processing and indexing PDFs..."):
            success, message, doc_count, chunk_count, filenames = index_documents(uploaded_files)
            if success:
                st.success(
                    f"{message} Indexed **{doc_count} document(s)** "
                    f"into **{chunk_count} chunks**."
                )
                st.write("**Documents indexed:**")
                for name in filenames:
                    st.write(f"- {name}")
            else:
                st.error(message)

# ── Mode selector ────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Ask Compliance Questions")

analysis_mode = st.radio(
    "Analysis mode",
    ["Standard RAG", "Agentic Compliance Analysis"],
    horizontal=True,
    help=(
        "**Standard RAG** — fast single-pass retrieval and generation.\n\n"
        "**Agentic Compliance Analysis** — multi-step LangGraph pipeline with "
        "evidence verification, citation validation and conflict detection."
    ),
)

user_question = st.chat_input("Ask a question across all indexed policy documents...")

# ═══════════════════════════════════════════════════════════════════════════════
# Standard RAG mode
# ═══════════════════════════════════════════════════════════════════════════════
if user_question and user_question.strip() and analysis_mode == "Standard RAG":
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("Cannot query without an NVIDIA API Key. Please add it to your .env file.")
    else:
        with st.spinner("Retrieving relevant policy sections and generating answer..."):
            try:
                answer, sources, elapsed = ask_question(user_question.strip())
                st.session_state["answer"] = answer
                st.session_state["sources"] = sources
                st.session_state["last_question"] = user_question.strip()
                st.session_state["elapsed"] = elapsed
                st.session_state.pop("agent_result", None)

                if snowflake_enabled and sources is not None:
                    src_filenames = ", ".join(
                        doc.metadata.get("source_filename", "Unknown") for doc in sources
                    )
                    src_pages = ", ".join(
                        str(doc.metadata.get("page", 0) + 1) for doc in sources
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
                        "Snowflake logging skipped because credentials are not configured.",
                    )

            except ValueError as e:
                st.error(str(e))
            except TimeoutError:
                st.error("The request timed out. Please try again or simplify your question.")
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Compliance Analysis mode
# ═══════════════════════════════════════════════════════════════════════════════
if user_question and user_question.strip() and analysis_mode == "Agentic Compliance Analysis":
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("Cannot query without an NVIDIA API Key. Please add it to your .env file.")
    else:
        with st.spinner("Running agentic compliance analysis..."):
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

                # Snowflake audit for agentic mode
                if snowflake_enabled:
                    sources = result.get("sources") or []
                    src_filenames = ", ".join(
                        doc.metadata.get("source_filename", "Unknown")
                        for doc in sources
                    )
                    src_pages = ", ".join(
                        str(doc.metadata.get("page", 0) + 1) for doc in sources
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
                        "Snowflake logging skipped because credentials are not configured.",
                    )

            except ValueError as e:
                st.error(str(e))
            except TimeoutError:
                st.error("The request timed out. Please try again or simplify your question.")
            except Exception as e:
                st.error(f"Agentic analysis encountered an error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Display — Standard RAG results
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("answer") and "agent_result" not in st.session_state:
    st.subheader("Answer")
    st.write(st.session_state["answer"])

    elapsed = st.session_state.get("elapsed")
    if elapsed is not None:
        st.caption(f"Response generated in {elapsed:.1f} seconds.")

    log_status = st.session_state.get("log_status")
    if log_status:
        log_ok, log_msg = log_status
        if log_ok is True:
            st.caption("Snowflake log saved successfully.")
        elif log_ok is False:
            st.warning(log_msg)
        else:
            st.caption(log_msg)

    sources = st.session_state.get("sources", [])
    if sources:
        st.subheader("Source Snippets")
        for i, doc in enumerate(sources, 1):
            filename = doc.metadata.get("source_filename", "Unknown")
            raw_page = doc.metadata.get("page", None)
            page_label = f"Page {raw_page + 1}" if raw_page is not None else "Page unknown"
            with st.expander(f"Source {i} — {filename} — {page_label}"):
                st.write(doc.page_content)

# ═══════════════════════════════════════════════════════════════════════════════
# Display — Agentic results
# ═══════════════════════════════════════════════════════════════════════════════
_ANSWER_TYPE_LABELS = {
    "grounded":           ("Verified Answer", "success"),
    "partial_grounded":   ("Partially Verified Answer", "warning"),
    "unverified_grounded":("Answer — Verification Incomplete", "warning"),
    "excerpt_fallback":   ("Answer from Policy Excerpts (unverified)", "warning"),
    "refusal":            ("Insufficient Evidence", "error"),
    "error":              ("Analysis Error", "error"),
}

_EVIDENCE_STATUS_LABELS = {
    "SUPPORTED":           ("Supported", "success"),
    "PARTIALLY_SUPPORTED": ("Partially Supported", "warning"),
    "INSUFFICIENT":        ("Insufficient", "error"),
    "CONFLICTING":         ("Conflicting", "warning"),
    "UNAVAILABLE":         ("Verification Unavailable", "info"),
    "":                    ("—", "info"),
}


def _render_badge(label: str, level: str) -> str:
    colours = {
        "success": "#1a7a4a",
        "warning": "#7a5c1a",
        "error":   "#7a1a1a",
        "info":    "#1a407a",
    }
    bg = colours.get(level, "#444")
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600">{label}</span>'
    )


if st.session_state.get("agent_result"):
    result = st.session_state["agent_result"]
    last_q = st.session_state.get("last_question", "")
    elapsed = st.session_state.get("elapsed")

    st.subheader("Agentic Compliance Analysis")

    # Question echo
    st.markdown(f"**Question:** {last_q}")

    # ── Answer type badge ────────────────────────────────────────────────────
    answer_type = result.get("answer_type", "")
    at_label, at_level = _ANSWER_TYPE_LABELS.get(answer_type, (answer_type or "Unknown", "info"))
    st.markdown(
        f"**Verification status:** {_render_badge(at_label, at_level)}",
        unsafe_allow_html=True,
    )

    # ── Evidence status badge ────────────────────────────────────────────────
    ev_status = result.get("evidence_status", "")
    ev_label, ev_level = _EVIDENCE_STATUS_LABELS.get(ev_status, (ev_status or "—", "info"))
    st.markdown(
        f"**Evidence status:** {_render_badge(ev_label, ev_level)}",
        unsafe_allow_html=True,
    )

    # ── Answer body ──────────────────────────────────────────────────────────
    answer = result.get("answer", "")
    if answer:
        st.markdown("**Answer:**")
        st.write(answer)
    else:
        st.info("No answer was generated.")

    # ── Human review notice ──────────────────────────────────────────────────
    if answer_type in ("partial_grounded", "unverified_grounded", "excerpt_fallback"):
        st.warning(
            "This answer is not fully verified against retrieved policy evidence. "
            "Human review is recommended before acting on this conclusion."
        )
    elif answer_type == "refusal":
        st.error(
            "The system could not find sufficient evidence in the indexed documents "
            "to answer this question. Consider uploading additional policy documents."
        )

    # ── Cross-document conflicts ─────────────────────────────────────────────
    conflict_status = result.get("conflict_status", "")
    conflicts = result.get("conflicts", []) or []
    if conflict_status == "CONFLICT_DETECTED" or conflicts:
        st.warning("**Cross-document conflicts detected**")
        for c in conflicts:
            st.markdown(f"- {c}")
    elif conflict_status == "UNKNOWN":
        st.info("Conflict analysis could not be completed for this query.")

    # ── Retrieval metadata ───────────────────────────────────────────────────
    retry_count = result.get("retry_count", 0)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Retrieval attempts", retry_count + 1)
    with col2:
        if elapsed is not None:
            st.metric("Response time", f"{elapsed:.1f}s")

    # ── Supporting passages ──────────────────────────────────────────────────
    passages = result.get("supporting_passages") or []
    sources = result.get("sources") or []

    if passages:
        st.subheader("Supporting Evidence")
        for i, p in enumerate(passages, 1):
            src = p.get("source", "Unknown")
            page = p.get("page")
            page_label = f"Page {page}" if page else "Page unknown"
            with st.expander(f"Passage {i} — {src} — {page_label}"):
                st.write(p.get("text", ""))

    # ── Retrieved documents ──────────────────────────────────────────────────
    if sources:
        st.subheader("Retrieved Policy Chunks")
        for i, doc in enumerate(sources, 1):
            filename = doc.metadata.get("source_filename", "Unknown")
            raw_page = doc.metadata.get("page", None)
            page_label = f"Page {raw_page + 1}" if raw_page is not None else "Page unknown"
            with st.expander(f"Chunk {i} — {filename} — {page_label}"):
                st.write(doc.page_content)
    elif result.get("retrieved_docs"):
        # Show retrieved docs even when not cited in answer
        st.subheader("Retrieved Policy Chunks")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            filename = doc.metadata.get("source_filename", "Unknown")
            raw_page = doc.metadata.get("page", None)
            page_label = f"Page {raw_page + 1}" if raw_page is not None else "Page unknown"
            with st.expander(f"Chunk {i} — {filename} — {page_label}"):
                st.write(doc.page_content)

    # ── Audit log status ─────────────────────────────────────────────────────
    log_status = st.session_state.get("log_status")
    if log_status:
        log_ok, log_msg = log_status
        if log_ok is True:
            st.caption("Snowflake log saved successfully.")
        elif log_ok is False:
            st.warning(f"Audit logging: {log_msg}")
        else:
            st.caption(log_msg)

    # ── Compliance disclaimer ────────────────────────────────────────────────
    st.markdown(
        "---\n*All compliance conclusions are subject to human review. "
        "This tool does not constitute legal advice.*"
    )
