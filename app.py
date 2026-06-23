import os
import streamlit as st
from dotenv import load_dotenv
from database import index_documents
from main import ask_question

load_dotenv()

st.set_page_config(page_title="Enterprise Policy Compliance Auditor", layout="wide")

st.title("Enterprise Policy Compliance Auditor")

with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        st.success("NVIDIA API Key detected")
    else:
        st.error("NVIDIA API Key not found. Add it to your .env file.")

    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload one or more policy PDFs")
    st.markdown("2. Click 'Index Documents'")
    st.markdown("3. Ask compliance questions across all indexed documents")

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
        st.session_state.pop("answer", None)
        st.session_state.pop("sources", None)
        st.session_state.pop("last_question", None)
        st.session_state.pop("elapsed", None)
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

st.markdown("---")
st.header("Ask Compliance Questions")

user_question = st.chat_input("Ask a question across all indexed policy documents...")

if user_question and user_question.strip():
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
            except ValueError as e:
                st.error(str(e))
            except TimeoutError:
                st.error("The request timed out. Please try again or simplify your question.")
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")

if st.session_state.get("answer"):
    st.subheader("Answer")
    st.write(st.session_state["answer"])

    elapsed = st.session_state.get("elapsed")
    if elapsed is not None:
        st.caption(f"Response generated in {elapsed:.1f} seconds.")

    sources = st.session_state.get("sources", [])
    if sources:
        st.subheader("Source Snippets")
        for i, doc in enumerate(sources, 1):
            filename = doc.metadata.get("source_filename", "Unknown")
            page = doc.metadata.get("page", None)
            page_label = f"Page {page}" if page is not None else "Page unknown"
            with st.expander(f"Source {i} — {filename} — {page_label}"):
                st.write(doc.page_content)
