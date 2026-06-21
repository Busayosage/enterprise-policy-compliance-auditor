import os
import streamlit as st
from dotenv import load_dotenv
from database import index_pdf
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
    st.markdown("1. Upload a policy PDF")
    st.markdown("2. Click 'Index Document'")
    st.markdown("3. Ask compliance questions")

st.header("Upload Policy Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    if st.button("Index Document"):
        with st.spinner("Processing and indexing PDF..."):
            success, message = index_pdf(uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)

st.markdown("---")
st.header("Ask Compliance Questions")

question = st.chat_input("Ask a question about the uploaded policy...")

if question:
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("Cannot query without an NVIDIA API Key. Please add it to your .env file.")
    else:
        with st.spinner("Retrieving relevant policy sections and generating answer..."):
            try:
                answer, sources = ask_question(question)

                st.subheader("Answer")
                st.write(answer)

                if sources:
                    st.subheader("Source Snippets")
                    for i, doc in enumerate(sources, 1):
                        with st.expander(f"Source {i} — Page {doc.metadata.get('page', 'N/A')}"):
                            st.write(doc.page_content)
            except ValueError as e:
                st.error(str(e))
            except TimeoutError:
                st.error("The request timed out. Please try again or simplify your question.")
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
