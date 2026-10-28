"""
Tests for the RAG pipeline: mocked LLM responses, unsupported-question refusal,
and the ask_question() contract.  No live NVIDIA API calls are made.
"""
import os
import sys
import shutil
import pytest
from unittest.mock import patch, MagicMock

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFUSAL_PHRASE = "I cannot find enough evidence"


def _build_store(uploaded_files, chroma_dir, temp_dir):
    """Build an isolated vector store, mirroring database.index_documents()."""
    os.makedirs(temp_dir, exist_ok=True)
    all_chunks = []
    for f in uploaded_files:
        tmp_path = os.path.join(temp_dir, f.name)
        with open(tmp_path, "wb") as fh:
            fh.write(f.getbuffer())
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_filename"] = f.name
        all_chunks.extend(documents)
        os.remove(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_chunks)
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=chroma_dir)


# ---------------------------------------------------------------------------
# 5. Unsupported-question refusal
# Refusal is triggered by the retriever returning nothing relevant OR by the LLM
# following its system prompt.  We test the retrieval-layer refusal path
# (no live LLM needed) by asserting the refusal string appears when the
# question has no match in the indexed documents.
# ---------------------------------------------------------------------------

class TestUnsupportedQuestionRefusal:

    def test_ask_question_no_store_returns_no_indexed_docs_message(self, tmp_path):
        """ask_question() with no store returns the 'no indexed documents' message."""
        missing_dir = str(tmp_path / "empty_chroma")
        with patch("main.load_retriever", return_value=None):
            import main
            answer, sources, elapsed = main.ask_question("What is the password policy?")
        assert "No indexed documents" in answer
        assert sources == []
        assert elapsed >= 0.0

    def test_ask_question_empty_retrieval_returns_refusal(self, tmp_path):
        """ask_question() returns refusal when the retriever yields no docs."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("main.load_retriever", return_value=mock_retriever):
            import main
            answer, sources, elapsed = main.ask_question("What is the cookie policy?")
        assert REFUSAL_PHRASE in answer
        assert sources == []

    def test_ask_question_returns_refusal_for_out_of_scope_topic(self):
        """A question on a topic not in any doc returns the refusal phrase when
        the retriever yields no matching chunks."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("main.load_retriever", return_value=mock_retriever):
            import main
            answer, sources, elapsed = main.ask_question(
                "What is the company's policy on dinosaur preservation?"
            )
        assert REFUSAL_PHRASE in answer


# ---------------------------------------------------------------------------
# 6. Mocked NVIDIA LLM responses
# ---------------------------------------------------------------------------

class TestMockedLLMResponse:

    def _mock_chain_response(self, mock_text: str):
        """Return a mock chain whose invoke() returns an AIMessage."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = AIMessage(content=mock_text)
        return mock_chain

    def _patch_llm_chain(self, mock_text: str):
        """Context manager stack that patches ChatNVIDIA and ChatPromptTemplate
        so that (prompt | llm).invoke(...) returns an AIMessage with mock_text."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = AIMessage(content=mock_text)
        mock_llm = MagicMock()
        mock_prompt_obj = MagicMock()
        # prompt | llm calls __or__ on the prompt object
        mock_prompt_obj.__or__ = MagicMock(return_value=mock_chain)
        return mock_llm, mock_prompt_obj, mock_chain

    def test_ask_question_returns_llm_answer_string(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        """ask_question() passes the LLM response back as the answer string."""
        _build_store([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vs = Chroma(persist_directory=tmp_chroma_dir, embedding_function=embeddings)
        real_retriever = vs.as_retriever(search_kwargs={"k": 2})

        mock_llm, mock_prompt_obj, mock_chain = self._patch_llm_chain(
            "Passwords must be at least 12 characters long."
        )
        with patch("main.load_retriever", return_value=real_retriever), \
             patch("main.get_llm", return_value=mock_llm), \
             patch("main.ChatPromptTemplate") as mock_cpt:
            mock_cpt.from_messages.return_value = mock_prompt_obj
            import main
            answer, sources, elapsed = main.ask_question("What is the password length requirement?")

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_ask_question_returns_sources_list(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        """ask_question() returns a non-empty sources list when retrieval succeeds."""
        _build_store([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vs = Chroma(persist_directory=tmp_chroma_dir, embedding_function=embeddings)
        real_retriever = vs.as_retriever(search_kwargs={"k": 2})

        mock_llm, mock_prompt_obj, mock_chain = self._patch_llm_chain(
            "MFA is required for all remote access."
        )
        with patch("main.load_retriever", return_value=real_retriever), \
             patch("main.get_llm", return_value=mock_llm), \
             patch("main.ChatPromptTemplate") as mock_cpt:
            mock_cpt.from_messages.return_value = mock_prompt_obj
            import main
            answer, sources, elapsed = main.ask_question("MFA requirements")

        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_ask_question_returns_elapsed_time(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        """ask_question() always returns a non-negative float for elapsed time."""
        _build_store([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vs = Chroma(persist_directory=tmp_chroma_dir, embedding_function=embeddings)
        real_retriever = vs.as_retriever(search_kwargs={"k": 2})

        mock_llm, mock_prompt_obj, mock_chain = self._patch_llm_chain("The VPN is required.")
        with patch("main.load_retriever", return_value=real_retriever), \
             patch("main.get_llm", return_value=mock_llm), \
             patch("main.ChatPromptTemplate") as mock_cpt:
            mock_cpt.from_messages.return_value = mock_prompt_obj
            import main
            answer, sources, elapsed = main.ask_question("VPN policy")

        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_ask_question_return_type_contract(self):
        """ask_question() always returns (str, list, float) regardless of outcome."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("main.load_retriever", return_value=mock_retriever):
            import main
            result = main.ask_question("anything")
        assert isinstance(result, tuple)
        assert len(result) == 3
        answer, sources, elapsed = result
        assert isinstance(answer, str)
        assert isinstance(sources, list)
        assert isinstance(elapsed, float)

    def test_get_llm_raises_without_api_key(self):
        """get_llm() raises ValueError when NVIDIA_API_KEY is absent."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=False):
            import main
            with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
                main.get_llm()
