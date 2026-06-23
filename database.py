import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DB_DIR = "chroma_db"
TEMP_DIR = "temp"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def index_documents(uploaded_files):
    if not uploaded_files:
        return False, "No files provided.", 0, 0, []

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)

        all_chunks = []
        filenames = []

        for uploaded_file in uploaded_files:
            temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_filename"] = uploaded_file.name

            all_chunks.extend(documents)
            filenames.append(uploaded_file.name)
            os.remove(temp_path)

        if not all_chunks:
            return False, "The uploaded PDFs appear to be empty or unreadable.", 0, 0, filenames

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(all_chunks)

        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)

        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=CHROMA_DB_DIR,
        )

        return True, "Indexing complete.", len(filenames), len(chunks), filenames

    except Exception as e:
        return False, f"Failed to process PDFs: {e}", 0, 0, []


def load_retriever(k=2):
    if not os.path.exists(CHROMA_DB_DIR):
        return None

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
