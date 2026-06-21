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


def index_pdf(uploaded_file):
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)

        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            return False, "The PDF appears to be empty or unreadable."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)

        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=CHROMA_DB_DIR,
        )

        os.remove(temp_path)

        return True, f"Successfully indexed {len(chunks)} chunks from '{uploaded_file.name}'."

    except Exception as e:
        return False, f"Failed to process PDF: {e}"


def load_retriever():
    if not os.path.exists(CHROMA_DB_DIR):
        return None

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})
