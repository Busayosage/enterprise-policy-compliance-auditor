import os
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from database import load_retriever

load_dotenv()

MODEL_NAME = "meta/llama-3.1-70b-instruct"
# Faster multi-document retrieval settings for MVP/demo use
MAX_CONTEXT_CHARS = 3000
TOP_K_CHUNKS = 2


def get_llm():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set. Add it to your .env file.")

    return ChatNVIDIA(
        model=MODEL_NAME,
        nvidia_api_key=api_key,
        temperature=0,
        max_tokens=350,
        timeout=30,
    )


SYSTEM_PROMPT = """You are a strict corporate compliance auditor. Your role is to answer
compliance questions using ONLY the policy document context provided below.

Rules:
- Base your answer strictly on the provided context.
- Quote or reference specific sections when possible.
- If the context does not contain enough information to answer, respond with:
  "I cannot find enough evidence in the uploaded policy documents to answer that."
- Do not speculate or use outside knowledge.

Context:
{context}"""


def ask_question(question: str):
    start = time.perf_counter()

    retriever = load_retriever(k=TOP_K_CHUNKS)
    if retriever is None:
        return "No indexed documents found. Please upload and index policy PDFs first.", [], 0.0

    sources = retriever.invoke(question)

    if not sources:
        elapsed = time.perf_counter() - start
        return "I cannot find enough evidence in the uploaded policy documents to answer that.", [], elapsed

    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source_filename', 'Unknown')} — Page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}"
        for doc in sources
    )
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    elapsed = time.perf_counter() - start
    return response.content, sources, elapsed
