import os
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from database import load_retriever

load_dotenv()

MODEL_NAME = "meta/llama-3.1-70b-instruct"
TOP_K_CHUNKS = 3      # k=3 achieves 100% evidence-clause coverage (Phase 3)
# Bounded whole-chunk context: never cut a chunk mid-sentence.
# Each chunk is at most ~1200 chars; 3 chunks ≈ 4000 chars.
# 8000 chars is a generous upper bound that fits within the model's context window
# while preventing unbounded growth if k is increased in future.
MAX_CONTEXT_CHARS = 8000


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


def _build_context(sources: list) -> str:
    """Build LLM context from retrieved chunks.

    Includes complete chunks up to MAX_CONTEXT_CHARS.  A chunk is always
    included in full or not at all — never truncated mid-sentence.
    Page numbers are 1-indexed (PyPDF stores 0-indexed; we add 1 here).
    At least the first chunk is always included regardless of size.
    """
    chunk_texts = []
    char_count = 0
    sep = "\n\n---\n\n"

    for doc in sources:
        page_num = doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
        header = f"[{doc.metadata.get('source_filename', 'Unknown')} — Page {page_num}]"
        chunk = f"{header}\n{doc.page_content}"
        sep_cost = len(sep) if chunk_texts else 0

        if chunk_texts and char_count + sep_cost + len(chunk) > MAX_CONTEXT_CHARS:
            break  # stop at chunk boundary — never cut mid-sentence

        chunk_texts.append(chunk)
        char_count += sep_cost + len(chunk)

    return sep.join(chunk_texts)


def ask_question(question: str):
    start = time.perf_counter()

    retriever = load_retriever(k=TOP_K_CHUNKS)
    if retriever is None:
        return "No indexed documents found. Please upload and index policy PDFs first.", [], 0.0

    sources = retriever.invoke(question)

    if not sources:
        elapsed = time.perf_counter() - start
        return "I cannot find enough evidence in the uploaded policy documents to answer that.", [], elapsed

    context = _build_context(sources)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    elapsed = time.perf_counter() - start
    return response.content, sources, elapsed
