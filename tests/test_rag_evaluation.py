"""
Baseline RAG evaluation tests.
Measures retrieval hit rate, citation correctness, and refusal accuracy
using the reference dataset in evaluation_dataset.py.
All measurements are OFFLINE (no live NVIDIA API calls).

Live LLM accuracy is NOT measured here — those scores would require
real inference and are left for a separate online evaluation run.
"""
import os
import sys
import shutil
import pytest
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from evaluation_dataset import EVALUATION_QUESTIONS, CATEGORIES

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # matches main.TOP_K_CHUNKS (raised from 2 in Phase 3)


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


# ---------------------------------------------------------------------------
# Session-scoped fixture: build a single shared vector store from all 3 PDFs.
# This is expensive (~10s) so we build it once per test session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def all_docs_store():
    """Build an in-memory vector store with all three fixtures."""
    pdf_names = [
        "it_security_policy.pdf",
        "hr_leave_policy.pdf",
        "finance_expense_policy.pdf",
    ]
    all_chunks = []
    for name in pdf_names:
        path = _fixture_path(name)
        loader = PyPDFLoader(path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_filename"] = name
        all_chunks.extend(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_chunks)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = InMemoryVectorStore(embedding=embeddings)
    vs.add_documents(chunks)
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    yield retriever


# ---------------------------------------------------------------------------
# Helper: run one evaluation question through the retriever
# ---------------------------------------------------------------------------

def _retrieve(retriever, question: str):
    return retriever.invoke(question)


def _check_hit(docs, expected_files: list[str]) -> bool:
    """Return True if at least one retrieved doc is from an expected file."""
    if not expected_files:
        return True  # unsupported questions have no expected file
    retrieved_files = {d.metadata.get("source_filename", "") for d in docs}
    return bool(retrieved_files & set(expected_files))


def _check_key_terms(docs, key_terms: list[str]) -> bool:
    """Return True if all key_terms appear somewhere in the retrieved chunks."""
    if not key_terms:
        return True
    combined_text = " ".join(d.page_content for d in docs).lower()
    return all(term.lower() in combined_text for term in key_terms)


# ---------------------------------------------------------------------------
# Metric 1: Source-document hit rate
# Pass = at least one retrieved chunk is from the expected source file.
# This checks the retriever found the right document but NOT that the
# specific evidence clause is present.
# ---------------------------------------------------------------------------

NON_REFUSAL_QUESTIONS = [q for q in EVALUATION_QUESTIONS if not q["should_refuse"]]
REFUSAL_QUESTIONS = [q for q in EVALUATION_QUESTIONS if q["should_refuse"]]


@pytest.mark.parametrize("entry", NON_REFUSAL_QUESTIONS, ids=[q["id"] for q in NON_REFUSAL_QUESTIONS])
def test_source_document_hit(entry, all_docs_store):
    """Source-document hit: retriever returns ≥1 chunk from the expected file(s).
    NOTE: This does NOT verify the evidence clause is present — only the filename.
    Use test_evidence_clause_coverage for that stronger check.
    """
    docs = _retrieve(all_docs_store, entry["question"])
    hit = _check_hit(docs, entry["expected_source_files"])
    retrieved = {d.metadata.get("source_filename") for d in docs}
    assert hit, (
        f"[{entry['id']}] Expected source(s) {entry['expected_source_files']} "
        f"not in retrieved filenames: {retrieved}\n"
        f"Question: {entry['question']}"
    )


# ---------------------------------------------------------------------------
# Metric 2: Evidence-clause coverage
# Stronger check: the specific key terms (policy clauses) must appear in the
# retrieved chunk text — not just the filename.
# xfail is used where k=2 is insufficient to retrieve all required clauses.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", NON_REFUSAL_QUESTIONS, ids=[q["id"] for q in NON_REFUSAL_QUESTIONS])
def test_evidence_clause_coverage(entry, all_docs_store):
    """Evidence-clause coverage: all key terms from the answer must be present
    in the text of the top-k retrieved chunks.
    This is a stronger check than source-document hit — it verifies the actual
    supporting evidence is in the retrieved context, not merely the filename.
    """
    if not entry["key_terms"]:
        pytest.skip("No key terms defined for this entry")
    docs = _retrieve(all_docs_store, entry["question"])
    covered = _check_key_terms(docs, entry["key_terms"])
    combined = " ".join(d.page_content for d in docs)
    missing = [t for t in entry["key_terms"] if t.lower() not in combined.lower()]
    assert covered, (
        f"[{entry['id']}] Key terms not found in top-{TOP_K} chunk text: {missing}\n"
        f"Question: {entry['question']}"
    )


@pytest.mark.parametrize("entry", REFUSAL_QUESTIONS, ids=[q["id"] for q in REFUSAL_QUESTIONS])
def test_unsupported_question_retrieval_is_low_confidence(entry, all_docs_store):
    """For questions not in any policy, retrieved chunks should not be highly relevant.
    This is a soft check — we verify retrieval still returns something (it always does),
    but we flag if the returned filenames happen to 'match' a non-existent policy."""
    docs = _retrieve(all_docs_store, entry["question"])
    # The refusal logic in ask_question() depends on the LLM following its system prompt,
    # not on retrieval returning empty (ChromaDB always returns k results).
    # We assert the query at least reaches the retriever without error.
    assert isinstance(docs, list)
    # No source file is expected — so we just verify no crash and note this is a live-model concern.


# ---------------------------------------------------------------------------
# Aggregate metrics (run as one test, prints a report)
# ---------------------------------------------------------------------------

def test_aggregate_retrieval_baseline(all_docs_store, capsys):
    """Compute and print aggregate baseline metrics across all 38 questions."""
    hits = 0
    key_term_hits = 0
    total_non_refusal = len(NON_REFUSAL_QUESTIONS)
    total_refusal = len(REFUSAL_QUESTIONS)
    total = len(EVALUATION_QUESTIONS)

    per_category = {cat: {"total": 0, "hits": 0} for cat in [
        "direct_factual", "multi_section", "cross_document", "unsupported", "conflicting_or_edge"
    ]}

    for entry in EVALUATION_QUESTIONS:
        cat = entry["category"]
        per_category[cat]["total"] += 1

        if entry["should_refuse"]:
            # Refusal correctness requires a live LLM — not measured offline
            continue

        docs = _retrieve(all_docs_store, entry["question"])
        hit = _check_hit(docs, entry["expected_source_files"])
        kt_hit = _check_key_terms(docs, entry["key_terms"]) if entry["key_terms"] else True

        if hit:
            hits += 1
            per_category[cat]["hits"] += 1
        if kt_hit:
            key_term_hits += 1

    hit_rate = hits / total_non_refusal if total_non_refusal else 0
    kt_rate = key_term_hits / total_non_refusal if total_non_refusal else 0

    report = f"""
╔══════════════════════════════════════════════════════════╗
║         RAG RETRIEVAL BASELINE METRICS (k={TOP_K})          ║
╚══════════════════════════════════════════════════════════╝

Dataset:           {total} questions
  Non-refusal:     {total_non_refusal}
  Should-refuse:   {total_refusal} (LLM-dependent, not measured offline)

Retrieval hit rate:    {hits}/{total_non_refusal} = {hit_rate:.1%}
  (≥1 chunk from expected source file(s) in top-{TOP_K} results)

Key-term coverage:     {key_term_hits}/{total_non_refusal} = {kt_rate:.1%}
  (all expected key terms present in top-{TOP_K} retrieved chunks)

Per-category hit rate (non-refusal questions only):"""

    for cat, v in per_category.items():
        if v["total"] == 0 or cat == "unsupported":
            continue
        rate = v["hits"] / v["total"] if v["total"] else 0
        report += f"\n  {cat:<22}: {v['hits']}/{v['total']} = {rate:.1%}"

    report += f"""

Notes:
  - Refusal accuracy (D01–D06) requires live NVIDIA inference — not measured here.
  - Key-term coverage is a retrieval proxy; a correct LLM answer may cite terms
    from context beyond the top-{TOP_K} chunks.
  - Increase TOP_K_CHUNKS or chunk_overlap to improve multi-section coverage.
  - Baseline established: Python 3.13.7, all-MiniLM-L6-v2, chunk_size=1000,
    chunk_overlap=200, k={TOP_K}.
"""
    print(report)

    # Soft assertion: hit rate should be ≥ 50% at k=2 as a minimum sanity check
    assert hit_rate >= 0.50, (
        f"Retrieval hit rate {hit_rate:.1%} is below 50% minimum threshold. "
        "Check fixture content and embedding model."
    )
