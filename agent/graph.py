"""
LangGraph graph definition for the Evidence-Aware Agentic RAG pipeline.

Stage 4a: deterministic node implementations (no live LLM calls).
Stage 4b will replace the classify_query and generate_grounded_answer stubs
with real LLM calls while keeping the graph topology and all other nodes.

Graph topology
--------------

    START
      │
      ▼
  classify_query
      │
      ▼
  plan_retrieval
      │
      ▼
  retrieve_policy_chunks
      │
      ▼
  assess_evidence ──────────────────────────┐
      │                                     │
      │ (conditional edge)                  │
      ├─ needs_retry & retry<2 ──► expand_retrieval
      │                                     │
      ├─ requires_cross_doc ──► check_cross_doc_conflicts
      │                               │
      └─ otherwise ───────────────────┘
                                      │
                                      ▼
                              generate_grounded_answer
                                      │
                                      ▼
                              log_audit_record
                                      │
                                      ▼
                                    END

Architecture safeguards (all enforced here):
  - At most three total retrieval calls: 1 initial + up to 2 expansions
    (expand_retrieval is gated by retry_count < _MAX_RETRIES=2).
  - Distinct terminal states: grounded answer, refusal, error.
  - No fabricated citations: sources list is always the literal retrieved docs.
  - Verification errors → answer_type="error", never answer_type="grounded".
  - Evidence coverage ≠ answer correctness; assessed separately.
  - No autonomous policy modification (graph is read-only w.r.t. ChromaDB).
  - No SharedSystemClient.clear_system_cache() calls inside graph nodes.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from agent.state import AgentState, QueryClassification, RetrievalPlan, EvidenceAssessment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2          # max expansions; total retrieval calls = 1 initial + _MAX_RETRIES
_EVIDENCE_THRESHOLD = 0.3 # minimum coverage score to consider evidence sufficient
_BASE_K = 3               # initial retrieval count
_EXPAND_K_INCREMENT = 2   # added to k on each retry

# Common English stopwords excluded from query-coverage calculation
_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "does", "did", "do",
    "the", "a", "an", "and", "or", "but", "nor",
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "that", "this", "these", "those",
    "our", "their", "its", "my", "your", "his", "her",
    "not", "no", "any", "all", "each", "both", "more",
    "than", "then", "there", "here", "can", "could", "will",
    "would", "should", "may", "might", "must", "shall",
    "according", "under", "about", "between", "policy",
})

_REFUSAL_MESSAGE = (
    "I cannot find enough evidence in the uploaded policy documents to answer that."
)

# Cross-document indicator phrases — used by the Stage 4a keyword classifier
_CROSS_DOC_PHRASES = (
    "both policies", "all policies", "it and hr", "hr and it",
    "it and finance", "finance and it", "hr and finance", "finance and hr",
    "across policies", "compare", "difference between",
)

# Unsupported-topic keywords — used by the Stage 4a keyword classifier
_UNSUPPORTED_KEYWORDS = (
    "salary", "payroll", "tax", "pension", "stock", "equity",
    "performance review", "promotion", "grievance", "disciplinary",
    "legal advice", "lawsuit",
)


# ---------------------------------------------------------------------------
# Node: classify_query
# ---------------------------------------------------------------------------

def classify_query(state: AgentState) -> dict:
    """Stage 4a: keyword-heuristic classifier (no LLM).

    Stage 4b replacement: call LLM with structured-output prompt to produce
    a QueryClassification. Keep the return contract identical.
    """
    query_lower = state["query"].lower()

    # Check unsupported topics first
    if any(kw in query_lower for kw in _UNSUPPORTED_KEYWORDS):
        classification: QueryClassification = {
            "category": "unsupported",
            "intent": "Question about a topic not covered by the indexed policies.",
            "requires_cross_doc": False,
        }
        return {"classification": classification}

    # Cross-document: explicit multi-doc phrasing or conjunction of two policy areas
    requires_cross_doc = any(phrase in query_lower for phrase in _CROSS_DOC_PHRASES)
    if requires_cross_doc:
        return {
            "classification": QueryClassification(
                category="cross_document",
                intent="Question that spans multiple policy documents.",
                requires_cross_doc=True,
            )
        }

    # Multi-section: long question or contains "and" with multiple clauses
    word_count = len(state["query"].split())
    has_conjunction = " and " in query_lower or ";" in query_lower
    if word_count > 20 and has_conjunction:
        return {
            "classification": QueryClassification(
                category="multi_section",
                intent="Question requiring multiple policy sections.",
                requires_cross_doc=False,
            )
        }

    # Default: direct factual
    return {
        "classification": QueryClassification(
            category="direct_factual",
            intent="Single-policy factual question.",
            requires_cross_doc=False,
        )
    }


# ---------------------------------------------------------------------------
# Node: plan_retrieval
# ---------------------------------------------------------------------------

def plan_retrieval(state: AgentState) -> dict:
    """Deterministic retrieval plan based on query classification."""
    category = (state["classification"] or {}).get("category", "direct_factual")

    k_map = {
        "direct_factual": _BASE_K,
        "multi_section": _BASE_K + 1,
        "cross_document": _BASE_K + 2,
        "unsupported": _BASE_K,
        "unknown": _BASE_K,
    }
    k = k_map.get(category, _BASE_K)

    query_lower = state["query"].lower()
    document_hints: list = []
    if "it" in query_lower or "security" in query_lower or "password" in query_lower:
        document_hints.append("it_security_policy.pdf")
    if "leave" in query_lower or "holiday" in query_lower or "hr" in query_lower:
        document_hints.append("hr_leave_policy.pdf")
    if "expense" in query_lower or "receipt" in query_lower or "finance" in query_lower:
        document_hints.append("finance_expense_policy.pdf")

    return {
        "retrieval_plan": RetrievalPlan(
            k=k,
            filter_hints=[],
            document_hints=document_hints,
        )
    }


# ---------------------------------------------------------------------------
# Node: retrieve_policy_chunks
# ---------------------------------------------------------------------------

def retrieve_policy_chunks(state: AgentState) -> dict:
    """Fetch chunks from ChromaDB via load_retriever().

    Uses retriever_override when set (test injection point).
    On any retrieval error, sets error and returns empty docs so assess_evidence
    can route to a graceful refusal without crashing.
    """
    k = (state["retrieval_plan"] or {}).get("k", _BASE_K)

    retriever = state.get("retriever_override")
    if retriever is None:
        try:
            from database import load_retriever
            retriever = load_retriever(k=k)
        except Exception as exc:
            return {"retrieved_docs": [], "error": f"Retriever load failed: {exc}"}

    if retriever is None:
        return {"retrieved_docs": [], "error": "No indexed documents. Upload and index PDFs first."}

    try:
        docs = retriever.invoke(state["query"])
        return {"retrieved_docs": docs or []}
    except Exception as exc:
        return {"retrieved_docs": [], "error": f"Retrieval failed: {exc}"}


# ---------------------------------------------------------------------------
# Node: assess_evidence
# ---------------------------------------------------------------------------

def assess_evidence(state: AgentState) -> dict:
    """Heuristic evidence assessment — no LLM required.

    Coverage is the proportion of query key-terms found in the retrieved text.
    Confidence is proxy-measured by doc count and source diversity.
    """
    docs = state.get("retrieved_docs") or []
    query_words = {w.lower() for w in state["query"].split() if len(w) > 3}

    if not docs or state.get("error"):
        assessment: EvidenceAssessment = {
            "coverage": 0.0,
            "confidence": 0.0,
            "sufficient": False,
        }
        needs_retry = (not state.get("error")) and (state.get("retry_count", 0) < _MAX_RETRIES)
        return {"evidence": assessment, "needs_retry": needs_retry}

    combined_text = " ".join(d.page_content for d in docs).lower()
    # Exclude stopwords and short tokens so question words don't dilute coverage
    content_words = {w for w in query_words if w.lower() not in _STOPWORDS and len(w) > 3}
    effective_words = content_words if content_words else query_words
    matched = sum(1 for w in effective_words if w in combined_text)
    coverage = matched / len(effective_words) if effective_words else 1.0

    unique_sources = len({d.metadata.get("source_filename", "") for d in docs})
    confidence = min(1.0, len(docs) / _BASE_K) * min(1.0, unique_sources / 1)

    sufficient = coverage >= _EVIDENCE_THRESHOLD and len(docs) >= 1

    needs_retry = (not sufficient) and (state.get("retry_count", 0) < _MAX_RETRIES)

    assessment = EvidenceAssessment(
        coverage=round(coverage, 3),
        confidence=round(confidence, 3),
        sufficient=sufficient,
    )
    return {"evidence": assessment, "needs_retry": needs_retry}


# ---------------------------------------------------------------------------
# Node: expand_retrieval
# ---------------------------------------------------------------------------

def expand_retrieval(state: AgentState) -> dict:
    """Increment k and re-fetch. Increments retry_count.

    After this node LangGraph routes back to assess_evidence.
    The retry_count cap is enforced in the router (_route_after_assessment),
    not here, so this node only runs when the router allowed it.
    """
    current_plan = state.get("retrieval_plan") or {}
    new_k = current_plan.get("k", _BASE_K) + _EXPAND_K_INCREMENT
    new_plan = RetrievalPlan(
        k=new_k,
        filter_hints=current_plan.get("filter_hints", []),
        document_hints=current_plan.get("document_hints", []),
    )

    retriever = state.get("retriever_override")
    if retriever is None:
        try:
            from database import load_retriever
            retriever = load_retriever(k=new_k)
        except Exception as exc:
            return {
                "retrieval_plan": new_plan,
                "retrieved_docs": [],
                "retry_count": state.get("retry_count", 0) + 1,
                "error": f"Retriever load failed on retry: {exc}",
            }

    if retriever is None:
        return {
            "retrieval_plan": new_plan,
            "retrieved_docs": [],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        docs = retriever.invoke(state["query"])
    except Exception as exc:
        return {
            "retrieval_plan": new_plan,
            "retrieved_docs": [],
            "retry_count": state.get("retry_count", 0) + 1,
            "error": f"Retrieval failed on retry: {exc}",
        }

    return {
        "retrieval_plan": new_plan,
        "retrieved_docs": docs or [],
        "retry_count": state.get("retry_count", 0) + 1,
        "needs_retry": False,  # reset; assess_evidence will set it again
    }


# ---------------------------------------------------------------------------
# Node: check_cross_doc_conflicts
# ---------------------------------------------------------------------------

def check_cross_doc_conflicts(state: AgentState) -> dict:
    """Stage 4a: detect obvious intra-corpus conflicts by source diversity.

    A conflict is flagged when the same topic appears in chunks from different
    source files with materially different numeric values or policy terms.
    Stage 4b will apply LLM-based semantic conflict detection.
    """
    docs = state.get("retrieved_docs") or []
    conflicts: list[str] = []

    sources_by_file: dict[str, list] = {}
    for doc in docs:
        fname = doc.metadata.get("source_filename", "unknown")
        sources_by_file.setdefault(fname, []).append(doc.page_content)

    if len(sources_by_file) > 1:
        # Stage 4a: note which files contribute (conflict detail deferred to 4b)
        file_list = ", ".join(sorted(sources_by_file.keys()))
        conflicts.append(
            f"Evidence spans multiple policy files ({file_list}). "
            "Verify that cited values are consistent across documents."
        )

    return {"conflicts": conflicts}


# ---------------------------------------------------------------------------
# Node: generate_grounded_answer
# ---------------------------------------------------------------------------

def generate_grounded_answer(state: AgentState) -> dict:
    """Stage 4a: build a grounded answer directly from retrieved chunk text.

    Stage 4b replacement: call LLM with retrieved chunks as context.
    This stub preserves the same return contract:
        answer      → str
        sources     → list[Document]
        answer_type → "grounded" | "refusal" | "error"
    """
    # Hard error from retrieval
    if state.get("error") and not state.get("retrieved_docs"):
        return {
            "answer": _REFUSAL_MESSAGE,
            "sources": [],
            "answer_type": "error",
        }

    evidence = state.get("evidence")
    docs = state.get("retrieved_docs") or []
    classification = state.get("classification") or {}

    # Insufficient evidence after retries → refusal
    if not docs or (evidence and not evidence.get("sufficient")):
        return {"answer": _REFUSAL_MESSAGE, "sources": [], "answer_type": "refusal"}

    # Stage 4a grounded answer: cite source headers and first excerpt of each chunk
    lines = []
    for doc in docs:
        fname = doc.metadata.get("source_filename", "Unknown")
        raw_page = doc.metadata.get("page", 0)
        page_label = f"Page {raw_page + 1}"
        excerpt = doc.page_content[:300].strip()
        if len(doc.page_content) > 300:
            excerpt += "…"
        lines.append(f"[{fname} — {page_label}]\n{excerpt}")

    conflicts = state.get("conflicts") or []
    conflict_note = ""
    if conflicts:
        conflict_note = (
            "\n\nNote: Evidence was retrieved from multiple policy documents. "
            "Please review all cited sources to verify consistency."
        )

    answer = (
        "Based on the indexed policy documents:\n\n"
        + "\n\n---\n\n".join(lines)
        + conflict_note
    )

    return {"answer": answer, "sources": docs, "answer_type": "grounded"}


# ---------------------------------------------------------------------------
# Node: log_audit_record
# ---------------------------------------------------------------------------

def log_audit_record(state: AgentState) -> dict:
    """Record the completed query in the audit metadata dict.

    In production, Stage 4b will call snowflake_logger.log_query_to_snowflake()
    here. For Stage 4a we record into audit_metadata only (no Snowflake call).
    This node must never raise — a logging failure must not abort the answer.
    """
    evidence = state.get("evidence") or {}
    try:
        record = {
            "query": state.get("query", ""),
            "answer_type": state.get("answer_type", ""),
            "category": (state.get("classification") or {}).get("category", "unknown"),
            "retrieved_doc_count": len(state.get("retrieved_docs") or []),
            "source_files": sorted({
                d.metadata.get("source_filename", "")
                for d in (state.get("sources") or [])
            }),
            "evidence_coverage": evidence.get("coverage", 0.0),
            "evidence_confidence": evidence.get("confidence", 0.0),
            "retry_count": state.get("retry_count", 0),
            "conflicts_flagged": len(state.get("conflicts") or []),
            "error": state.get("error"),
        }
    except Exception as exc:
        record = {"logging_error": str(exc)}

    return {"audit_metadata": record}


# ---------------------------------------------------------------------------
# Conditional edge router
# ---------------------------------------------------------------------------

def _route_after_assessment(state: AgentState) -> str:
    """Select the next node after assess_evidence completes.

    Routing priority:
      1. Retry if evidence insufficient AND retry budget remains.
      2. Conflict check if the query spans multiple documents.
      3. Generate answer (including graceful refusal if insufficient).
    """
    if state.get("needs_retry") and state.get("retry_count", 0) < _MAX_RETRIES:
        return "expand_retrieval"
    if (state.get("classification") or {}).get("requires_cross_doc"):
        return "check_cross_doc_conflicts"
    return "generate_grounded_answer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Compile and return the Evidence-Aware Agentic RAG graph.

    Call once per process and cache the result; compilation is not free.
    Returns a CompiledStateGraph whose .invoke() method accepts an AgentState.
    """
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("classify_query", classify_query)
    g.add_node("plan_retrieval", plan_retrieval)
    g.add_node("retrieve_policy_chunks", retrieve_policy_chunks)
    g.add_node("assess_evidence", assess_evidence)
    g.add_node("expand_retrieval", expand_retrieval)
    g.add_node("check_cross_doc_conflicts", check_cross_doc_conflicts)
    g.add_node("generate_grounded_answer", generate_grounded_answer)
    g.add_node("log_audit_record", log_audit_record)

    # Linear edges
    g.set_entry_point("classify_query")
    g.add_edge("classify_query", "plan_retrieval")
    g.add_edge("plan_retrieval", "retrieve_policy_chunks")
    g.add_edge("retrieve_policy_chunks", "assess_evidence")

    # Conditional edge after assess_evidence
    g.add_conditional_edges(
        "assess_evidence",
        _route_after_assessment,
        {
            "expand_retrieval": "expand_retrieval",
            "check_cross_doc_conflicts": "check_cross_doc_conflicts",
            "generate_grounded_answer": "generate_grounded_answer",
        },
    )

    # expand_retrieval loops back to assess_evidence
    g.add_edge("expand_retrieval", "assess_evidence")

    # check_cross_doc_conflicts always proceeds to generate
    g.add_edge("check_cross_doc_conflicts", "generate_grounded_answer")

    # generate → audit → END
    g.add_edge("generate_grounded_answer", "log_audit_record")
    g.add_edge("log_audit_record", END)

    return g.compile()
