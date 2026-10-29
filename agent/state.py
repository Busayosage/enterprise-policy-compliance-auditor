"""
AgentState: single immutable dict threaded through every node in the graph.

Each node receives the full state and returns a dict containing only the
fields it updated. LangGraph merges the partial update into the running state.

Design notes:
- All fields have safe defaults so nodes can read fields set by prior nodes
  without KeyError even when the graph is partially executed (e.g. tests).
- `retry_count` is the sole mutable counter; it is incremented only by
  expand_retrieval and never reset mid-run (guards the 2-attempt cap).
- `retriever_override` is an injection point for tests: when set, retrieve_
  policy_chunks uses it directly instead of calling database.load_retriever().
- `needs_retry` is set by assess_evidence and consumed by the conditional edge
  router; it is NOT a persistent preference — it is reset to False after use.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class QueryClassification(TypedDict):
    """Structured output of the classify_query node."""
    category: str           # direct_factual | multi_section | cross_document | unsupported | unknown
    intent: str             # one-line description of what the query asks
    requires_cross_doc: bool  # True when the answer likely spans multiple policy files


class RetrievalPlan(TypedDict):
    """Structured output of the plan_retrieval node."""
    k: int                       # number of top-k chunks to retrieve
    filter_hints: list           # keyword hints to bias retrieval (Stage 4b: applied as filters)
    document_hints: list         # source_filename hints (Stage 4b: applied as metadata filters)


class EvidenceAssessment(TypedDict):
    """Structured output of the assess_evidence node."""
    coverage: float   # 0.0–1.0 heuristic: how much of the query intent is covered
    confidence: float # 0.0–1.0 heuristic: overall confidence in the retrieved chunks
    sufficient: bool  # True when coverage and confidence meet the threshold for an answer


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str

    # ── Injection (tests / callers may provide a retriever directly) ────────
    retriever_override: Optional[Any]   # LangChain retriever | None

    # ── Computed by classify_query ──────────────────────────────────────────
    classification: Optional[QueryClassification]

    # ── Computed by plan_retrieval ──────────────────────────────────────────
    retrieval_plan: Optional[RetrievalPlan]

    # ── Computed by retrieve_policy_chunks / expand_retrieval ──────────────
    retrieved_docs: list   # list[Document]

    # ── Computed by assess_evidence ─────────────────────────────────────────
    evidence: Optional[EvidenceAssessment]
    needs_retry: bool       # consumed by the conditional edge router

    # ── Retry loop state ────────────────────────────────────────────────────
    retry_count: int        # incremented by expand_retrieval; capped at 2

    # ── Computed by check_cross_doc_conflicts ───────────────────────────────
    conflicts: list         # list[str] — human-readable conflict descriptions

    # ── Computed by generate_grounded_answer ────────────────────────────────
    answer: str
    sources: list           # list[Document] — the chunks cited in the answer
    answer_type: str        # "grounded" | "refusal" | "error"

    # ── Error channel ───────────────────────────────────────────────────────
    error: Optional[str]

    # ── Audit ────────────────────────────────────────────────────────────────
    audit_metadata: dict    # populated by log_audit_record before END


def make_initial_state(
    query: str,
    *,
    retriever=None,
) -> AgentState:
    """Return a fully initialised AgentState ready for graph.invoke()."""
    return {
        "query": query,
        "retriever_override": retriever,
        "classification": None,
        "retrieval_plan": None,
        "retrieved_docs": [],
        "evidence": None,
        "needs_retry": False,
        "retry_count": 0,
        "conflicts": [],
        "answer": "",
        "sources": [],
        "answer_type": "",
        "error": None,
        "audit_metadata": {},
    }
