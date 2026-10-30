"""
AgentState: single immutable dict threaded through every node in the graph.

Each node receives the full state and returns a dict containing only the
fields it updated. LangGraph merges the partial update into the running state.

Design notes (Stage 4b additions):
- llm_override: test injection point for mock LLMs; production leaves it None.
- available_sources: source_filename values discovered from ChromaDB metadata.
  Set by plan_retrieval; populated by _discover_indexed_sources() from the
  live index, or pre-set in make_initial_state() for tests.
- evidence_status: fine-grained status from LLM-based evidence evaluation.
  SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONFLICTING | UNAVAILABLE.
  UNAVAILABLE means the LLM assessment itself failed (not that evidence is bad).
- supporting_passages: LLM-validated passages [{text, source, page}].
  Only passages whose text appears in retrieved_docs are accepted.
- conflict_status: CONFLICT_DETECTED | NO_CONFLICT | UNKNOWN.
  UNKNOWN is returned when LLM conflict analysis fails — never NO_CONFLICT on error.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class QueryClassification(TypedDict):
    """Structured output of the classify_query node."""
    category: str           # direct_factual | multi_section | cross_document | potential_conflict | needs_clarification
    intent: str             # one-line description of what the query asks
    requires_cross_doc: bool  # True when the answer likely spans multiple policy files


class RetrievalPlan(TypedDict):
    """Structured output of the plan_retrieval node."""
    k: int                       # number of top-k chunks to retrieve
    filter_hints: list           # keyword hints (Stage 4c: applied as metadata filters)
    document_hints: list         # source_filename hints from actual indexed metadata


class EvidenceAssessment(TypedDict):
    """Numeric summary from the assess_evidence node.

    coverage and confidence are heuristic estimates — not calibrated probabilities.
    In Stage 4b, both are populated from EvidenceEvaluation.coverage_estimate.
    sufficient is the actionable gate for the routing decision.
    """
    coverage: float   # 0.0–1.0 heuristic coverage estimate
    confidence: float # 0.0–1.0 (same as coverage in Stage 4b; distinct in 4c)
    sufficient: bool  # True when evidence meets the threshold for generating an answer


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str

    # ── Injection points (tests / callers may provide these directly) ───────
    retriever_override: Optional[Any]   # LangChain retriever | None
    llm_override: Optional[Any]         # LLM instance (mock or real) | None
    available_sources: list             # pre-populated source_filename list for tests

    # ── Computed by classify_query ──────────────────────────────────────────
    classification: Optional[QueryClassification]

    # ── Computed by plan_retrieval ──────────────────────────────────────────
    retrieval_plan: Optional[RetrievalPlan]

    # ── Computed by retrieve_policy_chunks / expand_retrieval ──────────────
    retrieved_docs: list   # list[Document]

    # ── Computed by assess_evidence ─────────────────────────────────────────
    evidence: Optional[EvidenceAssessment]
    evidence_status: str    # SUPPORTED|PARTIALLY_SUPPORTED|INSUFFICIENT|CONFLICTING|UNAVAILABLE
    supporting_passages: list  # [{text: str, source: str, page: int}]
    needs_retry: bool           # consumed by the conditional edge router

    # ── Retry loop state ────────────────────────────────────────────────────
    retry_count: int        # incremented by expand_retrieval; capped at _MAX_RETRIES

    # ── Computed by check_cross_doc_conflicts ───────────────────────────────
    conflicts: list         # list[str] — human-readable conflict descriptions
    conflict_status: str    # CONFLICT_DETECTED | NO_CONFLICT | UNKNOWN

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
    llm_override=None,
    available_sources: Optional[list] = None,
) -> AgentState:
    """Return a fully initialised AgentState ready for graph.invoke().

    Args:
        query:             The compliance question to answer.
        retriever:         Optional LangChain retriever override (test injection).
        llm_override:      Optional mock or real LLM override (test injection).
        available_sources: Pre-populated list of indexed source_filename values.
                           When None, plan_retrieval will discover from ChromaDB.
    """
    return {
        "query": query,
        "retriever_override": retriever,
        "llm_override": llm_override,
        "available_sources": available_sources or [],
        "classification": None,
        "retrieval_plan": None,
        "retrieved_docs": [],
        "evidence": None,
        "evidence_status": "",
        "supporting_passages": [],
        "needs_retry": False,
        "retry_count": 0,
        "conflicts": [],
        "conflict_status": "",
        "answer": "",
        "sources": [],
        "answer_type": "",
        "error": None,
        "audit_metadata": {},
    }
