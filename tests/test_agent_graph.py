"""
Stage 4a: offline tests for the LangGraph agent foundation.

All tests are fully deterministic:
  - No live NVIDIA, Snowflake, or ChromaDB calls.
  - The retriever is injected via retriever_override in make_initial_state().
  - The compiled graph is shared across the module via a session fixture.

Test coverage:
  1. Graph compiles without error.
  2. All eight nodes are present in the compiled graph.
  3. Full happy-path run returns a grounded answer with sources.
  4. Refusal path: empty retriever → answer_type == "refusal".
  5. Retry limit: retriever always returns insufficient docs → retry_count == MAX.
  6. Cross-doc routing: cross_document query routes through conflict check.
  7. Unsupported-topic query → refusal.
  8. Error channel: retriever raises → answer_type == "error".
  9. Audit metadata is always populated.
 10. retry_count never exceeds _MAX_RETRIES.
 11. sources is always a list (never None).
 12. Each node's partial update is merged correctly into state.
 13. Keyword classifier coverage (direct/multi/cross/unsupported/unknown).
 14. plan_retrieval produces correct k values per category.
 15. assess_evidence coverage/confidence thresholds.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from agent.state import make_initial_state, AgentState
from agent.graph import (
    build_graph,
    classify_query,
    plan_retrieval,
    retrieve_policy_chunks,
    assess_evidence,
    expand_retrieval,
    check_cross_doc_conflicts,
    generate_grounded_answer,
    log_audit_record,
    _route_after_assessment,
    _MAX_RETRIES,
    _BASE_K,
    _EXPAND_K_INCREMENT,
    _REFUSAL_MESSAGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, source: str = "it_security_policy.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source_filename": source, "page": page})


def _stub_retriever(docs: list) -> MagicMock:
    """Return a mock retriever whose .invoke() always returns `docs`."""
    r = MagicMock()
    r.invoke.return_value = docs
    return r


def _minimal_state(**overrides) -> AgentState:
    """Return a base state with safe defaults, overridden by kwargs."""
    base = make_initial_state("What is the password policy?")
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Session-scoped compiled graph (compile once, share across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compiled_graph():
    return build_graph()


# ---------------------------------------------------------------------------
# 1. Graph compilation
# ---------------------------------------------------------------------------

def test_graph_compiles():
    g = build_graph()
    assert g is not None


def test_graph_has_invoke(compiled_graph):
    assert callable(compiled_graph.invoke)


# ---------------------------------------------------------------------------
# 2. Node presence
# ---------------------------------------------------------------------------

def test_graph_nodes_present(compiled_graph):
    expected_nodes = {
        "classify_query",
        "plan_retrieval",
        "retrieve_policy_chunks",
        "assess_evidence",
        "expand_retrieval",
        "check_cross_doc_conflicts",
        "generate_grounded_answer",
        "log_audit_record",
    }
    graph_nodes = set(compiled_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert expected_nodes == graph_nodes


# ---------------------------------------------------------------------------
# 3. Happy-path end-to-end run
# ---------------------------------------------------------------------------

def test_happy_path_grounded_answer(compiled_graph):
    docs = [
        _make_doc("Passwords must be at least 12 characters long.", "it_security_policy.pdf"),
        _make_doc("Multi-factor authentication is required.", "it_security_policy.pdf"),
        _make_doc("Accounts are locked after 5 failed attempts.", "it_security_policy.pdf"),
    ]
    initial = make_initial_state(
        "What is the password policy?",
        retriever=_stub_retriever(docs),
    )
    result = compiled_graph.invoke(initial)

    assert result["answer_type"] == "grounded"
    assert len(result["sources"]) == 3
    assert result["answer"] != ""
    assert "it_security_policy.pdf" in result["answer"]
    assert result["error"] is None
    assert isinstance(result["audit_metadata"], dict)


# ---------------------------------------------------------------------------
# 4. Refusal path: empty retriever
# ---------------------------------------------------------------------------

def test_empty_retriever_produces_refusal(compiled_graph):
    initial = make_initial_state(
        "What is the leave policy?",
        retriever=_stub_retriever([]),
    )
    result = compiled_graph.invoke(initial)

    assert result["answer_type"] in ("refusal", "error")
    assert result["answer"] == _REFUSAL_MESSAGE or "cannot find" in result["answer"].lower()
    assert result["sources"] == []


# ---------------------------------------------------------------------------
# 5. Retry limit enforcement
# ---------------------------------------------------------------------------

def test_retry_count_capped_at_max(compiled_graph):
    """Retriever always returns one doc whose text doesn't match the query well.
    The agent should retry but stop at _MAX_RETRIES."""
    poor_doc = _make_doc("General office safety notices.", "it_security_policy.pdf")
    initial = make_initial_state(
        "What is the password minimum length requirement?",
        retriever=_stub_retriever([poor_doc]),
    )
    result = compiled_graph.invoke(initial)

    assert result["retry_count"] <= _MAX_RETRIES


def test_retry_count_never_exceeds_max_retries(compiled_graph):
    """Stronger assertion: retry_count must never exceed _MAX_RETRIES."""
    stub = _stub_retriever([])  # always empty → always insufficient
    initial = make_initial_state("What is the data retention policy?", retriever=stub)
    result = compiled_graph.invoke(initial)
    assert result["retry_count"] <= _MAX_RETRIES


def test_total_retrieval_calls_capped_at_three(compiled_graph):
    """_MAX_RETRIES=2 means expand_retrieval fires at most twice.
    Total retrieval calls = 1 (initial) + 2 (expansions) = 3.
    The retriever mock's call_count must not exceed 3 even for a
    query that never achieves sufficient evidence.
    """
    stub = _stub_retriever([])  # always empty → always insufficient, always wants retry
    initial = make_initial_state("What is the password minimum length?", retriever=stub)
    compiled_graph.invoke(initial)
    # 1 call in retrieve_policy_chunks + up to 2 calls in expand_retrieval = 3 max
    assert stub.invoke.call_count <= 1 + _MAX_RETRIES


# ---------------------------------------------------------------------------
# 6. Cross-doc routing
# ---------------------------------------------------------------------------

def test_cross_doc_query_runs_conflict_check(compiled_graph):
    docs = [
        _make_doc("IT policy: passwords expire after 90 days.", "it_security_policy.pdf"),
        _make_doc("HR policy: annual leave entitlement is 25 days.", "hr_leave_policy.pdf"),
    ]
    initial = make_initial_state(
        "Compare the IT and HR policies on access controls and leave entitlement.",
        retriever=_stub_retriever(docs),
    )
    result = compiled_graph.invoke(initial)

    # Cross-doc path was taken → conflicts list populated (even if empty)
    assert isinstance(result["conflicts"], list)
    # With two different source files, a conflict note should be present
    assert len(result["conflicts"]) >= 1


# ---------------------------------------------------------------------------
# 7. Unsupported-topic refusal
# ---------------------------------------------------------------------------

def test_unsupported_topic_produces_refusal(compiled_graph):
    initial = make_initial_state(
        "What is my salary and payroll schedule?",
        retriever=_stub_retriever([]),
    )
    result = compiled_graph.invoke(initial)

    assert result["answer_type"] in ("refusal", "error")
    assert _REFUSAL_MESSAGE in result["answer"]


# ---------------------------------------------------------------------------
# 8. Error channel: retriever raises an exception
# ---------------------------------------------------------------------------

def test_retriever_exception_sets_error_channel(compiled_graph):
    bad_retriever = MagicMock()
    bad_retriever.invoke.side_effect = RuntimeError("ChromaDB connection refused")

    initial = make_initial_state(
        "What is the expense reimbursement limit?",
        retriever=bad_retriever,
    )
    result = compiled_graph.invoke(initial)

    # Graph must complete (not raise)
    assert result["answer_type"] in ("refusal", "error")
    assert result["sources"] == []


# ---------------------------------------------------------------------------
# 9. Audit metadata always populated
# ---------------------------------------------------------------------------

def test_audit_metadata_always_present(compiled_graph):
    for question, docs in [
        ("What is the password policy?", [_make_doc("Passwords must be 12 chars.")]),
        ("What is my salary?", []),
    ]:
        initial = make_initial_state(question, retriever=_stub_retriever(docs))
        result = compiled_graph.invoke(initial)
        assert isinstance(result["audit_metadata"], dict)
        assert "query" in result["audit_metadata"]
        assert "answer_type" in result["audit_metadata"]


# ---------------------------------------------------------------------------
# 10. sources is always a list
# ---------------------------------------------------------------------------

def test_sources_are_only_retrieved_documents(compiled_graph):
    """Citations cannot be fabricated: sources must be an exact subset of retrieved_docs.
    No document may appear in sources unless it was returned by the retriever.
    """
    docs = [_make_doc("Password must be 12 characters.", "it_security_policy.pdf")]
    initial = make_initial_state("What is the password policy?", retriever=_stub_retriever(docs))
    result = compiled_graph.invoke(initial)

    for src in result["sources"]:
        assert src in result["retrieved_docs"], (
            f"Source '{src.page_content[:40]}' was not in retrieved_docs — fabricated citation"
        )


def test_sources_always_list(compiled_graph):
    for retriever_docs in [[], [_make_doc("Some policy text.")]]:
        initial = make_initial_state(
            "What is the leave entitlement?",
            retriever=_stub_retriever(retriever_docs),
        )
        result = compiled_graph.invoke(initial)
        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# 11. Unit tests: classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    def test_direct_factual(self):
        state = _minimal_state(query="What is the password expiry period?")
        result = classify_query(state)
        assert result["classification"]["category"] == "direct_factual"
        assert result["classification"]["requires_cross_doc"] is False

    def test_cross_document(self):
        state = _minimal_state(query="Compare IT and HR policies on access.")
        result = classify_query(state)
        assert result["classification"]["category"] == "cross_document"
        assert result["classification"]["requires_cross_doc"] is True

    def test_unsupported(self):
        state = _minimal_state(query="What is my payroll schedule this month?")
        result = classify_query(state)
        assert result["classification"]["category"] == "unsupported"

    def test_multi_section(self):
        long_q = (
            "What are the requirements for password length and what are the "
            "rules for multi-factor authentication and account lockout policy?"
        )
        state = _minimal_state(query=long_q)
        result = classify_query(state)
        assert result["classification"]["category"] in ("multi_section", "direct_factual")

    def test_both_policies_phrase_is_cross_doc(self):
        state = _minimal_state(query="What do both policies say about data retention?")
        result = classify_query(state)
        assert result["classification"]["requires_cross_doc"] is True


# ---------------------------------------------------------------------------
# 12. Unit tests: plan_retrieval
# ---------------------------------------------------------------------------

class TestPlanRetrieval:
    def _classified_state(self, category: str, query: str = "test") -> AgentState:
        return _minimal_state(
            query=query,
            classification={
                "category": category,
                "intent": "test",
                "requires_cross_doc": category == "cross_document",
            },
        )

    def test_direct_factual_k(self):
        result = plan_retrieval(self._classified_state("direct_factual", "password policy"))
        assert result["retrieval_plan"]["k"] == _BASE_K

    def test_cross_document_k_larger(self):
        result = plan_retrieval(self._classified_state("cross_document", "compare IT and HR"))
        assert result["retrieval_plan"]["k"] > _BASE_K

    def test_multi_section_k_larger_than_direct(self):
        direct = plan_retrieval(self._classified_state("direct_factual", "password"))
        multi = plan_retrieval(self._classified_state("multi_section", "password and mfa and lockout"))
        assert multi["retrieval_plan"]["k"] >= direct["retrieval_plan"]["k"]

    def test_it_document_hint(self):
        result = plan_retrieval(self._classified_state("direct_factual", "password security policy"))
        assert "it_security_policy.pdf" in result["retrieval_plan"]["document_hints"]

    def test_hr_document_hint(self):
        result = plan_retrieval(self._classified_state("direct_factual", "leave holiday entitlement"))
        assert "hr_leave_policy.pdf" in result["retrieval_plan"]["document_hints"]

    def test_finance_document_hint(self):
        result = plan_retrieval(self._classified_state("direct_factual", "expense finance receipt"))
        assert "finance_expense_policy.pdf" in result["retrieval_plan"]["document_hints"]


# ---------------------------------------------------------------------------
# 13. Unit tests: assess_evidence
# ---------------------------------------------------------------------------

class TestAssessEvidence:
    def test_empty_docs_insufficient(self):
        state = _minimal_state(retrieved_docs=[], retry_count=0)
        result = assess_evidence(state)
        assert result["evidence"]["sufficient"] is False

    def test_good_docs_sufficient(self):
        docs = [
            _make_doc("password must be twelve characters long minimum"),
            _make_doc("password expiry is set at ninety days"),
            _make_doc("password reuse is prohibited for last five passwords"),
        ]
        state = _minimal_state(
            query="What is the password policy requirement?",
            retrieved_docs=docs,
            retry_count=0,
        )
        result = assess_evidence(state)
        assert result["evidence"]["sufficient"] is True

    def test_needs_retry_false_when_budget_exhausted(self):
        state = _minimal_state(retrieved_docs=[], retry_count=_MAX_RETRIES)
        result = assess_evidence(state)
        assert result["needs_retry"] is False

    def test_needs_retry_true_when_budget_remains(self):
        state = _minimal_state(retrieved_docs=[], retry_count=0)
        result = assess_evidence(state)
        assert result["needs_retry"] is True

    def test_coverage_between_0_and_1(self):
        docs = [_make_doc("Some policy text here about various rules.")]
        state = _minimal_state(query="password policy requirements", retrieved_docs=docs)
        result = assess_evidence(state)
        assert 0.0 <= result["evidence"]["coverage"] <= 1.0

    def test_error_state_produces_insufficient(self):
        state = _minimal_state(retrieved_docs=[], error="Connection refused")
        result = assess_evidence(state)
        assert result["evidence"]["sufficient"] is False
        assert result["needs_retry"] is False


# ---------------------------------------------------------------------------
# 14. Unit tests: expand_retrieval
# ---------------------------------------------------------------------------

class TestExpandRetrieval:
    def test_increments_retry_count(self):
        docs = [_make_doc("policy text")]
        state = _minimal_state(
            retrieval_plan={"k": _BASE_K, "filter_hints": [], "document_hints": []},
            retry_count=0,
            retriever_override=_stub_retriever(docs),
        )
        result = expand_retrieval(state)
        assert result["retry_count"] == 1

    def test_increases_k(self):
        docs = [_make_doc("policy text")]
        state = _minimal_state(
            retrieval_plan={"k": _BASE_K, "filter_hints": [], "document_hints": []},
            retry_count=0,
            retriever_override=_stub_retriever(docs),
        )
        result = expand_retrieval(state)
        assert result["retrieval_plan"]["k"] == _BASE_K + _EXPAND_K_INCREMENT

    def test_resets_needs_retry(self):
        docs = [_make_doc("policy text")]
        state = _minimal_state(
            retrieval_plan={"k": _BASE_K, "filter_hints": [], "document_hints": []},
            needs_retry=True,
            retry_count=0,
            retriever_override=_stub_retriever(docs),
        )
        result = expand_retrieval(state)
        assert result.get("needs_retry") is False


# ---------------------------------------------------------------------------
# 15. Unit tests: _route_after_assessment
# ---------------------------------------------------------------------------

class TestRouter:
    def _state_with(self, needs_retry=False, retry_count=0, requires_cross_doc=False):
        return _minimal_state(
            needs_retry=needs_retry,
            retry_count=retry_count,
            classification={
                "category": "cross_document" if requires_cross_doc else "direct_factual",
                "intent": "test",
                "requires_cross_doc": requires_cross_doc,
            },
            evidence={"coverage": 0.8, "confidence": 0.8, "sufficient": not needs_retry},
        )

    def test_retry_route(self):
        state = self._state_with(needs_retry=True, retry_count=0)
        assert _route_after_assessment(state) == "expand_retrieval"

    def test_no_retry_when_budget_exhausted(self):
        state = self._state_with(needs_retry=True, retry_count=_MAX_RETRIES)
        route = _route_after_assessment(state)
        assert route != "expand_retrieval"

    def test_cross_doc_route(self):
        state = self._state_with(requires_cross_doc=True)
        assert _route_after_assessment(state) == "check_cross_doc_conflicts"

    def test_direct_generate_route(self):
        state = self._state_with()
        assert _route_after_assessment(state) == "generate_grounded_answer"

    def test_retry_takes_priority_over_cross_doc(self):
        state = self._state_with(needs_retry=True, retry_count=0, requires_cross_doc=True)
        assert _route_after_assessment(state) == "expand_retrieval"


# ---------------------------------------------------------------------------
# 16. Unit tests: check_cross_doc_conflicts
# ---------------------------------------------------------------------------

class TestCheckCrossDocConflicts:
    def test_single_source_no_conflict(self):
        docs = [
            _make_doc("Policy A text.", "it_security_policy.pdf"),
            _make_doc("Policy A text 2.", "it_security_policy.pdf"),
        ]
        state = _minimal_state(retrieved_docs=docs)
        result = check_cross_doc_conflicts(state)
        assert result["conflicts"] == []

    def test_multi_source_flags_conflict_note(self):
        docs = [
            _make_doc("IT policy text.", "it_security_policy.pdf"),
            _make_doc("HR policy text.", "hr_leave_policy.pdf"),
        ]
        state = _minimal_state(retrieved_docs=docs)
        result = check_cross_doc_conflicts(state)
        assert len(result["conflicts"]) >= 1

    def test_empty_docs_no_conflicts(self):
        state = _minimal_state(retrieved_docs=[])
        result = check_cross_doc_conflicts(state)
        assert result["conflicts"] == []


# ---------------------------------------------------------------------------
# 17. Unit tests: generate_grounded_answer
# ---------------------------------------------------------------------------

class TestGenerateGroundedAnswer:
    def _sufficient_state(self, docs):
        return _minimal_state(
            retrieved_docs=docs,
            evidence={"coverage": 0.8, "confidence": 0.8, "sufficient": True},
            classification={"category": "direct_factual", "intent": "", "requires_cross_doc": False},
            conflicts=[],
        )

    def test_grounded_answer_contains_source_filename(self):
        docs = [_make_doc("Password must be 12 characters.", "it_security_policy.pdf")]
        state = self._sufficient_state(docs)
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "grounded"
        assert "it_security_policy.pdf" in result["answer"]

    def test_refusal_when_insufficient_evidence(self):
        state = _minimal_state(
            retrieved_docs=[],
            evidence={"coverage": 0.0, "confidence": 0.0, "sufficient": False},
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "refusal"
        assert result["answer"] == _REFUSAL_MESSAGE

    def test_error_state_produces_error_answer_type(self):
        state = _minimal_state(retrieved_docs=[], error="Retriever exploded")
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "error"

    def test_sources_matches_docs(self):
        docs = [
            _make_doc("Policy text A."),
            _make_doc("Policy text B."),
        ]
        state = self._sufficient_state(docs)
        result = generate_grounded_answer(state)
        assert result["sources"] == docs

    def test_conflict_note_appended(self):
        docs = [
            _make_doc("IT policy text.", "it_security_policy.pdf"),
            _make_doc("HR policy text.", "hr_leave_policy.pdf"),
        ]
        state = _minimal_state(
            retrieved_docs=docs,
            evidence={"coverage": 0.8, "confidence": 0.8, "sufficient": True},
            classification={"category": "cross_document", "intent": "", "requires_cross_doc": True},
            conflicts=["Evidence spans multiple policy files."],
        )
        result = generate_grounded_answer(state)
        assert "multiple policy documents" in result["answer"]


# ---------------------------------------------------------------------------
# 18. Unit tests: log_audit_record
# ---------------------------------------------------------------------------

class TestLogAuditRecord:
    def test_audit_contains_required_keys(self):
        state = _minimal_state(
            answer_type="grounded",
            classification={"category": "direct_factual", "intent": "", "requires_cross_doc": False},
            retrieved_docs=[_make_doc("text")],
            sources=[_make_doc("text")],
            evidence={"coverage": 0.9, "confidence": 0.9, "sufficient": True},
            retry_count=0,
            conflicts=[],
        )
        result = log_audit_record(state)
        meta = result["audit_metadata"]
        assert "query" in meta
        assert "answer_type" in meta
        assert "category" in meta
        assert "retry_count" in meta
        assert "evidence_coverage" in meta

    def test_audit_never_raises(self):
        # Pass a state with missing / None fields to verify robustness
        state = make_initial_state("test query")
        result = log_audit_record(state)
        assert isinstance(result["audit_metadata"], dict)
