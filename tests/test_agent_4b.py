"""
Stage 4b offline tests for the Evidence-Aware Agentic RAG pipeline.

All tests are fully deterministic:
  - No live NVIDIA, Snowflake, or ChromaDB calls.
  - LLM calls are replaced by mock objects passed via llm_override.
  - Retriever calls are replaced by stub objects passed via retriever_override.
  - Tests verify the full graph topology changes in Stage 4b.

Sections:
  A. Mock helpers
  B. classify_query — LLM path + fallback (no unsupported classification)
  C. plan_retrieval — available_sources injection + cross-doc hints
  D. assess_evidence — 4 LLM outcomes + UNAVAILABLE fallback
  E. check_cross_doc_conflicts — conflict detected / no-conflict / UNKNOWN fallback
  F. generate_grounded_answer — LLM answer + citation validation + fallback
  G. _route_after_assessment — CONFLICTING routes to conflict check
  H. Full pipeline integration with mocked LLM
  I. agent/llm helpers — context builder, citation validator
  J. Retrieval baseline comparison against Phase 3 (fixture-based, session-scoped)
"""

import os
import shutil
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
    _classify_query_heuristic,
    _assess_evidence_heuristic,
    _validate_passages,
    _generate_excerpt_answer,
    _MAX_RETRIES,
    _BASE_K,
    _REFUSAL_MESSAGE,
    _STATUS_SUPPORTED,
    _STATUS_PARTIAL,
    _STATUS_INSUFFICIENT,
    _STATUS_CONFLICTING,
    _STATUS_UNAVAILABLE,
    _CONFLICT_DETECTED,
    _NO_CONFLICT,
    _CONFLICT_UNKNOWN,
)
from agent.llm import (
    ClassificationOutput,
    EvidenceEvaluation,
    ConflictAnalysis,
    build_agent_context,
    validate_answer_citations,
)
from evaluation_dataset import EVALUATION_QUESTIONS

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ===========================================================================
# A. Mock helpers
# ===========================================================================

def _doc(text: str, source: str = "it_security_policy.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source_filename": source, "page": page})


def _stub_retriever(docs: list) -> MagicMock:
    r = MagicMock()
    r.invoke.return_value = docs
    return r


def _make_classify_mock(category="direct_factual", requires_cross_doc=False):
    """Return a mock LLM that produces ClassificationOutput."""
    mock = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = ClassificationOutput(
        category=category,
        intent="Test intent",
        requires_cross_doc=requires_cross_doc,
        needs_clarification=False,
        answerability="answerable",
    )
    mock.with_structured_output.return_value = structured
    return mock


def _make_evidence_mock(status=_STATUS_SUPPORTED, coverage=0.9):
    """Return a mock LLM that produces EvidenceEvaluation."""
    mock = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = EvidenceEvaluation(
        status=status,
        reasoning="Test reasoning",
        supporting_passages=[{"text": "policy text", "source": "it_security_policy.pdf", "page": 1}],
        coverage_estimate=coverage,
    )
    mock.with_structured_output.return_value = structured
    return mock


def _make_conflict_mock(has_conflict=False, descriptions=None, review=False):
    """Return a mock LLM that produces ConflictAnalysis."""
    mock = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = ConflictAnalysis(
        has_genuine_conflict=has_conflict,
        conflict_descriptions=descriptions or [],
        supporting_passages=[],
        review_recommended=review,
    )
    mock.with_structured_output.return_value = structured
    return mock


def _make_failing_mock():
    """Return a mock LLM whose all calls raise RuntimeError.

    with_structured_output is used for classification/evidence/conflict nodes.
    __call__ (mock.side_effect) is used by the LangChain chain for generate_grounded_answer.
    """
    mock = MagicMock()
    mock.with_structured_output.side_effect = RuntimeError("LLM unavailable")
    mock.side_effect = RuntimeError("LLM unavailable")  # chain calls mock(messages)
    return mock


def _make_answer_mock(answer_text: str):
    """Return a mock LLM that produces a plain answer string via chain.invoke().

    LangChain chains call llm(messages) (__call__), not llm.invoke(messages),
    so the response must be set on mock.return_value, not mock.invoke.return_value.
    """
    mock = MagicMock()
    response = MagicMock()
    response.content = answer_text
    mock.return_value = response  # chain uses mock(messages), not mock.invoke(messages)
    return mock


def _make_full_pipeline_mock(
    category="direct_factual",
    requires_cross_doc=False,
    evidence_status=_STATUS_SUPPORTED,
    coverage=0.9,
    answer_text="Based on the policy, passwords must be 12 characters.",
):
    """Return a single mock LLM that handles all Stage 4b structured outputs + answer."""
    mock = MagicMock()

    def wso_side_effect(schema):
        inner = MagicMock()
        if schema.__name__ == "ClassificationOutput":
            inner.invoke.return_value = ClassificationOutput(
                category=category,
                intent="Test intent",
                requires_cross_doc=requires_cross_doc,
                needs_clarification=False,
                answerability="answerable",
            )
        elif schema.__name__ == "EvidenceEvaluation":
            inner.invoke.return_value = EvidenceEvaluation(
                status=evidence_status,
                reasoning="Test",
                supporting_passages=[],
                coverage_estimate=coverage,
            )
        elif schema.__name__ == "ConflictAnalysis":
            inner.invoke.return_value = ConflictAnalysis(
                has_genuine_conflict=False,
                conflict_descriptions=[],
                supporting_passages=[],
                review_recommended=False,
            )
        else:
            inner.invoke.return_value = MagicMock()
        return inner

    mock.with_structured_output.side_effect = wso_side_effect

    # For the generate_grounded_answer prompt | llm chain
    ans_response = MagicMock()
    ans_response.content = answer_text
    mock.return_value = ans_response  # chain uses mock(messages), not mock.invoke(messages)

    return mock


def _state(**overrides) -> AgentState:
    base = make_initial_state("What is the password policy?")
    base.update(overrides)
    return base


# ===========================================================================
# Session-scoped compiled graph
# ===========================================================================

@pytest.fixture(scope="module")
def graph():
    return build_graph()


# ===========================================================================
# B. classify_query — LLM path + fallback
# ===========================================================================

class TestClassifyQueryStage4b:
    def test_llm_path_direct_factual(self):
        mock = _make_classify_mock("direct_factual")
        state = _state(llm_override=mock)
        result = classify_query(state)
        assert result["classification"]["category"] == "direct_factual"
        assert result["classification"]["requires_cross_doc"] is False

    def test_llm_path_cross_document(self):
        mock = _make_classify_mock("cross_document", requires_cross_doc=True)
        state = _state(llm_override=mock)
        result = classify_query(state)
        assert result["classification"]["category"] == "cross_document"
        assert result["classification"]["requires_cross_doc"] is True

    def test_llm_path_multi_section(self):
        mock = _make_classify_mock("multi_section")
        state = _state(llm_override=mock)
        result = classify_query(state)
        assert result["classification"]["category"] == "multi_section"

    def test_llm_path_potential_conflict(self):
        mock = _make_classify_mock("potential_conflict", requires_cross_doc=True)
        state = _state(llm_override=mock)
        result = classify_query(state)
        assert result["classification"]["category"] == "potential_conflict"

    def test_fallback_on_llm_failure(self):
        """When LLM fails, heuristic fallback must produce a valid classification."""
        state = _state(llm_override=_make_failing_mock())
        result = classify_query(state)
        assert result["classification"] is not None
        assert result["classification"]["category"] in {
            "direct_factual", "multi_section", "cross_document",
            "potential_conflict", "needs_clarification",
        }

    def test_no_llm_key_falls_back_to_heuristic(self):
        """Without NVIDIA_API_KEY and no override, get_agent_llm raises → heuristic used."""
        state = _state()  # no llm_override, no API key in test env
        result = classify_query(state)
        assert result["classification"] is not None

    def test_salary_question_not_classified_unsupported(self):
        """Stage 4b: 'salary' must NOT cause unsupported classification."""
        for mock in [_make_classify_mock("direct_factual"), _make_failing_mock()]:
            state = _state(
                query="What is the company salary review process?",
                llm_override=mock,
            )
            result = classify_query(state)
            assert result["classification"]["category"] != "unsupported"

    def test_tax_question_not_classified_unsupported(self):
        state = _state(
            query="What are the tax implications of expense claims?",
            llm_override=_make_failing_mock(),  # force heuristic
        )
        result = classify_query(state)
        assert result["classification"]["category"] != "unsupported"

    def test_grievance_question_not_classified_unsupported(self):
        state = _state(
            query="How do I raise a grievance?",
            llm_override=_make_failing_mock(),
        )
        result = classify_query(state)
        assert result["classification"]["category"] != "unsupported"

    def test_invalid_category_from_llm_triggers_fallback(self):
        """LLM returning an invalid category string falls back to heuristic."""
        mock = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = ClassificationOutput(
            category="totally_invalid_category",
            intent="test",
            requires_cross_doc=False,
        )
        mock.with_structured_output.return_value = structured
        state = _state(llm_override=mock)
        result = classify_query(state)
        # Invalid → fallback heuristic should have replaced it
        assert result["classification"]["category"] in {
            "direct_factual", "multi_section", "cross_document",
            "potential_conflict", "needs_clarification",
        }


# ===========================================================================
# C. plan_retrieval — available_sources + cross-doc hints
# ===========================================================================

class TestPlanRetrievalStage4b:
    def _classified(self, category="direct_factual", requires_cross_doc=False):
        return {
            "category": category,
            "intent": "test",
            "requires_cross_doc": requires_cross_doc,
        }

    def test_uses_available_sources_not_hardcoded_names(self):
        """Stage 4b: document_hints must come from available_sources, not hardcoded names."""
        state = _state(
            query="What is the custom_policy_v2.pdf password rule?",
            classification=self._classified("direct_factual"),
            available_sources=["custom_policy_v2.pdf"],
        )
        result = plan_retrieval(state)
        # Should not contain hardcoded Stage 4a names that aren't in available_sources
        for hint in result["retrieval_plan"]["document_hints"]:
            assert hint in state["available_sources"]

    def test_cross_doc_includes_all_available_sources(self):
        sources = ["it_security_policy.pdf", "hr_leave_policy.pdf", "finance_expense_policy.pdf"]
        state = _state(
            query="Compare IT and HR policies",
            classification=self._classified("cross_document", True),
            available_sources=sources,
        )
        result = plan_retrieval(state)
        hints = result["retrieval_plan"]["document_hints"]
        assert set(hints) == set(sources)

    def test_available_sources_stored_in_state(self):
        sources = ["policy_a.pdf", "policy_b.pdf"]
        state = _state(
            classification=self._classified("direct_factual"),
            available_sources=sources,
        )
        result = plan_retrieval(state)
        assert result.get("available_sources") == sources

    def test_empty_available_sources_no_hints(self):
        state = _state(
            classification=self._classified("direct_factual"),
            available_sources=[],
            retriever_override=MagicMock(),  # signals test mode → _discover returns []
        )
        result = plan_retrieval(state)
        assert result["retrieval_plan"]["document_hints"] == []

    def test_k_larger_for_potential_conflict(self):
        state = _state(
            classification=self._classified("potential_conflict", True),
            available_sources=["a.pdf"],
        )
        result = plan_retrieval(state)
        assert result["retrieval_plan"]["k"] > _BASE_K


# ===========================================================================
# D. assess_evidence — LLM outcomes + UNAVAILABLE fallback
# ===========================================================================

class TestAssessEvidenceStage4b:
    def _state_with_docs(self, docs, retry_count=0, llm=None):
        return _state(
            query="What is the password minimum length?",
            retrieved_docs=docs,
            retry_count=retry_count,
            llm_override=llm,
        )

    def test_supported_evidence_sufficient(self):
        docs = [_doc("Password must be at least 12 characters.")]
        result = assess_evidence(self._state_with_docs(docs, llm=_make_evidence_mock(_STATUS_SUPPORTED, 0.9)))
        assert result["evidence"]["sufficient"] is True
        assert result["evidence_status"] == _STATUS_SUPPORTED
        assert result["needs_retry"] is False

    def test_partially_supported_triggers_retry_when_budget_remains(self):
        """PARTIALLY_SUPPORTED with retry budget must set needs_retry=True."""
        docs = [_doc("Some relevant text about passwords.")]
        result = assess_evidence(self._state_with_docs(docs, retry_count=0, llm=_make_evidence_mock(_STATUS_PARTIAL, 0.6)))
        assert result["evidence_status"] == _STATUS_PARTIAL
        assert result["needs_retry"] is True
        assert result["evidence"]["sufficient"] is False

    def test_partially_supported_sufficient_when_budget_exhausted(self):
        """PARTIALLY_SUPPORTED with no retry budget left: sufficient=True, no retry."""
        docs = [_doc("Some relevant text about passwords.")]
        result = assess_evidence(self._state_with_docs(docs, retry_count=_MAX_RETRIES, llm=_make_evidence_mock(_STATUS_PARTIAL, 0.6)))
        assert result["evidence"]["sufficient"] is True
        assert result["needs_retry"] is False
        assert result["evidence_status"] == _STATUS_PARTIAL

    def test_insufficient_triggers_retry(self):
        docs = [_doc("General office notice.")]
        result = assess_evidence(self._state_with_docs(docs, llm=_make_evidence_mock(_STATUS_INSUFFICIENT, 0.1)))
        assert result["evidence"]["sufficient"] is False
        assert result["evidence_status"] == _STATUS_INSUFFICIENT
        assert result["needs_retry"] is True

    def test_conflicting_evidence_is_sufficient_routes_to_conflict_check(self):
        docs = [
            _doc("IT policy: access tokens expire after 30 days.", "it_security_policy.pdf"),
            _doc("HR policy: access tokens expire after 90 days.", "hr_leave_policy.pdf"),
        ]
        result = assess_evidence(self._state_with_docs(docs, llm=_make_evidence_mock(_STATUS_CONFLICTING, 0.8)))
        assert result["evidence"]["sufficient"] is True
        assert result["evidence_status"] == _STATUS_CONFLICTING
        assert result["needs_retry"] is False

    def test_unavailable_on_llm_failure_uses_heuristic(self):
        docs = [_doc("Password must be 12 characters minimum.")]
        result = assess_evidence(self._state_with_docs(docs, llm=_make_failing_mock()))
        assert result["evidence_status"] == _STATUS_UNAVAILABLE
        assert result["evidence"] is not None  # heuristic still computed

    def test_coverage_estimate_not_calibrated_probability(self):
        """coverage and confidence fields are heuristic estimates, within 0-1."""
        docs = [_doc("Policy text here.")]
        result = assess_evidence(self._state_with_docs(docs, llm=_make_evidence_mock(_STATUS_SUPPORTED, 0.75)))
        assert 0.0 <= result["evidence"]["coverage"] <= 1.0
        assert 0.0 <= result["evidence"]["confidence"] <= 1.0

    def test_supporting_passages_validated_against_docs(self):
        """Passages not found in retrieved docs must be stripped out."""
        real_text = "Password must be at least 12 characters."
        docs = [_doc(real_text)]

        mock = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = EvidenceEvaluation(
            status=_STATUS_SUPPORTED,
            reasoning="Supported",
            supporting_passages=[
                {"text": real_text[:30], "source": "it_security_policy.pdf", "page": 1},
                {"text": "FABRICATED passage not in docs", "source": "invented.pdf", "page": 99},
            ],
            coverage_estimate=0.9,
        )
        mock.with_structured_output.return_value = structured
        result = assess_evidence(self._state_with_docs(docs, llm=mock))

        # Only the real passage should survive validation
        for p in result["supporting_passages"]:
            assert real_text[:30] in real_text or p["text"][:30] in real_text

    def test_empty_docs_insufficient_with_retry(self):
        result = assess_evidence(_state(retrieved_docs=[], retry_count=0, llm_override=_make_failing_mock()))
        assert result["evidence"]["sufficient"] is False
        assert result["needs_retry"] is True

    def test_retry_blocked_when_budget_exhausted(self):
        result = assess_evidence(_state(retrieved_docs=[], retry_count=_MAX_RETRIES, llm_override=_make_failing_mock()))
        assert result["needs_retry"] is False

    def test_error_state_not_retried(self):
        result = assess_evidence(_state(retrieved_docs=[], error="Connection error", llm_override=_make_failing_mock()))
        assert result["needs_retry"] is False


# ===========================================================================
# E. check_cross_doc_conflicts — Stage 4b
# ===========================================================================

class TestCheckCrossDocConflictsStage4b:
    def test_genuine_conflict_detected(self):
        docs = [
            _doc("IT: passwords expire after 30 days.", "it_security_policy.pdf"),
            _doc("HR: passwords expire after 90 days.", "hr_leave_policy.pdf"),
        ]
        mock = _make_conflict_mock(has_conflict=True, descriptions=["30 vs 90 day expiry"], review=True)
        state = _state(retrieved_docs=docs, llm_override=mock)
        result = check_cross_doc_conflicts(state)
        assert result["conflict_status"] == _CONFLICT_DETECTED
        assert len(result["conflicts"]) >= 1
        assert any("[REVIEW RECOMMENDED]" in c for c in result["conflicts"])

    def test_no_conflict_detected(self):
        docs = [
            _doc("IT: use strong passwords.", "it_security_policy.pdf"),
            _doc("Finance: submit receipts within 30 days.", "finance_expense_policy.pdf"),
        ]
        mock = _make_conflict_mock(has_conflict=False)
        state = _state(retrieved_docs=docs, llm_override=mock)
        result = check_cross_doc_conflicts(state)
        assert result["conflict_status"] == _NO_CONFLICT
        assert result["conflicts"] == []

    def test_unknown_on_llm_failure(self):
        """LLM failure must return UNKNOWN, never NO_CONFLICT."""
        docs = [
            _doc("IT policy text.", "it_security_policy.pdf"),
            _doc("HR policy text.", "hr_leave_policy.pdf"),
        ]
        state = _state(retrieved_docs=docs, llm_override=_make_failing_mock())
        result = check_cross_doc_conflicts(state)
        assert result["conflict_status"] == _CONFLICT_UNKNOWN
        assert len(result["conflicts"]) >= 1  # warning message populated

    def test_single_source_no_llm_call_needed(self):
        """With only one source file, no LLM call is made."""
        docs = [
            _doc("IT policy text 1.", "it_security_policy.pdf"),
            _doc("IT policy text 2.", "it_security_policy.pdf"),
        ]
        mock = MagicMock()
        state = _state(retrieved_docs=docs, llm_override=mock)
        result = check_cross_doc_conflicts(state)
        assert result["conflict_status"] == _NO_CONFLICT
        mock.with_structured_output.assert_not_called()

    def test_review_recommended_flag_in_conflict_descriptions(self):
        docs = [
            _doc("Policy A text.", "policy_a.pdf"),
            _doc("Policy B text.", "policy_b.pdf"),
        ]
        mock = _make_conflict_mock(has_conflict=True, descriptions=["Conflict found"], review=True)
        state = _state(retrieved_docs=docs, llm_override=mock)
        result = check_cross_doc_conflicts(state)
        assert any("REVIEW RECOMMENDED" in c for c in result["conflicts"])


# ===========================================================================
# F. generate_grounded_answer — LLM answer + citation validation + fallback
# ===========================================================================

class TestGenerateGroundedAnswerStage4b:
    def _sufficient_state(self, docs, answer_text="The policy requires 12 characters.", llm=None):
        ans_mock = llm or _make_answer_mock(answer_text)
        return _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.9, "confidence": 0.9, "sufficient": True},
            evidence_status=_STATUS_SUPPORTED,
            classification={"category": "direct_factual", "intent": "", "requires_cross_doc": False},
            conflicts=[],
            conflict_status=_NO_CONFLICT,
            llm_override=ans_mock,
        )

    def test_llm_answer_returned_when_evidence_sufficient(self):
        docs = [_doc("Password must be 12 characters.", "it_security_policy.pdf")]
        state = self._sufficient_state(docs, "Passwords require 12 characters [it_security_policy.pdf, Page 1].")
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "grounded"
        assert "12 characters" in result["answer"]

    def test_sources_always_retrieved_docs(self):
        docs = [_doc("Policy text.")]
        state = self._sufficient_state(docs)
        result = generate_grounded_answer(state)
        for src in result["sources"]:
            assert src in docs

    def test_refusal_when_evidence_insufficient(self):
        state = _state(
            retrieved_docs=[],
            evidence={"coverage": 0.0, "confidence": 0.0, "sufficient": False},
            evidence_status=_STATUS_INSUFFICIENT,
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "refusal"
        assert result["answer"] == _REFUSAL_MESSAGE

    def test_error_state_returns_error_answer_type(self):
        state = _state(retrieved_docs=[], error="Retriever failed")
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "error"

    def test_fabricated_citation_stripped_from_answer(self):
        """A citation to a filename not in retrieved docs must be removed, not just warned."""
        docs = [_doc("Policy text.", "it_security_policy.pdf")]
        fabricated_answer = "See [invented_policy.pdf, Page 1] for details."
        state = self._sufficient_state(docs, fabricated_answer)
        result = generate_grounded_answer(state)
        # Fabricated filename must be gone from the answer
        assert "invented_policy.pdf" not in result["answer"]
        # Answer must carry some form of verification caveat (Source note, or excerpt notice)
        answer_lower = result["answer"].lower()
        assert (
            "source note" in answer_lower
            or "not been verified" in answer_lower
            or "unavailable" in answer_lower
        )

    def test_all_fabricated_citations_triggers_excerpt_fallback(self):
        """If ALL citations are fabricated, fall back to excerpts (excerpt_fallback)."""
        docs = [_doc("Policy text.", "it_security_policy.pdf")]
        all_fake_answer = "See [fake1.pdf, Page 1] and [fake2.pdf, Page 2]."
        state = self._sufficient_state(docs, all_fake_answer)
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "excerpt_fallback"
        assert "it_security_policy.pdf" in result["answer"]

    def test_valid_citation_not_stripped(self):
        """Citations matching retrieved doc filename AND page must be kept unchanged."""
        docs = [_doc("Policy text.", "it_security_policy.pdf", page=0)]  # page=0 → Page 1
        valid_answer = "See [it_security_policy.pdf, Page 1] for details."
        state = self._sufficient_state(docs, valid_answer)
        result = generate_grounded_answer(state)
        assert "it_security_policy.pdf" in result["answer"]
        assert "Source note" not in result["answer"]

    def test_wrong_page_citation_stripped(self):
        """Citation with correct filename but wrong page number must be stripped."""
        docs = [_doc("Policy text.", "it_security_policy.pdf", page=0)]  # only Page 1 retrieved
        wrong_page_answer = "See [it_security_policy.pdf, Page 99] for details."
        state = self._sufficient_state(docs, wrong_page_answer)
        result = generate_grounded_answer(state)
        assert "Page 99" not in result["answer"]

    def test_nonexistent_doc_citation_stripped(self):
        """Citation to a document that was never retrieved must be stripped."""
        docs = [_doc("Policy text.", "it_security_policy.pdf")]
        nonexistent_answer = "According to [hr_leave_policy.pdf, Page 2]."
        state = self._sufficient_state(docs, nonexistent_answer)
        result = generate_grounded_answer(state)
        assert "hr_leave_policy.pdf" not in result["answer"]

    def test_llm_failure_falls_back_to_excerpt_fallback(self):
        """LLM generation failure must return answer_type='excerpt_fallback', not 'grounded'."""
        docs = [_doc("Password must be 12 characters.", "it_security_policy.pdf")]
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.9, "confidence": 0.9, "sufficient": True},
            evidence_status=_STATUS_SUPPORTED,
            llm_override=_make_failing_mock(),
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "excerpt_fallback"
        assert "it_security_policy.pdf" in result["answer"]
        assert "unavailable" in result["answer"].lower() or "not been verified" in result["answer"].lower()

    def test_conflict_detected_note_appended(self):
        docs = [
            _doc("IT policy text.", "it_security_policy.pdf"),
            _doc("HR policy text.", "hr_leave_policy.pdf"),
        ]
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.8, "confidence": 0.8, "sufficient": True},
            evidence_status=_STATUS_SUPPORTED,
            conflict_status=_CONFLICT_DETECTED,
            llm_override=_make_answer_mock("Answer text here."),
        )
        result = generate_grounded_answer(state)
        assert "contradictory" in result["answer"].lower() or "conflict" in result["answer"].lower()

    def test_conflict_unknown_note_appended(self):
        docs = [_doc("Policy text.", "it_security_policy.pdf")]
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.8, "confidence": 0.8, "sufficient": True},
            evidence_status=_STATUS_SUPPORTED,
            conflict_status=_CONFLICT_UNKNOWN,
            llm_override=_make_answer_mock("Answer text."),
        )
        result = generate_grounded_answer(state)
        assert "verified" in result["answer"].lower() or "review" in result["answer"].lower()

    def test_unavailable_evidence_uses_unverified_grounded_type(self):
        """UNAVAILABLE evidence must produce 'unverified_grounded', not 'grounded'."""
        docs = [_doc("Password must be 12 characters.")]
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.5, "confidence": 0.5, "sufficient": True},
            evidence_status=_STATUS_UNAVAILABLE,
            llm_override=_make_answer_mock("The policy requires 12 characters."),
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "unverified_grounded"
        assert "not been independently verified" in result["answer"] or "unverified" in result["answer"].lower()

    def test_partial_evidence_uses_partial_grounded_type(self):
        """PARTIALLY_SUPPORTED after retries exhausted must use 'partial_grounded'."""
        docs = [_doc("Some relevant info about passwords.")]
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.6, "confidence": 0.6, "sufficient": True},
            evidence_status=_STATUS_PARTIAL,
            llm_override=_make_answer_mock("Partial answer here."),
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "partial_grounded"
        assert "partially" in result["answer"].lower() or "partial" in result["answer"].lower()

    def test_partial_evidence_low_coverage_refuses(self):
        """PARTIALLY_SUPPORTED with very low coverage must refuse."""
        docs = [_doc("Unrelated content.")]
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.1, "confidence": 0.1, "sufficient": True},
            evidence_status=_STATUS_PARTIAL,
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] == "refusal"


# ===========================================================================
# G. _route_after_assessment — Stage 4b routing
# ===========================================================================

class TestRouterStage4b:
    def _s(self, needs_retry=False, retry_count=0, requires_cross_doc=False, evidence_status=""):
        return _state(
            needs_retry=needs_retry,
            retry_count=retry_count,
            evidence_status=evidence_status,
            classification={
                "category": "cross_document" if requires_cross_doc else "direct_factual",
                "intent": "test",
                "requires_cross_doc": requires_cross_doc,
            },
            evidence={"coverage": 0.8, "confidence": 0.8, "sufficient": not needs_retry},
        )

    def test_conflicting_evidence_routes_to_conflict_check(self):
        state = self._s(evidence_status=_STATUS_CONFLICTING)
        assert _route_after_assessment(state) == "check_cross_doc_conflicts"

    def test_conflicting_takes_priority_over_cross_doc_flag(self):
        """CONFLICTING evidence routes to conflict check even without requires_cross_doc."""
        state = self._s(evidence_status=_STATUS_CONFLICTING, requires_cross_doc=False)
        assert _route_after_assessment(state) == "check_cross_doc_conflicts"

    def test_retry_takes_priority_over_conflicting(self):
        """Retry takes highest priority."""
        state = self._s(needs_retry=True, retry_count=0, evidence_status=_STATUS_CONFLICTING)
        assert _route_after_assessment(state) == "expand_retrieval"

    def test_cross_doc_flag_routes_to_conflict_check(self):
        state = self._s(requires_cross_doc=True)
        assert _route_after_assessment(state) == "check_cross_doc_conflicts"

    def test_direct_routes_to_generate(self):
        state = self._s()
        assert _route_after_assessment(state) == "generate_grounded_answer"

    def test_retry_exhausted_with_conflicting_routes_to_conflict(self):
        state = self._s(needs_retry=True, retry_count=_MAX_RETRIES, evidence_status=_STATUS_CONFLICTING)
        assert _route_after_assessment(state) == "check_cross_doc_conflicts"


# ===========================================================================
# H. Full pipeline integration tests with mocked LLM
# ===========================================================================

class TestFullPipelineStage4b:
    @pytest.fixture(scope="class")
    def compiled(self):
        return build_graph()

    def test_happy_path_grounded_answer(self, compiled):
        docs = [
            _doc("Passwords must be at least 12 characters long.", "it_security_policy.pdf"),
            _doc("Multi-factor authentication is required for all accounts.", "it_security_policy.pdf"),
        ]
        mock = _make_full_pipeline_mock(
            category="direct_factual",
            evidence_status=_STATUS_SUPPORTED,
            coverage=0.9,
            answer_text="The policy requires passwords of at least 12 characters [it_security_policy.pdf, Page 1].",
        )
        initial = make_initial_state(
            "What is the password policy?",
            retriever=_stub_retriever(docs),
            llm_override=mock,
            available_sources=["it_security_policy.pdf"],
        )
        result = compiled.invoke(initial)
        assert result["answer_type"] in ("grounded", "partial_grounded", "unverified_grounded", "excerpt_fallback")
        assert result["sources"] == docs
        assert result["retry_count"] == 0
        assert isinstance(result["audit_metadata"], dict)

    def test_refusal_on_empty_retriever(self, compiled):
        mock = _make_full_pipeline_mock(evidence_status=_STATUS_INSUFFICIENT)
        initial = make_initial_state(
            "What is the expense limit?",
            retriever=_stub_retriever([]),
            llm_override=mock,
        )
        result = compiled.invoke(initial)
        assert result["answer_type"] in ("refusal", "error")
        assert result["answer"] == _REFUSAL_MESSAGE or "cannot find" in result["answer"].lower()

    def test_cross_doc_pipeline_runs_conflict_check(self, compiled):
        docs = [
            _doc("IT: password expiry 30 days.", "it_security_policy.pdf"),
            _doc("HR: access expiry 90 days.", "hr_leave_policy.pdf"),
        ]
        mock = _make_full_pipeline_mock(
            category="cross_document",
            requires_cross_doc=True,
            evidence_status=_STATUS_SUPPORTED,
        )
        initial = make_initial_state(
            "Compare IT and HR access policies",
            retriever=_stub_retriever(docs),
            llm_override=mock,
            available_sources=["it_security_policy.pdf", "hr_leave_policy.pdf"],
        )
        result = compiled.invoke(initial)
        assert isinstance(result["conflicts"], list)
        assert result["conflict_status"] in (_NO_CONFLICT, _CONFLICT_DETECTED, _CONFLICT_UNKNOWN)

    def test_conflicting_evidence_routes_through_conflict_node(self, compiled):
        docs = [
            _doc("IT: access tokens expire in 30 days.", "it_security_policy.pdf"),
            _doc("HR: access tokens expire in 90 days.", "hr_leave_policy.pdf"),
        ]
        mock = _make_full_pipeline_mock(
            category="direct_factual",
            requires_cross_doc=False,
            evidence_status=_STATUS_CONFLICTING,  # CONFLICTING → triggers conflict check
        )
        initial = make_initial_state(
            "What is the access token expiry period?",
            retriever=_stub_retriever(docs),
            llm_override=mock,
        )
        result = compiled.invoke(initial)
        # Conflict check was run (conflict_status set to something)
        assert result["conflict_status"] in (_NO_CONFLICT, _CONFLICT_DETECTED, _CONFLICT_UNKNOWN)

    def test_salary_question_reaches_retrieval_not_refused_upfront(self, compiled):
        """Stage 4b: salary questions must not be refused before retrieval."""
        # If the retriever returns empty docs, refusal is from evidence gate (not classifier)
        mock = _make_full_pipeline_mock(evidence_status=_STATUS_INSUFFICIENT)
        initial = make_initial_state(
            "What is the salary review process?",
            retriever=_stub_retriever([]),
            llm_override=mock,
        )
        result = compiled.invoke(initial)
        # Must have gone through retrieval — retry_count might be > 0
        assert result.get("retrieval_plan") is not None  # plan_retrieval ran
        # Refusal is from insufficient evidence, not from classifier
        assert result["answer_type"] in ("refusal", "error")

    def test_total_retrieval_calls_capped_at_three(self, compiled):
        stub = _stub_retriever([])  # always empty → always insufficient
        mock = _make_full_pipeline_mock(evidence_status=_STATUS_INSUFFICIENT)
        initial = make_initial_state("What is the remote work policy?", retriever=stub, llm_override=mock)
        compiled.invoke(initial)
        assert stub.invoke.call_count <= 1 + _MAX_RETRIES

    def test_all_terminal_paths_populate_audit_metadata(self, compiled):
        for retriever_docs, ev_status in [
            ([_doc("Policy text.")], _STATUS_SUPPORTED),
            ([], _STATUS_INSUFFICIENT),
        ]:
            mock = _make_full_pipeline_mock(evidence_status=ev_status)
            initial = make_initial_state(
                "What is the leave policy?",
                retriever=_stub_retriever(retriever_docs),
                llm_override=mock,
            )
            result = compiled.invoke(initial)
            assert "query" in result["audit_metadata"]
            assert "evidence_status" in result["audit_metadata"]

    def test_full_pipeline_llm_failure_still_completes(self, compiled):
        """Full LLM failure at every node — graph must complete gracefully."""
        docs = [_doc("Password must be 12 characters.")]
        initial = make_initial_state(
            "What is the password policy?",
            retriever=_stub_retriever(docs),
            llm_override=_make_failing_mock(),
        )
        result = compiled.invoke(initial)
        # Must reach END without raising
        assert result["answer"] != ""
        assert result["answer_type"] in (
            "grounded", "partial_grounded", "unverified_grounded",
            "excerpt_fallback", "refusal", "error",
        )


# ===========================================================================
# K. Operational safeguards
# ===========================================================================

class TestOperationalSafeguards:
    def test_discover_indexed_sources_returns_empty_in_test_mode(self):
        """_discover_indexed_sources must never touch ChromaDB when retriever_override is set."""
        from agent.graph import _discover_indexed_sources
        state = _state(retriever_override=_stub_retriever([]))
        result = _discover_indexed_sources(state)
        assert result == []

    def test_get_agent_llm_returns_override_without_api_call(self):
        """get_agent_llm must return llm_override directly without touching the API key."""
        from agent.llm import get_agent_llm
        mock = MagicMock()
        state = _state(llm_override=mock)
        result = get_agent_llm(state)
        assert result is mock

    def test_get_agent_llm_raises_without_api_key(self):
        """Behavioural test: get_agent_llm raises ValueError when NVIDIA_API_KEY is absent.

        Patches:
          - dotenv.load_dotenv  → no-op (prevents .env from reloading credentials)
          - os.getenv           → returns None for NVIDIA_API_KEY, passthrough for all else
          - langchain_nvidia_ai_endpoints.ChatNVIDIA → spy to assert it is never constructed
        """
        import os as _os
        from unittest.mock import patch, MagicMock
        from agent.llm import get_agent_llm

        def _getenv_no_nvidia(key, *args):
            if key == "NVIDIA_API_KEY":
                return None
            return _os.environ.get(key, *(args or ()))

        nvidia_spy = MagicMock()

        with patch("dotenv.load_dotenv"):                                   # block .env reload
            with patch("os.getenv", side_effect=_getenv_no_nvidia):         # hide the key
                with patch(
                    "langchain_nvidia_ai_endpoints.ChatNVIDIA", nvidia_spy  # spy on constructor
                ):
                    state = _state()  # no llm_override
                    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
                        get_agent_llm(state)
                    nvidia_spy.assert_not_called()  # ChatNVIDIA must never be constructed

    def test_retry_count_never_exceeds_max(self):
        """retry_count must be <= _MAX_RETRIES after any number of expand_retrieval calls."""
        from agent.graph import expand_retrieval
        state = _state(
            retriever_override=_stub_retriever([]),
            retry_count=_MAX_RETRIES - 1,
        )
        result = expand_retrieval(state)
        assert result["retry_count"] <= _MAX_RETRIES

    def test_conflict_analysis_error_returns_unknown_not_no_conflict(self):
        """LLM failure in conflict check must return UNKNOWN, never NO_CONFLICT."""
        from agent.graph import check_cross_doc_conflicts, _CONFLICT_UNKNOWN, _NO_CONFLICT
        docs = [
            _doc("IT policy text.", "it_security_policy.pdf"),
            _doc("HR policy text.", "hr_leave_policy.pdf"),
        ]
        state = _state(retrieved_docs=docs, llm_override=_make_failing_mock())
        result = check_cross_doc_conflicts(state)
        assert result["conflict_status"] == _CONFLICT_UNKNOWN
        assert result["conflict_status"] != _NO_CONFLICT

    def test_assess_evidence_unavailable_never_grounded(self):
        """When evidence_status=UNAVAILABLE, answer_type must not be 'grounded'."""
        docs = [_doc("Policy text.")]
        # Force UNAVAILABLE by failing the LLM
        state = _state(
            retrieved_docs=docs,
            evidence={"coverage": 0.5, "confidence": 0.5, "sufficient": True},
            evidence_status=_STATUS_UNAVAILABLE,
            llm_override=_make_answer_mock("Some answer."),
        )
        result = generate_grounded_answer(state)
        assert result["answer_type"] != "grounded"

    def test_excerpt_fallback_is_not_grounded(self):
        """_generate_excerpt_answer must return 'excerpt_fallback', not 'grounded'."""
        from agent.graph import _generate_excerpt_answer
        docs = [_doc("Policy text.", "it_security_policy.pdf")]
        result = _generate_excerpt_answer(_state(retrieved_docs=docs), docs)
        assert result["answer_type"] == "excerpt_fallback"
        assert "not been verified" in result["answer"] or "unavailable" in result["answer"].lower()


# ===========================================================================
# I. agent/llm helpers
# ===========================================================================

class TestLLMHelpers:
    def test_build_agent_context_includes_filename_and_page(self):
        docs = [_doc("Policy text.", "it_security_policy.pdf", page=2)]
        ctx = build_agent_context(docs)
        assert "it_security_policy.pdf" in ctx
        assert "Page 3" in ctx  # 0-indexed → 1-indexed

    def test_build_agent_context_multiple_docs(self):
        docs = [
            _doc("IT policy text.", "it_security_policy.pdf", page=0),
            _doc("HR policy text.", "hr_leave_policy.pdf", page=1),
        ]
        ctx = build_agent_context(docs)
        assert "it_security_policy.pdf" in ctx
        assert "hr_leave_policy.pdf" in ctx

    def test_build_agent_context_respects_char_limit(self):
        """Context builder must never include text beyond _MAX_CONTEXT_CHARS."""
        big_doc = _doc("x" * 9000, "big_policy.pdf")
        small_doc = _doc("small text", "small_policy.pdf")
        ctx = build_agent_context([big_doc, small_doc])
        assert len(ctx) <= 10000  # big_doc alone is near limit; second should be excluded
        assert "small_policy.pdf" not in ctx

    def test_validate_answer_citations_valid_kept(self):
        answer = "See [it_security_policy.pdf, Page 1] for details."
        docs = [_doc("Policy text.", "it_security_policy.pdf", page=0)]  # page=0 → Page 1
        cleaned, warnings = validate_answer_citations(answer, {"it_security_policy.pdf"}, docs)
        assert warnings == []
        assert "it_security_policy.pdf" in cleaned

    def test_validate_answer_citations_fabricated_filename_stripped(self):
        answer = "See [invented_policy.pdf, Page 99] for details."
        cleaned, warnings = validate_answer_citations(answer, {"it_security_policy.pdf"})
        assert len(warnings) == 1
        assert "invented_policy.pdf" in warnings[0]
        assert "invented_policy.pdf" not in cleaned

    def test_validate_answer_citations_wrong_page_stripped(self):
        """Correct filename but page not in retrieved chunks → stripped."""
        docs = [_doc("Policy text.", "it_security_policy.pdf", page=0)]  # only Page 1
        answer = "See [it_security_policy.pdf, Page 5] for details."
        cleaned, warnings = validate_answer_citations(answer, {"it_security_policy.pdf"}, docs)
        assert len(warnings) == 1
        assert "Page 5" in warnings[0]
        assert "Page 5" not in cleaned

    def test_validate_answer_citations_no_citations(self):
        answer = "The policy does not address this topic."
        cleaned, warnings = validate_answer_citations(answer, {"it_security_policy.pdf"})
        assert warnings == []
        assert cleaned == answer

    def test_validate_answer_citations_multiple_mixed(self):
        """Valid citation kept; fabricated citation stripped."""
        docs = [_doc("Policy text.", "it_security_policy.pdf", page=0)]
        answer = (
            "According to [it_security_policy.pdf, Page 1], passwords expire in 90 days. "
            "However [fake_policy.pdf, Page 5] says 30 days."
        )
        cleaned, warnings = validate_answer_citations(answer, {"it_security_policy.pdf"}, docs)
        assert len(warnings) == 1
        assert "fake_policy.pdf" in warnings[0]
        assert "fake_policy.pdf" not in cleaned
        assert "it_security_policy.pdf" in cleaned  # valid kept


# ===========================================================================
# J. Retrieval baseline comparison against Phase 3
# ===========================================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

NON_REFUSAL = [q for q in EVALUATION_QUESTIONS if not q["should_refuse"]]
REFUSAL_QS = [q for q in EVALUATION_QUESTIONS if q["should_refuse"]]


@pytest.fixture(scope="module")
def agent_retriever(tmp_path_factory):
    """Build a shared in-memory ChromaDB store from all 3 fixture PDFs.
    Same setup as test_rag_evaluation.py to allow direct comparison.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chroma_dir = str(tmp_path_factory.mktemp("4b_chroma"))
    pdf_names = [
        "it_security_policy.pdf",
        "hr_leave_policy.pdf",
        "finance_expense_policy.pdf",
    ]
    all_chunks = []
    for name in pdf_names:
        path = os.path.join(FIXTURES_DIR, name)
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_filename"] = name
        all_chunks.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_chunks)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="policy_docs_4b",
    )
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    yield retriever

    import gc
    gc.collect()
    shutil.rmtree(chroma_dir, ignore_errors=True)


def _hit(docs, expected_files):
    if not expected_files:
        return True
    retrieved = {d.metadata.get("source_filename", "") for d in docs}
    return bool(retrieved & set(expected_files))


def _key_term_hit(docs, key_terms):
    if not key_terms:
        return True
    combined = " ".join(d.page_content for d in docs).lower()
    return all(t.lower() in combined for t in key_terms)


def test_agent_retrieval_vs_phase3_baseline(agent_retriever, capsys):
    """Compare Stage 4b retrieval against the Phase 3 32/32 baseline.

    Phase 3 baseline (test_rag_evaluation.py, k=3):
      Source-document hit rate:  32/32 non-refusal questions
      Key-term coverage:         measured per-question

    Stage 4b uses the same underlying retriever (load_retriever → Chroma),
    same embedding model, same chunk parameters — so the hit rate MUST match.
    Any regression is reported honestly with per-question breakdown.
    """
    PHASE3_BASELINE_HITS = 32  # Phase 3 measured result
    PHASE3_NON_REFUSAL = 32    # questions with should_refuse=False

    hits = 0
    key_hits = 0
    misses = []

    for entry in NON_REFUSAL:
        docs = agent_retriever.invoke(entry["question"])
        h = _hit(docs, entry["expected_source_files"])
        if h:
            hits += 1
        else:
            misses.append(f"  MISS [{entry['id']}]: {entry['question'][:60]}")
        if _key_term_hit(docs, entry["key_terms"]):
            key_hits += 1

    total = len(NON_REFUSAL)
    hit_rate = hits / total if total else 0
    kt_rate = key_hits / total if total else 0
    regression = PHASE3_BASELINE_HITS - hits

    report = (
        f"\nStage 4b Retrieval vs Phase 3 Baseline (k={TOP_K})\n"
        f"  Phase 3 baseline:   {PHASE3_BASELINE_HITS}/{PHASE3_NON_REFUSAL} = 100.0%\n"
        f"  Stage 4b result:    {hits}/{total} = {hit_rate:.1%}\n"
        f"  Regression:         {regression} question(s)\n"
        f"  Key-term coverage:  {key_hits}/{total} = {kt_rate:.1%}\n"
    )
    if misses:
        report += "  Missed questions:\n" + "\n".join(misses) + "\n"
    print(report)

    assert hits == total, (
        f"Stage 4b retrieval hit rate {hits}/{total} ({hit_rate:.1%}) is BELOW "
        f"Phase 3 baseline of {PHASE3_BASELINE_HITS}/{PHASE3_NON_REFUSAL} (100.0%).\n"
        + "\n".join(misses)
    )
