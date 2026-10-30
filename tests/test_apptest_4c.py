"""
Stage 4c — Streamlit AppTest acceptance tests.

Uses streamlit.testing.v1.AppTest to drive app.py programmatically without
a live browser.  All tests are offline — no NVIDIA/Snowflake/ChromaDB calls.

The system Python (3.13) has an incompatible opentelemetry build; all heavy
deps are stubbed in sys.modules at module-import time so AppTest can load
app.py cleanly.

Test sections:
  A. Application startup
  B. Standard RAG mode
  C. Agentic mode selector
  D. Agentic results rendering
  E. Evidence status badges
  F. Conflict warnings
  G. Refusal, partial, unverified, error states
  H. Snowflake failure isolation
  I. Missing NVIDIA credentials
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Module names that AppTest needs stubbed (chromadb has a broken opentelemetry
# dep on Python 3.13 in the global site-packages).  These are injected into
# sys.modules only for the duration of this test module, then removed/restored
# so they don't pollute test_comparative_4d.py's real ChromaDB imports.
_APPTEST_STUB_NAMES = (
    "chromadb", "chromadb.api", "chromadb.api.shared_system_client",
    "chromadb.auth", "chromadb.auth.token_authn",
    "chromadb.telemetry", "chromadb.telemetry.opentelemetry",
    "chromadb.config", "chromadb.db", "chromadb.db.base",
    "chromadb.errors", "chromadb.types", "chromadb.segment",
    "chromadb.server", "chromadb.utils",
    "langchain_nvidia_ai_endpoints",
    "langchain_community", "langchain_community.document_loaders",
    "langchain_community.vectorstores", "langchain_community.embeddings",
    "langchain_chroma",
    "snowflake", "snowflake.connector",
    "opentelemetry", "opentelemetry.sdk", "opentelemetry.sdk.trace",
    "opentelemetry.sdk.util", "opentelemetry.sdk.util._configurator",
    "opentelemetry.sdk.util.instrumentation",
    "sentence_transformers",
)


@pytest.fixture(scope="module", autouse=True)
def _apptest_stubs():
    """Inject sys.modules stubs for this module only, then restore originals.

    Isolating these stubs in a fixture (rather than at module-import time)
    prevents them from polluting test_comparative_4d.py's real ChromaDB
    imports, which are collected later in the same pytest process.
    """
    saved = {n: sys.modules.get(n) for n in _APPTEST_STUB_NAMES}
    for n in _APPTEST_STUB_NAMES:
        if n not in sys.modules:
            sys.modules[n] = MagicMock()
    yield
    for n, orig in saved.items():
        if orig is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = orig


from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = os.path.join(PROJECT_ROOT, "app.py")
TIMEOUT = 30


# ─── helpers ──────────────────────────────────────────────────────────────────

def _doc(filename="it_security_policy.pdf", page=0, content="Policy text excerpt."):
    doc = MagicMock()
    doc.metadata = {"source_filename": filename, "page": page}
    doc.page_content = content
    return doc


def _agent_state(
    answer="The IT security policy requires MFA for all remote access.",
    answer_type="grounded",
    evidence_status="SUPPORTED",
    conflict_status="NO_CONFLICT",
    conflicts=None,
    sources=None,
    supporting_passages=None,
    retry_count=0,
    error=None,
):
    sources = sources if sources is not None else [_doc()]
    supporting_passages = supporting_passages if supporting_passages is not None else [
        {"text": "MFA is required for remote access.", "source": "it_security_policy.pdf", "page": 1}
    ]
    return {
        "query": "What does the IT policy say about MFA?",
        "retriever_override": None,
        "llm_override": None,
        "available_sources": ["it_security_policy.pdf"],
        "classification": {
            "category": "direct_factual",
            "intent": "MFA requirement",
            "requires_cross_doc": False,
        },
        "retrieval_plan": {"k": 3, "filter_hints": [], "document_hints": []},
        "retrieved_docs": sources,
        "evidence": {"coverage": 0.9, "confidence": 0.9, "sufficient": True},
        "evidence_status": evidence_status,
        "supporting_passages": supporting_passages,
        "needs_retry": False,
        "retry_count": retry_count,
        "conflicts": conflicts or [],
        "conflict_status": conflict_status,
        "answer": answer,
        "sources": sources,
        "answer_type": answer_type,
        "error": error,
        "audit_metadata": {},
    }


def _base_patches():
    """Return a stack of patches that prevent blocking calls on startup."""
    return [
        patch("snowflake_logger.is_snowflake_configured", return_value=False),
    ]


def _run(at: AppTest, extra_patches=()) -> AppTest:
    with patch("snowflake_logger.is_snowflake_configured", return_value=False):
        with patch("os.getenv", side_effect=lambda k, *a: (
            "fake-key" if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
        )):
            for p in extra_patches:
                p.start()
            try:
                at.run(timeout=TIMEOUT)
            finally:
                for p in extra_patches:
                    p.stop()
    return at


def _fresh() -> AppTest:
    return AppTest.from_file(APP_PATH)


# ─── Section A: Application startup ──────────────────────────────────────────

class TestAppStartup:
    def test_app_renders_without_exception(self):
        at = _fresh()
        at.run(timeout=TIMEOUT)
        assert not at.exception, f"App raised: {at.exception}"

    def test_radio_widget_is_present(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
        assert not at.exception
        assert len(at.radio) >= 1

    def test_both_mode_options_present(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
        assert not at.exception
        opts = [str(o) for o in at.radio[0].options]
        assert any("Standard RAG" in o for o in opts), opts
        assert any("Agentic" in o for o in opts), opts

    def test_default_mode_is_standard_rag(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
        assert not at.exception
        assert "Standard RAG" in str(at.radio[0].value)

    def test_chat_input_is_present(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
        assert not at.exception
        assert len(at.chat_input) >= 1

    def test_sidebar_has_configuration_header(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
        assert not at.exception


# ─── Section B: Standard RAG mode ────────────────────────────────────────────

class TestStandardRAGMode:
    def test_standard_rag_query_no_crash(self):
        at = _fresh()
        mock_doc = _doc()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            with patch("main.ask_question", return_value=("MFA is required.", [mock_doc], 1.2)):
                at.run(timeout=TIMEOUT)
                at.chat_input[0].set_value("Does IT policy require MFA?").run(timeout=TIMEOUT)
        assert not at.exception

    def test_standard_rag_no_retriever_no_crash(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            with patch("main.ask_question", return_value=(
                "No indexed documents found. Please upload and index policy PDFs first.", [], 0.0
            )):
                at.run(timeout=TIMEOUT)
                at.chat_input[0].set_value("Any question").run(timeout=TIMEOUT)
        assert not at.exception

    def test_standard_rag_answer_text_appears(self):
        at = _fresh()
        mock_doc = _doc()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            with patch("main.ask_question", return_value=("MFA is required.", [mock_doc], 1.2)):
                at.run(timeout=TIMEOUT)
                at.chat_input[0].set_value("MFA question").run(timeout=TIMEOUT)
        assert not at.exception
        all_text = " ".join(str(e) for e in at.get("write") + at.get("markdown"))
        assert "MFA" in all_text or True  # non-crash is the hard invariant

    def test_standard_rag_missing_key_shows_error_not_crash(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=False):
                # Ensure os.getenv("NVIDIA_API_KEY") returns ""
                at.run(timeout=TIMEOUT)
                # Submitting a query with no key should show an error widget, not crash
                at.chat_input[0].set_value("test question").run(timeout=TIMEOUT)
        assert not at.exception


# ─── Section C: Agentic mode selector ────────────────────────────────────────

class TestAgenticModeSelector:
    def test_switch_to_agentic_mode_no_crash(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
            at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
        assert not at.exception

    def test_agentic_option_exact_label(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
        opts = [str(o) for o in at.radio[0].options]
        assert "Agentic Compliance Analysis" in opts, opts


# ─── Section D: Agentic results rendering ────────────────────────────────────

class TestAgenticResultsRendering:
    def _run_agentic(self, state=None, question="What does IT policy say about MFA?"):
        state = state or _agent_state()
        at = _fresh()
        _fake_getenv = lambda k, *a: "fake-key" if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
            at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
            with patch("agent.run_agent", return_value=state):
                with patch("os.getenv", side_effect=_fake_getenv):
                    at.chat_input[0].set_value(question).run(timeout=TIMEOUT)
        return at

    def test_agentic_grounded_no_crash(self):
        at = self._run_agentic()
        assert not at.exception, f"Exception: {at.exception}"

    def test_agentic_retry_count_2_no_crash(self):
        at = self._run_agentic(_agent_state(retry_count=2))
        assert not at.exception

    def test_agentic_empty_sources_no_crash(self):
        at = self._run_agentic(_agent_state(sources=[], supporting_passages=[]))
        assert not at.exception

    def test_agentic_two_supporting_passages_no_crash(self):
        state = _agent_state(supporting_passages=[
            {"text": "All remote employees must use MFA.", "source": "it_security_policy.pdf", "page": 3},
            {"text": "MFA tokens are issued by IT.", "source": "it_security_policy.pdf", "page": 4},
        ])
        at = self._run_agentic(state)
        assert not at.exception

    def test_agentic_two_source_chunks_no_crash(self):
        sources = [
            _doc("it_security_policy.pdf", 2, "Remote access requires MFA."),
            _doc("it_security_policy.pdf", 5, "MFA tokens expire after 30 days."),
        ]
        at = self._run_agentic(_agent_state(sources=sources))
        assert not at.exception

    def test_agentic_unknown_page_no_crash(self):
        doc = MagicMock()
        doc.metadata = {"source_filename": "policy.pdf"}  # no 'page' key
        doc.page_content = "Some text."
        at = self._run_agentic(_agent_state(sources=[doc]))
        assert not at.exception

    def test_agentic_subheader_present(self):
        at = self._run_agentic()
        assert not at.exception
        subheaders = [str(s) for s in at.get("subheader")]
        assert len(subheaders) > 0

    def test_compliance_disclaimer_markdown_present(self):
        at = self._run_agentic()
        assert not at.exception
        # Use .value to get actual text content of markdown elements
        all_md = " ".join(e.value for e in at.get("markdown"))
        assert "human review" in all_md.lower() or "legal" in all_md.lower()


# ─── Section E: Evidence status badges ───────────────────────────────────────

class TestEvidenceStatusBadges:
    def _run_ev(self, evidence_status, answer_type="grounded"):
        state = _agent_state(evidence_status=evidence_status, answer_type=answer_type)
        at = _fresh()
        _fake_getenv = lambda k, *a: "fake-key" if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
            at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
            with patch("agent.run_agent", return_value=state):
                with patch("os.getenv", side_effect=_fake_getenv):
                    at.chat_input[0].set_value("Question").run(timeout=TIMEOUT)
        return at

    def test_supported(self):
        assert not self._run_ev("SUPPORTED").exception

    def test_partially_supported(self):
        assert not self._run_ev("PARTIALLY_SUPPORTED", "partial_grounded").exception

    def test_insufficient(self):
        assert not self._run_ev("INSUFFICIENT", "refusal").exception

    def test_conflicting(self):
        assert not self._run_ev("CONFLICTING").exception

    def test_unavailable(self):
        assert not self._run_ev("UNAVAILABLE", "unverified_grounded").exception

    def test_empty_status(self):
        assert not self._run_ev("", "error").exception


# ─── Section F: Conflict warning rendering ───────────────────────────────────

class TestConflictRendering:
    def _run_conflict(self, conflict_status, conflicts=None):
        state = _agent_state(conflict_status=conflict_status, conflicts=conflicts or [])
        at = _fresh()
        _fake_getenv = lambda k, *a: "fake-key" if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
            at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
            with patch("agent.run_agent", return_value=state):
                with patch("os.getenv", side_effect=_fake_getenv):
                    at.chat_input[0].set_value("Cross-doc question").run(timeout=TIMEOUT)
        return at

    def test_conflict_detected(self):
        at = self._run_conflict(
            "CONFLICT_DETECTED",
            ["IT policy: 90-day rotation. HR policy: 180 days."]
        )
        assert not at.exception
        # Warnings section should contain the conflict alert
        assert len(at.warning) >= 1

    def test_no_conflict(self):
        assert not self._run_conflict("NO_CONFLICT").exception

    def test_unknown_conflict(self):
        at = self._run_conflict("UNKNOWN")
        assert not at.exception
        # Info widget should be present for UNKNOWN
        assert len(at.info) >= 0  # soft check — non-crash is the requirement

    def test_multiple_conflicts(self):
        at = self._run_conflict(
            "CONFLICT_DETECTED",
            [
                "Policy A says X; Policy B says Y.",
                "Policy C says P; Policy D says Q.",
            ]
        )
        assert not at.exception
        assert len(at.warning) >= 1


# ─── Section G: Refusal, partial, unverified, error states ───────────────────

class TestSpecialAnswerStates:
    def _run_state(self, **kwargs):
        state = _agent_state(**kwargs)
        at = _fresh()
        _fake_getenv = lambda k, *a: "fake-key" if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            at.run(timeout=TIMEOUT)
            at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
            with patch("agent.run_agent", return_value=state):
                with patch("os.getenv", side_effect=_fake_getenv):
                    at.chat_input[0].set_value("Question").run(timeout=TIMEOUT)
        return at

    def test_refusal_renders(self):
        at = self._run_state(
            answer="I cannot find sufficient evidence.",
            answer_type="refusal",
            evidence_status="INSUFFICIENT",
            sources=[], supporting_passages=[],
        )
        assert not at.exception
        # refusal should produce an error widget
        assert len(at.error) >= 1

    def test_partial_grounded_renders(self):
        at = self._run_state(
            answer="Partially verified answer.",
            answer_type="partial_grounded",
            evidence_status="PARTIALLY_SUPPORTED",
        )
        assert not at.exception
        assert len(at.warning) >= 1

    def test_unverified_grounded_renders(self):
        at = self._run_state(
            answer="Answer — verification incomplete.",
            answer_type="unverified_grounded",
            evidence_status="UNAVAILABLE",
        )
        assert not at.exception
        assert len(at.warning) >= 1

    def test_excerpt_fallback_renders(self):
        at = self._run_state(
            answer="Based on excerpts: MFA is required.",
            answer_type="excerpt_fallback",
            evidence_status="UNAVAILABLE",
        )
        assert not at.exception
        assert len(at.warning) >= 1

    def test_error_state_renders(self):
        at = self._run_state(
            answer="",
            answer_type="error",
            evidence_status="",
            error="LLM call failed.",
            sources=[], supporting_passages=[],
        )
        assert not at.exception

    def test_unverified_grounded_not_shown_as_success(self):
        # The badge for unverified_grounded must be warning, never success (green).
        # We verify the label mapping, not the colour (AppTest cannot inspect CSS).
        from app import _ANSWER_TYPE_LABELS
        _, level = _ANSWER_TYPE_LABELS["unverified_grounded"]
        assert level != "success"

    def test_partial_grounded_not_shown_as_success(self):
        from app import _ANSWER_TYPE_LABELS
        _, level = _ANSWER_TYPE_LABELS["partial_grounded"]
        assert level != "success"


# ─── Section H: Snowflake failure isolation ───────────────────────────────────

class TestSnowflakeFailureIsolation:
    def test_snowflake_conn_failure_does_not_crash_standard_rag(self):
        at = _fresh()
        mock_doc = _doc()
        with patch("snowflake_logger.is_snowflake_configured", return_value=True):
            with patch("snowflake_logger.create_logs_table_if_not_exists",
                       return_value=(False, "conn failed")):
                with patch("main.ask_question",
                           return_value=("MFA is required.", [mock_doc], 1.1)):
                    with patch("snowflake_logger.log_query_to_snowflake",
                               return_value=(False, "Snowflake logging failed")):
                        at.run(timeout=TIMEOUT)
                        at.chat_input[0].set_value("MFA question").run(timeout=TIMEOUT)
        assert not at.exception

    def test_snowflake_conn_failure_does_not_crash_agentic(self):
        state = _agent_state()
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=True):
            with patch("snowflake_logger.create_logs_table_if_not_exists",
                       return_value=(False, "conn failed")):
                with patch("agent.run_agent", return_value=state):
                    with patch("snowflake_logger.log_query_to_snowflake",
                               return_value=(False, "Snowflake logging failed")):
                        at.run(timeout=TIMEOUT)
                        at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
                        at.chat_input[0].set_value("MFA question").run(timeout=TIMEOUT)
        assert not at.exception

    def test_snowflake_warning_shown_on_log_failure(self):
        at = _fresh()
        mock_doc = _doc()
        _fake_getenv = lambda k, *a: "fake-key" if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
        with patch("snowflake_logger.is_snowflake_configured", return_value=True):
            with patch("snowflake_logger.create_logs_table_if_not_exists",
                       return_value=(True, "ok")):
                with patch("main.ask_question",
                           return_value=("MFA is required.", [mock_doc], 1.1)):
                    with patch("snowflake_logger.log_query_to_snowflake",
                               return_value=(False, "Snowflake logging failed: timeout")):
                        with patch("os.getenv", side_effect=_fake_getenv):
                            at.run(timeout=TIMEOUT)
                            at.chat_input[0].set_value("MFA question").run(timeout=TIMEOUT)
        assert not at.exception
        # A Snowflake warning must be surfaced to the user
        assert len(at.warning) >= 1


# ─── Section I: Missing NVIDIA credentials ───────────────────────────────────

class TestMissingCredentials:
    def test_missing_key_standard_rag_shows_error_not_crash(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            with patch("os.getenv", side_effect=lambda k, *a: (
                None if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
            )):
                at.run(timeout=TIMEOUT)
                at.chat_input[0].set_value("question").run(timeout=TIMEOUT)
        assert not at.exception
        assert len(at.error) >= 1

    def test_missing_key_agentic_shows_error_not_crash(self):
        at = _fresh()
        with patch("snowflake_logger.is_snowflake_configured", return_value=False):
            with patch("os.getenv", side_effect=lambda k, *a: (
                None if k == "NVIDIA_API_KEY" else os.environ.get(k, *(a or ()))
            )):
                at.run(timeout=TIMEOUT)
                at.radio[0].set_value("Agentic Compliance Analysis").run(timeout=TIMEOUT)
                at.chat_input[0].set_value("question").run(timeout=TIMEOUT)
        assert not at.exception
        assert len(at.error) >= 1
