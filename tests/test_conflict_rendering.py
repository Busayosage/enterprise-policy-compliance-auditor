"""Regression tests for conflict card parsing and HTML rendering.

All tests are fully offline — no Streamlit runtime, no LLM calls, no ChromaDB.
Tests cover:
  A. parse_conflict_item — all input variants
  B. conflict_card_html — structured and plain-text output
  C. XSS safety — HTML-escaped output
"""

import pytest
from conflict_renderer import parse_conflict_item, conflict_card_html


# ─────────────────────────────────────────────────────────────────────────────
# A. parse_conflict_item
# ─────────────────────────────────────────────────────────────────────────────

class TestParseConflictItem:

    def test_plain_dict_with_full_fields(self):
        item = {
            "topic": "IT Security sign-off",
            "policy_a": "it_security_policy.pdf",
            "page_a": 1,
            "excerpt_a": "All software must be approved.",
            "policy_b": "fast_track_policy.pdf",
            "page_b": 1,
            "excerpt_b": "Finance approval alone is sufficient.",
            "explanation": "These two policies directly contradict each other.",
            "review_flag": True,
        }
        result = parse_conflict_item(item)
        assert result["topic"] == "IT Security sign-off"
        assert result["policy_a"] == "it_security_policy.pdf"
        assert result["page_a"] == 1
        assert result["excerpt_a"] == "All software must be approved."
        assert result["policy_b"] == "fast_track_policy.pdf"
        assert result["review_flag"] is True
        assert result["structured"] is True

    def test_plain_dict_alternate_keys(self):
        """Handles LLM dicts that use non-standard field names."""
        item = {
            "conflict_topic": "Approval chain",
            "documents": ["policy_a.pdf", "policy_b.pdf"],
            "contradiction_details": "The two policies disagree.",
            "review_recommended": True,
        }
        result = parse_conflict_item(item)
        assert result["topic"] == "Approval chain"
        assert result["policy_a"] == "policy_a.pdf"
        assert result["policy_b"] == "policy_b.pdf"
        assert result["explanation"] == "The two policies disagree."
        assert result["review_flag"] is True

    def test_plain_dict_minimal(self):
        item = {"topic": "Minimal conflict"}
        result = parse_conflict_item(item)
        assert result["topic"] == "Minimal conflict"
        assert result["policy_a"] is None
        assert result["policy_b"] is None
        assert result["structured"] is False

    def test_json_string_dict(self):
        """Handles JSON-encoded dict strings (double-quoted keys)."""
        raw = '{"topic": "Data retention", "policy_a": "gdpr_policy.pdf", "explanation": "Conflict found."}'
        result = parse_conflict_item(raw)
        assert result["topic"] == "Data retention"
        assert result["policy_a"] == "gdpr_policy.pdf"
        assert result["explanation"] == "Conflict found."

    def test_python_literal_dict_string(self):
        """Handles Python repr of a dict (single-quoted keys — what str(dict) produces)."""
        raw = "{'topic': 'Software approval', 'policy_a': 'it_sec.pdf', 'review_flag': True}"
        result = parse_conflict_item(raw)
        assert result["topic"] == "Software approval"
        assert result["policy_a"] == "it_sec.pdf"
        assert result["review_flag"] is True

    def test_plain_text_with_colon_split(self):
        raw = "IT Security Policy vs Fast-Track Policy: These policies directly contradict each other."
        result = parse_conflict_item(raw)
        assert result["topic"] == "IT Security Policy vs Fast-Track Policy"
        assert "directly contradict" in result["explanation"]
        assert result["structured"] is False
        assert result["review_flag"] is False

    def test_plain_text_without_colon(self):
        raw = "These policies directly contradict each other with no colon in the text."
        result = parse_conflict_item(raw)
        assert result["topic"] == "Policy Conflict"
        assert "contradict" in result["explanation"]
        assert result["structured"] is False

    def test_review_recommended_prefix(self):
        raw = "[REVIEW RECOMMENDED] HR Policy vs Leave Policy: Annual leave entitlement differs."
        result = parse_conflict_item(raw)
        assert result["review_flag"] is True
        assert result["topic"] == "HR Policy vs Leave Policy"
        assert "Annual leave" in result["explanation"]

    def test_review_recommended_prefix_stripped_from_topic(self):
        """The [REVIEW RECOMMENDED] sentinel must not appear in topic or explanation."""
        raw = "[REVIEW RECOMMENDED] Finance Policy: Contradictory expense rules."
        result = parse_conflict_item(raw)
        assert "[REVIEW RECOMMENDED]" not in result["topic"]
        assert "[REVIEW RECOMMENDED]" not in result["explanation"]

    def test_empty_string_fallback(self):
        result = parse_conflict_item("")
        assert result["topic"] == "Policy Conflict"
        assert result["structured"] is False

    def test_non_string_non_dict(self):
        result = parse_conflict_item(42)
        assert result["topic"] == "Policy Conflict"
        assert result["structured"] is False

    def test_long_topic_before_colon_not_split(self):
        """A ':' occurring after >120 chars should not trigger the split."""
        long_prefix = "A" * 130
        raw = f"{long_prefix}: some explanation"
        result = parse_conflict_item(raw)
        assert result["topic"] == "Policy Conflict"
        assert long_prefix in result["explanation"]


# ─────────────────────────────────────────────────────────────────────────────
# B. conflict_card_html
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictCardHtml:

    def test_structured_card_contains_two_policy_boxes(self):
        parsed = {
            "topic": "IT Security sign-off",
            "policy_a": "it_security_policy.pdf",
            "page_a": 1,
            "excerpt_a": "All software must be approved.",
            "policy_b": "fast_track_policy.pdf",
            "page_b": 2,
            "excerpt_b": "Finance approval alone is sufficient.",
            "explanation": "Direct contradiction on approval requirement.",
            "review_flag": True,
            "structured": True,
        }
        html = conflict_card_html(parsed, 1)
        assert "conflict-compare" in html
        assert "conflict-policy-box" in html
        assert "it_security_policy.pdf" in html
        assert "fast_track_policy.pdf" in html
        assert "Page 1" in html
        assert "Page 2" in html
        assert "All software must be approved." in html

    def test_plain_text_card_no_compare_grid(self):
        parsed = {
            "topic": "HR vs Finance Policy",
            "explanation": "Contradictory leave entitlement rules.",
            "review_flag": False,
            "structured": False,
            "policy_a": None, "page_a": None, "excerpt_a": None,
            "policy_b": None, "page_b": None, "excerpt_b": None,
        }
        html = conflict_card_html(parsed, 1)
        assert "conflict-compare" not in html
        assert "HR vs Finance Policy" in html
        assert "Contradictory leave entitlement rules." in html

    def test_review_flag_true_shows_review_required(self):
        parsed = {
            "topic": "Conflict A",
            "explanation": "Some disagreement.",
            "review_flag": True,
            "structured": False,
            "policy_a": None, "page_a": None, "excerpt_a": None,
            "policy_b": None, "page_b": None, "excerpt_b": None,
        }
        html = conflict_card_html(parsed, 1)
        assert "Human review required" in html

    def test_review_flag_false_shows_recommended(self):
        parsed = {
            "topic": "Conflict B",
            "explanation": "Some disagreement.",
            "review_flag": False,
            "structured": False,
            "policy_a": None, "page_a": None, "excerpt_a": None,
            "policy_b": None, "page_b": None, "excerpt_b": None,
        }
        html = conflict_card_html(parsed, 1)
        assert "Human review recommended" in html

    def test_card_wraps_in_conflict_card_div(self):
        parsed = {
            "topic": "Test",
            "explanation": "Explanation.",
            "review_flag": False,
            "structured": False,
            "policy_a": None, "page_a": None, "excerpt_a": None,
            "policy_b": None, "page_b": None, "excerpt_b": None,
        }
        html = conflict_card_html(parsed, 1)
        assert html.strip().startswith('<div class="conflict-card">')
        assert html.strip().endswith("</div>")

    def test_no_page_number_when_page_is_none(self):
        parsed = {
            "topic": "Test",
            "policy_a": "policy.pdf",
            "page_a": None,
            "excerpt_a": None,
            "policy_b": "other.pdf",
            "page_b": None,
            "excerpt_b": None,
            "explanation": "",
            "review_flag": False,
            "structured": True,
        }
        html = conflict_card_html(parsed, 1)
        assert "Page" not in html


# ─────────────────────────────────────────────────────────────────────────────
# C. XSS safety
# ─────────────────────────────────────────────────────────────────────────────

class TestXssSafety:

    def test_script_tag_in_topic_is_escaped(self):
        item = {"topic": "<script>alert('xss')</script>", "explanation": "Safe."}
        parsed = parse_conflict_item(item)
        html = conflict_card_html(parsed, 1)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_script_tag_in_explanation_is_escaped(self):
        item = {"topic": "Safe Topic", "explanation": "<img src=x onerror=alert(1)>"}
        parsed = parse_conflict_item(item)
        html = conflict_card_html(parsed, 1)
        assert "<img" not in html
        assert "&lt;img" in html

    def test_script_tag_in_policy_name_is_escaped(self):
        item = {
            "topic": "T",
            "policy_a": "<script>bad()</script>",
            "policy_b": "safe.pdf",
            "structured": True,
        }
        parsed = parse_conflict_item(item)
        html = conflict_card_html(parsed, 1)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_plain_text_with_html_is_escaped(self):
        raw = "Policy A vs Policy B: <b>bold text</b> in conflict description."
        result = parse_conflict_item(raw)
        html = conflict_card_html(result, 1)
        assert "<b>" not in html
        assert "&lt;b&gt;" in html

    def test_script_tag_in_page_number_is_escaped(self):
        """Page values that contain HTML must be escaped in the card."""
        parsed = {
            "topic": "Approval Timing",
            "policy_a": "policy_a.pdf",
            "page_a": "<script>bad()</script>",
            "excerpt_a": "Must be approved first.",
            "policy_b": "policy_b.pdf",
            "page_b": 3,
            "excerpt_b": "Immediate approval allowed.",
            "explanation": "Conflict.",
            "review_flag": False,
            "structured": True,
        }
        html = conflict_card_html(parsed, 1)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_in_page_number_rendered_as_text(self):
        """An HTML-bearing page value appears as literal text, not markup."""
        parsed = {
            "topic": "T",
            "policy_a": "a.pdf",
            "page_a": '<img src=x onerror=alert(1)>',
            "excerpt_a": "Quote.",
            "policy_b": "b.pdf",
            "page_b": None,
            "excerpt_b": None,
            "explanation": "",
            "review_flag": False,
            "structured": True,
        }
        html = conflict_card_html(parsed, 1)
        assert "<img" not in html
        assert "&lt;img" in html


# ─────────────────────────────────────────────────────────────────────────────
# D. _pill / _badge filename escaping (app.py helpers)
# ─────────────────────────────────────────────────────────────────────────────

class TestAppHtmlHelpers:
    """Import _pill and _badge from app module via importlib to stay offline."""

    @staticmethod
    def _load_helpers():
        import importlib
        import sys
        import types

        # Stub every heavy dependency so app.py can be imported without
        # Streamlit, dotenv, or the project's own modules.
        stubs = [
            "streamlit", "dotenv", "database", "main",
            "snowflake_logger", "conflict_renderer",
            "langchain_community", "langchain_community.document_loaders",
            "chromadb", "openai",
        ]
        saved = {}
        for name in stubs:
            saved[name] = sys.modules.get(name)
            if name not in sys.modules:
                mod = types.ModuleType(name)
                # Give streamlit the attributes app.py calls at module level
                if name == "streamlit":
                    for attr in (
                        "set_page_config", "markdown", "sidebar", "session_state",
                        "query_params", "stop", "container", "columns",
                        "file_uploader", "button", "text_area", "metric",
                        "info", "warning", "error", "success", "expander",
                        "spinner", "write", "caption", "divider",
                    ):
                        setattr(mod, attr, lambda *a, **kw: None)
                    mod.session_state = {}
                    mod.query_params = {}
                if name == "dotenv":
                    mod.load_dotenv = lambda: None
                if name == "database":
                    for attr in (
                        "CHROMA_DB_DIR", "_COLLECTION_NAME",
                        "_detect_legacy_collection_name", "_is_legacy_index",
                        "_pointer_file", "_read_pointer", "_slot_dir",
                        "get_embeddings", "index_documents",
                    ):
                        setattr(mod, attr, None)
                if name == "main":
                    mod.ask_question = None
                    mod.MODEL_NAME = "mock"
                if name == "snowflake_logger":
                    mod.create_logs_table_if_not_exists = lambda: (True, "")
                    mod.is_snowflake_configured = lambda: False
                    mod.log_query_to_snowflake = lambda *a, **kw: None
                if name == "conflict_renderer":
                    mod.parse_conflict_item = lambda x: x
                    mod.conflict_card_html = lambda x, i: ""
                sys.modules[name] = mod
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "app_under_test",
                r"C:\Users\Admin\enterprise-compliance-rag\app.py",
            )
            app_mod = importlib.util.module_from_spec(spec)
            # Execute only the top-level function definitions, not st.* calls
            # by wrapping problematic code — instead, just grab the two helpers
            # after a partial exec up to the function definitions.
            # Simpler: parse the source and exec just the helper functions.
            import ast as _ast, html as _html_mod
            app_path = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")
            src = open(app_path, encoding="utf-8").read()
            tree = _ast.parse(src)
            # Collect only FunctionDef nodes named _pill or _badge
            ns = {"_html": _html_mod}
            for node in tree.body:
                if isinstance(node, _ast.FunctionDef) and node.name in ("_pill", "_badge"):
                    exec(compile(_ast.Module(body=[node], type_ignores=[]), "<app>", "exec"), ns)
            return ns["_pill"], ns["_badge"]
        finally:
            for name, orig in saved.items():
                if orig is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = orig

    def test_pill_escapes_script_tag_in_label(self):
        _pill, _ = self._load_helpers()
        html = _pill("<script>alert('xss')</script>", "info")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_pill_escapes_html_in_filename(self):
        _pill, _ = self._load_helpers()
        html = _pill('📄 <img src=x onerror=alert(1)>.pdf', "info")
        assert "<img" not in html
        assert "&lt;img" in html

    def test_badge_escapes_script_tag_in_label(self):
        _, _badge = self._load_helpers()
        html = _badge("<b>bold</b>", "info")
        assert "<b>" not in html
        assert "&lt;b&gt;" in html

    def test_pill_normal_filename_renders_correctly(self):
        """A safe filename passes through and is visible in the output."""
        _pill, _ = self._load_helpers()
        html = _pill("📄 policy_handbook_v2.pdf", "info")
        assert "policy_handbook_v2.pdf" in html
        assert 'class="pill' in html

    def test_badge_normal_label_renders_correctly(self):
        _, _badge = self._load_helpers()
        html = _badge("Compliant", "success")
        assert "Compliant" in html
        assert 'class="badge' in html
