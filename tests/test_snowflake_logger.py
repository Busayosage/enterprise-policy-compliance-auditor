"""
Tests for snowflake_logger.py graceful failure handling.
No live Snowflake connections are made.
"""
import os
import sys
import importlib
import pytest
from unittest.mock import patch, MagicMock


def _reload_logger():
    """Reload snowflake_logger with a clean module cache."""
    for key in list(sys.modules.keys()):
        if "snowflake_logger" in key:
            del sys.modules[key]
    import snowflake_logger
    return snowflake_logger


def _reload_without_connector():
    """Reload snowflake_logger as if snowflake.connector is not installed."""
    for key in list(sys.modules.keys()):
        if "snowflake_logger" in key:
            del sys.modules[key]
    sys.modules["snowflake"] = None       # type: ignore
    sys.modules["snowflake.connector"] = None  # type: ignore
    try:
        import snowflake_logger
        importlib.reload(snowflake_logger)
        return snowflake_logger
    finally:
        sys.modules.pop("snowflake", None)
        sys.modules.pop("snowflake.connector", None)


BLANK_CREDS = {k: "" for k in [
    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_ROLE",
]}

FAKE_CREDS = {
    "SNOWFLAKE_ACCOUNT": "fake_acct",
    "SNOWFLAKE_USER": "fake_user",
    "SNOWFLAKE_PASSWORD": "fake_pass",
    "SNOWFLAKE_WAREHOUSE": "fake_wh",
    "SNOWFLAKE_DATABASE": "fake_db",
    "SNOWFLAKE_SCHEMA": "fake_schema",
    "SNOWFLAKE_ROLE": "fake_role",
}


# ---------------------------------------------------------------------------
# 7a. Missing package
# ---------------------------------------------------------------------------

class TestConnectorAbsent:

    def test_is_configured_returns_false(self):
        mod = _reload_without_connector()
        assert mod._CONNECTOR_AVAILABLE is False
        assert mod.is_snowflake_configured() is False

    def test_log_query_returns_false_tuple(self):
        mod = _reload_without_connector()
        ok, msg = mod.log_query_to_snowflake(
            "doc.pdf", "question?", "answer.", "doc.pdf", "1", True
        )
        assert ok is False
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_create_table_returns_false_tuple(self):
        mod = _reload_without_connector()
        ok, msg = mod.create_logs_table_if_not_exists()
        assert ok is False
        assert isinstance(msg, str)

    def test_test_connection_returns_false_tuple(self):
        mod = _reload_without_connector()
        ok, msg = mod.test_snowflake_connection()
        assert ok is False
        assert isinstance(msg, str)

    def test_error_message_mentions_package(self):
        mod = _reload_without_connector()
        _, msg = mod.log_query_to_snowflake("d", "q", "a", "d", "1", False)
        assert "snowflake" in msg.lower()


# ---------------------------------------------------------------------------
# 7b. Connector present, credentials missing
# ---------------------------------------------------------------------------

class TestMissingCredentials:

    def test_is_configured_returns_false_when_blank(self):
        with patch.dict(os.environ, BLANK_CREDS):
            for key in list(sys.modules.keys()):
                if "snowflake_logger" in key:
                    del sys.modules[key]
            import snowflake_logger as mod
            assert mod.is_snowflake_configured() is False

    def test_log_query_returns_false_when_no_creds(self):
        with patch.dict(os.environ, BLANK_CREDS):
            for key in list(sys.modules.keys()):
                if "snowflake_logger" in key:
                    del sys.modules[key]
            import snowflake_logger as mod
            ok, msg = mod.log_query_to_snowflake("d", "q", "a", "d", "1", False)
        assert ok is False
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# 7c. Connector present, credentials set, but connection fails
# ---------------------------------------------------------------------------

class TestConnectionFailure:

    def _make_failing_connector(self, error_message="Network error"):
        mock_conn = MagicMock()
        mock_conn.connect.side_effect = Exception(error_message)
        return mock_conn

    def _setup_module_with_failing_connector(self, error_message="Network error"):
        mock_connector = self._make_failing_connector(error_message)
        for key in list(sys.modules.keys()):
            if "snowflake_logger" in key:
                del sys.modules[key]
        with patch.dict(sys.modules, {
            "snowflake.connector": mock_connector,
            "snowflake": MagicMock(),
        }):
            with patch.dict(os.environ, FAKE_CREDS):
                import snowflake_logger as mod
                mod._sf_connector = mock_connector
                mod._CONNECTOR_AVAILABLE = True
                return mod, mock_connector

    def test_log_query_returns_false_on_connect_error(self):
        mod, _ = self._setup_module_with_failing_connector()
        with patch.dict(os.environ, FAKE_CREDS):
            ok, msg = mod.log_query_to_snowflake("d", "q", "a", "d", "1", True)
        assert ok is False
        assert isinstance(msg, str)

    def test_create_table_returns_false_on_connect_error(self):
        mod, _ = self._setup_module_with_failing_connector()
        with patch.dict(os.environ, FAKE_CREDS):
            ok, msg = mod.create_logs_table_if_not_exists()
        assert ok is False
        assert isinstance(msg, str)

    def test_test_connection_returns_false_on_connect_error(self):
        mod, _ = self._setup_module_with_failing_connector()
        with patch.dict(os.environ, FAKE_CREDS):
            ok, msg = mod.test_snowflake_connection()
        assert ok is False
        assert isinstance(msg, str)

    def test_error_message_is_informative(self):
        mod, _ = self._setup_module_with_failing_connector("Auth token expired")
        with patch.dict(os.environ, FAKE_CREDS):
            ok, msg = mod.log_query_to_snowflake("d", "q", "a", "d", "1", True)
        assert ok is False
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# 7d. Return-type contract — all public functions always return (bool, str)
# ---------------------------------------------------------------------------

class TestReturnTypeContract:

    @pytest.mark.parametrize("fn_name,kwargs", [
        ("log_query_to_snowflake", {
            "document_names": "d", "user_question": "q", "ai_answer": "a",
            "source_filenames": "d", "source_pages": "1", "evidence_found": False,
        }),
        ("create_logs_table_if_not_exists", {}),
        ("test_snowflake_connection", {}),
    ])
    def test_always_returns_bool_str_tuple(self, fn_name, kwargs):
        mod = _reload_without_connector()
        fn = getattr(mod, fn_name)
        result = fn(**kwargs)
        assert isinstance(result, tuple), f"{fn_name} must return tuple"
        assert len(result) == 2, f"{fn_name} must return 2-element tuple"
        ok, msg = result
        assert isinstance(ok, bool), f"{fn_name} first element must be bool"
        assert isinstance(msg, str), f"{fn_name} second element must be str"
