"""
Shared fixtures for the RAG evaluation test suite.
All tests use temporary directories — never chroma_db/ or temp/.
"""
import os
import sys
import shutil
import tempfile
import pytest
# Ensure project root is on sys.path so application modules are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture_path(filename: str) -> str:
    return os.path.join(FIXTURES_DIR, filename)


@pytest.fixture()
def tmp_chroma_dir(tmp_path):
    """A temporary directory for an isolated ChromaDB collection.

    Windows limitation: ChromaDB holds HNSW file handles open after use, so
    shutil.rmtree may fail with PermissionError (WinError 32) on teardown.
    pytest's tmp_path machinery retries cleanup on subsequent runs so this
    does not accumulate permanent disk usage.  We suppress teardown errors
    here so they do not mask real test failures, but they remain visible in
    pytest's teardown ERROR section when they occur.
    """
    chroma_dir = str(tmp_path / "chroma_test")
    yield chroma_dir
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir, ignore_errors=True)


@pytest.fixture()
def tmp_temp_dir(tmp_path):
    """A temporary directory that replaces the application's temp/ folder."""
    t = str(tmp_path / "temp")
    os.makedirs(t, exist_ok=True)
    return t


@pytest.fixture()
def it_policy_path():
    return _fixture_path("it_security_policy.pdf")


@pytest.fixture()
def hr_policy_path():
    return _fixture_path("hr_leave_policy.pdf")


@pytest.fixture()
def finance_policy_path():
    return _fixture_path("finance_expense_policy.pdf")


from helpers import FakeUploadedFile


@pytest.fixture()
def it_policy_file(it_policy_path):
    return FakeUploadedFile(it_policy_path)


@pytest.fixture()
def hr_policy_file(hr_policy_path):
    return FakeUploadedFile(hr_policy_path)


@pytest.fixture()
def finance_policy_file(finance_policy_path):
    return FakeUploadedFile(finance_policy_path)
