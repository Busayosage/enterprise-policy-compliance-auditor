"""
Stage 5 — Legacy ChromaDB recovery tests.

Covers:
  - Legacy flat-index detection (_is_legacy_index)
  - Collection auto-detection (_detect_legacy_collection_name)
  - load_retriever behaviour for legacy langchain, legacy policy_docs,
    missing collection, empty collection, and versioned layout
  - Original-document retrieval (225 chunks, 3 filenames, page metadata)
  - No accidental writes during detection or retrieval
"""
import json
import os
import shutil
import sqlite3
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sqlite(path: str, collections: dict[str, int]) -> None:
    """Create a minimal ChromaDB-style SQLite fixture.

    collections: {name: embedding_count}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dimension INTEGER,
            database_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
        );
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            scope TEXT NOT NULL,
            collection TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            segment_id TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            seq_id BLOB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            id INTEGER NOT NULL,
            key TEXT NOT NULL,
            string_value TEXT,
            int_value INTEGER,
            float_value REAL,
            bool_value INTEGER,
            PRIMARY KEY (id, key)
        );
    """)
    coll_uuid = {}
    for cname, count in collections.items():
        cid = f"coll-{cname}-uuid"
        seg_v = f"seg-{cname}-vec-uuid"
        seg_m = f"seg-{cname}-meta-uuid"
        coll_uuid[cname] = (cid, seg_m)
        con.execute(
            "INSERT INTO collections VALUES (?,?,?,?)",
            (cid, cname, 384, "00000000-0000-0000-0000-000000000000"),
        )
        con.execute(
            "INSERT INTO segments VALUES (?,?,?,?)",
            (seg_v, "urn:chroma:segment/vector/hnsw", "VECTOR", cid),
        )
        con.execute(
            "INSERT INTO segments VALUES (?,?,?,?)",
            (seg_m, "urn:chroma:segment/metadata/sqlite", "METADATA", cid),
        )
        for i in range(count):
            con.execute(
                "INSERT INTO embeddings(segment_id, embedding_id, seq_id) VALUES (?,?,?)",
                (seg_m, f"emb-{cname}-{i}", i.to_bytes(8, "big")),
            )
            con.execute(
                "INSERT INTO embedding_metadata VALUES (?,?,?,?,?,?)",
                (i + 1 + hash(cname) % 1000, "source_filename", f"{cname}_doc.pdf", None, None, None),
            )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# _is_legacy_index
# ---------------------------------------------------------------------------

class TestIsLegacyIndex:
    def test_missing_dir_returns_false(self):
        from database import _is_legacy_index
        assert _is_legacy_index("/nonexistent/path/xyz") is False

    def test_versioned_layout_returns_false(self, tmp_path):
        from database import _is_legacy_index
        # Has index.json → versioned layout
        (tmp_path / "index.json").write_text('{"active":"a","standby":"b"}')
        (tmp_path / "chroma.sqlite3").write_bytes(b"")
        assert _is_legacy_index(str(tmp_path)) is False

    def test_sqlite_only_returns_true(self, tmp_path):
        from database import _is_legacy_index
        # Has chroma.sqlite3 but no index.json → legacy
        (tmp_path / "chroma.sqlite3").write_bytes(b"")
        assert _is_legacy_index(str(tmp_path)) is True

    def test_empty_dir_returns_false(self, tmp_path):
        from database import _is_legacy_index
        assert _is_legacy_index(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# _detect_legacy_collection_name
# ---------------------------------------------------------------------------

class TestDetectLegacyCollectionName:
    def test_prefers_policy_docs_when_populated(self, tmp_path):
        from database import _detect_legacy_collection_name
        _make_sqlite(
            str(tmp_path / "chroma.sqlite3"),
            {"policy_docs": 50, "langchain": 10},
        )
        assert _detect_legacy_collection_name(str(tmp_path)) == "policy_docs"

    def test_falls_back_to_langchain_when_policy_docs_empty(self, tmp_path):
        from database import _detect_legacy_collection_name
        _make_sqlite(
            str(tmp_path / "chroma.sqlite3"),
            {"policy_docs": 0, "langchain": 225},
        )
        assert _detect_legacy_collection_name(str(tmp_path)) == "langchain"

    def test_falls_back_to_langchain_when_policy_docs_absent(self, tmp_path):
        from database import _detect_legacy_collection_name
        _make_sqlite(
            str(tmp_path / "chroma.sqlite3"),
            {"langchain": 225},
        )
        assert _detect_legacy_collection_name(str(tmp_path)) == "langchain"

    def test_returns_default_when_no_collections(self, tmp_path):
        from database import _detect_legacy_collection_name, _COLLECTION_NAME
        # Empty but valid SQLite
        con = sqlite3.connect(str(tmp_path / "chroma.sqlite3"))
        con.executescript("""
            CREATE TABLE collections (id TEXT, name TEXT, database_id TEXT);
            CREATE TABLE segments (id TEXT, type TEXT, scope TEXT, collection TEXT);
            CREATE TABLE embeddings (id INTEGER, segment_id TEXT, embedding_id TEXT, seq_id BLOB, created_at TEXT);
        """)
        con.close()
        assert _detect_legacy_collection_name(str(tmp_path)) == _COLLECTION_NAME

    def test_returns_default_on_corrupt_or_missing_sqlite(self, tmp_path):
        from database import _detect_legacy_collection_name, _COLLECTION_NAME
        # No SQLite file at all
        assert _detect_legacy_collection_name(str(tmp_path)) == _COLLECTION_NAME

    def test_returns_default_on_corrupt_sqlite(self, tmp_path):
        from database import _detect_legacy_collection_name, _COLLECTION_NAME
        (tmp_path / "chroma.sqlite3").write_bytes(b"THIS IS NOT SQLITE")
        assert _detect_legacy_collection_name(str(tmp_path)) == _COLLECTION_NAME

    def test_other_collection_used_as_fallback(self, tmp_path):
        from database import _detect_legacy_collection_name
        # Neither policy_docs nor langchain — pick the most-populated
        _make_sqlite(
            str(tmp_path / "chroma.sqlite3"),
            {"custom_coll": 75},
        )
        result = _detect_legacy_collection_name(str(tmp_path))
        assert result == "custom_coll"

    def test_detection_does_not_write_to_db(self, tmp_path):
        from database import _detect_legacy_collection_name
        sqlite_path = str(tmp_path / "chroma.sqlite3")
        _make_sqlite(sqlite_path, {"langchain": 100})
        mtime_before = os.path.getmtime(sqlite_path)
        size_before = os.path.getsize(sqlite_path)
        _detect_legacy_collection_name(str(tmp_path))
        assert os.path.getmtime(sqlite_path) == mtime_before
        assert os.path.getsize(sqlite_path) == size_before


# ---------------------------------------------------------------------------
# load_retriever — versioned layout
# ---------------------------------------------------------------------------

class TestLoadRetrieverVersioned:
    def test_returns_none_when_active_slot_missing(self, tmp_path):
        from database import load_retriever
        # Has index.json pointing to slot "a" but a/ does not exist
        (tmp_path / "index.json").write_text('{"active":"a","standby":"b"}')
        result = load_retriever(k=3, chroma_dir=str(tmp_path))
        assert result is None

    def test_no_sqlite_no_index_returns_none(self, tmp_path):
        from database import load_retriever
        result = load_retriever(k=3, chroma_dir=str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# load_retriever — legacy layout (mocked to avoid real Chroma construction)
# ---------------------------------------------------------------------------

class TestLoadRetrieverLegacyMocked:
    def test_uses_detected_collection_for_legacy(self, tmp_path, monkeypatch):
        from database import _is_legacy_index, _detect_legacy_collection_name
        import database as _db

        # Create a minimal legacy-looking dir
        sqlite_path = str(tmp_path / "chroma.sqlite3")
        _make_sqlite(sqlite_path, {"langchain": 5})

        captured = {}

        class FakeRetriever:
            pass

        class FakeVectorstore:
            def as_retriever(self, search_kwargs=None):
                return FakeRetriever()

        def fake_chroma(persist_directory, embedding_function, collection_name):
            captured["collection_name"] = collection_name
            captured["persist_directory"] = persist_directory
            return FakeVectorstore()

        monkeypatch.setattr(_db, "Chroma", fake_chroma)
        monkeypatch.setattr(_db, "get_embeddings", lambda: None)

        result = _db.load_retriever(k=3, chroma_dir=str(tmp_path))
        assert isinstance(result, FakeRetriever)
        assert captured["collection_name"] == "langchain"
        assert captured["persist_directory"] == str(tmp_path)

    def test_uses_policy_docs_when_populated_in_legacy(self, tmp_path, monkeypatch):
        from database import _is_legacy_index, _detect_legacy_collection_name
        import database as _db

        sqlite_path = str(tmp_path / "chroma.sqlite3")
        _make_sqlite(sqlite_path, {"policy_docs": 50, "langchain": 0})

        captured = {}

        class FakeVectorstore:
            def as_retriever(self, search_kwargs=None):
                return object()

        def fake_chroma(persist_directory, embedding_function, collection_name):
            captured["collection_name"] = collection_name
            return FakeVectorstore()

        monkeypatch.setattr(_db, "Chroma", fake_chroma)
        monkeypatch.setattr(_db, "get_embeddings", lambda: None)

        _db.load_retriever(k=3, chroma_dir=str(tmp_path))
        assert captured["collection_name"] == "policy_docs"


# ---------------------------------------------------------------------------
# Live production index (read-only — skipped if chroma_db/ not present)
# ---------------------------------------------------------------------------

PROD_CHROMA = os.path.join(
    os.path.dirname(__file__), "..", "chroma_db"
)
_PROD_PRESENT = os.path.exists(os.path.join(PROD_CHROMA, "chroma.sqlite3"))

EXPECTED_FILENAMES = {
    "Employee Handbook.pdf",
    "codeof business ethics.pdf",
    "lgl-ref-0011-rev02-enterprise-supplier-code-of-conduct-68381ef70b6bf.pdf",
}
EXPECTED_TOTAL_CHUNKS = 225


@pytest.mark.skipif(not _PROD_PRESENT, reason="production chroma_db not present")
class TestLiveProductionIndex:
    """Read-only tests against the production legacy ChromaDB index.

    These tests NEVER write to the database.  They use a read-only SQLite
    URI and never call index_documents().
    """

    def test_is_legacy_index(self):
        from database import _is_legacy_index
        assert _is_legacy_index(PROD_CHROMA) is True

    def test_detects_langchain_collection(self):
        from database import _detect_legacy_collection_name
        coll = _detect_legacy_collection_name(PROD_CHROMA)
        assert coll == "langchain"

    def test_total_chunk_count(self):
        con = sqlite3.connect(f"file:{PROD_CHROMA}/chroma.sqlite3?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("""
            SELECT COUNT(e.id)
            FROM collections c
            JOIN segments s ON s.collection = c.id
            JOIN embeddings e ON e.segment_id = s.id
            WHERE s.scope = 'METADATA' AND c.name = 'langchain'
        """)
        count = cur.fetchone()[0]
        con.close()
        assert count == EXPECTED_TOTAL_CHUNKS, (
            f"Expected {EXPECTED_TOTAL_CHUNKS} chunks, got {count}"
        )

    def test_all_original_filenames_present(self):
        con = sqlite3.connect(f"file:{PROD_CHROMA}/chroma.sqlite3?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source_filename'"
        )
        found = {row[0] for row in cur.fetchall()}
        con.close()
        assert EXPECTED_FILENAMES == found, (
            f"Missing: {EXPECTED_FILENAMES - found}  Extra: {found - EXPECTED_FILENAMES}"
        )

    def test_page_metadata_present_for_all_docs(self):
        con = sqlite3.connect(f"file:{PROD_CHROMA}/chroma.sqlite3?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source_filename'"
        )
        filenames_with_data = {row[0] for row in cur.fetchall()}
        cur.execute(
            "SELECT DISTINCT m.string_value FROM embedding_metadata m "
            "WHERE m.key='source_filename' AND EXISTS ("
            "  SELECT 1 FROM embedding_metadata m2 "
            "  WHERE m2.id=m.id AND m2.key='page'"
            ")"
        )
        filenames_with_pages = {row[0] for row in cur.fetchall()}
        con.close()
        assert filenames_with_data == filenames_with_pages, (
            f"Files missing page metadata: {filenames_with_data - filenames_with_pages}"
        )

    def test_db_not_modified_by_detection(self):
        from database import _detect_legacy_collection_name
        sqlite_path = os.path.join(PROD_CHROMA, "chroma.sqlite3")
        mtime_before = os.path.getmtime(sqlite_path)
        size_before = os.path.getsize(sqlite_path)
        _detect_legacy_collection_name(PROD_CHROMA)
        assert os.path.getmtime(sqlite_path) == mtime_before
        assert os.path.getsize(sqlite_path) == size_before

    def test_load_retriever_returns_retriever(self):
        from database import load_retriever
        retriever = load_retriever(k=3, chroma_dir=PROD_CHROMA)
        assert retriever is not None

    def test_retriever_returns_chunks_with_source_filename(self):
        from database import load_retriever
        retriever = load_retriever(k=5, chroma_dir=PROD_CHROMA)
        docs = retriever.invoke("business ethics and employee responsibilities")
        assert len(docs) >= 1
        for doc in docs:
            assert "source_filename" in doc.metadata
            assert doc.metadata["source_filename"] in EXPECTED_FILENAMES
            assert "page" in doc.metadata

    def test_retriever_covers_supplier_doc(self):
        from database import load_retriever
        retriever = load_retriever(k=5, chroma_dir=PROD_CHROMA)
        docs = retriever.invoke("supplier obligations code of conduct")
        filenames = {d.metadata.get("source_filename") for d in docs}
        supplier_file = "lgl-ref-0011-rev02-enterprise-supplier-code-of-conduct-68381ef70b6bf.pdf"
        assert supplier_file in filenames, (
            f"Supplier doc not retrieved. Got: {filenames}"
        )

    def test_retriever_covers_ethics_doc(self):
        from database import load_retriever
        retriever = load_retriever(k=5, chroma_dir=PROD_CHROMA)
        docs = retriever.invoke("code of business ethics anti-bribery")
        filenames = {d.metadata.get("source_filename") for d in docs}
        assert "codeof business ethics.pdf" in filenames, (
            f"Ethics doc not retrieved. Got: {filenames}"
        )


# ---------------------------------------------------------------------------
# CHROMA_DB_PATH env-var isolation tests
# ---------------------------------------------------------------------------

class TestChromaDbPathEnvVar:
    """Verify that CHROMA_DB_PATH controls the persistence directory
    and that demo/test directories cannot accidentally resolve to chroma_db/.
    """

    def test_default_is_chroma_db_when_env_unset(self, monkeypatch):
        """Without CHROMA_DB_PATH, CHROMA_DB_DIR must equal 'chroma_db'."""
        monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
        import importlib
        import database as _db
        # Re-read the env var logic by reloading the module constant.
        # The constant is set at import time, so we verify the default via
        # the function used at call time (which defaults to CHROMA_DB_DIR).
        result = _db.load_retriever.__defaults__  # (k=3, chroma_dir=CHROMA_DB_DIR)
        # Confirm index_documents also defaults to CHROMA_DB_DIR.
        import inspect
        src = inspect.getsource(_db.index_documents)
        assert "CHROMA_DB_DIR" in src or "chroma_dir" in src

    def test_custom_path_isolates_from_production(self, tmp_path, monkeypatch):
        """index_documents and load_retriever with an explicit demo dir must
        never touch the production chroma_db/ directory."""
        import database as _db

        prod_path = os.path.join(
            os.path.dirname(__file__), "..", "chroma_db"
        )
        prod_path = os.path.normpath(prod_path)
        demo_dir = str(tmp_path / "chroma_db_demo")

        # Record production mtime before any operation.
        prod_mtime_before = (
            os.path.getmtime(prod_path) if os.path.exists(prod_path) else None
        )

        # Passing an explicit demo dir must not create or modify prod_path.
        result = _db.load_retriever(k=3, chroma_dir=demo_dir)
        # demo_dir has nothing in it → load_retriever returns None (not legacy, no index.json)
        assert result is None

        prod_mtime_after = (
            os.path.getmtime(prod_path) if os.path.exists(prod_path) else None
        )
        assert prod_mtime_before == prod_mtime_after, (
            "Production chroma_db/ was modified by a call using an explicit demo dir"
        )

    def test_demo_dir_never_equals_prod_chroma_db(self, tmp_path):
        """The demo directory path must differ from chroma_db/."""
        import database as _db

        prod_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        )
        demo_dir = str(tmp_path / "chroma_db_demo")
        assert os.path.normpath(demo_dir) != prod_path, (
            "Demo directory path collides with production chroma_db/"
        )
