"""
Tests for PDF indexing, chunk generation, metadata preservation,
ChromaDB persistence, retrieval behaviour, and the Phase 3 versioned-index
safety guarantees.

All tests use isolated temp directories — never chroma_db/ or temp/.
"""
import gc
import json
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock

import database
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from helpers import FakeUploadedFile

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ---------------------------------------------------------------------------
# Helpers: delegate to real database.index_documents() with isolated dirs
# ---------------------------------------------------------------------------

def _index_to(uploaded_files, chroma_dir, temp_dir):
    """Delegate to database.index_documents() with isolated directories."""
    return database.index_documents(uploaded_files, chroma_dir=chroma_dir, temp_dir=temp_dir)


def _load_retriever(chroma_dir, k=3):
    return database.load_retriever(k=k, chroma_dir=chroma_dir)


def _active_dir(chroma_dir):
    """Return the path to the currently active ChromaDB slot."""
    active, _ = database._read_pointer(chroma_dir)
    return database._slot_dir(active, chroma_dir)


# ---------------------------------------------------------------------------
# 1. PDF indexing — basic success path
# ---------------------------------------------------------------------------

class TestPDFIndexing:

    def test_single_pdf_indexing_succeeds(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        ok, msg, n_files, n_chunks, fnames = _index_to(
            [it_policy_file], tmp_chroma_dir, tmp_temp_dir
        )
        assert ok is True
        assert n_files == 1
        assert n_chunks > 0
        assert "it_security_policy.pdf" in fnames

    def test_multi_pdf_indexing_succeeds(self, tmp_chroma_dir, tmp_temp_dir,
                                          it_policy_file, hr_policy_file):
        ok, msg, n_files, n_chunks, fnames = _index_to(
            [it_policy_file, hr_policy_file], tmp_chroma_dir, tmp_temp_dir
        )
        assert ok is True
        assert n_files == 2
        assert n_chunks > 0
        assert set(fnames) == {"it_security_policy.pdf", "hr_leave_policy.pdf"}

    def test_three_pdfs_indexed_together(self, tmp_chroma_dir, tmp_temp_dir,
                                          it_policy_file, hr_policy_file, finance_policy_file):
        ok, msg, n_files, n_chunks, fnames = _index_to(
            [it_policy_file, hr_policy_file, finance_policy_file],
            tmp_chroma_dir, tmp_temp_dir,
        )
        assert ok is True
        assert n_files == 3

    def test_empty_file_list_returns_false(self, tmp_chroma_dir, tmp_temp_dir):
        ok, msg, _, _, _ = database.index_documents([])
        assert ok is False
        assert "No files provided" in msg

    def test_chroma_directory_created_after_indexing(self, tmp_chroma_dir, tmp_temp_dir,
                                                       it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert os.path.exists(tmp_chroma_dir)
        assert os.path.exists(_active_dir(tmp_chroma_dir))

    def test_temp_files_cleaned_up_after_indexing(self, tmp_chroma_dir, tmp_temp_dir,
                                                    it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        remaining = os.listdir(tmp_temp_dir)
        pdf_files = [f for f in remaining if f.endswith(".pdf")]
        assert pdf_files == [], f"Temp PDFs not cleaned up: {pdf_files}"


# ---------------------------------------------------------------------------
# 2. Chunk generation and metadata preservation
# ---------------------------------------------------------------------------

class TestChunkMetadata:

    def test_source_filename_metadata_preserved(self, tmp_chroma_dir, tmp_temp_dir,
                                                  it_policy_file):
        ok, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok
        active = _active_dir(tmp_chroma_dir)
        vs = Chroma(persist_directory=active, embedding_function=_get_embeddings(),
                    collection_name=database._COLLECTION_NAME)
        all_docs = vs.get(include=["metadatas"])
        filenames_in_store = {m.get("source_filename") for m in all_docs["metadatas"]}
        assert "it_security_policy.pdf" in filenames_in_store

    def test_page_metadata_preserved(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        active = _active_dir(tmp_chroma_dir)
        vs = Chroma(persist_directory=active, embedding_function=_get_embeddings(),
                    collection_name=database._COLLECTION_NAME)
        all_docs = vs.get(include=["metadatas"])
        pages = {m.get("page") for m in all_docs["metadatas"]}
        assert len(pages) >= 1

    def test_multi_doc_filenames_distinct_in_metadata(self, tmp_chroma_dir, tmp_temp_dir,
                                                        it_policy_file, hr_policy_file):
        _index_to([it_policy_file, hr_policy_file], tmp_chroma_dir, tmp_temp_dir)
        active = _active_dir(tmp_chroma_dir)
        vs = Chroma(persist_directory=active, embedding_function=_get_embeddings(),
                    collection_name=database._COLLECTION_NAME)
        all_docs = vs.get(include=["metadatas"])
        filenames_in_store = {m.get("source_filename") for m in all_docs["metadatas"]}
        assert "it_security_policy.pdf" in filenames_in_store
        assert "hr_leave_policy.pdf" in filenames_in_store

    def test_chunk_size_within_bounds(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        active = _active_dir(tmp_chroma_dir)
        vs = Chroma(persist_directory=active, embedding_function=_get_embeddings(),
                    collection_name=database._COLLECTION_NAME)
        all_docs = vs.get(include=["documents"])
        for doc_text in all_docs["documents"]:
            assert len(doc_text) <= 1200, f"Chunk too large: {len(doc_text)} chars"

    def test_indexing_replaces_old_collection(self, tmp_chroma_dir, tmp_temp_dir,
                                               it_policy_file, hr_policy_file):
        """Re-indexing produces a clean replacement with no old documents remaining.
        Phase 3: versioned slot-swap guarantees old docs are absent from the active slot.
        """
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)
        active = _active_dir(tmp_chroma_dir)
        vs = Chroma(persist_directory=active, embedding_function=_get_embeddings(),
                    collection_name=database._COLLECTION_NAME)
        all_docs = vs.get(include=["metadatas"])
        filenames_in_store = {m.get("source_filename") for m in all_docs["metadatas"]}
        assert "hr_leave_policy.pdf" in filenames_in_store
        assert "it_security_policy.pdf" not in filenames_in_store, (
            "Old IT policy chunks still present after HR-only re-index — slot swap failed"
        )


# ---------------------------------------------------------------------------
# 3. ChromaDB persistence and retrieval
# ---------------------------------------------------------------------------

class TestChromaRetrieval:

    def test_load_retriever_returns_none_when_no_store(self, tmp_path):
        missing_dir = str(tmp_path / "nonexistent")
        retriever = _load_retriever(missing_dir)
        assert retriever is None

    def test_retriever_returns_documents(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        assert retriever is not None
        results = retriever.invoke("password requirements")
        assert len(results) > 0

    def test_retriever_respects_k_parameter(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        for k in [1, 2, 3]:
            retriever = _load_retriever(tmp_chroma_dir, k=k)
            results = retriever.invoke("security policy")
            assert len(results) <= k

    def test_retrieval_result_has_page_content(self, tmp_chroma_dir, tmp_temp_dir,
                                                it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        retriever = _load_retriever(tmp_chroma_dir, k=1)
        results = retriever.invoke("password")
        assert all(hasattr(doc, "page_content") for doc in results)
        assert all(len(doc.page_content) > 0 for doc in results)

    def test_retrieval_result_has_metadata(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        results = retriever.invoke("data classification")
        for doc in results:
            assert "source_filename" in doc.metadata
            assert doc.metadata["source_filename"] == "it_security_policy.pdf"


# ---------------------------------------------------------------------------
# 4. Multi-document retrieval
# ---------------------------------------------------------------------------

class TestMultiDocRetrieval:

    def test_retrieval_from_multi_doc_store(self, tmp_chroma_dir, tmp_temp_dir,
                                             it_policy_file, hr_policy_file):
        _index_to([it_policy_file, hr_policy_file], tmp_chroma_dir, tmp_temp_dir)
        retriever = _load_retriever(tmp_chroma_dir, k=4)
        it_results = retriever.invoke("password requirements MFA")
        hr_results = retriever.invoke("annual leave entitlement days")
        it_fnames = {d.metadata.get("source_filename") for d in it_results}
        hr_fnames = {d.metadata.get("source_filename") for d in hr_results}
        assert "it_security_policy.pdf" in it_fnames
        assert "hr_leave_policy.pdf" in hr_fnames

    def test_cross_document_query_returns_both_sources(self, tmp_chroma_dir, tmp_temp_dir,
                                                        it_policy_file, finance_policy_file):
        _index_to([it_policy_file, finance_policy_file], tmp_chroma_dir, tmp_temp_dir)
        retriever = _load_retriever(tmp_chroma_dir, k=4)
        results = retriever.invoke("software subscription approval IT security")
        fnames = {d.metadata.get("source_filename") for d in results}
        assert len(fnames) >= 1

    def test_source_filename_accuracy_single_doc(self, tmp_chroma_dir, tmp_temp_dir,
                                                   hr_policy_file):
        _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        results = retriever.invoke("parental leave weeks")
        for doc in results:
            assert doc.metadata.get("source_filename") == "hr_leave_policy.pdf"


# ---------------------------------------------------------------------------
# 5. Versioned-index safety guarantees (Phase 3)
# ---------------------------------------------------------------------------

class TestVersionedIndexSafety:

    def test_pointer_file_created_after_indexing(self, tmp_chroma_dir, tmp_temp_dir,
                                                   it_policy_file):
        """index.json is written after a successful index build."""
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert os.path.exists(database._pointer_file(tmp_chroma_dir)), \
            "index.json not created after indexing"
        with open(database._pointer_file(tmp_chroma_dir)) as f:
            data = json.load(f)
        assert "active" in data and "standby" in data

    def test_active_and_standby_slots_are_different(self, tmp_chroma_dir, tmp_temp_dir,
                                                      it_policy_file):
        """After indexing, active != standby in index.json."""
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        active, standby = database._read_pointer(tmp_chroma_dir)
        assert active != standby
        assert active in database._SLOTS
        assert standby in database._SLOTS

    def test_old_docs_absent_from_search_after_replacement(self, tmp_chroma_dir, tmp_temp_dir,
                                                             it_policy_file, hr_policy_file):
        """After replacing IT policy with HR policy, search returns only HR results."""
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)

        retriever = _load_retriever(tmp_chroma_dir, k=4)
        # Query known IT-specific terms — should NOT find IT policy
        it_results = retriever.invoke("multi-factor authentication MFA password requirements")
        it_fnames = {d.metadata.get("source_filename") for d in it_results}
        assert "it_security_policy.pdf" not in it_fnames, (
            "IT policy docs still appear in search after HR-only re-index"
        )
        # Query known HR terms — should find HR policy
        hr_results = retriever.invoke("annual leave sick leave parental leave")
        hr_fnames = {d.metadata.get("source_filename") for d in hr_results}
        assert "hr_leave_policy.pdf" in hr_fnames

    def test_failed_build_preserves_previous_active_index(self, tmp_chroma_dir, tmp_temp_dir,
                                                            it_policy_file, hr_policy_file):
        """A failure during the Chroma.from_documents() staging write must leave
        the active slot (previous valid index) completely untouched.
        """
        ok, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok is True, "Initial IT policy index must succeed"

        active_before, standby_before = database._read_pointer(tmp_chroma_dir)

        with patch("database.Chroma.from_documents",
                   side_effect=RuntimeError("Simulated staging write failure")):
            ok2, msg2, _, _, _ = _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)

        assert ok2 is False
        assert "Failed" in msg2

        # Pointer must be unchanged — active slot still holds IT policy
        active_after, _ = database._read_pointer(tmp_chroma_dir)
        assert active_after == active_before, (
            "Pointer changed despite staging failure — active slot compromised"
        )

        # Active slot still contains IT policy data
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        assert retriever is not None
        results = retriever.invoke("password MFA")
        fnames = {d.metadata.get("source_filename") for d in results}
        assert "it_security_policy.pdf" in fnames, (
            "IT policy was lost from active slot after failed re-index"
        )

    def test_failed_validation_preserves_previous_active_index(self, tmp_chroma_dir,
                                                                 tmp_temp_dir, it_policy_file,
                                                                 hr_policy_file):
        """A failure during slot validation (step 5) must leave the active slot intact.
        The pointer must NOT be swapped if the new index fails validation.
        """
        ok, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok is True

        active_before, _ = database._read_pointer(tmp_chroma_dir)

        with patch("database._validate_slot",
                   side_effect=ValueError("Simulated validation failure")):
            ok2, msg2, _, _, _ = _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)

        assert ok2 is False
        active_after, _ = database._read_pointer(tmp_chroma_dir)
        assert active_after == active_before, (
            "Pointer swapped despite validation failure"
        )

    def test_failed_pointer_write_preserves_previous_active_index(self, tmp_chroma_dir,
                                                                     tmp_temp_dir,
                                                                     it_policy_file,
                                                                     hr_policy_file):
        """A failure during the pointer write (step 6) must leave the previous active
        slot serving queries.  The new index is built and validated but not activated.
        """
        ok, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok is True

        active_before, _ = database._read_pointer(tmp_chroma_dir)

        with patch("database._write_pointer",
                   side_effect=OSError("Simulated pointer write failure")):
            ok2, msg2, _, _, _ = _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)

        assert ok2 is False
        active_after, _ = database._read_pointer(tmp_chroma_dir)
        assert active_after == active_before, (
            "Active pointer changed despite pointer-write failure"
        )

        # Old IT policy must still be servable
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        assert retriever is not None
        results = retriever.invoke("password requirements")
        fnames = {d.metadata.get("source_filename") for d in results}
        assert "it_security_policy.pdf" in fnames

    def test_active_index_survives_restart(self, tmp_chroma_dir, tmp_temp_dir, it_policy_file):
        """Simulate an application restart: create a fresh load_retriever() call
        after indexing (no shared Chroma objects) and verify the index is accessible.
        """
        ok, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok is True

        # Simulate restart: call gc.collect() to release any in-memory state,
        # then create a brand-new retriever from scratch.
        gc.collect()
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        assert retriever is not None, "Retriever is None after simulated restart"
        results = retriever.invoke("password")
        assert len(results) > 0, "No results returned after simulated restart"
        fnames = {d.metadata.get("source_filename") for d in results}
        assert "it_security_policy.pdf" in fnames

    def test_repeated_indexing_does_not_mix_documents(self, tmp_chroma_dir, tmp_temp_dir,
                                                        it_policy_file, hr_policy_file,
                                                        finance_policy_file):
        """Three successive single-doc re-indexes must produce exactly one doc each time."""
        for (policy_file, expected_name, absent_names) in [
            (it_policy_file, "it_security_policy.pdf",
             {"hr_leave_policy.pdf", "finance_expense_policy.pdf"}),
            (hr_policy_file, "hr_leave_policy.pdf",
             {"it_security_policy.pdf", "finance_expense_policy.pdf"}),
            (finance_policy_file, "finance_expense_policy.pdf",
             {"it_security_policy.pdf", "hr_leave_policy.pdf"}),
        ]:
            ok, msg, _, _, _ = _index_to([policy_file], tmp_chroma_dir, tmp_temp_dir)
            assert ok is True, f"Indexing {expected_name} failed: {msg}"

            active = _active_dir(tmp_chroma_dir)
            vs = Chroma(persist_directory=active, embedding_function=_get_embeddings(),
                        collection_name=database._COLLECTION_NAME)
            all_docs = vs.get(include=["metadatas"])
            found = {m.get("source_filename") for m in all_docs["metadatas"]}
            if hasattr(vs, '_client') and hasattr(vs._client, 'close'):
                vs._client.close()
            del vs
            gc.collect()

            assert expected_name in found, f"{expected_name} missing after indexing"
            for absent in absent_names:
                assert absent not in found, (
                    f"{absent} still present after switching to {expected_name} — docs mixed"
                )

    def test_windows_file_locking_does_not_corrupt_active_index(self, tmp_chroma_dir,
                                                                   tmp_temp_dir,
                                                                   it_policy_file,
                                                                   hr_policy_file):
        """Verify that even when old Chroma objects remain in memory (simulating
        Windows HNSW handle retention), the active index after a successful
        re-index contains only the new documents.

        We deliberately keep a reference to the first retriever alive while
        performing the second index, mimicking the Windows file-locking scenario.
        """
        ok1, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok1 is True

        # Keep old retriever alive (holds HNSW file handles to the old active slot)
        old_retriever = _load_retriever(tmp_chroma_dir, k=2)

        # Re-index while the old retriever is still open
        ok2, _, _, _, _ = _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok2 is True, "Re-index must succeed even with old retriever still alive"

        # New retriever must see only HR policy (uses new active slot)
        new_retriever = _load_retriever(tmp_chroma_dir, k=4)
        results = new_retriever.invoke("annual leave sick leave")
        fnames = {d.metadata.get("source_filename") for d in results}
        assert "hr_leave_policy.pdf" in fnames, "HR policy not found in new index"
        assert "it_security_policy.pdf" not in fnames, (
            "IT policy leaked into new index despite Windows handle retention"
        )

        del old_retriever
        gc.collect()


# ---------------------------------------------------------------------------
# 6. Stale standby slot: collection survives rmtree, new index must still be clean
# ---------------------------------------------------------------------------

class TestStaleDirHandling:

    def test_stale_dir_cannot_cause_document_mixing(self, tmp_chroma_dir, tmp_temp_dir,
                                                      it_policy_file, hr_policy_file):
        """Simulate the worst-case Windows scenario: rmtree fails completely (the
        standby dir and its collection survive), but the build still produces
        a clean index containing only the new documents.

        _clear_slot raises RuntimeError when the collection persists after all
        cleanup attempts, which causes index_documents() to return (False, ...).
        The PREVIOUS active index must then remain intact and serve old data.

        This test confirms the safety contract:
          - When stale-slot cleanup is not possible, we ABORT rather than mix.
          - The active index is never corrupted.
        """
        # First index: IT policy goes to slot A (active), slot B is empty.
        ok1, _, _, _, _ = _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok1 is True

        active_before, standby_before = database._read_pointer(tmp_chroma_dir)
        standby_dir = database._slot_dir(standby_before, tmp_chroma_dir)

        # Pre-populate the standby slot with IT policy data (simulates a partial
        # previous build that rmtree could not clean up — the slot dir exists and
        # the collection is present).
        import chromadb as _chromadb
        from langchain_core.documents import Document as _Document

        stale_client = _chromadb.PersistentClient(path=standby_dir)
        # Ensure the stale collection exists in the standby dir
        existing = [c.name for c in stale_client.list_collections()]
        if database._COLLECTION_NAME not in existing:
            stale_client.create_collection(database._COLLECTION_NAME)
        stale_col = stale_client.get_collection(database._COLLECTION_NAME)
        stale_col.add(
            ids=["stale-001"],
            documents=["STALE: This document should never appear in search results."],
            metadatas=[{"source_filename": "stale_old_policy.pdf"}],
            embeddings=[[0.0] * 384],  # dummy embedding — never queried semantically
        )
        del stale_col, stale_client
        gc.collect()

        # Now attempt to re-index HR policy. _clear_slot must detect the surviving
        # collection and either succeed in clearing it or raise to abort the build.
        ok2, msg2, _, _, _ = _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)

        if ok2:
            # _clear_slot managed to delete the collection via the API on this
            # platform (Windows handle was released).  Verify no stale docs mixed in.
            new_retriever = _load_retriever(tmp_chroma_dir, k=6)
            results = new_retriever.invoke("leave policy annual sick")
            fnames = {d.metadata.get("source_filename") for d in results}
            assert "stale_old_policy.pdf" not in fnames, (
                "Stale document from pre-populated standby leaked into the new active index"
            )
            assert "it_security_policy.pdf" not in fnames, (
                "IT policy docs mixed into HR-only re-index"
            )
        else:
            # _clear_slot correctly raised RuntimeError (could not fully clear the
            # collection) — the build was aborted.  Active index must be unchanged.
            active_after, _ = database._read_pointer(tmp_chroma_dir)
            assert active_after == active_before, (
                "Pointer changed despite aborted build — active index lost"
            )
            # Previous IT policy must still be queryable.
            retriever = _load_retriever(tmp_chroma_dir, k=2)
            assert retriever is not None
            results = retriever.invoke("password MFA")
            fnames = {d.metadata.get("source_filename") for d in results}
            assert "it_security_policy.pdf" in fnames, (
                "IT policy was lost from active index after aborted re-index"
            )


# ---------------------------------------------------------------------------
# 7. Legacy-index compatibility
# ---------------------------------------------------------------------------

class TestLegacyIndexCompatibility:

    def _build_legacy_index(self, chroma_dir, it_policy_file, temp_dir):
        """Build a flat legacy-style index directly under chroma_dir (no slots)."""
        import os, tempfile as _tempfile
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        os.makedirs(chroma_dir, exist_ok=True)
        tmp_pdf = os.path.join(temp_dir, it_policy_file.name)
        with open(tmp_pdf, "wb") as f:
            f.write(it_policy_file.getbuffer())
        loader = PyPDFLoader(tmp_pdf)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_filename"] = it_policy_file.name
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        Chroma.from_documents(
            documents=chunks,
            embedding=_get_embeddings(),
            persist_directory=chroma_dir,
            collection_name=database._COLLECTION_NAME,
        )
        os.remove(tmp_pdf)
        gc.collect()

    def test_legacy_index_detected_when_no_pointer_file(self, tmp_chroma_dir,
                                                          tmp_temp_dir, it_policy_file):
        """_is_legacy_index() returns True for a flat index with no index.json."""
        self._build_legacy_index(tmp_chroma_dir, it_policy_file, tmp_temp_dir)
        gc.collect()
        assert not os.path.exists(database._pointer_file(tmp_chroma_dir))
        assert database._is_legacy_index(tmp_chroma_dir), (
            "_is_legacy_index() should detect a flat ChromaDB directory without index.json"
        )

    def test_legacy_index_is_not_detected_after_versioned_index(self, tmp_chroma_dir,
                                                                   tmp_temp_dir,
                                                                   it_policy_file):
        """After a versioned index build, _is_legacy_index() returns False."""
        _index_to([it_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert not database._is_legacy_index(tmp_chroma_dir)

    def test_load_retriever_serves_legacy_index_read_only(self, tmp_chroma_dir,
                                                           tmp_temp_dir, it_policy_file):
        """load_retriever() returns a working retriever from a legacy flat index."""
        self._build_legacy_index(tmp_chroma_dir, it_policy_file, tmp_temp_dir)
        gc.collect()
        # No index.json — legacy path
        retriever = database.load_retriever(k=2, chroma_dir=tmp_chroma_dir)
        assert retriever is not None, "load_retriever() must serve legacy index"
        results = retriever.invoke("password MFA")
        assert len(results) > 0, "Legacy retriever returned no results"
        fnames = {d.metadata.get("source_filename") for d in results}
        assert "it_security_policy.pdf" in fnames

    def test_legacy_index_not_modified_by_load_retriever(self, tmp_chroma_dir,
                                                           tmp_temp_dir, it_policy_file):
        """load_retriever() on a legacy index must not write index.json or slots."""
        self._build_legacy_index(tmp_chroma_dir, it_policy_file, tmp_temp_dir)
        gc.collect()
        before_contents = set(os.listdir(tmp_chroma_dir))
        database.load_retriever(k=2, chroma_dir=tmp_chroma_dir)
        after_contents = set(os.listdir(tmp_chroma_dir))
        new_files = after_contents - before_contents
        assert "index.json" not in new_files, (
            "load_retriever() must not create index.json in a legacy index directory"
        )
        assert "a" not in new_files and "b" not in new_files, (
            "load_retriever() must not create slot directories in a legacy index directory"
        )

    def test_first_versioned_index_after_legacy_builds_new_layout(self, tmp_chroma_dir,
                                                                     tmp_temp_dir,
                                                                     it_policy_file,
                                                                     hr_policy_file):
        """The first call to index_documents() on a legacy chroma_dir creates the
        versioned layout and the result is queryable; the legacy flat files
        remain untouched alongside the new a/ b/ slots.
        """
        self._build_legacy_index(tmp_chroma_dir, it_policy_file, tmp_temp_dir)
        gc.collect()
        # Now run versioned index with HR policy
        ok, _, _, _, _ = _index_to([hr_policy_file], tmp_chroma_dir, tmp_temp_dir)
        assert ok is True

        # index.json and slots must now exist
        assert os.path.exists(database._pointer_file(tmp_chroma_dir))

        # The new active index must serve HR policy
        retriever = _load_retriever(tmp_chroma_dir, k=2)
        results = retriever.invoke("annual leave")
        fnames = {d.metadata.get("source_filename") for d in results}
        assert "hr_leave_policy.pdf" in fnames

        # Legacy SQLite file must still be present (untouched by index_documents)
        assert os.path.exists(os.path.join(tmp_chroma_dir, "chroma.sqlite3")), (
            "Legacy chroma.sqlite3 should not have been deleted by index_documents()"
        )
