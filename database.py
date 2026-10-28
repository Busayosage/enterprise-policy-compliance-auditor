"""
Versioned ChromaDB indexing with atomic slot-swap.

Directory layout:
    chroma_db/
    ├── index.json   ← {"active": "a", "standby": "b"} — atomically replaced
    ├── a/           ← ChromaDB slot A
    └── b/           ← ChromaDB slot B

Build strategy:
  1. Identify active and standby slots from index.json.
  2. Build new index entirely in the standby slot.
  3. Validate standby contains exactly the intended documents.
  4. Atomically swap the pointer file (os.replace — atomic on both Windows and Linux).
  5. Old active data remains in the (now standby) slot until the next build.

Rollback guarantees:
  - A failure at any step before the pointer swap leaves the active slot untouched.
  - A failure after the pointer swap is not possible because os.replace is atomic.
  - The standby slot is cleared via the chromadb client API before each build,
    which releases HNSW file handles on Windows before any filesystem operations.

Concurrency limitations (documented):
  - Single-writer: two concurrent index_documents() calls may race on the standby
    slot and the pointer file. Streamlit's single-threaded reruns mean this cannot
    happen in the production app. For multi-worker deployments, add a file-lock.
  - Single-reader per slot: load_retriever() reads index.json once per call and
    creates a new Chroma object. A concurrent swap will be picked up on the next
    call to load_retriever(). In-flight queries hold their retriever for their
    lifetime and are unaffected by a concurrent swap.

Restart behaviour:
  - index.json survives restarts. load_retriever() always reads the current pointer.
  - If index.json is absent or corrupt, slot "a" is used as the default active slot.
    If slot "a" is also absent, load_retriever() returns None (no indexed documents).
"""

import gc
import json
import os
import shutil

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DB_DIR = "chroma_db"
TEMP_DIR = "temp"
_COLLECTION_NAME = "policy_docs"
_POINTER_FILE = os.path.join(CHROMA_DB_DIR, "index.json")
_SLOTS = ("a", "b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _slot_dir(slot: str, base: str = CHROMA_DB_DIR) -> str:
    return os.path.join(base, slot)


def _pointer_file(base: str = CHROMA_DB_DIR) -> str:
    return os.path.join(base, "index.json")


def _read_pointer(base: str = CHROMA_DB_DIR) -> tuple[str, str]:
    """Return (active_slot, standby_slot).  Defaults to ("a", "b") if absent."""
    try:
        with open(_pointer_file(base), "r") as f:
            data = json.load(f)
        active = data.get("active", "a")
        standby = data.get("standby", "b")
        if active not in _SLOTS or standby not in _SLOTS or active == standby:
            raise ValueError(f"Corrupt index.json: {data}")
        return active, standby
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
        return "a", "b"


def _write_pointer(active: str, standby: str, base: str = CHROMA_DB_DIR) -> None:
    """Atomically replace the pointer file using os.replace (atomic on Windows+Linux)."""
    tmp = _pointer_file(base) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"active": active, "standby": standby}, f)
    os.replace(tmp, _pointer_file(base))


def _clear_slot(slot_dir: str) -> None:
    """Ensure the standby slot contains no stale collection data before a build.

    Strategy (handles Windows HNSW file-locking):
    1. Open a chromadb client and delete the named collection via the API.
       delete_collection() releases HNSW mmap file handles, which shutil.rmtree
       cannot do on Windows (WinError 32).
    2. del client + gc.collect() to drop Python references.
    3. Best-effort rmtree to remove stale SQLite/WAL files.
    4. CRITICAL verification: open a fresh client and confirm the collection is
       absent.  Chroma.from_documents() appends to an existing collection rather
       than replacing it, so a leftover collection would silently mix old and new
       document chunks into the new index.  If the collection still exists after
       all our cleanup efforts, raise RuntimeError rather than proceed with
       a dirty standby.
    """
    if not os.path.exists(slot_dir):
        return

    # Step 1: delete the collection via client API (releases HNSW handles).
    try:
        client = chromadb.PersistentClient(path=slot_dir)
        names = [c.name for c in client.list_collections()]
        if _COLLECTION_NAME in names:
            client.delete_collection(_COLLECTION_NAME)
        del client
        gc.collect()
    except Exception:
        pass

    # Step 2: best-effort directory removal (not required for correctness).
    shutil.rmtree(slot_dir, ignore_errors=True)

    # Step 3: verify the collection is gone — open a fresh client to confirm.
    # If the directory survived (rmtree failed on locked files) AND the
    # collection still exists, Chroma.from_documents() would append to it.
    if os.path.exists(slot_dir):
        try:
            verify_client = chromadb.PersistentClient(path=slot_dir)
            still_present = [
                c.name for c in verify_client.list_collections()
                if c.name == _COLLECTION_NAME
            ]
            del verify_client
            gc.collect()
            if still_present:
                # One final attempt: delete via a fresh client.
                cleanup_client = chromadb.PersistentClient(path=slot_dir)
                cleanup_client.delete_collection(_COLLECTION_NAME)
                del cleanup_client
                gc.collect()
        except Exception:
            pass

        # Final check: if collection persists, abort — do not build on stale data.
        try:
            final_client = chromadb.PersistentClient(path=slot_dir)
            remaining = [
                c.name for c in final_client.list_collections()
                if c.name == _COLLECTION_NAME
            ]
            del final_client
            gc.collect()
        except Exception:
            remaining = []
        if remaining:
            raise RuntimeError(
                f"Cannot clear standby slot {slot_dir!r}: collection "
                f"{_COLLECTION_NAME!r} still present after cleanup. "
                "Windows HNSW file handles may still be open. "
                "Aborting build to prevent stale document mixing."
            )


def _validate_slot(slot_dir: str, expected_filenames: list[str]) -> None:
    """Raise ValueError if the standby slot does not exactly match expected_filenames.

    Checks both directions:
    - Every expected filename must appear in at least one chunk's metadata.
    - No unexpected filenames (from stale data) may appear.

    Raising on unexpected filenames is the safety net for the case where
    _clear_slot could not fully remove a previous collection: Chroma.from_documents
    appends to an existing collection, so stale docs would silently mix in.
    """
    vs = Chroma(
        persist_directory=slot_dir,
        embedding_function=get_embeddings(),
        collection_name=_COLLECTION_NAME,
    )
    result = vs.get(include=["metadatas"])
    del vs
    gc.collect()

    metadatas = result.get("metadatas", [])
    if not metadatas:
        raise ValueError("Validation failed: new index contains no chunks.")

    found = {m.get("source_filename") for m in metadatas}
    expected_set = set(expected_filenames)

    missing = expected_set - found
    if missing:
        raise ValueError(
            f"Validation failed: new index is missing documents: {missing}"
        )

    unexpected = found - expected_set
    if unexpected:
        raise ValueError(
            f"Validation failed: new index contains unexpected (stale) documents: "
            f"{unexpected}. The standby slot was not fully cleared before the build."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_documents(
    uploaded_files,
    chroma_dir: str = CHROMA_DB_DIR,
    temp_dir: str = TEMP_DIR,
):
    """Index PDFs using a versioned slot-swap strategy.

    Steps:
      1. Load PDFs into memory from uploaded_files.
      2. Identify the standby slot.
      3. Clear the standby slot via chromadb client API (releases handles).
      4. Build new index in standby slot.
      5. Validate standby contains exactly the expected documents.
      6. Atomically swap pointer: standby becomes active.

    The active slot is never touched until step 6, so any failure in steps 1–5
    leaves the previous active index intact.
    """
    if not uploaded_files:
        return False, "No files provided.", 0, 0, []

    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # ---- Step 1: load PDFs ----
        all_chunks = []
        filenames = []

        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_filename"] = uploaded_file.name

            all_chunks.extend(documents)
            filenames.append(uploaded_file.name)
            os.remove(temp_path)

        if not all_chunks:
            return False, "The uploaded PDFs appear to be empty or unreadable.", 0, 0, filenames

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(all_chunks)

        # ---- Step 2: identify standby slot ----
        active_slot, standby_slot = _read_pointer(chroma_dir)
        standby_dir = _slot_dir(standby_slot, chroma_dir)

        # ---- Step 3: clear standby (releases HNSW handles via chromadb API) ----
        _clear_slot(standby_dir)

        # ---- Step 4: build new index in standby slot ----
        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=standby_dir,
            collection_name=_COLLECTION_NAME,
        )

        # ---- Step 5: validate standby ----
        _validate_slot(standby_dir, filenames)

        # ---- Step 6: atomic pointer swap ----
        _write_pointer(standby_slot, active_slot, chroma_dir)

        return True, "Indexing complete.", len(filenames), len(chunks), filenames

    except Exception as e:
        return False, f"Failed to process PDFs: {e}", 0, 0, []


def _is_legacy_index(chroma_dir: str) -> bool:
    """Return True if chroma_dir contains a legacy flat index (pre-versioned layout).

    A legacy index stores ChromaDB files directly under chroma_dir/ with no
    index.json and no a/ or b/ subdirectories.  We detect it by the presence of
    ChromaDB's SQLite database file alongside the absence of index.json.
    """
    if not os.path.exists(chroma_dir):
        return False
    pointer = _pointer_file(chroma_dir)
    if os.path.exists(pointer):
        return False  # versioned layout already in place
    # ChromaDB 1.x stores its SQLite database as 'chroma.sqlite3'
    sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")
    return os.path.exists(sqlite_path)


def load_retriever(k: int = 3, chroma_dir: str = CHROMA_DB_DIR):
    """Return a retriever backed by the current active slot.

    Legacy-index compatibility: if chroma_dir contains a pre-versioned flat
    index (ChromaDB files directly under chroma_dir/ with no index.json), the
    legacy directory is served read-only.  The first call to index_documents()
    will migrate to the versioned layout; until then existing data is preserved.

    Reads index.json on every call so a concurrent index swap is picked up
    immediately for the next query.
    """
    if _is_legacy_index(chroma_dir):
        # Serve the legacy flat index read-only without modifying it.
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(),
            collection_name=_COLLECTION_NAME,
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})

    active_slot, _ = _read_pointer(chroma_dir)
    active_dir = _slot_dir(active_slot, chroma_dir)

    if not os.path.exists(active_dir):
        return None

    vectorstore = Chroma(
        persist_directory=active_dir,
        embedding_function=get_embeddings(),
        collection_name=_COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
