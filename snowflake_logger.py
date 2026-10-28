import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

TABLE_FQN = "RAG_COMPLIANCE_DB.AUDIT_LOGS.RAG_QUERY_LOGS"

SNOWFLAKE_KEYS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_ROLE",
]

# Check once at import time whether the connector package is installed.
# All functions check this flag before attempting a connection so that
# a missing package produces a clear message rather than an ImportError.
try:
    import snowflake.connector as _sf_connector
    _CONNECTOR_AVAILABLE = True
except ImportError:
    _sf_connector = None
    _CONNECTOR_AVAILABLE = False


def is_snowflake_configured() -> bool:
    """Return True only if both the connector package and all credentials are present."""
    if not _CONNECTOR_AVAILABLE:
        return False
    return all(os.getenv(k) for k in SNOWFLAKE_KEYS)


def _get_connection():
    """Open and return a Snowflake connection with autocommit enabled.

    Raises RuntimeError with a human-readable message if the connector
    package is absent or a connection cannot be established.
    """
    if not _CONNECTOR_AVAILABLE:
        raise RuntimeError(
            "snowflake-connector-python is not installed. "
            "Run: pip install snowflake-connector-python"
        )
    missing = [k for k in SNOWFLAKE_KEYS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing Snowflake credentials: {', '.join(missing)}"
        )
    try:
        return _sf_connector.connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            autocommit=True,
        )
    except Exception as e:
        raise RuntimeError(f"Snowflake connection failed: {e}") from e


def test_snowflake_connection() -> tuple[bool, str]:
    """Return (True, message) on success or (False, safe_message) on any failure."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_TIMESTAMP()")
        cur.close()
        conn.close()
        return True, "Snowflake connection successful."
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Snowflake connection check failed: {e}"


def create_logs_table_if_not_exists() -> tuple[bool, str]:
    """Create RAG_QUERY_LOGS if it does not exist. Returns (success, message)."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_FQN} (
                LOG_ID STRING,
                CREATED_AT TIMESTAMP_NTZ,
                DOCUMENT_NAMES STRING,
                USER_QUESTION STRING,
                AI_ANSWER STRING,
                SOURCE_FILENAMES STRING,
                SOURCE_PAGES STRING,
                EVIDENCE_FOUND BOOLEAN,
                MODEL_NAME STRING,
                RESPONSE_TIME_SECONDS FLOAT
            )
        """)
        cur.close()
        conn.close()
        return True, "Logs table ready."
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Snowflake table creation failed: {e}"


def log_query_to_snowflake(
    document_names: str,
    user_question: str,
    ai_answer: str,
    source_filenames: str,
    source_pages: str,
    evidence_found: bool,
    model_name: str = "",
    response_time_seconds: float = 0.0,
) -> tuple[bool, str]:
    """Insert one audit row. Returns (True, message) on success or (False, safe_message)."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {TABLE_FQN} (
                LOG_ID, CREATED_AT, DOCUMENT_NAMES, USER_QUESTION, AI_ANSWER,
                SOURCE_FILENAMES, SOURCE_PAGES, EVIDENCE_FOUND,
                MODEL_NAME, RESPONSE_TIME_SECONDS
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).replace(tzinfo=None),
                document_names,
                user_question,
                ai_answer,
                source_filenames,
                source_pages,
                evidence_found,
                model_name,
                float(response_time_seconds),
            ),
        )
        cur.close()
        conn.close()
        return True, "Query logged to Snowflake."
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Snowflake logging failed: {e}"
