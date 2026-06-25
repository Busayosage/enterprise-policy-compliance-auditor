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


def is_snowflake_configured():
    return all(os.getenv(k) for k in SNOWFLAKE_KEYS)


def _get_connection():
    import snowflake.connector

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        autocommit=True,
    )


def test_snowflake_connection():
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_TIMESTAMP()")
        cur.close()
        conn.close()
        return True, "Snowflake connection successful."
    except Exception as e:
        return False, f"Snowflake connection failed: {e}"


def create_logs_table_if_not_exists():
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
    except Exception as e:
        return False, f"Snowflake table creation failed: {e}"


def log_query_to_snowflake(
    document_names,
    user_question,
    ai_answer,
    source_filenames,
    source_pages,
    evidence_found,
    model_name="",
    response_time_seconds=0.0,
):
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
                response_time_seconds,
            ),
        )
        cur.close()
        conn.close()
        return True, "Query logged to Snowflake."
    except Exception as e:
        return False, f"Snowflake logging failed: {e}"
