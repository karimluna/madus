"""Optional Databricks Delta Lake _sink_. Only activates
when DATABRICKS_HOST is set. Delta Lake gives ACID trans-
actions and time travel so you can audit what the system
knew at any point in time"""

import logging

from core.models import DocumentState
from core.config import get_settings

logger = logging.getLogger(__name__)


def write_to_kb(state: DocumentState) -> None:
    """Write analysis result to Databricks knowledge base.
    no-op if Databricks are not configured"""
    s = get_settings()
    if not s.databricks_host:
        return

    try:
        from databricks import sql

        with sql.connect(
            server_hostname=s.databricks_host,
            http_path=s.databricks_http_path,
            access_token=s.databricks_token,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS madus.knowledge_base (
                            doc_id STRING, 
                            question STRING,
                            final_answer STRING,
                            confidence DOUBLE,
                            created_at TIMESTAMP, 
                    ) USING DELTA
                """)
                cur.execute(
                    """
                            INSERT INTO madus.knowledge_base (
                                doc_id, question, final_answer, confidence, created_at
                            ) VALUES (?, ?, ?, ?, current_timestamp())
                """,
                    [
                        state.doc_id,
                        state.question,
                        state.final_answer,
                        state.confidence,
                    ],
                )
        logger.info("Wrote doc %s to Databricks KB", state.doc_id)
    except Exception as e:
        logger.warning("Databricks write failed: %s", e)
