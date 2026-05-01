import logging
import time
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Manages low-level Neo4j database operations and connection lifecycle.
    Provides generic query execution methods.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    def __enter__(self):
        """Establishes connection and returns self for use in 'with' statements."""
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._driver.verify_connectivity()  # Verify connection immediately
        logger.info(
            f"Neo4j connection established at {self.uri} with user {self.user}."
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        """Closes the connection when exiting the 'with' block."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def check_connection(self) -> bool:
        """Verifies connectivity to the Neo4j database.

        Opens the driver if not yet open and leaves it open for subsequent
        queries.  Never closes the driver — callers that want lifecycle
        management should use ``Neo4jManager`` as a context manager.
        """
        try:
            if self._driver is None:
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                logger.info(
                    f"Neo4j connection established at {self.uri} with user {self.user}."
                )
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connection check failed: {e}")
            # If the driver is in a bad state, close and clear it so callers may recreate
            try:
                if self._driver:
                    self._driver.close()
            except Exception:
                pass
            self._driver = None
            return False

    def _ensure_driver(self):
        """Ensure there is an active driver and connectivity is verified."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            logger.info(f"Neo4j driver created for {self.uri}.")
            self._driver.verify_connectivity()

    def execute_read_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes a read-only Cypher query and returns a list of result records."""
        # Ensure driver exists
        self._ensure_driver()
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with self._driver.session() as session:
                    result = session.run(cypher, parameters=params)
                    return [record.data() for record in result]
            except DatabaseError as db_err:
                logger.error(
                    f"Neo4j DatabaseError during read query (attempt {attempt}/{max_retries}): {db_err}"
                )
                # If not last attempt, recreate driver and retry with backoff
                if attempt < max_retries:
                    sleep_sec = 1 << (attempt - 1)
                    logger.info(
                        f"Recreating driver and retrying after {sleep_sec}s backoff..."
                    )
                    try:
                        if self._driver:
                            self._driver.close()
                    except Exception:
                        pass
                    self._driver = None
                    try:
                        time.sleep(sleep_sec)
                        self._ensure_driver()
                        continue
                    except Exception:
                        logger.exception(
                            "Failed to recreate Neo4j driver during retry."
                        )
                        break
                else:
                    logger.critical(
                        "Neo4j read query failed after retries. The database may need a restart."
                    )
                    raise

    def execute_write_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Executes a write Cypher query and returns the summary counters."""
        # Ensure driver exists
        self._ensure_driver()
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with self._driver.session() as session:
                    result = session.run(cypher, parameters=params)
                    return result.consume().counters
            except DatabaseError as db_err:
                logger.error(
                    f"Neo4j DatabaseError during write query (attempt {attempt}/{max_retries}): {db_err}."
                )
                if attempt < max_retries:
                    sleep_sec = 1 << (attempt - 1)
                    logger.info(
                        f"Recreating driver and retrying after {sleep_sec}s backoff..."
                    )
                    try:
                        if self._driver:
                            self._driver.close()
                    except Exception:
                        pass
                    self._driver = None
                    try:
                        time.sleep(sleep_sec)
                        self._ensure_driver()
                        continue
                    except Exception:
                        logger.exception(
                            "Failed to recreate Neo4j driver during retry."
                        )
                        break
                else:
                    logger.critical(
                        "Neo4j write query failed after retries. The database may need to be restarted."
                    )
                    raise

    def get_schema(self) -> List[Dict[str, Any]]:
        """Retrieves the current schema of the Neo4j database."""
        with self._driver.session() as session:
            result = session.run("CALL db.schema.visualization()")
            return [record.data() for record in result]
