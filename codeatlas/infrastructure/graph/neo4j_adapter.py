from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger
from neo4j import GraphDatabase

from codeatlas.domain.base.patterns import SingletonMeta
from codeatlas.settings import settings


class Neo4jAdapter(metaclass=SingletonMeta):
    """Thin wrapper over the Neo4j driver.

    Reads/writes raise on failure. An earlier version swallowed every exception and
    returned an empty list, which made a failed write indistinguishable from a query
    that legitimately matched nothing — idempotent ingestion is impossible on top of
    that, so write failures must surface.
    """

    def __init__(
        self,
        uri: str | None = None,
        auth: tuple[str, str] | None = None,
        database: str | None = None,
    ):
        self._driver = None
        self._uri = uri or settings.NEO4J_URI
        self._auth = auth or (settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        self._database = database or settings.NEO4J_DATABASE
        self.connect()

    def connect(self):
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(self._uri, auth=self._auth)
                self.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self._uri} (database={self._database}).")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    def verify_connectivity(self):
        self._driver.verify_connectivity()

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a read query. Raises on failure."""
        if not self._driver:
            self.connect()

        records, _summary, _keys = self._driver.execute_query(
            query, parameters or {}, database_=self._database
        )
        return [record.data() for record in records]

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Run a write query. Raises on failure and returns the update counters.

        Never swallow the exception here — a silently failed write corrupts the graph in
        a way that only shows up much later, as a missing CALL edge and therefore a test
        that impact analysis forgets to run.
        """
        if not self._driver:
            self.connect()

        _records, summary, _keys = self._driver.execute_query(
            query, parameters or {}, database_=self._database
        )
        counters = summary.counters
        return {
            "nodes_created": counters.nodes_created,
            "relationships_created": counters.relationships_created,
            "properties_set": counters.properties_set,
        }

    def execute_write_batch(
        self,
        query: str,
        rows: Sequence[Dict[str, Any]],
        batch_size: int | None = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """UNWIND `rows` through `query` in batches.

        `query` must consume the parameter `$rows`, e.g.
            UNWIND $rows AS row MERGE (f:Function {qualified_name: row.qualified_name})
        """
        batch_size = batch_size or settings.INGESTION_BATCH_SIZE
        totals = {"nodes_created": 0, "relationships_created": 0, "properties_set": 0}

        for start in range(0, len(rows), batch_size):
            chunk = list(rows[start : start + batch_size])
            params = {"rows": chunk, **(extra_params or {})}
            counters = self.execute_write(query, params)
            for key in totals:
                totals[key] += counters[key]

        return totals

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Deprecated alias for `execute_read`. Kept for existing callers; raises now."""
        return self.execute_read(query, parameters)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def reset_singleton(cls) -> None:
        """Drop the cached singleton so a differently-configured instance can be built.

        Only intended for tests and for the ingestion CLI, which may need to point at a
        non-default Neo4j instance after `settings` has already been read.
        """
        instance = SingletonMeta._instances.pop(cls, None)
        if instance is not None:
            try:
                instance.close()
            except Exception as e:  # pragma: no cover - best effort cleanup
                logger.warning(f"Error closing previous Neo4j singleton: {e}")


def iter_batches(items: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
