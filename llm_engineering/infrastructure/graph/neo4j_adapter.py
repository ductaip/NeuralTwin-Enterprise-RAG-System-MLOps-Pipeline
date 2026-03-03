from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase
from loguru import logger
from llm_engineering.domain.base.patterns import SingletonMeta

class Neo4jAdapter(metaclass=SingletonMeta):
    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "password")):
        self._driver = None
        self._uri = uri
        self._auth = auth
        self.connect()

    def connect(self):
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(self._uri, auth=self._auth)
                self.verify_connectivity()
                logger.info("Connected to Neo4j successfully.")
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

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._driver:
            self.connect()
        
        try:
            records, summary, keys = self._driver.execute_query(query, parameters, database_="neo4j")
            return [record.data() for record in records]
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
