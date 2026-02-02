from typing import List, Dict
from loguru import logger
from llm_engineering.infrastructure.graph.neo4j_adapter import Neo4jAdapter
from llm_engineering.domain.base.patterns import SingletonMeta

class GraphIngestor(metaclass=SingletonMeta):
    def __init__(self, mock: bool = True):
        self._neo4j = Neo4jAdapter()
        self._mock = mock

    def ingest(self, document_id: str, content: str):
        """
        Extracts entities and relationships from content and stores them in Neo4j.
        """
        logger.info(f"Ingesting document {document_id} into Graph DB...")
        
        entities, relationships = self._extract_knowledge(content)
        
        # Store in Neo4j
        self._store_in_graph(document_id, entities, relationships)
        
        logger.info(f"Graph ingestion complete for {document_id}.")

    def _extract_knowledge(self, content: str):
        if self._mock:
             return self._mock_extraction(content)
        return [], []

    def _mock_extraction(self, content: str):
        # Mock logic to extract tech stack entities
        keywords = ["JWT", "OAuth2", "FastAPI", "Redis", "Kafka", "MongoDB", "Qdrant", "Prometheus", "Grafana", "Neo4j"]
        found_entities = []
        relationships = []
        
        content_lower = content.lower()
        
        for kw in keywords:
            if kw.lower() in content_lower:
                found_entities.append({"name": kw, "type": "Technology"})
        
        # Mock relationships
        if "jwt" in content_lower and "auth" in content_lower:
            relationships.append({"source": "JWT", "target": "Authentication", "type": "USED_FOR"})
        if "fastapi" in content_lower and "api" in content_lower:
             relationships.append({"source": "FastAPI", "target": "API", "type": "IS_A"})
        if "kafka" in content_lower and "event" in content_lower:
             relationships.append({"source": "Kafka", "target": "EventStreaming", "type": "HANDLES"})

        return found_entities, relationships

    def _store_in_graph(self, doc_id: str, entities: List[Dict], relationships: List[Dict]):
        # Create Document Node
        self._neo4j.execute_query(
            "MERGE (d:Document {id: $id})", {"id": doc_id}
        )
        
        # Create Entity Nodes and RELATES_TO_DOCUMENT relationships
        for entity in entities:
            query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.type = $type
            MERGE (d:Document {id: $doc_id})
            MERGE (d)-[:MENTIONS]->(e)
            """
            self._neo4j.execute_query(query, {"name": entity["name"], "type": entity["type"], "doc_id": doc_id})
            
        # Create Entity-Entity Relationships
        for rel in relationships:
            query = f"""
            MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
            MERGE (a)-[:{rel['type']}]->(b)
            """
            # Ensure target exists too (simple mock fix)
            self._neo4j.execute_query(
                "MERGE (e:Entity {name: $name})", {"name": rel["target"]}
            )
            
            self._neo4j.execute_query(query, {"source": rel["source"], "target": rel["target"]})
