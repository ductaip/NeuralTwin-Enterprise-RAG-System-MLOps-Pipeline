from loguru import logger
from llm_engineering.application.graph.ingestor import GraphIngestor

def main():
    logger.info("Starting Graph Ingestion from Mock Data...")
    ingestor = GraphIngestor(mock=True)
    
    # Mock documents to ingest
    documents = [
        {"id": "doc1", "content": "Our system uses JWT for Authentication in the FastAPI backend."},
        {"id": "doc2", "content": "FastAPI supports async operations and is a modern Python web framework."},
        {"id": "doc3", "content": "Kafka handles event streaming between microservices."},
        {"id": "doc4", "content": "Redis is used for rate limiting and caching."},
    ]
    
    for doc in documents:
        ingestor.ingest(doc["id"], doc["content"])
        
    logger.success("Graph Ingestion Completed Successfully.")

if __name__ == "__main__":
    main()
