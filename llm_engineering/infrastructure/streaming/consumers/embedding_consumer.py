import json
import random
from typing import List

from loguru import logger

from llm_engineering.infrastructure.streaming.kafka_config import KafkaConsumer


class EmbeddingConsumer:
    def __init__(self) -> None:
        self._consumer = KafkaConsumer()
        # Different group ID to allow independent scaling/processing from document processor
        self._consumer.initialize(group_id="embedding_consumer_group")
        self._consumer.subscribe(["embedding_requests"])

    def start(self):
        logger.info("Starting Embedding Consumer...")
        try:
            while True:
                msg = self._consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    data = json.loads(msg.value().decode("utf-8"))
                    self.process_embedding(data)
                except Exception as e:
                    logger.error(f"Failed to process message: {e!s}")

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self._consumer.close()

    def mock_embed(self, text: str) -> List[float]:
        """
        Simulate embedding generation.
        Returns a random vector of size 384 (common for smaller models).
        """
        # Simulate processing time
        # time.sleep(0.1) 
        return [random.random() for _ in range(384)]

    def process_embedding(self, data: dict):
        """
        Generate embedding and simulate storage in Qdrant.
        """
        content = data.get("content", "")
        doc_id = data.get("id")
        
        logger.info(f"Generating embedding for doc {doc_id}...")
        
        vector = self.mock_embed(content)
        
        # Simulate Qdrant Upsert
        logger.success(f"Successfully stored vector for {doc_id} in Qdrant (Simulated)")
        logger.debug(f"Vector preview: {vector[:5]}...")


if __name__ == "__main__":
    consumer = EmbeddingConsumer()
    consumer.start()
