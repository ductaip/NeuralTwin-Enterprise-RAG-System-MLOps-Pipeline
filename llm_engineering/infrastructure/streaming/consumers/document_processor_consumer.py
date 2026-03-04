import json
from loguru import logger

from llm_engineering.application.preprocessing.operations.cleaning import clean_text
from llm_engineering.infrastructure.streaming.kafka_config import KafkaConsumer, KafkaProducer


class DocumentProcessorConsumer:
    def __init__(self) -> None:
        self._consumer = KafkaConsumer()
        self._consumer.initialize(group_id="document_processor_group")
        self._consumer.subscribe(["raw_content_stream"])
        
        # We also need a producer to forward processed data to the next stage
        self._producer = KafkaProducer()
        self._next_topic = "embedding_requests"

    def start(self):
        logger.info("Starting Document Processor Consumer...")
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
                    self.process_document(data)
                except Exception as e:
                    logger.error(f"Failed to process message: {e!s}")

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self._consumer.close()

    def process_document(self, data: dict):
        """
        Clean text and forward to embedding stream.
        In a real app, this would also save to MongoDB here.
        """
        logger.info(f"Processing document {data.get('id', 'unknown')}")
        
        # 1. Clean Text
        raw_text = data.get("text", "")
        cleaned_text = clean_text(raw_text)
        
        # 2. Structure Processed Data
        processed_data = {
            "id": data.get("id"),
            "original_id": data.get("id"),
            "content": cleaned_text,
            "metadata": data.get("metadata", {}),
            "status": "processed"
        }
        
        # 3. Publish to Embedding Request Stream
        key = processed_data["id"]
        value = json.dumps(processed_data)
        
        logger.info(f"Forwarding processed document to {self._next_topic}")
        self._producer.produce(
            topic=self._next_topic,
            value=value,
            key=key
        )

if __name__ == "__main__":
    consumer = DocumentProcessorConsumer()
    consumer.start()
