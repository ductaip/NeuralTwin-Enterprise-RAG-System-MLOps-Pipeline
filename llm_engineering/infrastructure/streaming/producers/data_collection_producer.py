import json
import uuid
from datetime import datetime

from loguru import logger

from llm_engineering.infrastructure.streaming.kafka_config import KafkaProducer


class DataCollectionProducer:
    def __init__(self) -> None:
        self._producer = KafkaProducer()
        self._topic = "raw_content_stream"

    def delivery_report(self, err, msg):
        """
        Callback called once for each message produced to indicate delivery result.
        Triggered by poll() or flush().
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def publish(self, content: dict) -> None:
        """
        Publish raw content to the Kafka stream.
        """
        try:
            # Add metadata if missing
            if "id" not in content:
                content["id"] = str(uuid.uuid4())
            if "timestamp" not in content:
                content["timestamp"] = datetime.utcnow().isoformat()
            
            value = json.dumps(content)
            key = content["id"]
            
            logger.info(f"Publishing crawled content {key} to {self._topic}")
            
            self._producer.produce(
                topic=self._topic,
                value=value,
                key=key,
                callback=self.delivery_report
            )
        except Exception as e:
            logger.error(f"Failed to publish content: {e!s}")
            raise
