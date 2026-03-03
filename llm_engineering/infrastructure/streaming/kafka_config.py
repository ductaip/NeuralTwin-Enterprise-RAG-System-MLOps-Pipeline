from threading import Lock
from typing import ClassVar

from confluent_kafka import Consumer, Producer
from loguru import logger

from llm_engineering.domain.base.patterns import SingletonMeta


class KafkaProducer(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._producer: Producer | None = None
        self._config = {
            "bootstrap.servers": "localhost:9092",
            "client.id": "llm-engineering-producer",
        }

    @property
    def producer(self) -> Producer:
        """
        Lazily initialize the Kafka producer.
        """
        if self._producer is None:
            try:
                self._producer = Producer(self._config)
                logger.info("Kafka Producer initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka Producer: {e!s}")
                raise

        return self._producer

    def produce(self, topic: str, value: str, key: str | None = None, callback=None):
        """
        Produce a message to the specified topic.
        """
        try:
            self.producer.produce(topic, key=key, value=value, callback=callback)
            self.producer.poll(0)  # Trigger any available delivery report callbacks
        except BufferError:
            logger.warning(f"Local producer queue is full ({len(self.producer)} messages awaiting delivery)")
            self.producer.poll(1)  # Wait for some messages to be delivered
            self.producer.produce(topic, key=key, value=value, callback=callback)


class KafkaConsumer(metaclass=SingletonMeta):
    # Depending on use case, we might want different group IDs for different consumers
    # For simplicity in this demo, we'll allow passing config on init, but stick to SingletonMeta
    # If multiple distinct consumers are needed, we should refactor to a Factory pattern or named singletons.
    # For this architecture showcase, we will assume one main consumer configuration or handle re-init if needed.
    
    # Actually, for multiple consumers (DataProcessor, Embedding), Singleton might be too restrictive 
    # if they need different group_ids. 
    # Let's adjust: The Singleton here provides the *connection factory* or shared config, 
    # but individual consumers should probably be instantiated per use-case.
    # However, to stick to the requested "Thread-safe singleton for Kafka producer/consumer" requirement:
    # We will implement a Singleton that manages the *primary* shared resource or configuration,
    # OR we can make a Singleton Factory. 
    # Given the prompt asks for "Thread-safe singleton for Kafka producer/consumer", let's assume one active consumer per process 
    # OR make the class a wrapper that holds multiple consumers if needed.
    
    # A cleaner approach for "Singleton" requirement in this specific context (likely one consumer per container/process):
    # We'll implement it as a standard Singleton assuming 1 consumer per running service.
    
    def __init__(self) -> None:
        self._consumer: Consumer | None = None
        self._config = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "llm-engineering-group",
            "auto.offset.reset": "earliest",
        }

    def initialize(self, group_id: str | None = None):
        if group_id:
            self._config["group.id"] = group_id
            # Force re-initialization if config changes
            if self._consumer:
                self._consumer.close()
                self._consumer = None

    @property
    def consumer(self) -> Consumer:
        if self._consumer is None:
            try:
                self._consumer = Consumer(self._config)
                logger.info(f"Kafka Consumer initialized with group_id: {self._config['group.id']}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka Consumer: {e!s}")
                raise
        return self._consumer

    def subscribe(self, topics: list[str]):
        self.consumer.subscribe(topics)

    def poll(self, timeout: float = 1.0):
        return self.consumer.poll(timeout)

    def close(self):
        if self._consumer:
            self._consumer.close()
