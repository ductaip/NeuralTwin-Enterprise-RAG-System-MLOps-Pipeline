from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llm_engineering.domain.base.patterns import SingletonMeta
from llm_engineering.settings import settings


class MongoDatabaseConnector(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._host = settings.DATABASE_HOST

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            try:
                self._client = MongoClient(self._host)
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")

                raise

            logger.info(f"Connection to MongoDB with URI successful: {self._host}")

        return self._client

    def __call__(self) -> MongoClient:
        return self.client


connection = MongoDatabaseConnector().client
