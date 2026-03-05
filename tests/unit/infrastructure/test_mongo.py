from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import ConnectionFailure

from llm_engineering.domain.base.patterns import SingletonMeta
from llm_engineering.infrastructure.db.mongo import MongoDatabaseConnector


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset Singleton instances between tests to avoid state leakage."""
    SingletonMeta._instances = {}
    yield


@pytest.fixture
def mock_settings():
    with patch("llm_engineering.infrastructure.db.mongo.settings") as mock:
        mock.DATABASE_HOST = "mongodb://localhost:27017"
        yield mock


@pytest.fixture
def mock_mongo_client():
    with patch("llm_engineering.infrastructure.db.mongo.MongoClient") as mock:
        yield mock


def test_singleton_pattern(mock_settings, mock_mongo_client):
    """Test that MongoDatabaseConnector follows Singleton pattern."""
    connector1 = MongoDatabaseConnector()
    connector2 = MongoDatabaseConnector()
    
    assert connector1 is connector2


def test_connection_success(mock_settings, mock_mongo_client):
    """Test successful database connection."""
    connector = MongoDatabaseConnector()
    # Reset client to force reconnection
    connector._client = None
    
    client = connector.client
    
    assert client is mock_mongo_client.return_value
    mock_mongo_client.assert_called_with("mongodb://localhost:27017")


def test_connection_failure(mock_settings, mock_mongo_client):
    """Test database connection failure."""
    mock_mongo_client.side_effect = ConnectionFailure("Connection failed")
    connector = MongoDatabaseConnector()
    connector._client = None
    
    with pytest.raises(ConnectionFailure):
        _ = connector.client
