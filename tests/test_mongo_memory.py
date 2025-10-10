import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from mongo_memory import MongoMemory

@pytest.fixture(scope="function")
def mongo_memory_instance():
    """Fixture to create a MongoMemory instance with a mocked MongoClient."""
    with patch('mongo_memory.get_mongodb_uri_from_sources', return_value="mongodb://dummy_uri"), \
         patch('mongo_memory.MongoClient') as mock_mongo_client:
        # Mock the client and its attributes
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        # Mock the database and collections
        mock_db = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db

        mock_interaction_history = MagicMock()
        mock_personas = MagicMock()
        mock_session_summaries = MagicMock()

        mock_db.__getitem__.side_effect = lambda name: {
            "interaction_history": mock_interaction_history,
            "personas": mock_personas,
            "session_summaries": mock_session_summaries
        }.get(name)

        # Mock environment variables
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "test-location"

        # Instantiate MongoMemory
        db_name = "test_db"
        memory = MongoMemory(db_name)

        yield memory

def test_initialization(mongo_memory_instance: MongoMemory):
    """Test that the MongoMemory class initializes correctly with mocks."""
    assert mongo_memory_instance.client is not None
    assert mongo_memory_instance.db is not None
    assert mongo_memory_instance.interaction_history is not None
    assert mongo_memory_instance.personas is not None
    assert mongo_memory_instance.session_summaries is not None

def test_add_interaction(mongo_memory_instance: MongoMemory):
    """Test adding an interaction."""
    mongo_memory_instance.add_interaction("user1", "session1", "hello", "hi there")
    mongo_memory_instance.interaction_history.insert_one.assert_called_once()

def test_get_recent_interactions(mongo_memory_instance: MongoMemory):
    """Test retrieving recent interactions."""
    mongo_memory_instance.get_recent_interactions("user1", "session1")
    mongo_memory_instance.interaction_history.find.assert_called_once_with({"user_id": "user1", "session_id": "session1"})

def test_create_or_update_persona(mongo_memory_instance: MongoMemory):
    """Test creating or updating a persona."""
    mongo_memory_instance.create_or_update_persona("user1", {"name": "test"})
    mongo_memory_instance.personas.update_one.assert_called_once_with(
        {"user_id": "user1"},
        {"$set": {"name": "test"}, "$currentDate": {"last_updated": True}},
        upsert=True
    )

def test_get_persona(mongo_memory_instance: MongoMemory):
    """Test retrieving a persona."""
    mongo_memory_instance.get_persona("user1")
    mongo_memory_instance.personas.find_one.assert_called_once_with({"user_id": "user1"})

def test_save_session_summary(mongo_memory_instance: MongoMemory):
    """Test saving a session summary."""
    mongo_memory_instance.save_session_summary("user1", "session1", "summary")
    mongo_memory_instance.session_summaries.insert_one.assert_called_once()

def test_get_recent_session_summaries(mongo_memory_instance: MongoMemory):
    """Test retrieving recent session summaries."""
    mongo_memory_instance.get_recent_session_summaries("user1")
    mongo_memory_instance.session_summaries.find.assert_called_once_with({"user_id": "user1"})
