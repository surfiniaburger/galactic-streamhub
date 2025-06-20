# mongo_memory.py
from pymongo import MongoClient, DESCENDING, TEXT
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging
import os
import certifi # For TLS/SSL connections, especially with MongoDB Atlas
from google.cloud import secretmanager # For GCP Secret Manager

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Or your preferred level

# --- GCP Configuration ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not GCP_PROJECT_ID:
    logger.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Using default 'silver-455021'.")
    GCP_PROJECT_ID = "silver-455021"

# MongoDB Configuration from Secrets/Env
MONGODB_SECRET_ID = "MULTIMODAL_MONGODB_URI"
MONGODB_URI_ENV_VAR = "MONGODB_URI"

DEFAULT_MEMORY_DB_NAME = "adk_agent_memory"
DEFAULT_MEMORY_COLLECTION_NAME = "interaction_history"
DEFAULT_HISTORY_LIMIT = 5

def get_secret(secret_id: str, project_id: str, version_id: str = "latest") -> Optional[str]:
    if not project_id:
        logger.error("GCP_PROJECT_ID is not set. Cannot retrieve secret.")
        return None
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        logger.info(f"Successfully retrieved secret: {secret_id}")
        return secret_value
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_id} from project {project_id}: {e}", exc_info=True)
        return None

def get_mongodb_uri_from_sources() -> Optional[str]:
    mongodb_uri_env = os.environ.get(MONGODB_URI_ENV_VAR)
    if mongodb_uri_env:
        logger.info(f"Using MongoDB URI from environment variable '{MONGODB_URI_ENV_VAR}'.")
        return mongodb_uri_env

    logger.info(f"MongoDB URI not found in environment variable '{MONGODB_URI_ENV_VAR}'. Attempting to retrieve from Secret Manager.")
    mongodb_uri_secret = get_secret(MONGODB_SECRET_ID, GCP_PROJECT_ID)
    if mongodb_uri_secret:
        logger.info(f"Using MongoDB URI from Secret Manager ('{MONGODB_SECRET_ID}').")
        return mongodb_uri_secret

    logger.error(f"MongoDB URI could not be found in environment variable ('{MONGODB_URI_ENV_VAR}') or Secret Manager ('{MONGODB_SECRET_ID}').")
    return None


class MongoMemory:
    def __init__(self, db_name: str, collection_name: str):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        
        actual_mongo_uri = get_mongodb_uri_from_sources()

        if not actual_mongo_uri:
            logger.error("MongoDB URI is not available. MongoMemory will not be functional.")
            return

        try:
            self.client = MongoClient(actual_mongo_uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
            self.client.admin.command('ping')
            logger.info("Pinged your deployment. You successfully connected to MongoDB!")

            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            logger.info(f"Connected to MongoDB and using database: '{db_name}', collection: '{collection_name}'")
            self._ensure_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB at resolved URI or initialize: {e}", exc_info=True)
            self.client = None
            self.db = None
            self.collection = None

    def _ensure_indexes(self):
        # CORRECTED CHECK
        if self.collection is None:
            logger.warning("MongoDB collection not available. Skipping index creation.")
            return
        try:
            self.collection.create_index(
                [("user_id", DESCENDING), ("session_id", DESCENDING), ("timestamp", DESCENDING)],
                name="user_session_timestamp_idx",
                background=True
            )
            self.collection.create_index(
                [("user_input", TEXT), ("agent_response", TEXT)],
                name="interaction_text_idx",
                default_language="english",
                background=True
            )
            logger.info("MongoDB indexes ensured (or creation initiated in background).")
        except Exception as e:
            logger.error(f"Error ensuring MongoDB indexes: {e}", exc_info=True)

    def add_interaction(self, user_id: str, session_id: str, user_input: str, agent_response: str, turn_sequence: int):
        # CORRECTED CHECK
        if self.collection is None:
            logger.error("MongoDB collection not available. Cannot add interaction.")
            return

        interaction = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc),
            "turn_sequence": turn_sequence,
            "user_input": user_input,
            "agent_response": agent_response,
        }
        try:
            self.collection.insert_one(interaction)
            logger.debug(f"Added interaction to MongoDB for user {user_id}, session {session_id}")
        except Exception as e:
            logger.error(f"Error adding interaction to MongoDB: {e}", exc_info=True)

    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        # CORRECTED CHECK
        if self.collection is None:
            logger.error("MongoDB collection not available. Cannot get recent interactions.")
            return []
        try:
            cursor = self.collection.find(
                {"user_id": user_id, "session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit)
            
            interactions = list(cursor)
            interactions.reverse() 
            logger.debug(f"Retrieved {len(interactions)} recent interactions for user {user_id}, session {session_id}")
            return interactions
        except Exception as e:
            logger.error(f"Error getting recent interactions from MongoDB: {e}", exc_info=True)
            return []

    def search_interactions_by_keyword(self, user_id: str, session_id: Optional[str], query: str, limit: int = 3) -> List[Dict[str, Any]]:
        # CORRECTED CHECK
        if self.collection is None:
            logger.warning("MongoDB collection not available. Cannot search interactions by keyword.")
            return []
        try:
            mongo_db_query: Dict[str, Any] = {"user_id": user_id, "$text": {"$search": query}}
            if session_id:
                mongo_db_query["session_id"] = session_id

            cursor = self.collection.find(mongo_db_query, {"score": {"$meta": "textScore"}}) \
                                   .sort([("score", {"$meta": "textScore"})]) \
                                   .limit(limit)
            interactions = list(cursor)
            interactions.reverse()
            logger.debug(f"Keyword search for '{query}' found {len(interactions)} interactions for user {user_id}.")
            return interactions
        except Exception as e:
            logger.error(f"Error searching interactions by keyword: {e}", exc_info=True)
            return []

# Instantiate the memory service
mongo_memory_service = MongoMemory(
    db_name=DEFAULT_MEMORY_DB_NAME,
    collection_name=DEFAULT_MEMORY_COLLECTION_NAME
)