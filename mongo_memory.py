# mongo_memory.py
from pymongo import MongoClient, DESCENDING, TEXT
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging
import os
import certifi # For TLS/SSL connections, especially with MongoDB Atlas
from pymongo.operations import SearchIndexModel # Import for Atlas Search Index
import vertexai # For Vertex AI initialization
from vertexai.language_models import TextEmbeddingModel # For text embeddings
from google.adk.memory.base_memory_service import BaseMemoryService
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from google.adk.sessions.session import Session # Import Session type
from google.cloud import secretmanager # For GCP Secret Manager
from typing import AsyncGenerator

# Configure logging for this module
from google.api_core.exceptions import ServiceUnavailable, InternalServerError, Aborted, DeadlineExceeded
logger = logging.getLogger(__name__)
logger.info("--- MongoMemory module loaded (Version 2025-06-24) ---") # Added for debugging
logger.setLevel(logging.INFO) # Or your preferred level

# --- GCP Configuration ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1") # Define GCP_LOCATION
if not GCP_PROJECT_ID:
    logger.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Using default 'silver-455021'.")
    GCP_PROJECT_ID = "silver-455021"

# MongoDB Configuration from Secrets/Env
MONGODB_SECRET_ID = "MULTIMODAL_MONGODB_URI"
MONGODB_URI_ENV_VAR = "MONGODB_URI"

DEFAULT_MEMORY_DB_NAME = "adk_agent_memory" # Consider making this configurable if needed
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
    """Retrieves the MongoDB URI from environment variables or GCP Secret Manager."""
    if mongodb_uri := os.environ.get(MONGODB_URI_ENV_VAR):
        logger.info(f"Using MongoDB URI from environment variable '{MONGODB_URI_ENV_VAR}'.")
        return mongodb_uri

    logger.info(f"MongoDB URI not found in environment. Attempting to retrieve from Secret Manager.")
    if mongodb_uri := get_secret(MONGODB_SECRET_ID, GCP_PROJECT_ID):
        logger.info(f"Using MongoDB URI from Secret Manager ('{MONGODB_SECRET_ID}').")
        return mongodb_uri

    logger.error(f"MongoDB URI not found in environment variable or Secret Manager.")
    return None


class MongoMemory(BaseMemoryService):
    """A memory service that uses MongoDB as a backend."""

    def __init__(self, db_name: str):
        super().__init__()
        self.client: Optional[MongoClient] = None
        self.db = None
        self.embedding_model = None
        self._initialize(db_name)

    def _initialize(self, db_name: str):
        """Initializes the MongoDB connection and collections."""
        if not (actual_mongo_uri := get_mongodb_uri_from_sources()):
            logger.error("MongoDB URI is not available. MongoMemory will not be functional.")
            return

        try:
            self.client = MongoClient(actual_mongo_uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB.")

            self.db = self.client[db_name]
            self.interaction_history = self.db["interaction_history"]
            self.personas = self.db["personas"]
            self.session_summaries = self.db["session_summaries"]
            self.toolbox = self.db["toolbox"]
            self.workflows = self.db["workflows"]
            self.collection = self.interaction_history  # For backward compatibility

            self._initialize_embedding_model()
            self._ensure_all_indexes()
        except Exception as e:
            logger.error(f"Failed to initialize MongoMemory: {e}", exc_info=True)
            self.client = None
            self.db = None

    def _initialize_embedding_model(self):
        """Initializes the Vertex AI TextEmbeddingModel."""
        try:
            vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            logger.info("Vertex AI TextEmbeddingModel initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI TextEmbeddingModel: {e}", exc_info=True)

    def _ensure_all_indexes(self):
        # CORRECTED CHECK
        if self.interaction_history is None:
            logger.warning("MongoDB collection not available. Skipping index creation.")
            return
        try:
            # Index for interaction history
            self.interaction_history.create_index(
                [("user_id", DESCENDING), ("session_id", DESCENDING), ("timestamp", DESCENDING)],
                name="user_session_timestamp_idx",
                background=True
            )
            self.interaction_history.create_index(
                [("user_input", TEXT), ("agent_response", TEXT)],
                name="interaction_text_idx",
                default_language="english",
                background=True
            )

            # Index for personas (unique per user)
            self.personas.create_index("user_id", unique=True, background=True)

            # Index for toolbox (for semantic search on tool descriptions)
            # This would be an Atlas Vector Search index similar to your interaction_history
            # For simplicity, we'll just log it here.

            logger.info("MongoDB indexes for all memory types ensured.")
        except Exception as e:
            logger.error(f"Error ensuring MongoDB indexes: {e}", exc_info=True)

        # Check and create Atlas Search Index for hybrid search
        atlas_search_index_name = "default" # As used in hybrid_search_interactions
        try:
            existing_search_indexes = list(self.interaction_history.list_search_indexes())
            if any(idx.get('name') == atlas_search_index_name for idx in existing_search_indexes):
                logger.info(f"Atlas Search Index '{atlas_search_index_name}' already exists. Skipping creation.")
            else:
                logger.info(f"Atlas Search Index '{atlas_search_index_name}' not found. Creating it...")
                index_definition = {
                  "mappings": {
                    "dynamic": True, # Set to false if you want a strict schema, but dynamic is more flexible
                    "fields": {
                      "user_input": {
                        "type": "string"
                      },
                      "user_id": {
                        "type": "token"
                      },
                      "session_id": {
                        "type": "token"
                      },
                      "agent_response": {
                        "type": "string"
                      },
                      "embedding": {
                        "type": "knnVector",
                        "dimensions": 768, # Must match the output dimension of text-embedding-005
                        "similarity": "cosine" # Or dotProduct / euclidean
                      }
                    }
                  },
                  "name": atlas_search_index_name
                }
                search_index_model = SearchIndexModel(definition=index_definition, name=atlas_search_index_name)
                self.interaction_history.create_search_index(model=search_index_model)
                logger.info(f"Successfully initiated creation of Atlas Search Index '{atlas_search_index_name}'. It may take a few minutes to become active.")
        except Exception as e:
            logger.error(f"Error ensuring Atlas Search Index '{atlas_search_index_name}': {e}", exc_info=True)

    # --- NEW: Persona Memory Functions ---
    def get_persona(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the persona document for a given user."""
        if not user_id:
            logger.warning("get_persona called with an empty user_id.")
            return None
        try:
            logger.info(f"Retrieving persona for user_id: {user_id}")
            persona = self.personas.find_one({"user_id": user_id})
            if persona:
                persona["_id"] = str(persona["_id"])  # For JSON serialization
                return persona
            return None
        except Exception as e:
            logger.error(f"Error retrieving persona for user {user_id}: {e}", exc_info=True)
            return None

    def create_or_update_persona(self, user_id: str, persona_data: Dict[str, Any]):
        """Creates or updates a user's persona."""
        if not user_id:
            logger.warning("create_or_update_persona called with an empty user_id.")
            return
        try:
            logger.info(f"Updating persona for user_id: {user_id}")
            self.personas.update_one(
                {"user_id": user_id},
                {"$set": persona_data, "$currentDate": {"last_updated": True}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving persona for user {user_id}: {e}", exc_info=True)

    def save_session_summary(self, user_id: str, session_id: str, summary: str):
        """Saves the summary of a session."""
        if not user_id or not session_id:
            logger.warning("save_session_summary called with an empty user_id or session_id.")
            return
        try:
            logger.info(f"Saving session summary for session_id: {session_id}")
            summary_data = {
                "user_id": user_id,
                "session_id": session_id,
                "summary": summary,
                "timestamp": datetime.now(timezone.utc),
            }
            self.session_summaries.insert_one(summary_data)
        except Exception as e:
            logger.error(f"Error saving session summary for session {session_id}: {e}", exc_info=True)

    async def add_session_to_memory(self, session: Session) -> None:
        """Ingests a session's events into long-term memory."""
        logger.info(f"Adding session {session.id} to memory for user {session.user_id}.")
        user_message_content = None
        for event in session.events:
            if event.author == "user" and event.content and event.content.parts:
                user_message_content = event.content.parts[0].text
            elif event.author == "model" and event.content and event.content.parts and user_message_content:
                agent_response_content = event.content.parts[0].text
                self.add_interaction(
                    user_id=session.user_id,
                    session_id=session.id,
                    user_input=user_message_content,
                    agent_response=agent_response_content
                )
                user_message_content = None  # Reset for the next turn
        logger.info(f"Finished adding session {session.id} to memory.")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for a given text."""
        if not self.embedding_model:
            return None
        try:
            embeddings_response = self.embedding_model.get_embeddings([text])
            return embeddings_response[0].values if embeddings_response else None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            return None

    def add_interaction(self, user_id: str, session_id: str, user_input: str, agent_response: str):
        """Adds a user-agent interaction to the database."""
        if not self.interaction_history or not user_id:
            logger.warning("Interaction not added due to missing collection or user_id.")
            return

        try:
            turn_sequence = self.interaction_history.count_documents({"user_id": user_id, "session_id": session_id}) + 1
            combined_text = f"User: {user_input}\nAgent: {agent_response}"
            interaction = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc),
                "turn_sequence": turn_sequence,
                "user_input": user_input,
                "agent_response": agent_response,
                "embedding": self._get_embedding(combined_text),
            }
            self.interaction_history.insert_one(interaction)
            logger.debug(f"Added interaction for user {user_id}, session {session_id}")
        except Exception as e:
            logger.error(f"Error adding interaction to MongoDB: {e}", exc_info=True)

    async def save_memory(self, ctx: InvocationContext, user_id: str, session_id: str, event: Event) -> None:
        """Saves an event to memory."""
        user_input, agent_response = "", ""
        if event.author == "user" and event.content and event.content.parts:
            user_input = event.content.parts[0].text or ""
        elif event.author == "model" and event.content and event.content.parts:
            agent_response = event.content.parts[0].text or ""
        
        self.add_interaction(user_id, session_id, user_input, agent_response)

    def get_recent_interactions(self, user_id: str, session_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """Retrieves recent interactions for a given user and session."""
        if not self.interaction_history or not user_id or not session_id:
            logger.warning("get_recent_interactions called with missing arguments.")
            return []

        try:
            cursor = self.interaction_history.find(
                {"user_id": user_id, "session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit)
            return list(reversed(list(cursor)))
        except Exception as e:
            logger.error(f"Error getting recent interactions: {e}", exc_info=True)
            return []

    def get_recent_session_summaries(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieves recent session summaries for a given user."""
        if not self.session_summaries or not user_id:
            logger.warning("get_recent_session_summaries called with missing arguments.")
            return []

        try:
            cursor = self.session_summaries.find(
                {"user_id": user_id}
            ).sort("timestamp", DESCENDING).limit(limit)
            return list(reversed(list(cursor)))
        except Exception as e:
            logger.error(f"Error getting recent session summaries: {e}", exc_info=True)
            return []

    async def load_memory(self, ctx: InvocationContext, user_id: str, session_id: str, limit: Optional[int] = None) -> AsyncGenerator[Event, None]:
        """Loads memory and yields it as a stream of events."""
        limit = limit or DEFAULT_HISTORY_LIMIT
        for interaction in self.get_recent_interactions(user_id, session_id, limit):
            if user_input := interaction.get("user_input"):
                yield Event(invocation_id=ctx.invocation_id, author="user", content=Content(parts=[Part(text=user_input)]))
            if agent_response := interaction.get("agent_response"):
                yield Event(invocation_id=ctx.invocation_id, author="model", content=Content(parts=[Part(text=agent_response)]))

    def search_interactions_by_keyword(self, user_id: str, session_id: Optional[str], query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Searches interactions by keyword."""
        if not self.interaction_history or not user_id:
            logger.warning("search_interactions_by_keyword called with missing arguments.")
            return []

        try:
            mongo_query = {"user_id": user_id, "$text": {"$search": query}}
            if session_id:
                mongo_query["session_id"] = session_id

            cursor = self.interaction_history.find(mongo_query, {"score": {"$meta": "textScore"}}) \
                                   .sort([("score", {"$meta": "textScore"})]) \
                                   .limit(limit)
            return list(reversed(list(cursor)))
        except Exception as e:
            logger.error(f"Error searching interactions by keyword: {e}", exc_info=True)
            return []

    async def search_memory(self, ctx: InvocationContext, user_id: str, session_id: str, query: str, limit: Optional[int] = None) -> AsyncGenerator[Event, None]:
        """Searches memory and yields it as a stream of events."""
        limit = limit or 3
        for interaction in self.search_interactions_by_keyword(user_id, session_id, query, limit):
            if user_input := interaction.get("user_input"):
                yield Event(invocation_id=ctx.invocation_id, author="user", content=Content(parts=[Part(text=user_input)]))
            if agent_response := interaction.get("agent_response"):
                yield Event(invocation_id=ctx.invocation_id, author="model", content=Content(parts=[Part(text=agent_response)]))

    async def clear_memory(self, ctx: InvocationContext, user_id: str, session_id: str) -> None:
        """Clears memory for a given user and session."""
        if not self.interaction_history or not user_id or not session_id:
            logger.warning("clear_memory called with missing arguments.")
            return
        try:
            result = self.interaction_history.delete_many({"user_id": user_id, "session_id": session_id})
            logger.info(f"Cleared {result.deleted_count} interactions for session {session_id}.")
        except Exception as e:
            logger.error(f"Error clearing memory for session {session_id}: {e}", exc_info=True)

    async def vector_search_interactions(self, user_id: str, session_id: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Performs a vector search on conversation history."""
        if not self.interaction_history or not user_id:
            logger.warning("vector_search_interactions called with missing arguments.")
            return []
        if not self.embedding_model:
            logger.warning("Embedding model not initialized. Falling back to keyword search.")
            return self.search_interactions_by_keyword(user_id, session_id, query_text, limit)

        try:
            query_embedding = self._get_embedding(query_text)
            if not query_embedding:
                return self.search_interactions_by_keyword(user_id, session_id, query_text, limit)

            pipeline = [
                {"$vectorSearch": {
                    "index": "default", "path": "embedding", "queryVector": query_embedding,
                    "numCandidates": limit + 50, "limit": limit, "filter": {"user_id": user_id}
                }},
                {"$match": {"user_input": {"$not": {"$regex": "^remember our", "$options": "i"}}}},
                {"$project": {
                    "_id": 0, "user_id": 1, "session_id": 1, "timestamp": 1, "turn_sequence": 1,
                    "user_input": 1, "agent_response": 1, "score": {"$meta": "searchScore"}
                }},
                {"$sort": {"score": -1}},
                {"$limit": limit}
            ]
            return list(self.interaction_history.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error performing vector search: {e}", exc_info=True)
            return []

    async def search_persona_and_interactions(self, user_id: str, session_id: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Searches both persona data and interaction history."""
        results = []
        if persona_data := self.get_persona(user_id):
            if "name" in query_text.lower() or "who am i" in query_text.lower():
                results.append({"source": "persona", "data": persona_data})

        enriched_query = f"A conversation about: {query_text}"
        interaction_results = await self.vector_search_interactions(user_id, session_id, enriched_query, limit)
        results.extend([{"source": "interaction", "data": item} for item in interaction_results])

        return results



# Instantiate the memory service
mongo_memory_service = MongoMemory(
    db_name=DEFAULT_MEMORY_DB_NAME
)
