# In your callbacks.py
# CORRECTED IMPORT
from google.adk.agents.callback_context import CallbackContext # Only import the general context
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part           # For LLM interaction types
from typing import Optional, List, Any
import logging
import re # For keyword matching

# Assuming mongo_memory_service is initialized in mongo_memory.py and imported here
# from .mongo_memory import mongo_memory_service, HISTORY_LIMIT, MongoMemory 
# For simplicity, if mongo_memory.py is in the same directory:
from mongo_memory import mongo_memory_service, DEFAULT_HISTORY_LIMIT # Make sure MongoMemory class is also available if needed

logger = logging.getLogger(__name__)


HISTORY_LIMIT = 5

# --- Callback to SAVE interaction AFTER the agent processes a turn ---
async def save_interaction_after_model_callback( # Changed to async def
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """
    Saves the user's input and the agent's final response to MongoDB.
    This runs *after* the model generates a response.
    """
    # Add an unmissable print statement for debugging. This will print to your server's console.
    if not mongo_memory_service or mongo_memory_service.collection is None:
        logger.warning("MongoMemoryService not available in save_interaction_callback. Skipping save.")
        return None

    try:
        # Access the session through the internal invocation_context
        user_id = callback_context._invocation_context.session.user_id
        session_id = callback_context._invocation_context.session.id
        logger.info(f"Save Callback - User: {user_id}, Session: {session_id}")

        # --- Get the last user input from the session events ---
        last_user_input_text = "User input not found"
        for event in reversed(callback_context._invocation_context.session.events):
            if event.author == "user" and event.content and event.content.parts:
                last_user_input_text = event.content.parts[0].text or "User provided non-text input" # Access text property
                break
        logger.info(f"Save Callback - Found User Input: '{last_user_input_text[:100]}'")

        # --- Get the Agent's final response from the session events ---
        agent_final_response_text = "Agent response not found"
        # Get the agent's response directly from the llm_response object.
        if llm_response.content and llm_response.content.parts:
            agent_final_response_text = llm_response.content.parts[0].text or "Agent provided non-text response"
        logger.info(f"Save Callback - Found Agent Response: '{agent_final_response_text[:100]}'") # Access text property

        # Correctly get current turn count for the sequence
        current_turn_count_in_db = 0
        if mongo_memory_service.collection is not None: # Check again before using
            current_turn_count_in_db = mongo_memory_service.collection.count_documents(
                {"user_id": user_id, "session_id": session_id}
            )
        turn_sequence = current_turn_count_in_db + 1

        logger.info(f"[Callback: AfterAgent] SAVING - User: '{last_user_input_text[:50]}...', Agent: '{agent_final_response_text[:50]}...' for user {user_id}, session {session_id}, turn {turn_sequence}")
        
        mongo_memory_service.add_interaction(
            user_id=user_id,
            session_id=session_id,
            user_input=last_user_input_text,
            agent_response=agent_final_response_text,
            turn_sequence=turn_sequence
        )

    except Exception as e:
        logger.error(f"Error in save_interaction_after_agent_callback: {e}", exc_info=True)
    
    # Return None to indicate we are not modifying the LLM's response.
    return None

# --- Callback to LOAD memory BEFORE the model is called ---
async def load_memory_before_model_callback( # Changed to async def
    # Type hint uses the general CallbackContext
    callback_context: CallbackContext, 
    llm_request: LlmRequest # ADK provides this specific arg for before_model
) -> Optional[LlmResponse]:
    """
    Retrieves recent interactions from MongoDB and prepends them to the LLM request.
    """
    if not mongo_memory_service or mongo_memory_service.collection is None:
        logger.warning("MongoMemoryService not available in load_memory_before_model_callback. Skipping memory load.")
        return None 

    try:
        user_id = callback_context._invocation_context.session.user_id
        session_id = callback_context._invocation_context.session.id
        
        # Get the current user query from llm_request.contents
        current_user_query_text = ""
        if llm_request.contents and llm_request.contents[-1].role == "user":
            # Assuming the last content is the current user's message
            for part in llm_request.contents[-1].parts:
                if part.text:
                    current_user_query_text = part.text
                    break

        # Heuristic to decide between recent history and hybrid search
        # Trigger hybrid search if query is long, or contains specific keywords
        trigger_hybrid_search = False
        if current_user_query_text:
            # Keywords that might indicate a need for deeper memory recall
            memory_keywords = ["remember", "recall", "what did i say", "last time", "previously", "my favorite"]
            if len(current_user_query_text.split()) > 10 or any(kw in current_user_query_text.lower() for kw in memory_keywords):
                trigger_hybrid_search = True

        if trigger_hybrid_search:
            logger.info(f"[Callback: BeforeModel] Triggering vector search for query: '{current_user_query_text[:50]}...'")
            recent_interactions = await mongo_memory_service.vector_search_interactions(user_id, session_id, current_user_query_text, limit=DEFAULT_HISTORY_LIMIT)
        else:
            logger.info(f"[Callback: BeforeModel] Loading recent interactions (limit={DEFAULT_HISTORY_LIMIT}) for user {user_id}, session {session_id}")
            recent_interactions = mongo_memory_service.get_recent_interactions(user_id, session_id, limit=DEFAULT_HISTORY_LIMIT)
        
        if recent_interactions:
            logger.info(f"[Callback: BeforeModel] Loaded {len(recent_interactions)} past interactions into prompt.")
            
            history_contents: List[Content] = [] # Changed from history_parts to history_contents for clarity
            for interaction in recent_interactions:
                if interaction.get("user_input"):
                    history_contents.append(Content(role="user", parts=[Part.from_text(text=interaction["user_input"])]))
                if interaction.get("agent_response"):
                    history_contents.append(Content(role="model", parts=[Part.from_text(text=interaction["agent_response"])]))
            
            # Prepend the loaded history to the current request's contents.
            if llm_request.contents:
                llm_request.contents = history_contents + llm_request.contents
            else:
                # This case is unlikely if a user just sent a message, but for safety:
                llm_request.contents = history_contents
        else:
            logger.info(f"[Callback: BeforeModel] No relevant past interactions found for user {user_id}, session {session_id}.")
    except Exception as e:
        logger.error(f"Error in load_memory_before_model_callback: {e}", exc_info=True)

    return None