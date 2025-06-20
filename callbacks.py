# In your callbacks.py
# CORRECTED IMPORT
from google.adk.agents.callback_context import CallbackContext # Only import the general context
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part           # For LLM interaction types
from typing import Optional, List, Dict, Any # Added for Mongo types
import logging

# Assuming mongo_memory_service is initialized in mongo_memory.py and imported here
# from .mongo_memory import mongo_memory_service, HISTORY_LIMIT, MongoMemory 
# For simplicity, if mongo_memory.py is in the same directory:
from mongo_memory import mongo_memory_service # Make sure MongoMemory class is also available if needed

logger = logging.getLogger(__name__)


HISTORY_LIMIT = 5

# --- Callback to SAVE interaction AFTER the agent processes a turn ---
def save_interaction_after_agent_callback(
    # Type hint uses the general CallbackContext
    callback_context: CallbackContext, 
    final_response_content: Content # ADK provides this specific arg for after_agent
) -> Optional[Content]:
    """
    Saves the user's input and the agent's final response to MongoDB.
    This runs *after* the agent's main logic for a turn is complete.
    """
    logger.info("--- save_interaction_after_agent_callback: ENTERED ---") 
    print("--- save_interaction_after_agent_callback: ENTERED ---")
    if not mongo_memory_service or mongo_memory_service.collection is None:
        logger.warning("MongoMemoryService not available in save_interaction_callback. Skipping save.")
        print("MongoMemoryService not available in save_interaction_callback. Skipping save.")
        return final_response_content

    try:
        user_id = callback_context.user_id
        session_id = callback_context.session.id # Correct way to get session_id
        logger.info(f"Save Callback - User ID: {user_id}, Session ID: {session_id}")
        print(f"Save Callback - User ID: {user_id}, Session ID: {session_id}")

        last_user_input_text = "User input not found"
        logger.info(f"Save Callback - Extracted User Input: '{last_user_input_text[:100]}'")
        print(f"Save Callback - Extracted User Input: '{last_user_input_text[:100]}'")

        # Access session events to find the last user input
        # Note: session.events is a list. The actual last user input might not be the absolute last event
        # if other system events occurred. A more robust way might be to iterate backwards.
        # For this example, assuming the latest relevant USER event is what we need.
        user_events = [event for event in callback_context.session.events if event.author == "USER"]
        if user_events:
            last_user_event = user_events[-1]
            if last_user_event.content and last_user_event.content.parts:
                last_user_input_text = last_user_event.content.parts[0].text or "User provided non-text input"
        
        agent_final_response_text = "Agent response not found"
        if final_response_content and final_response_content.parts:
            agent_final_response_text = final_response_content.parts[0].text or "Agent provided non-text response"
            logger.info(f"Save Callback - Extracted Agent Response: '{agent_final_response_text[:100]}'")
            print(f"Save Callback - Extracted Agent Response: '{agent_final_response_text[:100]}'")

        # Correctly get current turn count for the sequence
        current_turn_count_in_db = 0
        if mongo_memory_service.collection is not None: # Check again before using
            current_turn_count_in_db = mongo_memory_service.collection.count_documents(
                {"user_id": user_id, "session_id": session_id}
            )
        turn_sequence = current_turn_count_in_db + 1

        logger.info(f"[Callback: AfterAgent] User: '{last_user_input_text[:50]}...', Agent: '{agent_final_response_text[:50]}...' for user {user_id}, session {session_id}, turn {turn_sequence}")
        
        mongo_memory_service.add_interaction(
            user_id=user_id,
            session_id=session_id,
            user_input=last_user_input_text,
            agent_response=agent_final_response_text,
            turn_sequence=turn_sequence
        )

    except Exception as e:
        logger.error(f"Error in save_interaction_after_agent_callback: {e}", exc_info=True)
    
    return final_response_content


# --- Callback to LOAD memory BEFORE the model is called ---
def load_memory_before_model_callback(
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
        user_id = callback_context.user_id
        session_id = callback_context.session.id
        
        recent_interactions = mongo_memory_service.get_recent_interactions(user_id, session_id, limit=HISTORY_LIMIT)

        if recent_interactions:
            logger.info(f"[Callback: BeforeModel] Loading {len(recent_interactions)} past interactions into prompt for user {user_id}, session {session_id}")
            
            history_contents: List[Content] = [] # Changed from history_parts to history_contents for clarity
            for interaction in recent_interactions:
                if interaction.get("user_input"):
                    history_contents.append(Content(role="user", parts=[Part.from_text(interaction["user_input"])]))
                if interaction.get("agent_response"):
                    history_contents.append(Content(role="model", parts=[Part.from_text(interaction["agent_response"])]))
            
            if llm_request.contents:
                # Ensure current_user_query is a list of Content objects if llm_request.contents is a list
                current_user_query_list = [llm_request.contents[-1]] if isinstance(llm_request.contents[-1], Content) else llm_request.contents[-1]
                llm_request.contents = history_contents + current_user_query_list
            else:
                # If llm_request.contents was empty, and current query is passed differently (unlikely for typical user turn)
                # This branch might need adjustment based on how llm_request is structured if empty.
                # For safety, let's assume we're adding history and the current query will be appended by ADK or is already there.
                # However, the typical pattern is that llm_request.contents already contains the current user query as the last item.
                # If llm_request.contents could be None or an empty list initially:
                if not llm_request.contents: # if it was None or empty list
                    # This case implies the current user query is not yet in llm_request.contents.
                    # This would be unusual if the callback is triggered for a user's turn.
                    # For safety, let's just prepend history. The current query should be added by ADK later or be there.
                    llm_request.contents = history_contents
                else: # if it was an empty list but not None.
                     llm_request.contents = history_contents + llm_request.contents # Should be rare
    except Exception as e:
        logger.error(f"Error in load_memory_before_model_callback: {e}", exc_info=True)

    return None