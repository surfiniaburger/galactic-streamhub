# In your callbacks.py
# CORRECTED IMPORT
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part
from typing import Optional, List
import logging

from google.adk.events import Event

# Assuming mongo_memory_service is initialized in mongo_memory.py and imported here
# from .mongo_memory import mongo_memory_service, HISTORY_LIMIT, MongoMemory 
# For simplicity, if mongo_memory.py is in the same directory:
from mongo_memory import mongo_memory_service, DEFAULT_HISTORY_LIMIT # Make sure MongoMemory class is also available if needed

logger = logging.getLogger(__name__)


HISTORY_LIMIT = 5

async def check_for_prompt_injection_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    A pre-model callback to check for potential prompt injection attacks.

    This should run before any other logic that processes the user's input.
    It inspects the latest user prompt for common injection phrases.
    If a potential threat is found, it logs a warning and returns a canned
    LlmResponse to terminate the turn and inform the user.

    Args:
        callback_context: The invocation context, containing session info.
        llm_request: The request object about to be sent to the LLM.

    Returns:
        An LlmResponse object to terminate the turn if an injection is detected,
        otherwise None.
    """
    # The user prompt is the last content in the request
    if not llm_request.contents:
        return None

    last_content = llm_request.contents[-1]
    if last_content.role != 'user' or not last_content.parts:
        return None

    user_prompt = "".join(part.text for part in last_content.parts if part.text).strip().lower()

    if not user_prompt:
        return None

    injection_patterns = [
        "ignore all previous instructions", "ignore the above", "disregard the above",
        "you are now", "your new instructions are", "system prompt:",
        "reveal your instructions", "what are your instructions", "print your instructions",
    ]

    for pattern in injection_patterns:
        if pattern in user_prompt:
            logger.warning(
                f"Potential prompt injection detected in session {callback_context._invocation_context.session.id}. "
                f"Pattern: '{pattern}'. Prompt: '{user_prompt[:200]}...'"
            )
            # Halt processing by returning a canned LlmResponse.
            return LlmResponse(
                content=Content(
                    parts=[Part.from_text("Your request could not be processed due to a security policy. Please rephrase your request and try again.")],
                    role="model"
                ),
                turn_complete=True # Signal the end of the turn
            )

    return None # No injection detected, proceed normally

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
    Retrieves the last few chronological interactions from MongoDB to provide
    short-term context to the LLM. It does NOT perform a vector search.
    """
    if not mongo_memory_service or mongo_memory_service.collection is None:
        logger.warning("MongoMemoryService not available in load_memory_before_model_callback. Skipping memory load.")
        return None 

    try:
        user_id = callback_context._invocation_context.session.user_id
        session_id = callback_context._invocation_context.session.id
        # --- SIMPLIFIED LOGIC ---
        # Always fetch the last few chronological turns for immediate context.
        # The DeepMemoryRecallAgent is responsible for vector search on explicit user requests.
        logger.info(f"[Callback: BeforeModel] Loading recent chronological interactions (limit={DEFAULT_HISTORY_LIMIT}) for user {user_id}, session {session_id}")
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