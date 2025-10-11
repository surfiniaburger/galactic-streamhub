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
from security import sanitize_prompt_with_model_armor
# For simplicity, if mongo_memory.py is in the same directory:
from google.adk.memory import VertexAiMemoryBankService
from session_processor import SessionProcessorAgent
import json

logger = logging.getLogger(__name__)


HISTORY_LIMIT = 5

async def security_check_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    A pre-model callback to check for security violations using Model Armor.
    
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

    # --- NEW: Use Model Armor for sanitization ---
    sanitization_result = sanitize_prompt_with_model_armor(user_prompt)
    
    if not sanitization_result.get("is_safe"):
        reason = sanitization_result.get("reason", "unspecified security concern")
        logger.warning(
            f"Model Armor blocked prompt in session {callback_context._invocation_context.session.id}. "
            f"Reason: '{reason}'. Prompt: '{user_prompt[:200]}...'"
        )
        # Halt processing by returning a canned LlmResponse.
        return LlmResponse(
            content=Content(
                parts=[Part(text="Your request could not be processed due to a security policy. Please rephrase your request and try again.")],
                role="model"
            ),
            turn_complete=True # Signal the end of the turn
        )
    
    logger.info("Prompt passed Model Armor security check.")
    return None # No injection detected, proceed normally

# --- Callback to SAVE interaction AFTER the agent processes a turn ---
async def save_interaction_after_model_callback( # Changed to async def
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """
    Saves the user's input and the agent's final response to the Memory Bank.
    This runs *after* the model generates a response.
    """
    try:
        # Access the session through the internal invocation_context
        user_id = callback_context._invocation_context.session.user_id
        session_id = callback_context._invocation_context.session.id
        
        # Directly use the memory_service from the invocation context
        memory_service = callback_context._invocation_context.runner.memory_service
        if isinstance(memory_service, VertexAiMemoryBankService):
            # The ADK session object is automatically updated with the latest turn.
            # We can pass the whole session to the memory service.
            await memory_service.add_session_to_memory(callback_context._invocation_context.session)
            logger.info(f"Save Callback - Session {session_id} sent to Memory Bank for processing.")
        else:
            logger.warning("VertexAiMemoryBankService not available in save_interaction_callback. Skipping save.")

    except Exception as e:
        logger.error(f"Error in save_interaction_after_model_callback: {e}", exc_info=True)
    
    # Return None to indicate we are not modifying the LLM's response.
    return None

# --- Callback to LOAD memory BEFORE the model is called ---
async def load_memory_before_model_callback( # Changed to async def
    # Type hint uses the general CallbackContext
    callback_context: CallbackContext, 
    llm_request: LlmRequest # ADK provides this specific arg for before_model
) -> Optional[LlmResponse]:
    """
    Retrieves recent interactions and the user's persona from the Memory Bank to provide
    context to the LLM.
    """
    try:
        user_id = callback_context._invocation_context.session.user_id
        app_name = callback_context._invocation_context.runner.app_name

        # Directly use the memory_service from the invocation context
        memory_service = callback_context._invocation_context.runner.memory_service
        if not isinstance(memory_service, VertexAiMemoryBankService):
            logger.warning("VertexAiMemoryBankService not available in load_memory_before_model_callback. Skipping memory load.")
            return None

        # Get the last user query
        last_user_query = ""
        if llm_request.contents:
            last_content = llm_request.contents[-1]
            if last_content.role == 'user' and last_content.parts:
                last_user_query = last_content.parts[0].text or ""

        # Search for relevant memories
        retrieved_memories = await memory_service.search_memory(
            user_id=user_id,
            app_name=app_name,
            query=last_user_query,
            limit=5  # Or your desired limit
        )

        if retrieved_memories:
            # Format memories and prepend them to the prompt
            memory_summary = "[System Note: Here are some relevant facts from past conversations]:\n"
            for fact in retrieved_memories:
                memory_summary += f"- {fact}\n"
            
            memory_content = Content(role="user", parts=[Part.from_text(text=memory_summary)])
            llm_request.contents = [memory_content] + llm_request.contents
            logger.info(f"Loaded {len(retrieved_memories)} memories for user {user_id}.")
        else:
            logger.info(f"No relevant memories found for user {user_id}.")

    except Exception as e:
        logger.error(f"Error in load_memory_before_model_callback: {e}", exc_info=True)
    return None

async def end_of_session_callback(
    callback_context: CallbackContext,
) -> None:
    """
    Processes the session's conversation history at the end of the session.
    """
    try:
        user_id = callback_context._invocation_context.session.user_id
        session_id = callback_context._invocation_context.session.id

        # Directly use the memory_service from the invocation context
        memory_service = callback_context._invocation_context.runner.memory_service
        if isinstance(memory_service, VertexAiMemoryBankService):
            # The ADK session object is automatically updated with the latest turn.
            # We can pass the whole session to the memory service.
            await memory_service.add_session_to_memory(callback_context._invocation_context.session)
            logger.info(f"End of Session Callback - Session {session_id} sent to Memory Bank for processing.")
        else:
            logger.warning("VertexAiMemoryBankService not available in end_of_session_callback. Skipping session processing.")

    except Exception as e:
        logger.error(f"Error in end_of_session_callback: {e}", exc_info=True)