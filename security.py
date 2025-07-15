# /Users/surfiniaburger/Desktop/app/security.py
import logging
import os
from functools import lru_cache
from typing import Dict, Any, Optional

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import modelarmor_v1

logger = logging.getLogger(__name__)

# --- Model Armor Configuration ---
MODEL_ARMOR_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "us-central1")
MODEL_ARMOR_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ARMOR_TEMPLATE_ID = "galactic-streamhub-ma-template" # As per the user's example

@lru_cache(maxsize=1)
def get_model_armor_client() -> Optional[modelarmor_v1.ModelArmorClient]:
    """
    Initializes and returns a singleton ModelArmorClient instance.
    Caches the client to avoid re-initialization on every call.
    """
    try:
        logging.info("Initializing ModelArmorClient...")
        # The endpoint needs to be specified for the REST transport
        client_options = {"api_endpoint": f"modelarmor.{MODEL_ARMOR_LOCATION}.rep.googleapis.com"}
        client = modelarmor_v1.ModelArmorClient(transport="rest", client_options=client_options)
        logging.info("ModelArmorClient initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize ModelArmorClient: {e}", exc_info=True)
        return None

def sanitize_prompt_with_model_armor(prompt: str) -> Dict[str, Any]:
    """
    Uses Google Cloud Model Armor to check a prompt for safety violations.
    """
    client = get_model_armor_client()
    if not client:
        logging.error("Model Armor client is not available. Bypassing security check.")
        return {"is_safe": True, "reason": "Client unavailable"}

    try:
        user_prompt_data = modelarmor_v1.DataItem(text=prompt)
        template_path = f"projects/{MODEL_ARMOR_PROJECT_ID}/locations/{MODEL_ARMOR_LOCATION}/templates/{MODEL_ARMOR_TEMPLATE_ID}"
        request = modelarmor_v1.SanitizeUserPromptRequest(name=template_path, user_prompt_data=user_prompt_data)
        response = client.sanitize_user_prompt(request=request)

        if response.sanitization_result.filter_results["pi_and_jailbreak"].pi_and_jailbreak_filter_result.match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
            return {"is_safe": False, "reason": "PI/Jailbreak detected"}
        return {"is_safe": True, "reason": "Passed security check"}
    except GoogleAPICallError as e:
        logging.error(f"Model Armor API call failed: {e}", exc_info=True)
        # Fail open or closed? For security, failing closed is safer.
        return {"is_safe": False, "reason": f"API Error: {e}"}
    except Exception as e:
        logging.error(f"An unexpected error occurred during prompt sanitization: {e}", exc_info=True)
        return {"is_safe": False, "reason": f"Unexpected Error: {e}"}