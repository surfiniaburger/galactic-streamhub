# /Users/surfiniaburger/Desktop/app/tools/nanobana_tool.py
import logging
import os
import uuid
from typing import Optional
import google.generativeai as genai
from google.generativeai import types
import PIL.Image
import io

logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure the directory for generated images exists
STATIC_UPLOADS_DIR = "static/uploads"
os.makedirs(STATIC_UPLOADS_DIR, exist_ok=True)

# It's recommended to set the API key as an environment variable for security
# e.g., export GOOGLE_API_KEY="your_api_key_here"
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Google GenAI: {e}")
    # Handle the case where the API key is not set.
    # The tool will fail gracefully if the key is missing.

MODEL_ID = "gemini-2.5-flash-image-preview"

def generate_image_with_nanobana(
    prompt: str,
) -> Optional[str]:
    """
    Generates an image based on a text prompt using the Gemini 1.5 Flash model.

    Args:
        prompt: The text prompt describing the image to generate.

    Returns:
        The relative web URL of the saved image (e.g., "/static/uploads/image_uuid.png"),
        or None if an error occurs.
    """
    if not prompt:
        logger.warning("Nano Banana tool was called with an empty prompt.")
        return None

    if not GOOGLE_API_KEY:
        logger.error("Cannot generate image: GOOGLE_API_KEY is not configured.")
        return "Error: Image generation API key is not configured."

    try:
        logger.info(f"Generating image with Gemini 1.5 for prompt: '{prompt}'")

        # Initialize the client within the function if you expect the key to be
        # managed in a dynamic environment, or rely on the module-level configuration.
        client = genai.GenerativeModel(MODEL_ID)

        # Generate the content
        response = client.generate_content(
            contents=prompt,
            generation_config=types.GenerationConfig(
                response_mime_type='image/png' # Request a PNG image directly
            )
        )

        # Process the response to find and save the image
        for part in response.parts:
            if part.inline_data:
                # The response contains the image bytes directly
                image_bytes = part.inline_data.data

                # Create a PIL Image object from the bytes
                image = PIL.Image.open(io.BytesIO(image_bytes))

                # Save the image with a unique filename
                filename = f"image_{uuid.uuid4().hex}.png"
                filepath = os.path.join(STATIC_UPLOADS_DIR, filename)
                image.save(filepath)

                # Return the web-accessible URL
                web_url = f"/{STATIC_UPLOADS_DIR}/{filename}"
                logger.info(f"Image successfully generated and saved to {filepath}")
                return web_url

        logger.warning("Image generation call succeeded but no image data was returned.")
        return "No image could be generated for that prompt."

    except Exception as e:
        logger.error(f"Error in Gemini image generation tool: {e}", exc_info=True)
        return f"An error occurred during image generation: {e}"
