import os
import time
import logging
import asyncio
import uuid
from gradio_client import Client, handle_file


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def diagnose_plant_from_huggingface(image_path_on_server: str) -> str | None:
    """
    Connects to the Hugging Face Gradio server, sends an image,
    and gets back a diagnosis. Includes a retry mechanism for cold starts.

    Args:
        image_path_on_server: The local path to the image file to be sent for diagnosis.

    Returns:
        The diagnosis text from the server, or None if an error occurs.
    """
    # --- Configuration for the retry mechanism ---
    MAX_RETRIES = 5
    RETRY_DELAY_SECONDS = 120
    HF_SPACE_URL = "https://surfiniaburger-aura-mind-glow.hf.space/"

    def blocking_gradio_call():
        """This inner function contains the blocking I/O code."""
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(f"--- Attempt {attempt + 1} of {MAX_RETRIES} ---")
                logging.info(f"Connecting to Hugging Face Space: {HF_SPACE_URL}")
                
                client = Client(HF_SPACE_URL)
                
                logging.info("Connection successful. Sending image for diagnosis...")
                logging.info(f"Sending {image_path_on_server} for diagnosis...")
                
                result = client.predict(
                    uploaded_image=handle_file(image_path_on_server),
                    feedback="Automated diagnosis from ADK Agent",
                    api_name="/get_diagnosis_and_remedy"
                )

                logging.info("✅ Diagnosis received successfully!")
                logging.info(f"Result from server: {result}")
                
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    diagnosis_text = result[0]
                elif isinstance(result, str):
                    diagnosis_text = result
                else:
                    diagnosis_text = str(result)

                return diagnosis_text

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed. Error: {e}", exc_info=True)
                if attempt < MAX_RETRIES - 1:
                    logging.warning(f"Server may be experiencing a cold start. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error("All retry attempts have failed.")
                    return "Error: The diagnosis service is currently unavailable after multiple retries."
    
    # Run the blocking function in a separate thread to avoid blocking the main asyncio event loop.
    return await asyncio.to_thread(blocking_gradio_call)


async def diagnose_plant_tool(tool_context: ToolContext) -> str:
    """
    Retrieves the latest video frame from the session, saves it as a temporary file,
    and sends it for diagnosis.

    Args:
        tool_context: The context of the tool call, containing session state.

    Returns:
        The diagnosis text from the server, or an error message.
    """
    logging.info("Attempting to diagnose plant from video feed.")
    image_bytes = tool_context._invocation_context.session.state.get('latest_image_bytes')

    if not image_bytes:
        logging.warning("Diagnose tool called, but no image was found in the session context.")
        return "I am sorry, I could not see an image to diagnose. Please ensure your video is active."

    # Create a unique filename for the temporary image
    temp_dir = "static/uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_filename = f"{uuid.uuid4()}.jpg"
    temp_filepath = os.path.join(temp_dir, temp_filename)

    try:
        # Save the image bytes to the temporary file
        with open(temp_filepath, 'wb') as f:
            f.write(image_bytes)
        logging.info(f"Saved latest video frame to temporary file: {temp_filepath}")

        # Call the diagnosis function with the new image path
        diagnosis = await diagnose_plant_from_huggingface(temp_filepath)
        return diagnosis if diagnosis else "Failed to get a diagnosis."

    except Exception as e:
        logging.error(f"Error saving or diagnosing image from video feed: {e}", exc_info=True)
        return "I encountered an error while processing the image from the video feed."
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logging.info(f"Removed temporary file: {temp_filepath}")



