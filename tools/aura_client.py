import os
import time
import logging
import asyncio
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

if __name__ == '__main__':
    # This is a placeholder for a real image file path
    # In your app, this path would come from a user upload
    example_image_path = "sample_images/healthy_maize_test_1.jpg"

    # Create a dummy image file for testing if it doesn't exist
    if not os.path.exists(example_image_path):
        from PIL import Image
        print("Creating a dummy image for demonstration.")
        dummy_image_dir = os.path.dirname(example_image_path)
        if not os.path.exists(dummy_image_dir):
            os.makedirs(dummy_image_dir)
        dummy_image = Image.new('RGB', (100, 100), color = 'green')
        dummy_image.save(example_image_path)

    diagnosis = asyncio.run(diagnose_plant_from_huggingface(example_image_path))
    if diagnosis:
        print("\n--- Diagnosis ---")
        print(diagnosis)
    else:
        print("\nFailed to get a diagnosis.")