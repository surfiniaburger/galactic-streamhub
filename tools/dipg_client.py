import logging
import time
import asyncio
from gradio_client import Client
from google.adk.tools.tool_context import ToolContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_zk_jbfuzz_from_client(user_query: str) -> str | None:
    """
    Connects to the ZK-JBFuzz Hugging Face Space, sends a query,
    and gets back a generated answer. Includes a retry mechanism for cold starts.
    This is the blocking implementation that will be run in a separate thread.
    """
    # --- Configuration for the retry mechanism ---
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 60
    HF_SPACE_URL = "https://surfiniaburger-zk-jbfuzz.hf.space/"

    logger.info(f"Connecting to Hugging Face Space: {HF_SPACE_URL}...")

    # Loop for retry attempts
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"--- Attempt {attempt + 1} of {MAX_RETRIES} ---")

            # Connect to the public Hugging Face Space
            client = Client(HF_SPACE_URL)
            logger.info("Connection successful.")

            logger.info(f"Sending query: '{user_query}'...")

            # Call the specific function on the server.
            result = client.predict(
                user_query,
                fn_index=0
            )

            logger.info("✅ Answer received successfully!")
            logger.info(f"--- Generated Answer ---\n{result}\n------------------------")
            return str(result)

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed. Error: {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Server may be experiencing a cold start. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("All retry attempts have failed. The server might be unavailable or has an error.")
                return f"Error: The DIPG knowledge base is currently unavailable after {MAX_RETRIES} attempts."

async def dipg_knowledge_base_tool(tool_context: ToolContext, user_query: str) -> str:
    """
    A tool that queries the ZK-JBFuzz Hugging Face Space for information on DIPG.
    It handles blocking I/O by running the Gradio client in a separate thread.

    Args:
        tool_context: The context of the tool call, provided by the ADK.
        user_query: The user's question about DIPG.

    Returns:
        The answer from the knowledge base, or an error message.
    """
    logger.info(f"DIPG Knowledge Base Tool called with query: '{user_query}'")

    # Run the blocking function in a separate thread to avoid blocking the main asyncio event loop.
    result = await asyncio.to_thread(query_zk_jbfuzz_from_client, user_query)

    return result if result is not None else "An unknown error occurred while querying the DIPG knowledge base."