# main.py

import os
import json
import asyncio
import base64
import contextlib
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel

from google.genai.types import Part, Content, Blob
from google.adk.runners import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from google.cloud import secretmanager 

# Import agent configuration
from agent_config import create_streaming_agent_with_mcp_tools

# --- Configuration & Global Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APP_NAME = "ADK MCP Streaming App"
STATIC_DIR = Path("static")

# Initialize ADK services
session_service = InMemorySessionService()

# --- MCP Server Parameter Definitions & Pydantic Model ---
class AllServerConfigs(BaseModel):
    configs: Dict[str, StdioServerParameters]

weather_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/weather_server.py"],
)
ct_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/cocktail.py"],
)
# Ensure the Airbnb server is started externally if you use its tools
# The command below is for if ADK were to manage its lifecycle,
# but typically it's run as a separate service.
bnb_server_params = StdioServerParameters(
    command="npx", args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
)

server_configs_instance = AllServerConfigs(
    configs={
        "weather": weather_server_params,
        "ct": ct_server_params,
        # "bnb": bnb_server_params, # Uncomment if you want ADK to try and manage it
                                   # and you have Node.js/npx available.
                                   # Otherwise, ensure it runs separately and tools can connect.
    }
)

# --- Configuration & Global Setup ---
# ...
# NEW: Configuration for Google Maps API Key from Secret Manager
GOOGLE_MAPS_API_KEY_SECRET_NAME = os.environ.get("GOOGLE_MAPS_API_KEY_SECRET_NAME", "google-maps-api-key") # Default secret name
SECRET_MANAGER_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "") # Your GCP Project ID

# --- MCP Server Parameter Definitions & Pydantic Model ---
# ... (AllServerConfigs, weather_server_params, ct_server_params remain the same) ...

# Variable to hold the fetched API key
g_maps_api_key_value = None # Will be populated at startup



# --- MCP Tool Collection Logic (Corrected) ---
async def _collect_tools_stack(
    server_config_dict: AllServerConfigs,
) -> Tuple[Dict[str, Any], contextlib.AsyncExitStack]:
    all_tools: Dict[str, Any] = {}
    exit_stack = contextlib.AsyncExitStack()
    # stack_needs_closing is implicitly handled by whether exit_stack has items

    try:
        if not hasattr(server_config_dict, "configs") or not isinstance(
            server_config_dict.configs, dict
        ):
            logging.error("server_config_dict does not have a valid '.configs' dictionary.")
            return {}, exit_stack

        for key, server_params in server_config_dict.configs.items():
            try:
                # MCPToolset.from_server returns a tuple: (tools_list, individual_toolset_exit_stack)
                tools_list, individual_toolset_exit_stack = await MCPToolset.from_server(
                    connection_params=server_params
                )

                if individual_toolset_exit_stack:
                    # The individual_toolset_exit_stack is the async context manager
                    await exit_stack.enter_async_context(individual_toolset_exit_stack)
                    logging.info(f"Successfully entered context for MCP server: {key}")
                else:
                    logging.warning(f"No exit stack returned for MCP server: {key}. Tools might not be managed correctly.")

                if tools_list:
                    sample_tool = tools_list[0]
                    logging.info(f"Sample MCPTool: {sample_tool}, type: {type(sample_tool)}")
                    logging.info(f"Is callable: {callable(sample_tool)}")
                    logging.info(f"Attributes: {dir(sample_tool)}")
                    if hasattr(sample_tool, 'execute'): 
                        logging.info("Has 'execute' method")
                    all_tools[key] = tools_list # Store the list of tools
                    logging.info(f"Successfully collected {len(tools_list)} tools for MCP server: {key}")
                else:
                    logging.warning(f"Connection successful for key '{key}', but no tools returned.")

            except FileNotFoundError as file_error:
                logging.error(f"Command or script not found for MCP server '{key}': {file_error}. This server will be unavailable.")
            except ConnectionRefusedError as conn_refused:
                logging.error(f"Connection refused for MCP server '{key}': {conn_refused}. Ensure the server is running.")
            except TypeError as te: # Catching the specific TypeError if from_server doesn't unpack as expected
                logging.error(f"TypeError during MCP toolset setup for key '{key}': {te}. This might indicate an issue with MCPToolset.from_server's return value or usage.", exc_info=True)
            except Exception as e:
                logging.error(f"Failed to initialize MCP toolset for key '{key}': {e}", exc_info=True)
        
        if not all_tools:
            logging.warning("No tools were collected from any MCP server.")

        # Ensure expected keys exist, even if empty, for cleaner access later
        expected_keys = ["weather", "bnb", "ct", "maps"]
        for k in expected_keys:
            if k not in all_tools:
                all_tools[k] = [] # Initialize with an empty list if no tools were loaded

        return all_tools, exit_stack

    except Exception as e:
        logging.error(f"Unhandled exception in _collect_tools_stack: {e}", exc_info=True)
        # Attempt to clean up if partially successful before re-raising
        await exit_stack.aclose() # Ensure cleanup on any error from this function
        raise # Re-raise the exception to be caught by the lifespan manager




# --- FastAPI Application Lifespan (for loading/unloading MCP tools) ---
@asynccontextmanager
async def app_lifespan(app_instance: FastAPI) -> Any:
    global g_maps_api_key_value # Allow modification of the global variable
    logging.info("Application Lifespan: Startup initiated - Loading MCP Tools.")
    app_instance.state.mcp_tools = {}
    app_instance.state.mcp_tool_exit_stack = None
    # --- NEW: Fetch Google Maps API Key from Secret Manager ---
    if SECRET_MANAGER_PROJECT_ID and GOOGLE_MAPS_API_KEY_SECRET_NAME:
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_version_name = f"projects/{SECRET_MANAGER_PROJECT_ID}/secrets/{GOOGLE_MAPS_API_KEY_SECRET_NAME}/versions/latest"
            response = client.access_secret_version(request={"name": secret_version_name})
            g_maps_api_key_value = response.payload.data.decode("UTF-8")
            logging.info("Successfully fetched Google Maps API Key from Secret Manager.")
        except Exception as e:
            logging.error(f"Failed to fetch Google Maps API Key from Secret Manager: {e}. Maps tools may not be available.", exc_info=True)
            g_maps_api_key_value = None # Ensure it's None if fetching failed
    else:
        logging.warning("SECRET_MANAGER_PROJECT_ID or GOOGLE_MAPS_API_KEY_SECRET_NAME not set. Cannot fetch Google Maps API Key.")
        g_maps_api_key_value = None

    # --- Define MCP Server Configs (Moved here to use the fetched API key) ---
    current_server_configs = {
        "weather": weather_server_params,
        "ct": ct_server_params,
    }

    if g_maps_api_key_value:
        maps_server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-google-maps"],
            env={"GOOGLE_MAPS_API_KEY": g_maps_api_key_value} # Pass fetched key
        )
        current_server_configs["maps"] = maps_server_params # Add maps server config
        logging.info("Google Maps MCP server configured.")
    else:
        logging.warning("Google Maps API Key not available. Google Maps MCP server will not be configured.")

    # Use the dynamically built server_configs_instance
    dynamic_server_configs_instance = AllServerConfigs(configs=current_server_configs)
    # --- End of new MCP Server Configs definition ---


    logging.info("Application Lifespan: Loading MCP Tools.")
    try:
        collected_tools, tool_stack = await _collect_tools_stack( dynamic_server_configs_instance )
        app_instance.state.mcp_tools = collected_tools
        app_instance.state.mcp_tool_exit_stack = tool_stack
        if collected_tools:
             logging.info(
                f"Application Lifespan: MCP Toolset initialized. Loaded toolsets for: {list(app_instance.state.mcp_tools.keys())}"
            )
        else:
            logging.warning("Application Lifespan: MCP Toolset initialization completed, but no tools were loaded.")
    except Exception as e:
        logging.error(f"Application Lifespan: Failed to initialize MCP tools during startup: {e}", exc_info=True)
        # Depending on severity, you might want to prevent app startup or handle gracefully
        # For now, it will continue, but tools might be unavailable.

    yield # Application runs here

    logging.info("Application Lifespan: Shutdown initiated - Closing MCP Tool connections.")
    if hasattr(app_instance.state, 'mcp_tool_exit_stack') and app_instance.state.mcp_tool_exit_stack:
        try:
            await app_instance.state.mcp_tool_exit_stack.aclose()
            logging.info("Application Lifespan: MCP Toolset connections closed successfully.")
        except Exception as e:
            logging.error(f"Application Lifespan: Error closing MCP Toolset connections: {e}", exc_info=True)
    else:
        logging.warning("Application Lifespan: No MCP Toolset exit stack found to close.")

# Instantiate FastAPI app with the lifespan manager
app = FastAPI(lifespan=app_lifespan)

# --- ADK Streaming Agent Session ---
async def start_agent_session(session_id: str, loaded_mcp_tools: Dict[str, Any], is_audio: bool = False):
    """Starts an agent session with loaded MCP tools"""
    logging.info(f"Starting agent session {session_id}, audio: {is_audio}")
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=session_id,
        session_id=session_id,
        state={} # Initial empty state for the session
    )

    agent_instance = create_streaming_agent_with_mcp_tools(loaded_mcp_tools)
    logging.info(f"Agent instance created for session {session_id} with {len(agent_instance.tools or [])} tools.")


    runner = Runner(
        app_name=APP_NAME,
        agent=agent_instance,
        session_service=session_service,
    )

    modality = "AUDIO" if is_audio else "TEXT"
    run_config = RunConfig(response_modalities=[modality])
    live_request_queue = LiveRequestQueue()

    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    logging.info(f"ADK Runner.run_live initiated for session {session_id}.")
    return live_events, live_request_queue

# --- WebSocket Communication Logic (adapted from your streaming example) ---
async def agent_to_client_messaging(websocket: WebSocket, live_events, session_id: str):
    """Agent to client communication for streaming"""
    logging.info(f"Agent to client messaging task started for session {session_id}.")
    try:
        async for event in live_events:
            if event.turn_complete or event.interrupted:
                message = {
                    "turn_complete": event.turn_complete,
                    "interrupted": event.interrupted,
                }
                await websocket.send_text(json.dumps(message))
                logging.debug(f"[S:{session_id} AGENT TO CLIENT]: Turn status: {message}")
                continue

            part: Optional[Part] = event.content and event.content.parts and event.content.parts[0]
            if not part:
                continue

            is_audio = part.inline_data and part.inline_data.mime_type.startswith("audio/") # More general audio check
            if is_audio:
                audio_data = part.inline_data and part.inline_data.data
                if audio_data:
                    message = {
                        "mime_type": part.inline_data.mime_type, # Send original audio mime_type
                        "data": base64.b64encode(audio_data).decode("ascii")
                    }
                    await websocket.send_text(json.dumps(message))
                    logging.debug(f"[S:{session_id} AGENT TO CLIENT]: {part.inline_data.mime_type}: {len(audio_data)} bytes.")
                    continue
            
            if part.text and (event.partial or event.turn_complete): # Send text if partial or if it's the final part of a turn
                message = {
                    "mime_type": "text/plain",
                    "data": part.text
                }
                await websocket.send_text(json.dumps(message))
                logging.debug(f"[S:{session_id} AGENT TO CLIENT]: text/plain: {part.text[:100]}")
    except WebSocketDisconnect:
        logging.info(f"Agent to client messaging: WebSocket disconnected for session {session_id}.")
    except Exception as e:
        logging.error(f"Error in agent_to_client_messaging for session {session_id}: {e}", exc_info=True)
    finally:
        logging.info(f"Agent to client messaging task ended for session {session_id}.")


async def client_to_agent_messaging(websocket: WebSocket, live_request_queue: LiveRequestQueue, session_id: str):
    """Client to agent communication for streaming"""
    logging.info(f"Client to agent messaging task started for session {session_id}.")
    try:
        while True:
            message_json = await websocket.receive_text()
            message = json.loads(message_json)
            mime_type = message.get("mime_type")
            data = message.get("data")

            if mime_type == "text/plain":
                content = Content(role="user", parts=[Part.from_text(text=data)])
                live_request_queue.send_content(content=content)
                logging.debug(f"[S:{session_id} CLIENT TO AGENT]: text/plain: {data[:100]}")
            elif mime_type and mime_type.startswith("audio/"): # More general audio check
                decoded_data = base64.b64decode(data)
                live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
                logging.debug(f"[S:{session_id} CLIENT TO AGENT]: {mime_type}: {len(decoded_data)} bytes.")
            # --- NEW: Handle Video Frames (as images) ---
            elif mime_type in ["image/jpeg", "image/webp"]: # Add other image types if needed
                try:
                    # Assuming 'data' is a Base64 encoded string for the image
                    decoded_image_data = base64.b64decode(data)
                    # The Gemini API expects image data directly as bytes for supported formats.
                    # We send it as a Blob via send_realtime.
                    # The ADK/Live API should package this correctly for the model.
                    live_request_queue.send_realtime(
                        Blob(data=decoded_image_data, mime_type=mime_type)
                    )
                    logging.debug(f"[S:{session_id} CLIENT TO AGENT]: {mime_type} frame: {len(decoded_image_data)} bytes.")
                except Exception as e:
                    logging.error(f"Error processing image data for session {session_id}: {e}", exc_info=True)
            # --- End of Video Frame Handling ---                
            else:
                logging.warning(f"Mime type not supported for session {session_id}: {mime_type}")
                # Optionally send an error message back to client
    except WebSocketDisconnect:
        logging.info(f"Client to agent messaging: WebSocket disconnected for session {session_id}.")
        # Signal to live_request_queue that client has disconnected if applicable
        live_request_queue.close() # Indicate that no more requests will come
    except Exception as e:
        logging.error(f"Error in client_to_agent_messaging for session {session_id}: {e}", exc_info=True)
    finally:
        logging.info(f"Client to agent messaging task ended for session {session_id}.")


# --- FastAPI WebSocket Endpoint ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    is_audio: str = Query("false", description="Set to 'true' if client supports audio I/O")
):
    await websocket.accept()
    actual_is_audio = is_audio.lower() == "true"
    logging.info(f"Client #{session_id} connected via WebSocket, audio mode: {actual_is_audio}")

    # Access pre-loaded MCP tools from application state
    loaded_mcp_tools = websocket.app.state.mcp_tools
    mcp_stack_exists = hasattr(websocket.app.state, 'mcp_tool_exit_stack') and websocket.app.state.mcp_tool_exit_stack is not None

    if not mcp_stack_exists: # Check if the stack object exists, even if tools dict might be empty
        logging.error(f"MCP Tools not properly initialized. Cannot serve requests for session {session_id}.")
        error_message = json.dumps({"message": "Error: Server is not fully initialized. Please try again later."})
        try:
            await websocket.send_text(error_message)
        except WebSocketDisconnect:
            pass # Client already gone
        finally:
            await websocket.close(code=1011) # Internal server error
        return

    live_events, live_request_queue = None, None # Initialize to None
    agent_to_client_task, client_to_agent_task = None, None # Initialize to None

    try:
        live_events, live_request_queue =await start_agent_session(session_id, loaded_mcp_tools, actual_is_audio)

        agent_to_client_task = asyncio.create_task(
            agent_to_client_messaging(websocket, live_events, session_id)
        )
        client_to_agent_task = asyncio.create_task(
            client_to_agent_messaging(websocket, live_request_queue, session_id)
        )

        # Wait for either task to complete (e.g., disconnection or error)
        done, pending = await asyncio.wait(
            [agent_to_client_task, client_to_agent_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            logging.info(f"Cancelling pending task for session {session_id}: {task.get_name()}")
            task.cancel()
        # Await the cancelled tasks to allow them to process cancellation
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    except WebSocketDisconnect:
        logging.info(f"Client #{session_id} disconnected from WebSocket endpoint.")
    except Exception as e:
        logging.error(f"Unexpected error in WebSocket endpoint for session {session_id}: {e}", exc_info=True)
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_text(json.dumps({"message": "An unexpected server error occurred."}))
        except Exception:
            pass # Suppress errors during error reporting
    finally:
        logging.info(f"Cleaning up WebSocket tasks for session {session_id}.")
        # Ensure tasks are cancelled if not already done
        tasks_to_cancel = [t for t in [agent_to_client_task, client_to_agent_task] if t and not t.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        if live_request_queue:
            live_request_queue.close() # Ensure queue is closed

        # Ensure WebSocket is closed
        if websocket.client_state != websocket.client_state.DISCONNECTED:
            await websocket.close()
        logging.info(f"WebSocket processing finished for client #{session_id}.")


# --- Static Files & Root Endpoint ---
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.get("/")
async def root():
    """Serves the index.html from the static directory"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# To run this (after installing dependencies like uvicorn, fastapi, google-adk, etc.):
# Ensure your .env file has GOOGLE_API_KEY or Vertex AI configs.
# Ensure mcp_server/cocktail.py and mcp_server/weather_server.py are present and executable.
# If using Airbnb MCP server, ensure it's running separately or configure its management.
# Command: uvicorn main:app --reload