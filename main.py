# main.py

import io
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
from fastapi import FastAPI, Response, WebSocket, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel

from google.genai.types import Part, Content, Blob
from google.adk.runners import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from mongo_memory import MongoMemory
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams

from google.cloud import secretmanager 

import pydicom
import numpy as np
from PIL import Image




### --- NEW --- ###
# Import Firebase Admin SDK components
import firebase_admin
from firebase_admin import credentials, auth
### --- END NEW --- ###

# Import agent configuration
#from main_agent.agent import create_streaming_agent_with_mcp_tools
from agent_config import create_streaming_agent_with_mcp_tools
from mongo_memory import mongo_memory_service, DEFAULT_HISTORY_LIMIT 
# --- Configuration & Global Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APP_NAME = "ADK MCP Streaming App"
STATIC_DIR = Path("static")

# Initialize ADK services

session_service = InMemorySessionService()

# MongoDB Configuration from Secrets/Env
MONGODB_SECRET_ID = "MULTIMODAL_MONGODB_URI"
MONGODB_URI_ENV_VAR = "MONGODB_URI"

DEFAULT_MEMORY_DB_NAME = "adk_agent_memory" # Consider making this configurable if needed
DEFAULT_MEMORY_COLLECTION_NAME = "interaction_history"
DEFAULT_HISTORY_LIMIT = 5


# --- MCP Server Parameter Definitions & Pydantic Model ---
class AllServerConfigs(BaseModel):
    configs: Dict[str, StdioConnectionParams]

# --- Configuration & Global Setup ---
# ...
# NEW: Configuration for Google Maps API Key from Secret Manager
GOOGLE_MAPS_API_KEY_SECRET_NAME = os.environ.get("GOOGLE_MAPS_API_KEY_SECRET_NAME", "google-maps-api-key") # Default secret name
SECRET_MANAGER_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "") # Your GCP Project ID

def create_mongo_memory():
    return MongoMemory(db_name=DEFAULT_MEMORY_DB_NAME)

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
    # --- Initialize Firebase Admin SDK at application startup ---
    try:
        logging.info("Initializing Firebase Admin SDK...")
        # This check prevents re-initialization errors during hot-reloads
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_project_id = os.environ.get("FIREBASE_PROJECT_ID", "studio-l13dd")
            firebase_admin.initialize_app(cred, {
                'projectId': firebase_project_id,
            })
            logging.info(f"Firebase Admin SDK initialized successfully for project '{firebase_project_id}'.")
        else:
            logging.info("Firebase Admin SDK already initialized.")

    except google_exceptions.DefaultCredentialsError: # type: ignore
        logging.critical("FATAL: Could not find Google Cloud Application Default Credentials.")
        logging.critical("--> For local development, run 'gcloud auth application-default login'")
        logging.critical("--> For deployment, ensure the service account has the correct IAM roles.")
    except Exception as e:
        logging.critical(f"FATAL: Firebase Admin SDK failed to initialize: {e}", exc_info=True)

    # --- Define Base MCP Server Configurations ---
    # **FIXED**: The `command` and `args` are nested inside `server_params`.
    weather_server_params = StdioConnectionParams(
        server_params={"command": "python", "args": ["./mcp_server/weather_server.py"]}
    )
    ct_server_params = StdioConnectionParams(
        server_params={"command": "python", "args": ["./mcp_server/cocktail.py"]}
    )

    current_server_configs = {
        "weather": weather_server_params,
        "ct": ct_server_params,
    }

    # --- ADDED BACK: Dynamic Google Maps Server Configuration ---
    # This logic now correctly checks for the API key and adds the Maps server if present.
    google_maps_api_key_from_env = os.environ.get("GOOGLE_MAPS_API_KEY")
    if google_maps_api_key_from_env:
        logging.info("Google Maps API Key found in environment. Configuring Maps MCP server.")
        # **FIXED**: `command`, `args`, and `env` are all nested inside `server_params`.
        maps_server_params = StdioConnectionParams(
            server_params={
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-google-maps"],
                "env": {"GOOGLE_MAPS_API_KEY": google_maps_api_key_from_env}
            }
        )
        current_server_configs["maps"] = maps_server_params
    else:
        logging.warning("GOOGLE_MAPS_API_KEY environment variable not found. Google Maps MCP server will not be configured.")
    # --- END OF ADDED BACK LOGIC ---

    # Store toolsets on the app's state for access in endpoints
    app_instance.state.mcp_toolsets = []
    
    logging.info("Application Lifespan: Creating MCPToolsets.")
    for key, server_params in current_server_configs.items():
        try:
            logging.info(f"Initializing MCPToolset for key: {key}")
            toolset_instance = MCPToolset(connection_params=server_params)
            app_instance.state.mcp_toolsets.append(toolset_instance)
            logging.info(f"MCPToolset instance created for key: {key}")
        except Exception as e:
            logging.error(f"Failed to create MCPToolset for key '{key}': {e}", exc_info=True)
    
    if app_instance.state.mcp_toolsets:
        logging.info(f"Application Lifespan: {len(app_instance.state.mcp_toolsets)} MCPToolset(s) created.")
    else:
        logging.warning("Application Lifespan: No MCPToolsets were created. Check server configurations and paths.")

    # Yield control to the running FastAPI application
    yield

    # --- Shutdown Logic ---
    logging.info("Application Lifespan: Shutdown initiated - Closing MCPToolsets.")
    if hasattr(app_instance.state, 'mcp_toolsets'):
        for toolset in app_instance.state.mcp_toolsets:
            try:
                await toolset.close()
            except Exception as e:
                logging.error(f"Error closing an MCPToolset: {e}", exc_info=True)
    logging.info("Application Lifespan: All MCPToolsets closed.")

# Instantiate FastAPI app with the lifespan manager
app = FastAPI(lifespan=app_lifespan)

# --- ADK Streaming Agent Session ---
async def start_agent_session(session_id: str, app_state: Any, is_audio: bool = False): # Pass app_state
    logging.info(f"Starting agent session {session_id}, audio: {is_audio}")
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=session_id, session_id=session_id, state={}
    )

    # Get the list of MCPToolset instances and any other tools
    mcp_toolsets_from_state = getattr(app_state, 'mcp_toolsets', [])
    
    # Your agent_config.py will now receive these MCPToolsets
    # It also needs to know about the TaskExecutionAgentTool
    
    # --- This is where agent_config.py is called ---
    # We need to adjust what create_streaming_agent_with_mcp_tools expects.
    # Instead of a dict of tool lists, it will now get a list of MCPToolset objects.
    # Let's assume for now that create_streaming_agent_with_mcp_tools will be adapted
    # to take these toolsets and create the AgentTool for TaskExecutionAgent itself.

    # For agent_config.py's warnings, we pass the temporary raw tools list
    raw_mcp_tools_for_config = getattr(app_state, 'raw_mcp_tools_for_agent_config', {})

    agent_instance = create_streaming_agent_with_mcp_tools(
        loaded_mcp_toolsets=mcp_toolsets_from_state, # Pass the toolsets
        # We might need a different way to handle the TaskExecutionAgent creation now
        # or agent_config.py needs to be smarter.
        # For now, let's assume agent_config.py is also refactored.
        #raw_mcp_tools_lookup_for_warnings=raw_mcp_tools_for_config
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=agent_instance,
        memory_service=mongo_memory_service, # Pass the persistent memory service
        session_service=session_service, 
       
    )

    modality = "AUDIO" if is_audio else "TEXT"
    # Initialize parameters for RunConfig
    run_config_args = {
        "response_modalities": [modality] 
        # Other defaults like 'save_input_blobs_as_artifacts': False, 'support_cfc': False will apply
    }

    if is_audio:
        # Enable output audio transcription when in audio mode
        # The Vertex AI docs show this as an empty dict {} or a specific config object.
        # Let's try with an empty dict first, which usually means "enable with defaults".
        run_config_args["output_audio_transcription"] = {} 
        logging.info(f"Session {session_id}: Enabling output_audio_transcription for audio mode.")

        # Optionally, if you also want to transcribe the user's input audio:
        run_config_args["input_audio_transcription"] = {}

        # Enable sound recognition
        
        logging.info(f"Session {session_id}: Enabling sound recognition for audio mode.")

        # Optionally, if you want to configure the voice (example from ADK docs):
        # from google.genai.types import VoiceConfig, PrebuiltVoiceConfig, SpeechConfig # Make sure these are imported
        # voice_config = VoiceConfig(
        #     prebuilt_voice_config=PrebuiltVoiceConfig(voice_name='Aoede') # Example voice
        # )
        # adk_speech_config = SpeechConfig(voice_config=voice_config) # This is google.genai.types.SpeechConfig
        # run_config_args["speech_config"] = adk_speech_config
        # logging.info(f"Session {session_id}: Custom speech_config applied for audio mode.")


    run_config = RunConfig(**run_config_args) # Pass arguments as a dictionary



    live_request_queue = LiveRequestQueue()

    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    logging.info(f"ADK Runner.run_live initiated for session {session_id}.")
    return live_events, live_request_queue, session

# --- WebSocket Communication Logic (adapted from your streaming example) ---
async def agent_to_client_messaging(websocket: WebSocket, live_events, session_id: str, session: Any):
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
            
            if hasattr(event, 'sound_event') and event.sound_event:
                if 'sound_events' not in session.state:
                    session.state['sound_events'] = []
                session.state['sound_events'].append(event.sound_event.name)

                message = {
                    "mime_type": "sound_event",
                    "data": event.sound_event.name
                }
                await websocket.send_text(json.dumps(message))
                logging.debug(f"[S:{session_id} AGENT TO CLIENT]: sound_event: {event.sound_event.name}")
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


async def client_to_agent_messaging(websocket: WebSocket, live_request_queue: LiveRequestQueue, session_id: str, session: Any):
    """Client to agent communication for streaming"""
    logging.info(f"Client to agent messaging task started for session {session_id}.")
    try:
        while True:
            message_json = await websocket.receive_text()

            # Add a check to handle empty messages which can cause JSONDecodeError
            if not message_json:
                logging.warning(f"Received empty message from client for session {session_id}. Skipping.")
                continue

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
                    # Use the session object that was passed in
                    if session.state is not None:
                        session.state['latest_image_bytes'] = decoded_image_data
                        logging.debug(f"[S:{session_id}] Saved latest image frame to session state.")
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
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="Firebase Auth ID Token"),
    is_audio: str = Query("false", description="Set to 'true' if client supports audio I/O")
):
    uid = None
    try:
        # **CRITICAL SECURITY STEP: Verify the token**
        # This checks the token against Firebase's servers. If it's invalid, expired,
        # or fake, it will raise an exception.
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        user_name = decoded_token.get('name', 'Unknown User')
        logging.info(f"Client token verified. UID: {uid}, Name: {user_name}")

    except Exception as e:
        # If verification fails, log the error and refuse the connection.
        logging.error(f"Authentication failed: {e}", exc_info=False) # exc_info=False to not log the full token
        # Close with code 1011 (server error) to signal to the client.
        await websocket.close(code=1011)
        return

    # If verification succeeds, proceed with the connection.    
    await websocket.accept()
    actual_is_audio = is_audio.lower() == "true"
    logging.info(f"Client #{uid} connected via WebSocket, audio mode: {actual_is_audio}")

    # Access application state which should contain the MCPToolsets
    app_state = websocket.app.state

    # --- UPDATED CHECK for MCP Toolset Initialization ---
    mcp_toolsets_initialized_correctly = False
    if hasattr(app_state, 'mcp_toolsets') and isinstance(app_state.mcp_toolsets, list):
        # You might want to check if the expected number of toolsets are present,
        # e.g., if you expect weather, ct, and maps, len(app_state.mcp_toolsets) >= 2 or 3
        # For now, let's just check if the list was created and has at least one (if expected).
        # If you expect specific toolsets, you'd need a more robust check here or
        # rely on the logging during app_lifespan.
        # A simple check that toolsets list exists and is a list:
        mcp_toolsets_initialized_correctly = True 
        if not app_state.mcp_toolsets: # If list is empty, but you expect toolsets
             logging.warning(f"MCPToolsets list is empty for session {uid}, though app_state.mcp_toolsets exists.")
             # You might still consider this an error state depending on your app's requirements
             # mcp_toolsets_initialized_correctly = False # Uncomment if an empty list is an error

    if not mcp_toolsets_initialized_correctly:
        logging.error(f"MCP Tools (MCPToolsets) not properly initialized in app.state. Cannot serve requests for session {uid}.")
        error_message = json.dumps({"message": "Error: Server is not fully initialized. Please try again later."})
        try:
            await websocket.send_text(error_message)
        except WebSocketDisconnect:
            pass 
        finally:
            await websocket.close(code=1011) 
        return
    # --- END UPDATED CHECK ---

    live_events, live_request_queue, session = None, None, None # Initialize to None
    agent_to_client_task, client_to_agent_task, session = None, None, None # Initialize to None

    try:
        # **CRITICAL CHANGE: Use the verified 'uid' as the session ID.**
        live_events, live_request_queue, session =await start_agent_session(uid, app_state, actual_is_audio)

        agent_to_client_task = asyncio.create_task(
            agent_to_client_messaging(websocket, live_events, uid, session)
        )
        client_to_agent_task = asyncio.create_task(
            client_to_agent_messaging(websocket, live_request_queue, uid, session)
        )

        # Wait for either task to complete (e.g., disconnection or error)
        done, pending = await asyncio.wait(
            [agent_to_client_task, client_to_agent_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            logging.info(f"Cancelling pending task for session {uid}: {task.get_name()}")
            task.cancel()
        # Await the cancelled tasks to allow them to process cancellation
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    except WebSocketDisconnect:
        logging.info(f"Client #{uid} disconnected from WebSocket endpoint.")
    except Exception as e:
        logging.error(f"Unexpected error in WebSocket endpoint for session {uid}: {e}", exc_info=True)
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_text(json.dumps({"message": "An unexpected server error occurred."}))
        except Exception:
            pass # Suppress errors during error reporting
    finally:
        logging.info(f"Cleaning up WebSocket tasks for session {uid}.")
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
        logging.info(f"WebSocket processing finished for client #{uid}.")

# --- NEW: Configuration for the medical image data path ---
# This MUST match the path where your download_tcia_data.py script saves the data
MEDICAL_IMAGE_DATA_DIR = os.path.abspath(os.path.join('data', 'tcia_lidc_idri'))

# ... (your existing app = FastAPI() and other setup)

# --- NEW: Image Serving Endpoint for Medical Scans ---
@app.get("/static/medical_images/{series_uid}/{image_filename}")
async def serve_medical_image(series_uid: str, image_filename: str):
    """
    Serves a specific medical image (DICOM converted to PNG for browser).
    This endpoint finds the DICOM file, converts it on-the-fly to PNG,
    and sends it to the user's browser.
    """
    # Basic security to prevent directory traversal attacks
    if ".." in series_uid or ".." in image_filename:
        return Response(status_code=404)

    try:
        # Reconstruct the full path to the .dcm file
        # The agent will request a .png, but the file on disk is .dcm
        dicom_filename = image_filename.replace('.png', '.dcm')
        series_dir_path = os.path.join(MEDICAL_IMAGE_DATA_DIR, series_uid)
        dicom_filepath = os.path.join(series_dir_path, dicom_filename)

        if not os.path.exists(dicom_filepath):
            logging.error(f"Image file not found: {dicom_filepath}")
            return Response(status_code=404)

        # --- On-the-fly DICOM to PNG conversion ---
        dicom_data = pydicom.dcmread(dicom_filepath)
        pixels = dicom_data.pixel_array.astype(float)
        pixels_scaled = (np.maximum(pixels, 0) / pixels.max()) * 255.0
        image_array = pixels_scaled.astype(np.uint8)
        
        pil_image = Image.fromarray(image_array)
        
        # Save the PNG to an in-memory byte buffer
        img_io = io.BytesIO()
        pil_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Use StreamingResponse for sending image data
        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        logging.error(f"Error serving medical image {series_uid}/{image_filename}: {e}", exc_info=True)
        return Response(status_code=500)



# --- Static Files & Root Endpoint ---
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# This assumes 'assets' is a folder in the same directory as main.py
app.mount("/assets", StaticFiles(directory="assets", html=True), name="assets")

@app.get("/")
async def root():
    """Serves the index.html from the static directory"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# To run this (after installing dependencies like uvicorn, fastapi, google-adk, etc.):
# Ensure your .env file has GOOGLE_API_KEY or Vertex AI configs.
# Ensure mcp_server/cocktail.py and mcp_server/weather_server.py are present and executable.
# If using Airbnb MCP server, ensure it's running separately or configure its management.
# Command: uvicorn main:app --reload