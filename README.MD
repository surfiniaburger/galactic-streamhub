Okay, here's a comprehensive README.md for your new ADK Streaming application that integrates MCP tools.

```markdown
# ADK Streaming Agent with MCP Tool Integration

## Overview

This application demonstrates a real-time, streaming-capable agent built using the Google Agent Development Kit (ADK). The agent can interact with users via text and potentially audio (if client-side support is implemented fully for audio I/O), and leverages tools loaded from Model Context Protocol (MCP) servers to answer queries related to cocktails, weather, and Airbnb bookings.

The application uses FastAPI as the web backend and WebSockets for streaming communication. MCP tools are loaded dynamically at application startup and are made available to the ADK agent.

## Features

*   **Real-time Streaming:** Bidirectional communication with the ADK agent using WebSockets.
*   **MCP Tool Integration:** Dynamically loads and utilizes tools from local and remote MCP servers.
    *   **Cocktail MCP Server:** Provides tools for cocktail recipes and ingredients.
    *   **Weather MCP Server:** Provides tools for weather forecasts.
    *   **Airbnb MCP Server (Optional):** Can be configured to provide tools for Airbnb listings (requires separate setup or ADK management).
*   **Configurable Agent:** Agent behavior and instructions are defined in `agent_config.py`.
*   **FastAPI Backend:** Robust and modern Python web framework.
*   **Lifespan Management:** MCP tools are loaded on startup and gracefully shut down on application exit.
*   **Environment-based Configuration:** Uses a `.env` file for API keys and cloud project settings.

## Architecture

The application consists of:

1.  **FastAPI Server (`main.py`):**
    *   Manages WebSocket connections for streaming communication with clients.
    *   Uses a lifespan manager to initialize and close MCP tool connections.
    *   Handles HTTP requests (e.g., serving the frontend).
2.  **ADK Agent (`agent_config.py`):**
    *   An `LlmAgent` instance configured with a specific Gemini model.
    *   Provided with the loaded MCP tools to handle user queries.
3.  **ADK Runner & Streaming Components:**
    *   `Runner` orchestrates the agent's live interaction.
    *   `LiveRequestQueue` manages incoming user messages for the agent.
4.  **MCP Toolset & Servers (`mcp_server/`):**
    *   `MCPToolset` connects to MCP servers defined by `StdioServerParameters`.
    *   Local Python scripts (`cocktail.py`, `weather_server.py`) act as MCP servers.
    *   The Airbnb MCP server can be a publicly hosted one or run locally.
5.  **Static Frontend (`static/`):**
    *   `index.html` and associated JavaScript for client-side interaction (you'll need to implement this based on the WebSocket endpoint).

## Project Structure

```text
your_streaming_project_folder/
├── main.py                  # FastAPI app, WebSocket logic, MCP tool lifespan
├── agent_config.py          # Agent definition and MCP tool patching
├── mcp_server/
│   ├── cocktail.py          # Local Cocktail MCP server
│   └── weather_server.py     # Local Weather MCP server
├── static/                  # Frontend files
│   ├── index.html           # Main HTML page for client
│   └── client.js            # (Example) Client-side JavaScript
│   └── style.css            # (Example) CSS
├── .env                     # Environment variables
├── pyproject.toml           # Project dependencies (e.g., for poetry or uv)
├── requirements.txt         # (Alternative) Project dependencies (for pip)
├── README.md                # This file
└── Dockerfile               # (Optional) For containerization```

## Prerequisites

*   **Python 3.9+**
*   **`uv` (recommended)** or `pip` for package management.
    *   Install `uv`: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
*   **Google Cloud Project & Gemini API Access:**
    *   A Google Cloud Project with the Vertex AI API enabled if using Vertex AI models.
    *   Or, a Google AI Studio API key if using `generative-language.googleapis.com`.
*   **(Optional) Node.js & npx:** If you intend for ADK to manage the Airbnb MCP server locally via `StdioServerParameters`. Otherwise, the Airbnb server must be run as a separate process.

## Setup and Running Locally

### 1. Clone the Repository (or set up your project)

```bash
# If you have this in a git repository:
# git clone <your-repo-url>
# cd your_streaming_project_folder
```

### 2. Configure Environment Variables

Create a `.env` file in the root of the `your_streaming_project_folder/` directory. Populate it based on your chosen backend (Google AI Studio or Vertex AI):

```dotenv
# --- Option 1: Google AI Studio API Key ---
# GOOGLE_GENAI_USE_VERTEXAI=0 # Ensure this is 0 or commented out if using API key
# GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# --- Option 2: Vertex AI ---
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
GOOGLE_CLOUD_LOCATION="your-gcp-region" # e.g., us-central1

# --- Optional: For Audio Streaming with some clients/environments ---
# export SSL_CERT_FILE=$(python -m certifi) # If you encounter SSL issues with audio
```
*Replace placeholders with your actual values.*

### 3. Install Dependencies

It's recommended to use a virtual environment.

**Using `uv` (with `pyproject.toml` or `requirements.txt`):**

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # On macOS/Linux
# .venv\Scripts\activate # On Windows

# Install dependencies
# If you have pyproject.toml with dependencies listed:
uv pip install -r requirements.txt # or create requirements.txt and use this
# Example requirements.txt:
# fastapi
# uvicorn[standard]
# google-adk
# pydantic
# python-dotenv
# httpx # Often a dependency for API clients, good to have```

### 4. Ensure MCP Server Scripts are Executable

The `mcp_server/cocktail.py` and `mcp_server/weather_server.py` scripts will be run as subprocesses. Ensure they have execute permissions if necessary and that their Python shebangs (e.g., `#!/usr/bin/env python`) are correct or that they are invoked with `python ...` as configured in `StdioServerParameters`.

### 5. Run the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The application should now be accessible at `http://127.0.0.1:8000` (or your local IP if accessing from another device on the network).

## Example Usage

Open `http://127.0.0.1:8000` in your browser. You will need a client-side implementation (`static/index.html` and `static/client.js`) that connects to the WebSocket endpoint: `ws://127.0.0.1:8000/ws/{some_session_id}?is_audio=<true|false>`.

Once connected, you can send messages like:

*   `"What's the recipe for a Negroni?"`
*   `"Tell me the weather in Paris."`
*   `"I need a random cocktail."`
*   *(If Airbnb tools are configured and working)* `"Find an Airbnb in Tokyo for 2 people next month."`

The agent will use the loaded MCP tools to respond, and the responses will be streamed back to the client.

## Key Files

*   **`main.py`:**
    *   Contains the FastAPI application setup.
    *   Manages the application lifecycle (startup/shutdown) for loading and closing MCP tool connections.
    *   Defines the WebSocket endpoint (`/ws/{session_id}`) for streaming communication.
    *   Initializes and runs the ADK agent session for each connected client.
*   **`agent_config.py`:**
    *   Defines the `ROOT_AGENT_INSTRUCTION_STREAMING` for the ADK agent.
    *   Contains `create_streaming_agent_with_mcp_tools()` which:
        *   Instantiates the `LlmAgent`.
        *   **Crucially, patches `MCPTool` instances** by adding a `.func` attribute pointing to their `.run_async` method. This is a workaround for the ADK's streaming tool execution path expecting this attribute.
*   **`mcp_server/` directory:**
    *   `cocktail.py`: A Python script that acts as an MCP server, exposing cocktail-related tools.
    *   `weather_server.py`: A Python script that acts as an MCP server, exposing weather-related tools.
*   **`static/` directory:**
    *   Intended for your HTML, JavaScript, and CSS files to build the user interface that interacts with the WebSocket.

## Deployment (Example: Google Cloud Run)

To deploy this application to Google Cloud Run:

1.  **Create a `Dockerfile`:**
    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.11-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the requirements file into the container at /app
    COPY requirements.txt .

    # Install any needed packages specified in requirements.txt
    # Using uv for faster installs
    RUN pip install uv && uv pip install --system --no-cache-dir -r requirements.txt

    # Copy the rest of the application code into the container at /app
    COPY . .

    # Make port 8000 available to the world outside this container
    EXPOSE 8000

    # Define environment variable
    ENV GOOGLE_GENAI_USE_VERTEXAI=${GOOGLE_GENAI_USE_VERTEXAI}
    ENV GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
    ENV GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}
    ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
    # Add other environment variables needed by your app

    # Run uvicorn when the container launches
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
2.  **Build and Push the Container:**
    Use Google Cloud Build and Artifact Registry.
    ```bash
    export PROJECT_ID="your-gcp-project-id"
    export SERVICE_NAME="adk-streaming-mcp-agent"
    export REGION="your-gcp-region" # e.g., us-central1

    gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}" .
    ```
3.  **Deploy to Cloud Run:**
    ```bash
    gcloud run deploy "${SERVICE_NAME}" \
      --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}" \
      --platform="managed" \
      --region="${REGION}" \
      --allow-unauthenticated \
      --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}" # Set other env vars as needed
      # If using Vertex AI, ensure the Cloud Run service account has Vertex AI User role.
      # Consider CPU allocation, memory, concurrency settings.
    ```

## Important Notes & Troubleshooting

*   **MCPTool Patching:** The `agent_config.py` includes a patch for `MCPTool` instances. This is because the ADK's streaming tool execution mechanism (`functions.py`) currently expects a `.func` attribute on tool objects. This patch assigns the tool's `run_async` method to `.func`.
*   **MCP Server Availability:** Ensure your local MCP server scripts (`cocktail.py`, `weather_server.py`) are correctly implemented and can be started by the `StdioServerParameters`. If using remote MCP servers like Airbnb, ensure they are accessible.
*   **ADK Version:** This setup has been tested with certain versions of ADK. If you encounter issues, compatibility with the ADK version might be a factor.
*   **Firewall:** If running MCP servers on different machines or in containers, ensure firewalls are configured to allow communication.
*   **Client-Side Implementation:** This README focuses on the backend. A functional client (`static/index.html`, `static/client.js`) is required to interact with the WebSocket endpoint. Refer to ADK streaming examples for client-side WebSocket and audio handling.

```


audio sentiment analysis

natural language module 

export SERVICE_NAME='galactic-streamhub'
export LOCATION='us-central1'

# Replace with your Google Cloud Project ID
export PROJECT_ID='silver-455021'
```

In Cloud Shell, execute the following command:

```bash
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $LOCATION \
  --project $PROJECT_ID \
  --memory 4G \
  --allow-unauthenticated
```