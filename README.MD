# AVA - Galactic StreamHub: A Multimodal, Multi-Agent AI Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

**Hackathon Project for the Agent Development Kit Hackathon with Google Cloud #adkhackathon**

Welcome to the Galactic StreamHub, powered by AVA (Advanced Visual Assistant)! This project showcases a sophisticated multi-agent AI system built using Google's Agent Development Kit (ADK). AVA can interact via text, voice, and live video, understand complex user goals, perceive the user's environment, and orchestrate tasks using specialized tools and delegated agents.

**[https://galatic-streamhub-140457946058.us-central1.run.app/]** 

**[https://devpost.com/software/galactic-streamhub]**

**[https://medium.com/@James_Masciano/ava-building-a-glimpse-of-i-o-2025s-agentic-multimodal-future-with-google-s-adk-for-bddbaac17d3c]** 


![Galactic StreamHub UI](/assets/ui.png)

## Table of Contents

*   [Features](#features)
*   [Architecture Overview](#architecture-overview)
*   [Tech Stack](#tech-stack)
*   [Setup & Installation](#setup--installation)
    *   [Prerequisites](#prerequisites)
    *   [Clone Repository](#clone-repository)
    *   [Environment Configuration](#environment-configuration)
*   [Running the Application Locally](#running-the-application-locally)
*   [Running MCP Servers](#running-mcp-servers)
*   [Deployment (Example: Google Cloud Run)](#deployment-example-google-cloud-run)
*   [Project Structure](#project-structure)
*   [Key Learnings & Workarounds](#key-learnings--workarounds)
*   [Future Enhancements](#future-enhancements)
*   [Contributing](#contributing)
*   [License](#license)
*   [Acknowledgements](#acknowledgements)

## Features

*   **Multimodal Interaction:** Communicate with AVA via text, voice, and live video stream.
*   **Visual Understanding:** AVA can analyze objects and elements from your live webcam feed.
*   **Multi-Agent System:**
    *   A **Root Agent** (multimodal Gemini Flash) acts as the primary interface.
    *   A **ProactiveContextOrchestratorAgent** (custom `BaseAgent`) manages the core logic for proactive and reactive assistance. It delegates to:
        *   **EnvironmentalMonitorAgent** (`LlmAgent`): Analyzes visual context and user hints to identify proactive opportunities.
        *   **ContextualPrecomputationAgent** (`LlmAgent`): If a proactive opportunity is identified, this agent formulates suggestions and pre-fetches relevant information using tools.
        *   **ReactiveTaskDelegatorAgent** (`LlmAgent`): Handles explicit user tasks or executes actions based on accepted proactive suggestions.
*   **Proactive Assistance:** AVA can anticipate user needs based on visual context and general queries, offering timely suggestions.
*   **Tool Integration (MCP):** Leverages Model Context Protocol (MCP) tools for:
    *   Cocktail recipes
    *   Weather information
    *   Google Maps functionalities (geocoding, place search).
*   **Agent as a Tool:** A dedicated **GoogleSearchAgent** (wrapped as an `AgentTool`) is available to other agents for general information retrieval.
*   **Real-time Streaming:** Bidirectional streaming of audio and text using WebSockets.
*   **Dynamic UI:** A futuristic, dark-themed web interface with 3D animated elements and a space-themed background.

## Architecture Overview

AVA employs a multi-agent architecture orchestrated by the Google Agent Development Kit:

1.  **Client (Web Browser):** Provides the user interface for text, voice (Web Audio API), and video (WebRTC/HTML5 Media) input. Communicates with the backend via WebSockets.
2.  **Backend (FastAPI Server):**
    *   Manages WebSocket connections.
    *   Hosts the ADK `Runner` and `SessionService`.
    *   Initializes and manages MCPToolsets for external services.
    *   Defines and orchestrates the ADK agents.
3.  **Root Agent (`mcp_streaming_assistant`):**
    *   An `LlmAgent` (Gemini Flash Multimodal).
    *   Receives multimodal input from the client.
    *   Performs initial analysis and decides to either:
        *   Use one of its directly available MCP tools (via `MCPToolset`).
        *   Delegate to the `ProactiveContextOrchestrator` tool (which wraps the `ProactiveContextOrchestratorAgent`).
4.  **ProactiveContextOrchestratorAgent & its Sub-Agents:**
    *   This custom agent orchestrates the proactive/reactive flow.
    *   `EnvironmentalMonitorAgent`: Identifies contextual keywords from visual input and user hints.
    *   `ContextualPrecomputationAgent`: If a proactive context is identified, this agent formulates a suggestion and can use tools (including the `GoogleSearchAgentTool` or MCP tools like CocktailDB) to pre-fetch data.
    *   `ReactiveTaskDelegatorAgent`: Handles direct user requests or tasks following an accepted proactive suggestion. It can use MCP tools or the `GoogleSearchAgentTool` as needed.
5.  **GoogleSearchAgentTool:** An `LlmAgent` specifically for performing Google searches, made available as a tool to other agents.
5.  **MCP Servers:** External processes (Stdio servers for CocktailDB, Weather, Google Maps) that provide tool functionalities via the Model Context Protocol.


![Architecture Diagram](/assets/flowchart.jpg)

## Tech Stack

*   **AI Framework:** Google Agent Development Kit (ADK) for Python
*   **LLM:** Google Gemini (Flash for streaming/multimodal, Pro for some agent logic during development)
*   **Backend:** Python, FastAPI, Uvicorn
*   **Frontend:** HTML, CSS, JavaScript (with Vanta.js for background animation)
*   **Real-time Communication:** WebSockets
*   **External Tools:** MCP (Model Context Protocol) for CocktailDB, Weather, Google Maps
*   **Deployment:** Google Cloud Run, Docker
*   **Dependency Management/Runner (Local):** `uv`

## Setup & Installation

### Prerequisites

*   Python 3.13
*   `uv` (recommended for faster virtual environment and package management: `pip install uv`)
*   Node.js and `npx` (if you intend to run the Google Maps MCP server locally via ADK's StdioServerParameters).
*   Access to Google Cloud Platform project with:
    *   Vertex AI API enabled.
    *   Secret Manager API enabled (if using it for API keys).
*   Google Cloud CLI (`gcloud`) configured and authenticated.
*   A `.env` file configured with your Google Cloud Project ID, location, and potentially API keys (see `.env.example`).

### Clone Repository

```bash
git clone [https://github.com/surfiniaburger/galactic-streamhub]
cd [galactic-streamhub]
```

### Environment Configuration

1.  **Create a virtual environment (using `uv`):**
    ```bash
    uv venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate.bat # On Windows CMD
    # .venv\Scripts\Activate.ps1 # On Windows PowerShell
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

3.  **Set up your `.env` file:**
    Copy `.env.example` to `.env` and fill in your details:
    ```env
    # For Vertex AI
    GOOGLE_GENAI_USE_VERTEXAI=TRUE
    GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    GOOGLE_CLOUD_LOCATION="us-central1" # Or your preferred region

    # For Google Maps API Key via Secret Manager (optional, used by main.py)
    # GOOGLE_MAPS_API_KEY_SECRET_NAME="your-secret-name-for-maps-key"

    # If NOT using Secret Manager, and your Maps MCP server expects GOOGLE_MAPS_API_KEY directly:
    # GOOGLE_MAPS_API_KEY="your-actual-maps-api-key"
    ```
    *Note: The `main.py` is configured to fetch the Maps API key from Secret Manager. If you provide it directly as `GOOGLE_MAPS_API_KEY`, you'll need to adjust `main.py` or ensure the Maps MCP server consumes this variable.*

## Running the Application Locally

1.  **Ensure MCP Servers are Ready:**
    *   The `main.py` application will attempt to start the Weather and Cocktail MCP servers using `StdioServerParameters`. Ensure `mcp_server/weather_server.py` and `mcp_server/cocktail.py` are executable and their dependencies are met.
    *   The Google Maps MCP server (if configured in `main.py`) will also be started by `StdioServerParameters` (requires `npx`).

2.  **Start the FastAPI Application:**
    Navigate to the project root directory (where `main.py` is located) in your terminal and run:
    ```bash
    uv run uvicorn main:app --reload
    ```
    The application will typically be available at `http://127.0.0.1:8000`.

3.  **Access the Web Interface:**
    Open your browser and go to `http://127.0.0.1:8000`.

## Running MCP Servers

This application is configured to start the Weather, Cocktail, and (if API key is available) Google Maps MCP servers as subprocesses using ADK's `StdioServerParameters`.

*   **Weather Server:** `mcp_server/weather_server.py`
*   **Cocktail Server:** `mcp_server/cocktail.py`
*   **Google Maps Server:** Uses `npx -y @modelcontextprotocol/server-google-maps` (requires Node.js/npx and a `GOOGLE_MAPS_API_KEY` environment variable available to it, which `main.py` attempts to provide from Secret Manager).

If you encounter issues with these starting automatically, you may need to run them manually in separate terminals before starting `main.py` and adjust `main.py` to connect to them as existing services if necessary.

## Deployment (Example: Google Cloud Run)

To deploy this application to Google Cloud Run:

1.  **Create a `Dockerfile` (if not already present):**
    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.11-slim

    # Set the working directory in the container
    WORKDIR /app

    # Install uv first for faster dependency installation
    RUN pip install uv

    # Copy the requirements file into the container at /app
    COPY requirements.txt .

    # Install any needed packages specified in requirements.txt using uv
    RUN uv pip install --system --no-cache-dir -r requirements.txt

    # Copy the rest of the application code into the container at /app
    COPY . .

    # Make port 8000 (or the port Uvicorn runs on) available
    # Cloud Run will automatically use the PORT environment variable.
    # Uvicorn by default runs on 8000 if PORT is not set.
    # Let's ensure Uvicorn uses the PORT env var provided by Cloud Run.
    # EXPOSE 8080 # Or whatever PORT Cloud Run provides. This line is actually not strictly needed for Cloud Run.

    # Command to run the application using Uvicorn, listening on the port specified by Cloud Run's PORT env var.
    CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]
    ```

2.  **Set Environment Variables for Deployment:**
    In your Cloud Shell or local terminal (with `gcloud` CLI configured):
    ```bash
    export SERVICE_NAME='galactic-streamhub' # Or your preferred service name
    export LOCATION='us-central1'         # Or your preferred region
    export PROJECT_ID='your-gcp-project-id' # Replace with your Project ID
    ```

3.  **Deploy to Cloud Run:**
    Ensure you are in the project's root directory (where the `Dockerfile` is).
    ```bash
    gcloud run deploy $SERVICE_NAME \
      --source . \
      --region $LOCATION \
      --project $PROJECT_ID \
      --memory 4Gi \
      --cpu 2 \
      --concurrency 80 \
      --allow-unauthenticated \
      --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_MAPS_API_KEY_SECRET_NAME=your-secret-name-for-maps-key" # Add other necessary env vars
      # Ensure the service account for Cloud Run has access to Secret Manager if using it.
    ```
    *   Adjust `--memory`, `--cpu`, `--concurrency` as needed.
    *   `--allow-unauthenticated` makes the service publicly accessible. Remove if you need authentication.
    *   Use `--set-env-vars` to pass environment variables required by your application (like the Secret Manager name for the Maps API key).
    *   The service account running the Cloud Run instance will need permissions to access Secret Manager secrets if you're using that feature.

    Or simply run
```bash
    gcloud run deploy $SERVICE_NAME \
      --source . \
      --region $LOCATION \
      --project $PROJECT_ID \
      --memory 4G \
      --allow-unauthenticated
```

    On successful deployment, you will be provided a URL to the Cloud Run service

## Project Structure

```
.
├── .env.example                # Example environment variables
├── .venv/                      # Virtual environment (if created with uv venv)
├── Dockerfile                  # For containerization
├── README.md                   # This file
├── agent_config.py             # Defines ADK agents, tools, and their configurations
├── proactive_agents.py         # Defines the ProactiveContextOrchestratorAgent and its sub-agents
├── google_search_agent/        # Directory for the Google Search agent
│   └── agent.py
├── main.py                     # FastAPI application, WebSocket endpoints, ADK runner setup
├── mcp_server/                 # Directory for local MCP server scripts
│   ├── cocktail.py             # MCP server for cocktail recipes
│   └── weather_server.py
├── requirements.txt            # Python dependencies
└── static/                     # Frontend assets
    ├── css/
    │   └── style.css
    ├── js/
    │   ├── app.js              # Main client-side JavaScript
    │   ├── audio-player.js     # AudioWorklet for playback
    │   └── audio-recorder.js   # AudioWorklet for recording
    └── index.html              # Main HTML page
```

## Key Learnings & Workarounds

During development, especially when integrating various ADK components for a streaming multimodal experience, a few workarounds and insights were key:

*   **Patching ADK Tools (`.func` attribute):**
    *   Both `MCPTool` and `AgentTool` instances needed to be "patched" by adding a `.func` attribute that pointed to their respective `run_async` methods. This was necessary because the ADK's internal tool execution flow (in `google/adk/flows/llm_flows/functions.py`) appeared to expect this attribute for certain tool types in streaming scenarios.
*   **`AgentTool` Argument Handling (`KeyError: 'request'`):**
    *   The `AgentTool.run_async` method (when wrapping an agent like `ProactiveContextOrchestratorAgent`) expects its input arguments from the LLM to be bundled under a single key `args['request']`.
    *   To resolve this, the Root Agent was instructed to call the `ProactiveContextOrchestrator` tool by passing a single argument named `request`, whose value is a JSON string containing the actual parameters (`user_goal`, `seen_items`). The orchestrator then accesses these from session state, which the Root Agent populates.
*   **Secure WebSockets (`wss://`):** Ensured client-side JavaScript uses `wss://` when the application is deployed over HTTPS to prevent mixed content errors.
*   **Model Selection for Stability & Capability:** Using `gemini-2.0-flash` for most agents for speed and cost-effectiveness, while ensuring the `EnvironmentalMonitorAgent` uses a multimodal model if it's directly processing image data (though current flow has Root Agent do primary visual analysis).

## Future Enhancements

*   More sophisticated visual understanding (e.g., fine-grained object attribute recognition).
*   More robust proactive trigger logic (e.g., using confidence scores or a dedicated decision-making agent).
*   Enhanced error handling and feedback in the UI.
*   Persistent session storage for longer-term memory.
*   Integration of more diverse tools and agents.
*   Refined Vanta.js background with more dynamic elements (e.g., subtle nebulae).

## Contributing

This project was developed for the ADK Hackathon. While contributions are not actively sought at this moment, feel free to fork the repository, explore, and adapt the concepts for your own projects! If you find bugs or have significant improvement suggestions related to the ADK usage patterns demonstrated, feel free to raise an issue on this GitHub repo or (if applicable) on the official `google/adk-python` repository.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

*   The Google Agent Development Kit team for providing the framework.
*   The FastAPI and Uvicorn communities.
*   Vanta.js for the cool animated background.
*   Participants and organizers of the #adkhackathon.
