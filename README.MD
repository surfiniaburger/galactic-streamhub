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
        *   **PubMedRAGAgent** (`LlmAgent`): Orchestrates biomedical research by querying a local PubMed database, performing live web searches, querying the ClinicalTrials.gov API, and querying the OpenFDA API for drug adverse events. It synthesizes this information and can add new web-found articles to the local knowledge base.
        *   **VisualizationAgent** (`LlmAgent`): Creates various data visualizations (e.g., bar charts, line graphs) using data retrieved from other agents.
*   **Proactive Assistance:** AVA can anticipate user needs based on visual context and general queries, offering timely suggestions.
*   **Tool Integration (MCP):** Leverages Model Context Protocol (MCP) tools for:
    *   Cocktail recipes
    *   Weather information
    *   Google Maps functionalities (geocoding, place search).
*   **Agent as a Tool:** A dedicated `GoogleSearchAgent` (wrapped as an `AgentTool`) is available to other agents for general information retrieval.
*   **Knowledge Augmentation:** Can ingest new biomedical articles found via web search into its local knowledge base (MongoDB & BigQuery).
*   **Real-time Streaming:** Bidirectional streaming of audio and text using WebSockets.
*   **Dynamic UI:** A futuristic, dark-themed web interface with 3D animated elements and a space-themed background.
*   **Comprehensive Accessibility Suite:** Provides workflows for visual (scene description, OCR), auditory (speech sentiment, sound recognition), and cognitive (text simplification) assistance.

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
    *   **If General Research Synthesis**: Delegates to `MasterResearchSynthesizer`. This agent sequentially:
        *   Runs `DataGatheringAndConnectionAgent`: This involves parallel searches by `ResearchOrchestratorAgent` (across PubMed, web, local clinical trials DB, OpenFDA), followed by `KeyInsightExtractorAgent` and `TrialConnectorAgent`.
        *   Runs `ParallelSynthesisAgent`: This involves parallel work by `TextSynthesizerAgent` (generates text, can propose ingestion via `IngestionRouterAgentTool`), `ChartProducerAgent` (uses `VisualizationAgentTool`), and `ImageEvidenceProducerAgent` (uses `MultimodalEvidenceAgentTool` to find and prepare medical image URLs).
        *   Runs `FinalReportAggregatorAgent`: Combines all synthesized parts into the final response.
    *   `VisualizationAgent`: If the user requests a visualization, this agent is triggered to create a chart from the provided data.
5.  **Specialized Search Agents:**
    *   `GoogleSearchAgentTool`: An `LlmAgent` for general web searches, available as a tool.
    *   `ClinicalTrialsSearchAgent`: An `LlmAgent` for querying the live ClinicalTrials.gov API.
    *   `OpenFDASearchAgent`: An `LlmAgent` for querying the live OpenFDA API for drug adverse events.
6.  **MCP Servers:** External processes (Stdio servers for CocktailDB, Weather, Google Maps) that provide tool functionalities via the Model Context Protocol.

7.  **Research Workflow:**
    *   The `MasterResearchSynthesizer` is a sequential agent that manages a deep research pipeline. It coordinates parallel data gathering, insight extraction, and the generation of a final report composed of text, charts, and images from multiple synthesis agents.
8.  **Accessibility Workflow:**
    *   The `AccessibilityOrchestratorAgent` is triggered by accessibility-related queries (e.g., "what do you see?", "read this label").
    *   It analyzes the user's intent and delegates to one of two specialist agents:
        *   **`SceneDescriberAgent`**: Uses the list of visually identified items from the session state to construct a natural-language description of the environment.
        *   **`TextReaderAgent`**: Uses a dedicated tool (`perform_ocr_on_last_frame`) that calls the **Google Cloud Vision API** to accurately extract and read text from the live video feed.
9.  **Firebase Authentication:**
    *   The frontend uses the Firebase client SDK to handle Google Sign-In and retrieve a secure ID token.
    *   The backend uses the Firebase Admin SDK to verify this token upon a WebSocket connection attempt, ensuring only authenticated users can interact with the agent. The user's Firebase UID is used as the session ID.

10. **Auditory Accessibility Workflow:**
    *   The `AuditoryAssistanceOrchestratorAgent` is triggered by auditory-related queries (e.g., "how do I sound?", "what's that noise?").
    *   It analyzes the user's intent and delegates to one of two specialist agents:
        *   **`AudioSentimentAgent`**: Uses the Google Cloud Natural Language API to analyze the sentiment of the user's speech.
        *   **`SoundRecognitionAgent`**: Identifies and reports significant ambient sounds from the user's environment.

11. **Cognitive Accessibility Workflow:**
    *   The `CognitiveAssistanceOrchestratorAgent` is triggered when a user asks to simplify text.
    *   The root agent first uses the `set_text_for_simplification` tool to store the target text.
    *   The orchestrator then delegates to the `TextSimplificationAgent`, which rephrases the stored text in simple, clear language.




## Firebase Authentication

Galactic StreamHub now implements robust user authentication using Firebase, ensuring that only verified users can connect to the agent backend and interact with AVA.

### Frontend (Client-Side - `static/js/app.js`)

1.  **Firebase Initialization**:
    *   The Firebase app is initialized using your project's `firebaseConfig` (ensure your actual API key and other details are securely managed, e.g., via environment variables for deployment, though the provided `app.js` has placeholders).
2.  **Google Sign-In**:
    *   Users can sign in using their Google accounts via a "Sign In with Google" button.
    *   Firebase Authentication (`signInWithPopup` with `GoogleAuthProvider`) handles the OAuth flow.
3.  **Authentication State Management**:
    *   An `onAuthStateChanged` listener monitors the user's sign-in status.
    *   **If Signed In**:
        *   The user's display name and avatar are shown.
        *   An ID token is retrieved from Firebase using `user.getIdToken(true)`. This token is a secure credential.
        *   The WebSocket connection to the backend (`/ws`) is established, and the ID token is passed as a query parameter (`?token=<ID_TOKEN>`).
        *   UI elements (message input, send button, audio/video controls) are enabled.
    *   **If Signed Out**:
        *   A login gate is displayed, prompting the user to sign in.
        *   UI elements are disabled.
        *   Any existing WebSocket connection is closed.
4.  **Sign-Out**:
    *   A "Sign Out" button allows users to end their session.

### Backend (Server-Side - `main.py`)

1.  **Firebase Admin SDK Initialization**:
    *   The Firebase Admin SDK is initialized during the FastAPI application's startup (`app_lifespan`).
    *   It uses Application Default Credentials, which is suitable for Google Cloud environments (e.g., Cloud Run, Cloud Workstations). For local development, ensure you've run `gcloud auth application-default login`.
    *   The `projectId` is configured during initialization.
2.  **Token Verification in WebSocket Endpoint**:
    *   The `/ws` WebSocket endpoint now requires a `token` query parameter.
    *   When a client attempts to connect, the backend uses `auth.verify_id_token(token)` to verify the received ID token with Firebase's authentication servers.
    *   **If Verification Succeeds**:
        *   The connection is accepted.
        *   The `uid` (unique user ID) extracted from the decoded token is used as the `session_id` for the ADK agent session. This securely links the agent's activity to the authenticated user.
    *   **If Verification Fails** (e.g., token is invalid, expired, or tampered with):
        *   An error is logged.
        *   The WebSocket connection is refused (closed with code 1011).

### Security Benefits

*   **Secure Access**: Only authenticated users can establish a WebSocket connection and interact with the agent.
*   **User Identification**: The backend can reliably identify users based on their Firebase UID, allowing for potential future features like personalized experiences or data storage.
*   **Standardized Authentication**: Leverages Google's robust and secure Firebase Authentication platform.

## Auditory Accessibility

To enhance the user's awareness and provide valuable feedback, AVA includes a suite of auditory accessibility tools. These are managed by the `AuditoryAssistanceOrchestratorAgent`, which routes user requests to the appropriate specialist.

### Speech Sentiment Analysis

*   **Functionality**: Users can ask about their tone or how they sound (e.g., "Do I sound angry?", "How am I coming across?").
*   **Underlying Technology**: The `AudioSentimentAgent` takes the user's most recent speech, transcribes it to text, and uses the **Google Cloud Natural Language API** to perform detailed sentiment analysis.
*   **User Benefit**: Provides users with real-time feedback on their vocal tone, helping them understand how they may be perceived by others. The agent returns a clear description of the detected sentiment (positive, negative, or neutral) along with a confidence score.

### Ambient Sound Recognition

*   **Functionality**: The agent can listen to the user's environment and identify significant background noises (e.g., "What was that sound?", "Can you hear the doorbell?").
*   **Underlying Technology**: The `SoundRecognitionAgent` processes the incoming audio stream to detect and classify common ambient sounds, which are stored in the session state. The agent retrieves and reports these detected sounds upon request.
*   **User Benefit**: Enhances situational awareness by identifying important environmental cues that the user might have missed, such as a doorbell, a fire alarm, or a dog barking.


## Cognitive Accessibility

To assist users who may benefit from simplified information, AVA includes a cognitive accessibility workflow.

### Text Simplification

*   **Functionality**: Users can request to simplify a block of text by asking "can you make this easier to read?" or "explain this to me simply".
*   **Underlying Technology**: The `CognitiveAssistanceOrchestratorAgent` manages this task. The root agent first captures the text to be simplified using a dedicated tool. The orchestrator then delegates to a `TextSimplificationAgent` which uses Gemini to rephrase the content in clear, easy-to-understand language, removing jargon and complex sentence structures.
*   **User Benefit**: Makes complex information more accessible, aiding comprehension for a wider range of users.




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

    Synthesize the latest research on the diagnosis and treatment of non-small cell lung cancer. Find connections to ongoing clinical trials and show me a visual example of a lung nodule from a CT scan.
    The application will typically be available at `http://127.0.0.1:8000`.

3.  **Access the Web Interface:**
    Open your browser and go to `http://127.0.0.1:8000`.

## Agent Evaluation

This project uses the ADK's evaluation framework (`adk eval`) to test the performance and correctness of the agents against predefined conversation scenarios (eval sets). This ensures that as the agent's logic evolves, its behavior remains consistent and correct.

### Running Evaluations

This project contains evaluation sets for multiple agents to test specific capabilities in isolation.

**To evaluate the primary `main_agent`'s conversational and tool-use abilities:**

```bash
adk eval main_agent main_agent/cocktailsEval.evalset.json
```

This command will:
1.  Load the `root_agent` defined in the `main_agent` module.
2.  Run the conversation turns defined in `main_agent/cocktailsEval.evalset.json`.
3.  Compare the agent's actual tool calls and final responses against the expected "golden" responses defined in the eval set.
4.  Generate a detailed result file in `main_agent/.adk/eval_history/`.

The evaluation results help track metrics like `tool_trajectory_avg_score` (did the agent call the right tools?) and `response_match_score` (how similar was the agent's text response to the expected one?).

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
    export SERVICE_NAME='galatic-streamhub' # Or your preferred service name
    export LOCATION='us-central1'         # Or your preferred region
    export PROJECT_ID='silver-455021' # Replace with your Project ID
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

```
gcloud run services describe galatic-streamhub --format export --region us-central1 --project silver-455021 > galatic-streamhub-cloudrun.yaml

kubectl apply -f kubernetes-deployment.yaml -n default 

kubectl get pods -n default -w
kubectl rollout status deployment/galatic-streamhub -n default

kubectl logs deployment/galatic-streamhub -n default -f
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
├── google_search_agent/  
├── pubmed_pipeline.py          # Logic for querying PubMed and ingesting articles
├── ingest_clinical_trials.py   # Logic for ingesting clinical trial data
├── ingest_multimodal_data.py   # Logic for finding similar medical images      # Directory for the Google Search agent
│   └── agent.py
├── openfda_pipeline.py         # Logic for querying OpenFDA API
├── clinical_trials_pipeline.py # Logic for querying ClinicalTrials.gov API
├── main.py                     # FastAPI application, WebSocket endpoints, ADK runner setup
├── mcp_server/                 # Directory for local MCP server scripts
│   ├── cocktail.py             # MCP server for cocktail recipes
│   └── weather_server.py
├── tools/                      # Directory for agent tools
│   └── chart_tool.py           # Tool 
for creating data visualizations
    └── web_utils.py            # Utility for fetching web article text
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
    *   These findings and workarounds were detailed in GitHub Issue #1084 on google/adk-python.
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
