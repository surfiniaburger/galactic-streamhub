# Project Documentation: AVA - Galactic StreamHub

This document provides a detailed explanation of the key components of the AVA (Advanced Visual Assistant) project, a multimodal, multi-agent AI assistant.

## 1. Project Overview (from README.MD)

AVA is a sophisticated multi-agent AI system built using Google's Agent Development Kit (ADK). It is designed to interact with users through text, voice, and live video. The system can understand complex user goals, perceive the user's environment, and orchestrate tasks using specialized tools and delegated agents.

**Core Features:**

*   **Multimodal Interaction:** Engages via text, voice, and live video.
*   **Visual Understanding:** Analyzes objects from a live webcam feed.
*   **Multi-Agent System:** A root agent orchestrates a series of specialist agents for tasks like proactive assistance, environmental monitoring, research, and visualization.
*   **Proactive Assistance:** Anticipates user needs based on visual and conversational context.
*   **Tool Integration:** Leverages the Model Context Protocol (MCP) for external services like cocktail recipes, weather, and Google Maps.
*   **Security:** Integrates Google Cloud's Model Armor to sanitize prompts and prevent attacks.
*   **Privacy by Design:** Sensitive data like facial and vocal analysis is processed locally. All sessions are secured with Firebase Authentication.
*   **Accessibility Suite:** Includes workflows for visual, auditory, and cognitive assistance.

The project is deployed on Google Cloud Run and utilizes a tech stack including FastAPI, WebSockets, and MongoDB for memory.

## 2. `agent_config.py`

This is the central configuration file for the entire multi-agent system. It defines the instructions, tools, and interconnections of all the agents that make up AVA.

**Key Responsibilities:**

*   **Agent Definitions:** It instantiates all agents, including the main `root_agent` (the user-facing multimodal assistant), specialist agents for research (`PubMedRAGAgent`, `VisualizationAgent`), proactive assistance (`ProactiveContextOrchestratorAgent`), and accessibility (`AccessibilityOrchestratorAgent`, `AuditoryAssistanceOrchestratorAgent`, etc.).
*   **Agent Instructions:** This file contains the detailed instructional prompts for each agent, defining their roles, workflows, and constraints. For example, `ROOT_AGENT_INSTRUCTION_STREAMING` is a complex prompt that governs how the main agent behaves, how it delegates tasks, and how it uses its tools.
*   **Tool Aggregation:** It collects and configures all the tools available to the agents. This includes:
    *   **MCP Tools:** Tools for external services like weather and cocktails.
    *   **AgentTools:** Wraps other agents (like the `GoogleSearchAgent` or the `MasterResearchSynthesizer`) so they can be used as tools by the root agent.
    *   **FunctionTools:** Standard Python functions (like `query_pubmed_articles` or `generate_simple_bar_chart`) that are exposed as tools to the agents.
*   **Callback Configuration:** It sets up shared callbacks for functionalities that apply to multiple agents, such as loading conversation history from memory (`load_memory_before_model_callback`) and saving the interaction after a turn (`save_interaction_after_model_callback`).
*   **Agent Assembly:** It constructs the final agent hierarchy. For example, it builds the `MasterResearchSynthesizer` as a `SequentialAgent` that runs a series of other agents in a specific order to perform complex research tasks.

In essence, `agent_config.py` is the blueprint that wires the entire AI system together, turning individual components into a cohesive, intelligent application.

## 3. `main.py`

This file is the entry point for the application. It sets up the FastAPI web server that hosts the agent and handles all real-time communication with the user's browser.

**Key Responsibilities:**

*   **FastAPI Server:** Initializes the web server.
*   **WebSocket Endpoint (`/ws`):** This is the primary communication channel. It handles:
    *   **Authentication:** It uses the Firebase Admin SDK to verify the user's ID token, ensuring only authenticated users can connect. The user's Firebase UID is used as the session ID.
    *   **Session Management:** It initiates an ADK agent session for each connected user.
    *   **Message Handling:** It manages the bidirectional streaming of data (text, audio, video frames) between the client and the agent.
*   **Application Lifespan Management (`app_lifespan`):**
    *   **Startup:** Before the application starts accepting requests, it initializes the Firebase Admin SDK and dynamically configures and starts the MCP (Model Context Protocol) servers for tools like weather and cocktails.
    *   **Shutdown:** When the application is shutting down, it gracefully closes the connections to the MCP servers.
*   **Static File Serving:** It serves the frontend HTML, CSS, and JavaScript files to the user's browser.
*   **Image Serving:** It includes a special endpoint (`/static/medical_images/...`) to dynamically convert and serve medical DICOM images as PNGs, making them viewable in the browser.

## 4. `main_agent` Directory

This directory defines the primary user-facing agent module, making it discoverable by the ADK command-line tools for evaluation and testing.

*   **`agent.py`:** This file is a slightly older or alternative version of the main `agent_config.py`. It contains similar logic for defining and assembling the root agent and its sub-agents. Its primary purpose in the current project structure is to provide a self-contained agent definition that the ADK's evaluation tools (`adk eval`) can use.
*   **`__init__.py`:** This file makes the `main_agent` directory a Python package and exposes the `root_agent` instance from `agent.py`, which is necessary for the ADK tools to find and run it.
*   **`cocktailsEval.evalset.json`:** This is an evaluation set file for the ADK. It contains a series of predefined conversation turns, including user inputs and the expected "golden" responses and tool calls from the agent. It is used with the `adk eval` command to test the agent's behavior and ensure its performance remains consistent as the code changes.
*   **`.adk/eval_history/...`:** These JSON files are the results generated by running `adk eval`. They contain a detailed breakdown of an evaluation run, comparing the agent's actual behavior against the `cocktailsEval.evalset.json` and providing scores for metrics like tool accuracy and response similarity.

## 5. `proactive_agents.py`

This file defines the agents and logic responsible for AVA's proactive assistance capabilities. It allows the agent to anticipate user needs based on context rather than just reacting to direct commands.

**Key Components:**

*   **`ProactiveContextOrchestratorAgent`:** This is the central agent in this module. It doesn't directly interact with the user but instead orchestrates a sequence of sub-agents to decide if and how to be proactive.
*   **`EnvironmentalMonitorAgent`:** This agent analyzes the user's visual environment (from `seen_items`) and their query to identify "context keywords." For example, if it sees ingredients for a cocktail, it might output the keyword `cocktail_making`.
*   **`ContextualPrecomputationAgent`:** If the `EnvironmentalMonitorAgent` identifies a proactive opportunity, this agent steps in. It formulates a suggestion for the user (e.g., "I see you have gin and lime. Would you like a recipe for a Gimlet?") and pre-fetches the necessary information (the Gimlet recipe) using its tools.
*   **`ReactiveTaskDelegatorAgent`:** If no proactive opportunity is found, or if the user accepts a proactive suggestion, this agent is responsible for executing the task. It can use pre-fetched data to respond quickly or call tools to handle the user's direct request.

This proactive loop (Monitor -> Precompute -> Suggest/Execute) allows AVA to provide more intelligent and timely assistance.

## 6. `callbacks.py`

This file contains callback functions that are hooked into the agent's lifecycle to perform actions at specific points, such as before the AI model is called or after it generates a response.

**Key Callbacks:**

*   **`security_check_callback`:** This is a critical security function that runs *before* the model processes a prompt. It uses Google Cloud's **Model Armor** to inspect the user's input for malicious content like prompt injection or jailbreaking attempts. If a threat is detected, it blocks the request and prevents it from reaching the agent.
*   **`load_memory_before_model_callback`:** This function runs before the agent thinks. It fetches the user's recent conversation history and their saved persona (name, goals) from the MongoDB database. This information is then prepended to the prompt, giving the agent the necessary context to have a coherent and personalized conversation.
*   **`save_interaction_after_model_callback`:** This function runs *after* the agent has generated its response. It takes the user's last input and the agent's final answer and saves them to the MongoDB database, ensuring a persistent record of the conversation for future recall.

## 7. `mongo_memory.py`

This module implements the agent's long-term memory using a MongoDB database. It allows the agent to remember conversations and user preferences across different sessions.

**Key Features:**

*   **`MongoMemory` Class:** This class inherits from the ADK's `BaseMemoryService` and implements the core logic for interacting with the database.
*   **Connection Management:** It handles connecting to the MongoDB Atlas cluster, retrieving the connection URI securely from environment variables or Google Cloud Secret Manager.
*   **Specialized Collections:** It uses different collections to store different types of memory:
    *   `interaction_history`: Stores the turn-by-turn conversation history.
    *   `personas`: Stores user-specific information like their name and goals.
*   **CRUD Operations:** It provides methods to add, retrieve, and search for interactions and personas.
*   **Vector Search:** It integrates with Vertex AI's Text Embedding models. When an interaction is saved, it generates a vector embedding of the conversation. This enables powerful semantic search, allowing the agent to find past conversations based on meaning rather than just keywords.
*   **Index Management:** It ensures that the necessary database indexes (including a Vector Search index on Atlas) are created for efficient querying.

## 8. `security.py`

This file is dedicated to the security of the agent, specifically by integrating with **Google Cloud Model Armor**.

**Key Functionality:**

*   **`sanitize_prompt_with_model_armor`:** This is the core function. It takes a user's prompt and sends it to a predefined Model Armor template.
*   **Model Armor Integration:** It configures and initializes the `ModelArmorClient`. Model Armor is a GCP service that acts as a firewall for LLMs. It uses specialized models to detect and block harmful inputs.
*   **Threat Detection:** The function checks the response from Model Armor to see if a threat (like a prompt injection attack) was found.
*   **Safety Check:** It returns a simple `is_safe` status, which the `security_check_callback` in `callbacks.py` uses to decide whether to allow the prompt to proceed to the agent or to block it.

This module provides a critical layer of defense, protecting the agent from manipulation and ensuring its responses remain safe and on-topic.

## 9. `deployment/README.md`

This file is a detailed migration guide, documenting the process of moving the "Galatic Streamhub" application from its initial deployment on **Google Cloud Run** to a more robust and scalable environment on **Google Kubernetes Engine (GKE) Autopilot**.

**Key Migration Steps Documented:**

*   **GKE Cluster Setup:** Creating a GKE Autopilot cluster with a dedicated service account.
*   **Kubernetes Configuration:** Defining all the necessary Kubernetes resources as YAML files:
    *   **Secrets:** For securely managing API keys and database URIs.
    *   **Deployment:** To define the application pods, container image, and environment variables.
    *   **Service:** To expose the application within the cluster.
    *   **HPA (Horizontal Pod Autoscaler):** To automatically scale the application based on CPU load.
    *   **Ingress:** To manage external traffic, handle SSL termination, and support WebSockets.
*   **HTTPS and Domain Setup:**
    *   Reserving a static IP address.
    *   Using `ManagedCertificate` to automatically provision and renew SSL certificates.
    *   Configuring DNS to point the custom domain (`app.galactic-streamhub.com`) to the Ingress.
*   **Workload Identity:** Configuring Workload Identity to securely allow the application running on GKE to access other Google Cloud services (like Vertex AI and Secret Manager) without needing to manage service account keys.
*   **Troubleshooting:** It includes key learnings from the migration process, such as configuring WebSocket timeouts and resolving SSL-related issues.
