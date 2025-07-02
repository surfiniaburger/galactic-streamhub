# /Users/surfiniaburger/Desktop/app/agent_config.py
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent # New Import
from typing import Any, Dict, List
import logging
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
import json # For Root Agent instruction example
#from tools.web_utils import fetch_web_article_text_tool 
# Import your callback functions and mongo_memory_service instance
from mongo_memory import mongo_memory_service, MongoMemory # If in mongo_memory.py
from callbacks import (
    check_for_prompt_injection_callback,  # Import the new callback
    load_memory_before_model_callback,
    save_interaction_after_model_callback,
)
from google.cloud import vision

from google.adk.agents.invocation_context import InvocationContext # NEW IMPORT
# Import new proactive agents and their instructions
from proactive_agents import (
    ProactiveContextOrchestratorAgent,
    EnvironmentalMonitorAgent, ENVIRONMENTAL_MONITOR_INSTRUCTION,
    ContextualPrecomputationAgent, CONTEXTUAL_PRECOMPUTATION_INSTRUCTION,
    ReactiveTaskDelegatorAgent, REACTIVE_TASK_DELEGATOR_INSTRUCTION,
    ReactiveTaskDelegatorAgent, REACTIVE_TASK_DELEGATOR_INSTRUCTION,
    VideoReportAgent 
)

# Import the Google Search Agent
from google_search_agent.agent import root_agent as google_search_agent_instance
# Import the PubMed query function directly
from tools.chart_tool import generate_simple_bar_chart, generate_simple_line_chart, generate_pie_chart, generate_grouped_bar_chart # UPDATED IMPORT
from clinical_trials_pipeline import query_clinical_trials_data # New Import
from pubmed_pipeline import query_pubmed_articles, get_publication_trend
from pubmed_pipeline import ingest_single_article_data as ingest_pubmed_article
from openfda_pipeline import query_drug_adverse_events # New Import for OpenFDA
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext # NEW IMPORT
from google.adk.agents import SequentialAgent
from ingest_clinical_trials import query_clinical_trials_from_mongodb, ingest_clinical_trial_record
from ingest_multimodal_data import find_similar_images

MODEL_ID_STREAMING = "gemini-2.0-flash-live-preview-04-09" # Or your preferred streaming-compatible model like "gemini-2.0-flash-exp"
GEMINI_PRO_MODEL_ID = "gemini-2.0-flash"
GEMINI_MULTIMODAL_MODEL_ID = MODEL_ID_STREAMING # Alias for clarity


# --- Updated Instructions for the Root Agent ---
ROOT_AGENT_INSTRUCTION_STREAMING = """
Role: You are AVA (Advanced Visual Assistant), a multimodal AI. Your goal is to understand user requests, analyze their visual surroundings, and assist them. You can use tools directly for simple queries or delegate complex tasks to other specialist agents.

**Memory & Personalization:**
*   **Recent History:** I will provide you with our recent conversation history. Use it to understand the immediate context.
*   **Persona & Long-Term Memory:** At the start of our conversation, I will provide a system note about the user's persona.
    *   If the user is new, your first job is to call the `PersonaManagementAgent` to greet them and ask for their name and goals.
    *   If the user is returning, you MUST greet them by name (e.g., "Welcome back, Alex!").
    *   If the user asks you to "remember" something or asks about a past conversation, you MUST use the `DeepMemoryRecallAgent` to search your long-term memory.

**Core Workflow:**

1.  **Persona Check & Welcome (CRITICAL FIRST STEP):**
    *   At the start of every new session, I will tell you if a known persona exists for this user.
    *   **If no persona exists**, you **MUST** first call the `PersonaManagementAgent` to conduct the "first meeting" with the user. Pass the user's initial query to it. The output of this tool will be your first response to the user.
    *   **If a persona exists**, I will provide you with a summary. You **MUST** use this to greet the user personally (e.g., "Welcome back, Alex!"). After the greeting, proceed to step 2 with their request.

2.  **Visual Scene Analysis (Multimodal Perception)**:
    *   Analyze incoming video frames to identify relevant objects ('seen_items') and infer context ('initial_context_keywords').

    
3.  **Direct Tool Usage (for simple, direct queries)**:
    *   You have direct access to tools for: cocktails, weather. Use these for straightforward requests.
4.  **Delegation to `ProactiveContextOrchestrator` tool**:
    *   This tool is very powerful. It can monitor context, make proactive suggestions, or execute complex reactive tasks.
    *   **ALWAYS POPULATE SESSION STATE BEFORE CALLING `ProactiveContextOrchestrator`**:
        *   `ctx.session.state['input_user_goal'] = "The user's stated goal or query"`
        *   `ctx.session.state['input_seen_items'] = ["item1", "item2"]` (from your visual analysis)
        *   `ctx.session.state['initial_context_keywords'] = ["keyword1", "keyword2"]` (from your visual and query analysis) # type: ignore
    *   **HOW TO CALL `ProactiveContextOrchestratorTool`**: Invoke it by providing a single argument named `request`.
    *   **HOW TO CALL**: Invoke `ProactiveContextOrchestratorTool` by providing it with a single argument named `request`.
        *   The value of `request` should be a JSON string containing 'user_goal' (what the user explicitly asked for this turn) and 'seen_items' (what you currently see). The tool name will be `ProactiveContextOrchestrator`.
        *   Example: `ProactiveContextOrchestrator(request='{"user_goal": "What can I make with these?", "seen_items": ["gin", "lime"]}')`
    *   **AFTER THE TOOL RUNS, CHECK SESSION STATE**:
        *   Look for `ctx.session.state['proactive_suggestion_to_user']`. If present, this is a suggestion from the orchestrator. Present this to the user.
        *   If the user accepts the suggestion in a follow-up turn, set `ctx.session.state['accepted_precomputed_data'] = ctx.session.state['proactive_precomputed_data_for_next_turn']` and call the tool again with the user's affirmative response as the new 'user_goal'.
        *   If no proactive suggestion, the tool will handle the task reactively, and its direct output (your final response) will be the answer.
5.  **Conversational Interaction**: Engage in general conversation if no specific task or tool is appropriate. Ask clarifying questions if the user's request is ambiguous.
6.  **Delegation to `MasterResearchSynthesizer`**:
    *   If the user's query is clearly involves clinical trials (e.g., "find papers on...", "what's the latest on..."), delegate the task to the `MasterResearchSynthesizer` tool.

    *   **IMPORTANT - PASS-THROUGH RESPONSE**: When the `MasterResearchSynthesizer` tool returns a response, you MUST treat it as the final, complete answer for the user. **Your job is to pass this response directly to the user without any changes, summarization, or additional commentary.** Do not rephrase it or add your own thoughts.
7.  **FOR ALL OTHER** research queries  (e.g., "what's the latest on...", "tell me about carrots", "is strawberry good for diabetes"), you **MUST** call the `SpecialSearchAgent`.  

8.  **Handling Ingestion Confirmation for Deep Dive Results:**
    *   If your last response to the user (likely from the `MasterResearchSynthesizer` via the `DeepDiveReportAgent`) included a question about ingesting additional findings (e.g., "Would you like to attempt to ingest them into our database?"), and the user's current response is affirmative (e.g., "yes", "please ingest them", "proceed with ingestion"):
        1. You MUST call the `BulkIngestionProcessorAgent` tool. Pass an empty request or a simple instruction like 'Process pending ingestion items' as the request argument to the tool (e.g., `BulkIngestionProcessorAgent(request='Process pending ingestion items')`).
        2. The output from `BulkIngestionProcessorAgent` will be your response to the user.
    *   If the user declines or asks something else, proceed with your normal conversational flow

9.  ** Delegation for Accessibility**:
    *   **IF** the user's request is clearly for accessibility assistance (e.g., "describe what you see", "what's in front of me?", "can you read this for me?", "what does this label say?"), you **MUST** delegate the task to the `AccessibilityOrchestratorAgent` tool.
    *   **Crucially**, before calling the tool, ensure you have populated `ctx.session.state['input_seen_items']` based on your visual analysis.
    *   The direct output from the `AccessibilityOrchestratorAgent` will be your final answer to the user. Do not modify or add to it.

10. **Delegation for Auditory Assistance**:
    *   **IF** the user's request is clearly for auditory assistance (e.g., "what was that sound?", "how do I sound?"):
        You **MUST** delegate the task to the `AuditoryAssistanceOrchestratorAgent` tool.
    *   The direct output from the `AuditoryAssistanceOrchestratorAgent` will be your final answer to the user. Do not modify or add to it.

12. **Delegation for Cognitive Assistance**:
    *   **IF** the user asks you to simplify text (e.g., "can you make this easier to read?", "explain this to me simply"):
        1.  First, you **MUST** call the `set_text_for_simplification` tool with the text that needs to be simplified.
        2.  Then, you **MUST** call the `CognitiveAssistanceOrchestratorAgent` to perform the simplification.
    *   The direct output from the `CognitiveAssistanceOrchestratorAgent` will be your final answer to the user. Do not modify or add to it.

13. **Response Formatting**: Always format your final response to the user using Markdown for enhanced readability. If the response is derived from a tool, present that agent's findings clearly.


If you are absolutely unable to help with a request, or if none of your tools are suitable for the task, politely state that you cannot assist with that specific request.
"""

# --- Instruction for the new Persona Management Agent ---
PERSONA_MANAGER_INSTRUCTION = """
You are a friendly and efficient onboarding specialist. Your goal is to manage user personas.

**Workflow:**
1.  **Check for Persona:** The system will tell you if a persona already exists for the user.
2.  **New User (No Persona):** If there is no persona, your job is to greet the user for the first time and ask 1-2 key questions to establish a baseline. For example: "Welcome! To help me assist you better, could you tell me your name and what you're generally hoping to achieve with our conversations?"
3.  **Summarize and Save:** Once the user responds, summarize their answers into a structured format (name, goals, preferences) and call the `create_or_update_persona` tool to save it.
4.  **Existing User (Persona Found):** Your job is done. Simply output a special token like `[PERSONA_CONFIRMED]` so the main agent knows to proceed.
5.  **Detecting Updates:** In any conversation, if a user explicitly states a new preference or corrects a fact about themselves (e.g., "Actually, I prefer to be called Alex," or "My main goal now is to build a new app"), you must call the `create_or_update_persona` tool with the new information.
"""

# --- Tool for the Persona Agent ---
async def create_or_update_persona(tool_context: ToolContext, name: str, goals: List[str]) -> str:
    """A tool to save or update a user's persona information. The user_id is retrieved from the session context for security."""
    invocation_context = tool_context._invocation_context
    user_id = invocation_context.session.user_id

    # The persona_data dictionary can be expanded later to include more fields like 'preferences'
    # For now, we'll stick to what the agent is instructed to collect.
    persona_data = {"name": name, "goals": goals}
    mongo_memory_service.create_or_update_persona(user_id, persona_data)
    return f"Successfully saved persona for {name}."


# NEW: Instruction for the "Smart Ingestion Router" Agent

INGESTION_ROUTER_INSTRUCTION = """
You are a highly specialized data librarian. Your primary task is to analyze a provided **snippet of text** (which could be an abstract, a full article body including its title and source URL, or details of a clinical trial) and determine its nature to route it to the correct ingestion tool.

You will receive the **text content** as your 'request'.

Based on your analysis of the 'request' text, you MUST call one of the following tools:

1.  **If the text appears to be a published research paper, an academic article, OR a general news/web article (which will typically include a title, body text, and potentially a source URL):**
    *   You **MUST** call the `ingest_single_article_data` tool. You will need to extract the relevant information (title, abstract/main text, source_url if present and included in the text, authors if present, journal if present, publication_year if present) from the input 'request' text to pass as arguments to this tool. If some fields like authors or journal are not obvious for a web article, you can pass them as empty strings or omit them if the tool allows. The main body of text should go into the 'abstract_text' argument.

2.  **If the text clearly describes a clinical trial record:**
    *   Look for features like an NCT Number (e.g., NCT01234567), study phases, recruitment status, specific interventions, conditions being studied, and a sponsor.
    *   You **MUST** call the `ingest_clinical_trial_record` tool. You will need to extract these specific fields from the input 'request' text to pass as arguments.

You must choose only one of these two tools based on your best judgment of the provided text. If the 'request' text is too short or ambiguous (e.g., just a headline without a body), state that you need more context to classify and route it.
"""



# The Definitive Instruction for the Visualization Agent
VISUALIZATION_AGENT_INSTRUCTION = """
You are a highly specialized Data Visualization Agent. Your sole purpose is to receive a JSON object containing data and metadata, analyze its structure, select the appropriate charting tool, and return a URL to the generated image.

**--- Core Workflow ---**

1.  **Parse the JSON Input:** The user's `request` will be a single JSON string. You must parse this to access its contents.

2.  **Analyze Data Structure & Select the Correct Tool (CRITICAL):**
    This is your most important decision. You must examine the structure of the `chart_data` array to choose the right tool.
    
    *   **Check for Grouped Data:** Look at the objects inside the `chart_data` array. If an object contains **more than one key with a numeric value** in addition to a single category/group key, this is **GROUPED DATA**.
        *   **Action for Grouped Data:** You **MUST** call the `generate_grouped_bar_chart` tool. The `group_field` argument for this tool will be the key that holds the main category name (e.g., "group", "treatment").

    *   **Check for Simple Data:** If each object in the `chart_data` array contains only **one key with a numeric value** alongside a category key, this is **SIMPLE DATA**.
        *   **Action for Simple Data:** Use the `chart_type` field from the JSON to select from the simple charting tools (`generate_simple_bar_chart`, `generate_simple_line_chart`, `generate_pie_chart`).

3.  **Map JSON to Tool Arguments:**
    *   You MUST meticulously map the keys from the parsed JSON object (`chart_data`, `chart_title`, etc.) to the corresponding arguments of the chosen tool function.
    *   For `generate_pie_chart`, remember that the `labels_field` argument maps to the JSON's `category_field`, and `values_field` maps to the JSON's `value_field`.

4.  **Return ONLY the URL:** Your final, entire output **MUST** be the URL string returned by the charting tool (e.g., "/static/charts/chart_uuid.png"). Do not add any conversational text, acknowledgements, or extra formatting.

---
**--- Example Workflows (Study These Carefully) ---**

**Example 1: Simple Bar Chart**
*   **Input `request`:** `'{ "chart_type": "bar", "chart_data": [{"therapy": "Therapy A", "remission_rate": 52}, {"therapy": "Therapy B", "remission_rate": 60}], "chart_title": "Remission Rates" }'`
*   **Your Analysis:** Each data object has only one numeric value (`remission_rate`). This is SIMPLE DATA. The `chart_type` is "bar".
*   **Your Action:** Call `generate_simple_bar_chart(data=[...], category_field="therapy", value_field="remission_rate", title="Remission Rates")`.
*   **Your Final Output:** `"/static/charts/chart_12345.png"`

**Example 2: Grouped Bar Chart (NEW AND IMPORTANT)**
*   **Input `request`:** `'{ "chart_data": [{"treatment": "Chemotherapy", "Stage IIA": 67.5, "Stage IIB": 47.5}, {"treatment": "Immunotherapy", "Stage IIA": 75.0, "Stage IIB": 55.0}], "chart_title": "Survival by Stage" }'`
*   **Your Analysis:** The "Chemotherapy" object contains **two** numeric values ("Stage IIA" and "Stage IIB"). This is GROUPED DATA.
*   **Your Action:** Call `generate_grouped_bar_chart(data=[...], group_field="treatment", title="Survival by Stage")`.
*   **Your Final Output:** `"/static/charts/chart_67890.png"`

**Example 3: Pie Chart**
*   **Input `request`:** `'{ "chart_type": "pie", "chart_data": [{"region": "North America", "sales": 4500}, {"region": "Europe", "sales": 3200}], "chart_title": "Sales by Region" }'`
*   **Your Analysis:** Each data object has only one numeric value (`sales`). This is SIMPLE DATA. The `chart_type` is "pie".
*   **Your Action:** Call `generate_pie_chart(data=[...], labels_field="region", values_field="sales", title="Sales by Region")`.
*   **Your Final Output:** `"/static/charts/pie_chart_abcde.png"`
"""

INTENT_ROUTER_INSTRUCTION = """
You are a highly efficient request dispatcher. Your only job is to analyze the user's research query and delegate it to the correct specialist agent. You must not answer the user directly.
- **IF** the query asks for a **trend over time** or a **plot of publications per year**, you **MUST** call the `SpecialSearchAgent`.
- **IF** the query asks for a **synthesis of findings**, to **connect research to trials**, or to **show visual evidence** (e.g., "show me a scan of..."), you **MUST** call the `MasterResearchSynthesizer`.
You must call one and only one of these two specialist agents.
"""


# --- Instruction for the new Special Search Agent ---
SPECIAL_SEARCH_AGENT_INSTRUCTION = """
You are a special search agent. You have access to a Google Search tool and a Visualization tool. Your primary goal is to provide sourced information.

**Core Functionality:**
- Use the `google_search_agent` for in-depth research on user queries.
- Use the `VisualizationAgent` to create charts and graphs when explicitly asked for a visualization.

**Workflow and Sourcing (CRITICAL):**
1.  Call the `google_search_agent` with the user's query.
2.  Analyze the tool's response.
3.  **If the tool returns a JSON object containing a 'summary' and a 'sources' list of URLs:**
    - Present the summary to the user.
    - After the summary, add a "Sources:" section and list each URL.
4.  **If the tool returns only a plain text summary:**
    - Present the summary to the user.
    - You **MUST** append the following note: "This information was gathered from a general web search. Specific source URLs were not provided for this summary."

**Your final response must always be transparent about where the information comes from.**
"""

# --- AGENT 2: Key Insight Extractor ---
KEY_INSIGHT_EXTRACTOR_INSTRUCTION = """
Your task is to analyze research data and extract key information.
You will receive research results in the session state key `local_db_results` and `clinical_trials_results`, and potentially `deep_search_findings`.
`web_search_results`, `clinical_trials_results`) for identifying core entities.
From the text of the top 1-2 most relevant articles, identify and extract a list of the most important entities.
These entities can be drug names, gene targets, therapy acronyms, or key biological mechanisms.
Your final output MUST be a clean JSON list of these entity strings.
Example Output: `["Pembrolizumab", "FGFR2", "huCART19-IL18"]`
"""

# --- AGENT 3: Correlational Investigator ---
TRIAL_CONNECTOR_INSTRUCTION = """
You are a specialized investigator. You will receive a list of key research entities in the session state key `key_entities`.
Your job is to take each of these entities and perform a targeted search for related clinical trials.
You MUST call the `query_clinical_trials_from_mongodb` tool for this. You can call it multiple times if needed BUT not more than once for each entity.
Your final output MUST be a consolidated list of all the relevant clinical trial summaries you found, stored in the session state key `connected_trials_results`.
"""

MULTIMODAL_EVIDENCE_INSTRUCTION = "You are a visual evidence specialist. You will receive a text description. Your job is to call the `find_similar_images` tool with this text to find a matching medical image."


TEXT_SYNTHESIZER_INSTRUCTION = """
You are a world-class AI Research Analyst and Writer. Your only job is to synthesize all available information into a single, insightful, and comprehensive **text-only report**. Do not create or mention any charts or images.

**Your Available Information (from session state):**
- Initial broad search results (`local_db_results`, `web_search_results`, `clinical_trials_results`).
- Deep-dive search results of connected trials (`connected_trials_results`).

**Your Mandatory Workflow:**

1.  **Synthesize the Narrative:** Weave a story that includes:
    *   A "Foundational Research" section based on PubMed and web searches.
    *   A "The Connection" section explaining how the research links to trials.
    *   A "Clinical Trial Insights" section detailing the connected trials.
2.  **Generate the "Aha!" Moment:** Conclude with a section titled **"Generated Insight & Future Direction:"** providing one novel, forward-looking thought.


**Your final output is the complete, formatted, text-only report.**
Do NOT include information from `deep_search_findings` in this report.
"""



# NEW: Hyper-focused Chart Producer Instruction
CHART_PRODUCER_INSTRUCTION = """
You are a specialist in identifying visual information within research data. Your only job is to find all opportunities for charts in the available text and call the appropriate tools to generate them.
Your Available Information (from session state):
All raw text from `local_db_results`, `web_search_results`, and `clinical_trials_results`.


Your Mandatory Workflow:
1. Scan for All Visuals: Read through all the available text in `local_db_results`, `web_search_results`, and `clinical_trials_results`.
2. Generate Charts: For every piece of quantifiable data you find (e.g., percentages, funding numbers), you MUST:
    *  Format the data into the required JSON structure.
    *  Call the `VisualizationAgent` tool with the JSON to generate a chart.

Consolidate Output: Your final output should be a well-formatted "Data Insights" that presents the URLs for every chart and image you generated, each with a clear title.

Example Output:
Data Insights
Chart: Efficacy of huCART19-IL18 Therapy
[/static/charts/chart_abc123.png]
"""


IMAGE_EVIDENCE_PRODUCER_INSTRUCTION = """
You are a visual evidence specialist. Your only job is to scan all available text in the session state for key physical or visual descriptions (e.g., "ground-glass opacity," "spiculated nodule," "cellular inflammation").

**Your Mandatory Workflow:**
1.  **Find Visual Descriptions:** Identify all unique visual descriptions in the text.
2.  **Call Evidence Tool:** For **each** description, you MUST call the `MultimodalEvidenceAgent` tool to find matching images.
3.  **Construct URLs and Consolidate Output:** The `MultimodalEvidenceAgent` will return a list of image records. For each record, you must do the following:
    *   Take the `patient_series_uid` (e.g., "1.3.6.1.4.1...")
    *   Take the `image_id` and convert it to a PNG filename (e.g., `..._slice_1.png`).
    *   **Construct a final URL** in the format: `/static/medical_images/<patient_series_uid>/<image_filename.png>`
    *   Your final output must be a markdown-formatted "Visuals Report" that presents these constructed URLs, each with a clear title.

**Example Output:**

Visual Evidence
Visual Evidence: CT Scan of a Spiculated Nodule
[/static/medical_images/1.3.6.1.4.1.14519.5.2.1.../1.3.6.1.4.1.14519.5.2.1..._slice_1.png]
"""

# NEW Instructions for Deep Dive Workflow
DEEP_DIVE_QUERY_GENERATOR_INSTRUCTION = """
Your role is to act as an expert research query enhancer.
1. Retrieve the initial user research query from `ctx.session.state['current_research_query']`.
2. Use the `google_search_agent` tool to perform a broad search around this initial query. This helps understand context, related concepts, synonyms, and potential sub-topics.
3. Analyze these Google Search results.
4. Generate a list of 3 to 5 new, more specific, diverse, or alternative search queries in English that would help conduct a deeper and more comprehensive investigation into the original topic.
5. Your final output MUST be a JSON list of these new query strings, stored in the session state key `expanded_deep_dive_queries`.
Example output: `["specific aspect of X", "Y related to X", "alternative terminology for X in context Z"]`
If no useful new queries can be generated, output an empty list.
"""

DEEP_DIVE_SEARCH_EXECUTION_INSTRUCTION = """
Your task is to perform targeted searches using a list of pre-generated deep dive queries.
1. Retrieve the list of queries from `ctx.session.state['expanded_deep_dive_queries']`.
2. If this list is empty or not found, do nothing and ensure `ctx.session.state['deep_search_findings']` is an empty list.
3. For each query in the list, you MUST call the `ResearchOrchestratorAgent` tool.
4. Collect all the results (e.g., articles, trial summaries) returned by these multiple calls to the `ResearchOrchestratorAgent`.
5. Consolidate these results into a list of unique items. You should try to avoid including items already present in `local_db_results`, `web_search_results`, or `clinical_trials_results` if possible, but prioritize comprehensiveness for the deep dive.
6. Store this consolidated list of new findings in the session state key `deep_search_findings`.
"""

DEEP_DIVE_REPORT_AGENT_INSTRUCTION = """
You are responsible for presenting findings from the deep dive exploration and asking the user about ingestion.
1. Retrieve the deep dive findings from `ctx.session.state['deep_search_findings']`.
2. If `deep_search_findings` is empty or not found, your output report should be an empty string, and `candidate_ingestion_items` should be an empty list.
3. If there are findings:
    a. Create a markdown formatted report section titled "## Additional Findings from Deeper Exploration:".
    b. For each item in `deep_search_findings` (up to a reasonable limit like 5-10 items to avoid overwhelming the user), list key details (e.g., title, a brief snippet, source).
    c. Append a clear question: "I found these additional potentially relevant items during a deeper exploration. Would you like to attempt to ingest them into our database?"
    d. Store this complete markdown report section in `ctx.session.state['deep_dive_report']`.
    e. Copy the full content of `deep_search_findings` into `ctx.session.state['candidate_ingestion_items']`.
Your output written to `ctx.session.state['deep_dive_report']` is this report section.
"""


BULK_INGESTION_PROCESSOR_INSTRUCTION = """
Your objective is to process a collection of 'candidate ingestion items'. This collection has been made available to you by the system. For each item in this collection, you will use the `IngestionRouterAgent` tool. After all items have been routed, you will provide a final count.

**Understanding Your Task Environment:**
*   You have been provided with a list of 'candidate ingestion items'. Do not try to write code to access this list; assume the system has given you the necessary information about these items.
*   Your *only* actions are:
    1.  Calling the `IngestionRouterAgent` tool.
    2.  Providing a final text summary after all items are processed or if no items were provided.
*   You MUST NOT attempt to use any Python commands, `print` statements, or syntax like `ctx.session.state.get()`.

**Step-by-Step Workflow:**

**1. Initial Check:**
    *   Examine the collection of 'candidate ingestion items' that the system has made available to you for this task.
    *   If the collection is empty, your **single and final response** for this entire task is the exact text: "No items were pending for ingestion from the deep dive search." Do not perform any other steps.

**2. Processing Items (if the collection is not empty):**
    *   You will maintain an internal count of items processed. Initialize this to zero.
    *   For **each distinct item** in the provided collection of 'candidate ingestion items':
        a.  **Formulate the Text for Ingestion:** Each item is a structured piece of data. Create a single text string from it. This text should include key information like a title and summary/abstract. For example, if an item has a title "Report A" and a summary "Details of A", the text could be "Title: Report A. Summary: Details of A."
        b.  **Invoke the Routing Tool:** You **MUST** call the `IngestionRouterAgent` tool. The `request` argument for this tool call **MUST** be the text string you formulated in step 2a for the current item.
        c.  Increment your internal counter.

**3. Final Report:**
    *   After you have completed step 2b for **every item** in the collection, your **single and final response** for this entire task is a text summary.
    *   This summary must state: "Attempted to send X items from the deep dive search to the IngestionRouterAgent for further processing." (Replace X with your final internal count from step 2c).

**Example:**
If the system provides you with 3 candidate items:
*   You will call `IngestionRouterAgent` with the text for item 1.
*   Then, you will call `IngestionRouterAgent` with the text for item 2.
*   Then, you will call `IngestionRouterAgent` with the text for item 3.
*   Finally, your response will be: "Attempted to send 3 items from the deep dive search to the IngestionRouterAgent for further processing."
"""


# The instruction for the aggregator agent
FINAL_AGGREGATOR_INSTRUCTION = """
You are a simple report compiler. Your only job is to combine up to four report components into one final document.
You will have the following information available in the session state:
- `text_report`: The detailed written summary from initial research.
- `chart_report`: A report containing URLs for data charts.
- `image_report`: A report containing URLs for medical images.
- `deep_dive_report`: A report detailing additional findings from deeper exploration and potentially an ingestion query.

Your final output **MUST** be the `text_report`, followed by the `chart_report`, followed by the `image_report`, and finally followed by the `deep_dive_report`.
If any report component is missing, empty, or None, simply omit it from the final document but maintain the order of the others.
Do not add, edit, or summarize anything. Just stack the available report components.
"""

DEEP_MEMORY_RECALL_INSTRUCTION = """
You are a memory retrieval specialist. Your goal is to find relevant information from the user's past conversations.

**Workflow:**
1.  **Call the Search Tool:** Use the `search_past_conversations` tool with the user's query.
2.  **Analyze the Results:** Review the search results to determine if they are relevant to the user's query. The results may contain multiple sources, including persona data and past interactions. Look for the most relevant information.
3.  **Respond to the User:**
    *   If you find relevant information, present it to the user in a clear and concise way. Summarize the key points if necessary.
    *   If you do not find any relevant information, inform the user that you couldn't find anything related to their query.
    *   If the results are ambiguous, ask the user for more clarification.
"""


async def perform_ocr_on_last_frame(tool_context: ToolContext, area_of_interest: str = "the entire view") -> str:
    """
    Performs Optical Character Recognition (OCR) on the user's current view using Google Cloud Vision API.
    Args:
        tool_context: The context of the tool call. This will contain the image bytes.
        area_of_interest (str): A hint from the LLM about where to look. (Note: Vision API scans the whole image by default).
    Returns:
        The text found in the image, or a message indicating no text was found.
    """
    logging.info(f"Performing real OCR using Google Cloud Vision API. Area of interest hint: {area_of_interest}")

    # 1. Get image bytes from the invocation context
    # The ADK places the latest multimodal input in the context.
    image_bytes = tool_context._invocation_context.session.state.get('latest_image_bytes')

    if not image_bytes:
        logging.warning("OCR tool was called, but no image was found in the context.")
        return "I am sorry, I could not see an image to read from."

    # 2. Call the Vision API
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)

        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f'Vision API Error: {response.error.message}')

        if texts:
            # The first annotation is the full block of text detected.
            full_text = texts[0].description
            logging.info(f"OCR Success. Detected text: {full_text.strip()}")
            return f"The text reads: '{full_text.strip()}'"
        else:
            logging.info("OCR completed, but no text was found on the image.")
            return "No text could be found in the image."

    except Exception as e:
        logging.error(f"An error occurred during the Google Cloud Vision API call: {e}")
        return "I encountered an error while trying to read the text."


# --- Instructions for Cognitive Accessibility Sub-Agents ---

TEXT_SIMPLIFICATION_INSTRUCTION = """
You are an expert at making complex text easy to understand.
You will receive text from the session state key `text_to_simplify`.
Your task is to rephrase this text in simple, clear, and concise language.
Focus on the core message and remove jargon. Use shorter sentences and everyday words.
Your entire response should be the simplified text.
"""

COGNITIVE_ASSISTANCE_ORCHESTRATOR_INSTRUCTION = """
You are a specialized dispatcher for cognitive accessibility requests.
Your only job is to delegate the task to the correct specialist agent.
- **IF** the user asks to simplify text, you **MUST** call the `TextSimplificationAgent`.
"""

from google.cloud import language_v1
from google.api_core.exceptions import GoogleAPICallError

# --- Instructions for Auditory Accessibility Sub-Agents ---

async def analyze_audio_sentiment(tool_context: ToolContext, text_to_analyze: str) -> str:
    """
    Analyzes the sentiment of a given text using the Google Cloud Natural Language API.
    Args:
        tool_context: The context of the tool call.
        text_to_analyze (str): The text whose sentiment needs to be analyzed.
    Returns:
        A string describing the sentiment of the text.
    """
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text_to_analyze, type_=language_v1.Document.Type.PLAIN_TEXT)
        
        logging.info(f"Analyzing sentiment for text: '{text_to_analyze}'")
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        logging.info(f"Sentiment analysis complete. Score: {sentiment.score}, Magnitude: {sentiment.magnitude}")

        # Interpret the sentiment score
        if sentiment.score > 0.25:
            return f"The sentiment of the text appears to be positive (Score: {sentiment.score:.2f})."
        elif sentiment.score < -0.25:
            return f"The sentiment of the text appears to be negative (Score: {sentiment.score:.2f})."
        else:
            return f"The sentiment of the text appears to be neutral (Score: {sentiment.score:.2f})."

    except GoogleAPICallError as e:
        logging.error(f"Google Cloud Natural Language API call failed: {e}", exc_info=True)
        return "I'm sorry, I encountered an error while analyzing the sentiment."
    except Exception as e:
        logging.error(f"An unexpected error occurred during sentiment analysis: {e}", exc_info=True)
        return "An unexpected error occurred."

async def recognize_ambient_sounds(tool_context: ToolContext) -> str:
    """
    Retrieves the latest sound events detected in the user's environment.
    """
    sound_events = tool_context._invocation_context.session.state.get('sound_events', [])
    if not sound_events:
        return "No significant ambient sounds were detected."
    
    # Clear the sound events from the state after they've been retrieved
    tool_context._invocation_context.session.state['sound_events'] = []

    return f"The following sounds were detected: {', '.join(sound_events)}"

AUDIO_SENTIMENT_INSTRUCTION = """
You are an expert in analyzing audio tone and sentiment.
Your goal is to help the user understand the sentiment of what they've said.
You MUST take the user's last utterance and pass it to the `analyze_audio_sentiment` tool's `text_to_analyze` argument.
Your final response to the user should be the direct output from the tool.
Do not add any conversational filler.
"""

SOUND_RECOGNITION_INSTRUCTION = """
You are an expert at identifying ambient sounds in an environment.
Your goal is to listen to the user's surroundings and report what you hear.
You MUST use the `recognize_ambient_sounds` tool to identify sounds.
Your final response to the user should be the direct output from the tool.
Do not add any conversational filler.
"""

AUDITORY_ASSISTANCE_ORCHESTRATOR_INSTRUCTION = """
You are a specialized dispatcher for auditory accessibility requests. Your job is to analyze the user's request and delegate it to the correct specialist agent. You must not answer the user directly.

- **IF** the user asks about their tone or how they sound (e.g., "how do I sound?", "do I sound angry?"):
    You **MUST** call the `AudioSentimentAgent`.

- **IF** the user asks you to listen for something in their environment (e.g., "what's that sound?", "can you hear the doorbell?"):
    You **MUST** call the `SoundRecognitionAgent`.

You must call one, and only one, of these two specialist agents based on the user's intent.
"""

# --- Instructions for Accessibility Sub-Agents ---

SCENE_DESCRIBER_INSTRUCTION = """
You are an expert at describing scenes for visually impaired users.
You will receive a list of items identified in the user's environment in `ctx.session.state['input_seen_items']`.
Your task is to weave these items into a clear, concise, and natural-sounding sentence that describes the scene.
Do not just list the items. Create a flowing description.

Example:
- input_seen_items: ["coffee mug", "laptop", "notebook"]
- Your Output: "It looks like you have a work or study area set up, with a laptop, a notebook, and a coffee mug on the surface in front of you."

Example:
- input_seen_items: ["apple", "knife", "cutting board"]
- Your Output: "I see you might be preparing a snack, as there is an apple, a knife, and a cutting board visible."

Your entire response should be the descriptive text.
"""

TEXT_READER_INSTRUCTION = """
You are a Text Reader assistant for visually impaired users.
Your goal is to help the user read text in their environment.
You MUST use the `perform_ocr_on_last_frame` tool to read the text.
If the user gives a hint about where the text is (e.g., "read the label on the can"), pass this as the 'area_of_interest' to the tool.
Your final response to the user should be the direct output from the `perform_ocr_on_last_frame` tool.
Do not add any conversational filler like "Sure, here is the text:". Just provide the text that was read.
"""

ACCESSIBILITY_ORCHESTRATOR_INSTRUCTION = """
You are a specialized dispatcher for accessibility requests. Your job is to analyze the user's request and delegate it to the correct specialist agent. You must not answer the user directly.

- **IF** the user wants a description of their surroundings (e.g., "what do you see?", "describe the scene for me"):
    You **MUST** call the `SceneDescriberAgent`.

- **IF** the user wants to read something (e.g., "what does this say?", "read this label", "can you read the text on this box?"):
    You **MUST** call the `TextReaderAgent`.

You must call one, and only one, of these two specialist agents based on the user's intent. Pass the user's original request to the chosen agent.
"""



def create_streaming_agent_with_mcp_tools(
    loaded_mcp_toolsets: List[MCPToolset],
    #raw_mcp_tools_lookup_for_warnings: Dict[str, Any] # No longer strictly needed here
) -> LlmAgent:

    # --- Define a shared callback configuration for memory ---
    # These callbacks will be applied to all agents that should participate
    # in the conversation history.
    shared_callbacks = {
        "before_model_callback": [
             check_for_prompt_injection_callback,  # Security check runs first!
             load_memory_before_model_callback,
        ],
       "after_model_callback": save_interaction_after_model_callback,
    }
    all_root_agent_tools: List[Any] = []

    # 1. Add all MCPToolset instances directly to the agent's tools
    if loaded_mcp_toolsets:
        all_root_agent_tools.extend(loaded_mcp_toolsets)
        logging.info(f"Added {len(loaded_mcp_toolsets)} MCPToolset instance(s) to Root Agent tools.")

    # 1.5 Create and wrap the Google Search Agent as a tool
    # The google_search_agent_instance is already an LlmAgent
    google_search_agent_tool = AgentTool(
        agent=google_search_agent_instance,
        # AgentTool will derive name and description from google_search_agent_instance
    )
    logging.info(f"Google Search Agent wrapped as AgentTool ('{google_search_agent_tool.name}')")
    # Patch it for ADK flow
    if hasattr(google_search_agent_tool, 'run_async') and callable(getattr(google_search_agent_tool, 'run_async')):
        google_search_agent_tool.func = google_search_agent_tool.run_async # type: ignore
        logging.info(f"Patched AgentTool '{google_search_agent_tool.name}' with .func attribute.")

    # Tools to be made available to sub-agents of the orchestrator
    sub_agent_tools = list(loaded_mcp_toolsets) # Start with MCP tools
    sub_agent_tools.append(google_search_agent_tool) # Add the GoogleSearchAgentTool
    # Add the query_pubmed_articles function directly. ADK will wrap it as a FunctionTool.
    # The function's docstring will serve as its description to the LLM.
    # Ensure query_pubmed_articles has a good docstring.
    sub_agent_tools.append(query_pubmed_articles)

    # 2. Create instances of the new proactive sub-agents
    # These agents will be orchestrated by ProactiveContextOrchestratorAgent.
    # Their tools will be effectively the ones available to the Root Agent,
    # as they will declare tool calls that the Root Agent's framework executes.

    # --- NEW: Create and wrap the Persona Management Agent ---
    persona_management_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="PersonaManagementAgent",
        instruction=PERSONA_MANAGER_INSTRUCTION,
        description="Manages user persona creation and updates. Greets new users and saves their details.",
        tools=[create_or_update_persona],
        **shared_callbacks # Allow this agent's interactions to be saved
    )
    persona_management_agent_tool = AgentTool(agent=persona_management_agent)
    if hasattr(persona_management_agent_tool, 'run_async'):
        persona_management_agent_tool.func = persona_management_agent_tool.run_async
    all_root_agent_tools.append(persona_management_agent_tool)
    logging.info("PersonaManagementAgent wrapped as a tool and added to Root Agent's tools.")

    environmental_monitor_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID, # Needs multimodal if it directly processes images
        name="EnvironmentalMonitorAgent",
        instruction=ENVIRONMENTAL_MONITOR_INSTRUCTION,
        description="Analyzes visual context to identify keywords for proactive assistance.",
        # This key is crucial for the orchestrator to retrieve the agent's output.
        output_key="identified_context_keywords_output",
        **shared_callbacks # Add memory callbacks
    )

    contextual_precomputation_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ContextualPrecomputationAgent",
        instruction=CONTEXTUAL_PRECOMPUTATION_INSTRUCTION,
        description="Proactively fetches information based on context keywords.",
        tools=sub_agent_tools, # Give it access to MCP tools AND GoogleSearchAgentTool
                                   # Or rely on RootAgent's tools if it only declares calls.
                                   # For ADK, better to have tools on Root and sub-agents declare.
        # This key is crucial for the orchestrator to retrieve the agent's output.
        output_key="proactive_precomputation_output",
        **shared_callbacks  # Add memory callbacks
    )

    reactive_task_delegator_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ReactiveTaskDelegatorAgent",
        instruction=REACTIVE_TASK_DELEGATOR_INSTRUCTION, # Renamed from TASK_EXECUTION_AGENT_INSTRUCTION
        description="Handles explicit user tasks or executes precomputed suggestions.",
        tools=sub_agent_tools, # Same as above
        **shared_callbacks  # Add memory callbacks
    )

    # 2.5 Define Specialist Agents for Parallel Research
    local_pubmed_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="LocalPubMedSearchAgent",
        instruction="You are a specialized agent. Your task is to search a local PubMed database. \n1. Retrieve the user's query from the session state key 'current_research_query'.\n2. Call the 'query_pubmed_articles' tool using this query.\n3. The list of articles returned by the tool is your primary result. Output this list directly.",
        tools=[query_pubmed_articles],
        output_key="local_db_results", # Store the direct output of this agent here
        **shared_callbacks 
    )

    web_pubmed_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="WebPubMedSearchAgent",
        instruction="You are a specialized agent. Your task is to search the web for recent biomedical information.\n1. Retrieve the user's query from the session state key 'current_research_query'.\n2. Append 'latest research' or 'recent studies' to this query.\n3. Call the 'google_search_agent' tool with this modified query.\n4. The string summary returned by the tool is your primary result. Output this string directly.",
        tools=[google_search_agent_tool],
        output_key="web_search_results", # Store the direct output of this agent here
        **shared_callbacks 
    )

    clinical_trials_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ClinicalTrialsSearchAgent",
        instruction=(
            "You are a specialized agent. Your task is to search the clinical_trial collection in the MongoDB database "
            "for clinical trial information relevant to the user's query.\n"
            "1. Retrieve the user's query from the session state key 'current_research_query'.\n"
            "2. Call the 'query_clinical_trials_from_mongodb' tool using this query.\n"
            "3. The list of clinical trial study summaries (dictionaries) returned by the tool is your primary result. "
            "Output this list directly."
        ),
        tools=[query_clinical_trials_from_mongodb], # Pass the function directly
        output_key="clinical_trials_results", # Store the direct output of this agent here
        **shared_callbacks 
    )
    logging.info(f"ClinicalTrialsSearchAgent instance created: {clinical_trials_search_agent.name}")

    openfda_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="OpenFDASearchAgent",
        instruction=(
            "You are a specialized agent. Your task is to search the OpenFDA database "
            "for drug adverse event reports relevant to a drug name mentioned in the user's query.\n"
            "1. Retrieve the user's query from the session state key 'current_research_query'.\n"
            "2. Identify all distinct drug names mentioned in the query (e.g., 'Ozempic, Metformin, and Lisinopril').\n"
            "3. For each identified drug name, call the 'query_drug_adverse_events' tool. You might need to make multiple parallel tool calls if multiple drugs are mentioned.\n"
            "4. Your final output MUST be a single JSON dictionary where keys are the drug names and values are the lists of adverse event report summaries (dictionaries) returned by the tool for that drug.\n"
            "   Example output: `{\"Ozempic\": [{\"report_summary\": \"...\"}, ...], \"Metformin\": [{\"report_summary\": \"...\"}, ...]}`\n"
            "5. If no drug names are identified or no reports are found for any drug, output an empty JSON dictionary: `{}`."
        ),
        tools=[query_drug_adverse_events],
        output_key="openfda_adverse_event_results",
        **shared_callbacks 
    )

    logging.info(f"OpenFDASearchAgent instance created: {openfda_search_agent.name}")


    # 2.6 Create ParallelAgent for multi-source research (renamed for clarity)
    research_orchestrator_agent = ParallelAgent(
        name="ResearchOrchestratorAgent", # Renamed from DualSourceResearchAgent
        sub_agents=[local_pubmed_search_agent, web_pubmed_search_agent, clinical_trials_search_agent, openfda_search_agent],
        description="Concurrently searches local PubMed DB, the live web, ClinicalTrials.gov, and OpenFDA for biomedical information. Results are stored in session state."
    )
    logging.info(f"ResearchOrchestratorAgent instance created: {research_orchestrator_agent.name}")

    research_orchestrator_agent_tool = AgentTool(agent=research_orchestrator_agent) # Renamed
    if hasattr(research_orchestrator_agent_tool, 'run_async') and callable(getattr(research_orchestrator_agent_tool, 'run_async')):
        research_orchestrator_agent_tool.func = research_orchestrator_agent_tool.run_async # type: ignore

    # Agent 2: Key Insight Extractor
    key_insight_extractor_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="KeyInsightExtractorAgent",
        instruction=KEY_INSIGHT_EXTRACTOR_INSTRUCTION,
        output_key="key_entities", # Its output is a list of entities
        **shared_callbacks 
    )

    # Agent 3: Correlational Investigator
    trial_connector_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TrialConnectorAgent",
        instruction=TRIAL_CONNECTOR_INSTRUCTION,
        tools=[query_clinical_trials_from_mongodb],
        output_key="connected_trials_results", # Its output is the list of connected trials
        **shared_callbacks 
    )

    # Create the initial data-gathering sequence
    data_gathering_and_connection_agent = SequentialAgent(
        name="DataGatheringAndConnectionAgent",
        sub_agents=[
            research_orchestrator_agent,
            key_insight_extractor_agent,
            trial_connector_agent
        ]
    )

    deep_dive_query_generator_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="DeepDiveQueryGeneratorAgent",
        instruction=DEEP_DIVE_QUERY_GENERATOR_INSTRUCTION,
        tools=[google_search_agent_tool], # Needs Google Search
        output_key="expanded_deep_dive_queries",
        description="Generates expanded queries for a deeper search based on initial user query and broad Google search."
        # This agent doesn't need memory as it's a one-shot generator
    )

    deep_dive_search_execution_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="DeepDiveSearchExecutionAgent",
        instruction=DEEP_DIVE_SEARCH_EXECUTION_INSTRUCTION,
        tools=[research_orchestrator_agent_tool], # Uses the main research orchestrator
        output_key="deep_search_findings",
        description="Executes the generated deep dive queries using the ResearchOrchestratorAgent."
        # This agent also doesn't need conversational memory
    )
    
    deep_dive_report_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="DeepDiveReportAgent",
        instruction=DEEP_DIVE_REPORT_AGENT_INSTRUCTION,
        # No tools needed, it just processes session state and formats text.
        # It will populate 'deep_dive_report' and 'candidate_ingestion_items'
        output_key="deep_dive_report", # Explicitly stating its main text output to session
        description="Creates a report from deep dive findings and asks for ingestion confirmation."
        # This agent doesn't need conversational memory
    )
    
    visualization_agent = LlmAgent(
       model=GEMINI_PRO_MODEL_ID,
       name="VisualizationAgent",
       instruction=VISUALIZATION_AGENT_INSTRUCTION,
       tools=[generate_simple_bar_chart, generate_simple_line_chart, generate_pie_chart, generate_grouped_bar_chart], # UPDATED TOOLS
       output_key="visualization_output", # Or handle output directly
       **shared_callbacks 
    )
    logging.info(f"VisualizationAgent instance created: {visualization_agent.name}")

    visualization_agent_tool = AgentTool(agent=visualization_agent)
    if hasattr(visualization_agent_tool, 'run_async') and callable(getattr(visualization_agent_tool, 'run_async')):
       visualization_agent_tool.func = visualization_agent_tool.run_async # type: ignore

    # --- Create the Special Search Agent ---
    special_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="SpecialSearchAgent",
        instruction=SPECIAL_SEARCH_AGENT_INSTRUCTION,
        description="A special search agent that can perform in-depth research and create visualizations.",
        tools=[google_search_agent_tool, visualization_agent_tool],
        **shared_callbacks
    )
    special_search_agent_tool = AgentTool(agent=special_search_agent)
    if hasattr(special_search_agent_tool, 'run_async'):
        special_search_agent_tool.func = special_search_agent_tool.run_async
    all_root_agent_tools.append(special_search_agent_tool)
    logging.info("SpecialSearchAgent wrapped as a tool and added to Root Agent's tools.")

    trend_data_fetcher_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TrendDataFetcherAgent",
        instruction="Call the get_publication_trend tool.",
        tools=[get_publication_trend],
        output_key="trend_chart_json",
        **shared_callbacks )
    
    # NEW: The Multimodal Evidence Agent
    multimodal_evidence_agent = LlmAgent(model=GEMINI_PRO_MODEL_ID, name="MultimodalEvidenceAgent", instruction=MULTIMODAL_EVIDENCE_INSTRUCTION, tools=[find_similar_images], output_key="image_evidence_results")
    multimodal_evidence_tool = AgentTool(agent=multimodal_evidence_agent)
    if hasattr(multimodal_evidence_tool, 'run_async'):
        multimodal_evidence_tool.func = multimodal_evidence_tool.run_async
    
    # --- WORKFLOW 1: The Trend Analysis "Short Path" ---
    trend_analysis_agent = SequentialAgent(name="TrendAnalysisAgent", description="A simple workflow to fetch trend data and visualize it.", sub_agents=[trend_data_fetcher_agent, visualization_agent])
    trend_analysis_tool = AgentTool(agent=trend_analysis_agent)
    if hasattr(trend_analysis_tool, 'run_async'):
        trend_analysis_tool.func = trend_analysis_tool.run_async


    # NEW: Create the "Smart Ingestion Router" Agent
    ingestion_router_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="IngestionRouterAgent",
        instruction=INGESTION_ROUTER_INSTRUCTION,
        tools=[ingest_pubmed_article, ingest_clinical_trial_record],
        description="A smart data librarian that analyzes text and routes it to the correct database (PubMed or Clinical Trials).",
        **shared_callbacks 
    )
    ingestion_router_agent_tool = AgentTool(agent=ingestion_router_agent)
    if hasattr(ingestion_router_agent_tool, 'run_async'):
       ingestion_router_agent_tool.func = ingestion_router_agent_tool.run_async

    all_root_agent_tools.append(ingestion_router_agent_tool)
    logging.info(f"IngestionRouterAgent wrapped as AgentTool ('{ingestion_router_agent_tool.name}') and added to Root Agent's tools.")

    # Agent A: The Narrative Writer
    text_synthesizer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TextSynthesizerAgent",
        instruction=TEXT_SYNTHESIZER_INSTRUCTION,
        output_key="text_report",
        **shared_callbacks 
    )

    # Agent B: The Chart Producer (NEW)
    chart_producer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ChartProducerAgent",
        instruction=CHART_PRODUCER_INSTRUCTION,
        tools=[visualization_agent_tool], # Its only tool is the viz agent
        output_key="chart_report",
        **shared_callbacks 
    )

    # Agent C: The Image Detective (NEW)
    image_evidence_producer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ImageEvidenceProducerAgent",
        instruction=IMAGE_EVIDENCE_PRODUCER_INSTRUCTION,
        tools=[multimodal_evidence_tool], # Its only tool is the multimodal agent
        output_key="image_report",
        **shared_callbacks 
    )


    bulk_ingestion_processor_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="BulkIngestionProcessorAgent",
        instruction=BULK_INGESTION_PROCESSOR_INSTRUCTION,
        tools=[ingestion_router_agent_tool], # Uses the IngestionRouterAgent
        description="Processes a list of candidate items for ingestion after user confirmation, using the IngestionRouterAgent.",
        # Its direct output will be the summary message for AVA to relay.
        **shared_callbacks 
    )
    bulk_ingestion_processor_agent_tool = AgentTool(agent=bulk_ingestion_processor_agent)
    if hasattr(bulk_ingestion_processor_agent_tool, 'run_async'):
        bulk_ingestion_processor_agent_tool.func = bulk_ingestion_processor_agent_tool.run_async
    all_root_agent_tools.append(bulk_ingestion_processor_agent_tool) # Add to AVA's tools


    # NEW: Wrapper function to correctly handle context for the memory search tool
    async def search_past_conversations(tool_context: ToolContext, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches through all past conversations and persona data to recall specific details or facts previously mentioned by the user.
        Use this when the user asks you to 'remember' something, or asks about a personal preference they stated earlier,
        or refers to a past discussion beyond the immediate recent turns.
        Args:
            tool_context: The context of the tool call, automatically provided by the ADK.
            query_text (str): The specific query or detail the user is asking to recall.
            limit (int): The maximum number of relevant past interactions to retrieve.
        Returns:
            A list of dictionaries, each representing a relevant past interaction or persona data.
        """
        invocation_context = tool_context._invocation_context
        user_id = invocation_context.session.user_id
        session_id = invocation_context.session.id
        return await mongo_memory_service.search_persona_and_interactions(user_id, session_id, query_text, limit)

    # NEW: Create the Deep Memory Recall agent and wrap it as a tool
    deep_memory_recall_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="DeepMemoryRecallAgent",
        instruction=DEEP_MEMORY_RECALL_INSTRUCTION,
        tools=[search_past_conversations],
        description="A specialist agent that searches through all past conversations to recall specific details or facts previously mentioned by the user.",
        **shared_callbacks 
    )
    deep_memory_recall_tool = AgentTool(agent=deep_memory_recall_agent)
    if hasattr(deep_memory_recall_tool, 'run_async'):
       deep_memory_recall_tool.func = deep_memory_recall_tool.run_async

    all_root_agent_tools.append(deep_memory_recall_tool)
    logging.info(f"DeepMemoryRecallAgent wrapped as AgentTool and added to Root Agent's tools.")


    # NEW: The Parallel Synthesizer
    parallel_synthesis_agent = ParallelAgent(
        name="ParallelSynthesisAgent",
        description="Splits the final report generation into two parallel streams: text and visuals.",
        sub_agents=[
            text_synthesizer_agent,
            chart_producer_agent,
            image_evidence_producer_agent
        ]
        # The outputs "text_report" and "visual_report" will be available in session state
    )

    # NEW: The Final Aggregator Agent
    final_report_aggregator_agent = LlmAgent(
        name="FinalReportAggregatorAgent",
        model=GEMINI_PRO_MODEL_ID,
        instruction=FINAL_AGGREGATOR_INSTRUCTION,
        **shared_callbacks 
        # No tools needed, it just processes text from session state
    )
    

    # NEW: Define the Master Research Synthesizer as a Sequential Agent
    master_research_synthesizer = SequentialAgent(
        name="MasterResearchSynthesizer",
        description="A sequential research assembly line that generates novel insights by connecting published papers to clinical trials.",
        sub_agents=[
            data_gathering_and_connection_agent,
            deep_dive_query_generator_agent,     # Generate queries for deep dive
            deep_dive_search_execution_agent,  # Execute deep dive searches
            parallel_synthesis_agent,          # Synthesize reports from *initial* data
            deep_dive_report_agent,            # Create report for *deep_dive* findings & ask ingestion Q
            final_report_aggregator_agent 
        ]
    )
    master_research_synthesizer_tool = AgentTool(agent=master_research_synthesizer)
    if hasattr(master_research_synthesizer_tool, 'run_async'):
       master_research_synthesizer_tool.func = master_research_synthesizer_tool.run_async
    
    all_root_agent_tools.append(master_research_synthesizer_tool)
    logging.info("MasterResearchSynthesizer assembly line created and added to Root Agent's tools.")


    # --- THE DISPATCHER: The Intent Router Agent ---
    intent_router_agent = LlmAgent(
        name="IntentRouterAgent",
        model=GEMINI_PRO_MODEL_ID,
        instruction=INTENT_ROUTER_INSTRUCTION,
        description="The master research dispatcher. Analyzes user queries and routes them to the correct specialist workflow.",
        tools=[
            special_search_agent_tool,
            master_research_synthesizer_tool 
        ]
    )
    intent_router_agent_tool = AgentTool(agent=intent_router_agent)
    if hasattr(intent_router_agent_tool, 'run_async'):
        intent_router_agent_tool.func = intent_router_agent_tool.run_async

    all_root_agent_tools.append(intent_router_agent_tool)
    logging.info("IntentRouterAgent created and added to Root Agent's tools.")

    # --- Create Accessibility Agents (Following the specified ADK Style) ---

    # Agent 1: SceneDescriberAgent (Specialist)
    scene_describer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="SceneDescriberAgent",
        instruction=SCENE_DESCRIBER_INSTRUCTION,
        description="Describes a scene based on a list of visible objects for visually impaired users.",
        # This agent reads from session state, so it doesn't need tools.
        **shared_callbacks
    )
    logging.info(f"SceneDescriberAgent instance created: {scene_describer_agent.name}")

    # Wrap it in an AgentTool to be called by the orchestrator
    scene_describer_agent_tool = AgentTool(agent=scene_describer_agent)
    if hasattr(scene_describer_agent_tool, 'run_async') and callable(getattr(scene_describer_agent_tool, 'run_async')):
        scene_describer_agent_tool.func = scene_describer_agent_tool.run_async # type: ignore

    # Agent 2: TextReaderAgent (Specialist)
    text_reader_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TextReaderAgent",
        instruction=TEXT_READER_INSTRUCTION,
        description="Reads text from the user's environment using an OCR tool.",
        tools=[perform_ocr_on_last_frame],  # Give it the specific OCR tool
        **shared_callbacks
    )
    logging.info(f"TextReaderAgent instance created: {text_reader_agent.name}")

    # Wrap it in an AgentTool to be called by the orchestrator
    text_reader_agent_tool = AgentTool(agent=text_reader_agent)
    if hasattr(text_reader_agent_tool, 'run_async') and callable(getattr(text_reader_agent_tool, 'run_async')):
        text_reader_agent_tool.func = text_reader_agent_tool.run_async # type: ignore

    # Agent 3: AccessibilityOrchestratorAgent (Dispatcher)
    accessibility_orchestrator_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="AccessibilityOrchestratorAgent",
        instruction=ACCESSIBILITY_ORCHESTRATOR_INSTRUCTION,
        description="Dispatches accessibility-related tasks to the appropriate specialist agent (SceneDescriber or TextReader).",
        # This agent's tools are the other specialist agents
        tools=[scene_describer_agent_tool, text_reader_agent_tool],
        **shared_callbacks
    )
    logging.info(f"AccessibilityOrchestratorAgent instance created: {accessibility_orchestrator_agent.name}")

    # Wrap the orchestrator itself in an AgentTool so the RootAgent can call it
    accessibility_orchestrator_agent_tool = AgentTool(agent=accessibility_orchestrator_agent)
    if hasattr(accessibility_orchestrator_agent_tool, 'run_async') and callable(getattr(accessibility_orchestrator_agent_tool, 'run_async')):
        accessibility_orchestrator_agent_tool.func = accessibility_orchestrator_agent_tool.run_async # type: ignore

    # Add the main orchestrator tool to the root agent's tool list
    all_root_agent_tools.append(accessibility_orchestrator_agent_tool)
    logging.info("AccessibilityOrchestratorAgent wrapped as a tool and added to Root Agent's tools.")

    # --- Create Auditory Accessibility Agents ---

    # Agent 1: AudioSentimentAgent (Specialist)
    audio_sentiment_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="AudioSentimentAgent",
        instruction=AUDIO_SENTIMENT_INSTRUCTION,
        description="Analyzes the user's tone of voice to determine their emotional state.",
        tools=[analyze_audio_sentiment],
        **shared_callbacks
    )
    logging.info(f"AudioSentimentAgent instance created: {audio_sentiment_agent.name}")

    # Wrap it in an AgentTool to be called by the orchestrator
    audio_sentiment_agent_tool = AgentTool(agent=audio_sentiment_agent)
    if hasattr(audio_sentiment_agent_tool, 'run_async') and callable(getattr(audio_sentiment_agent_tool, 'run_async')):
        audio_sentiment_agent_tool.func = audio_sentiment_agent_tool.run_async # type: ignore

    # Agent 2: SoundRecognitionAgent (Specialist)
    sound_recognition_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="SoundRecognitionAgent",
        instruction=SOUND_RECOGNITION_INSTRUCTION,
        description="Listens for and identifies significant ambient sounds in the user's environment.",
        tools=[recognize_ambient_sounds],
        **shared_callbacks
    )
    logging.info(f"SoundRecognitionAgent instance created: {sound_recognition_agent.name}")

    # Wrap it in an AgentTool to be called by the orchestrator
    sound_recognition_agent_tool = AgentTool(agent=sound_recognition_agent)
    if hasattr(sound_recognition_agent_tool, 'run_async') and callable(getattr(sound_recognition_agent_tool, 'run_async')):
        sound_recognition_agent_tool.func = sound_recognition_agent_tool.run_async # type: ignore

    # Agent 3: AuditoryAssistanceOrchestratorAgent (Dispatcher)
    auditory_assistance_orchestrator_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="AuditoryAssistanceOrchestratorAgent",
        instruction=AUDITORY_ASSISTANCE_ORCHESTRATOR_INSTRUCTION,
        description="Dispatches auditory accessibility tasks to the appropriate specialist agent (AudioSentiment or SoundRecognition).",
        tools=[audio_sentiment_agent_tool, sound_recognition_agent_tool],
        **shared_callbacks
    )
    logging.info(f"AuditoryAssistanceOrchestratorAgent instance created: {auditory_assistance_orchestrator_agent.name}")

    # Wrap the orchestrator itself in an AgentTool so the RootAgent can call it
    auditory_assistance_orchestrator_agent_tool = AgentTool(agent=auditory_assistance_orchestrator_agent)
    if hasattr(auditory_assistance_orchestrator_agent_tool, 'run_async') and callable(getattr(auditory_assistance_orchestrator_agent_tool, 'run_async')):
        auditory_assistance_orchestrator_agent_tool.func = auditory_assistance_orchestrator_agent_tool.run_async # type: ignore

    # Add the main orchestrator tool to the root agent's tool list
    all_root_agent_tools.append(auditory_assistance_orchestrator_agent_tool)
    logging.info("AuditoryAssistanceOrchestratorAgent wrapped as a tool and added to Root Agent's tools.")

    # --- Create Cognitive Accessibility Agents ---

    async def set_text_for_simplification(tool_context: ToolContext, text: str) -> str:
        """
        A tool to place a body of text into the session state so that it can be simplified by another agent.
        Args:
            tool_context: The context of the tool call.
            text (str): The text to be simplified.
        Returns:
            A confirmation message.
        """
        logging.info(f"Setting text for simplification: {text[:100]}...")
        tool_context._invocation_context.session.state['text_to_simplify'] = text
        return "The text has been set. Now, please call the CognitiveAssistanceOrchestratorAgent to simplify it."

    # Agent 1: TextSimplificationAgent (Specialist)
    text_simplification_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TextSimplificationAgent",
        instruction=TEXT_SIMPLIFICATION_INSTRUCTION,
        description="Simplifies complex text to make it easier to understand.",
        # This agent reads from session state, so it doesn't need tools.
        **shared_callbacks
    )
    logging.info(f"TextSimplificationAgent instance created: {text_simplification_agent.name}")

    # Wrap it in an AgentTool to be called by the orchestrator
    text_simplification_agent_tool = AgentTool(agent=text_simplification_agent)
    if hasattr(text_simplification_agent_tool, 'run_async') and callable(getattr(text_simplification_agent_tool, 'run_async')):
        text_simplification_agent_tool.func = text_simplification_agent_tool.run_async # type: ignore

    # Agent 2: CognitiveAssistanceOrchestratorAgent (Dispatcher)
    cognitive_assistance_orchestrator_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="CognitiveAssistanceOrchestratorAgent",
        instruction=COGNITIVE_ASSISTANCE_ORCHESTRATOR_INSTRUCTION,
        description="Dispatches cognitive accessibility tasks to the appropriate specialist agent.",
        tools=[text_simplification_agent_tool],
        **shared_callbacks
    )
    logging.info(f"CognitiveAssistanceOrchestratorAgent instance created: {cognitive_assistance_orchestrator_agent.name}")

    # Wrap the orchestrator itself in an AgentTool so the RootAgent can call it
    cognitive_assistance_orchestrator_agent_tool = AgentTool(agent=cognitive_assistance_orchestrator_agent)
    if hasattr(cognitive_assistance_orchestrator_agent_tool, 'run_async') and callable(getattr(cognitive_assistance_orchestrator_agent_tool, 'run_async')):
        cognitive_assistance_orchestrator_agent_tool.func = cognitive_assistance_orchestrator_agent_tool.run_async # type: ignore

    # Add the main orchestrator tool and the setter tool to the root agent's tool list
    all_root_agent_tools.append(cognitive_assistance_orchestrator_agent_tool)
    all_root_agent_tools.append(set_text_for_simplification)
    logging.info("CognitiveAssistanceOrchestratorAgent and its tools have been added to the Root Agent's tools.")



    # 3. Create the ProactiveContextOrchestratorAgent instance
    proactive_orchestrator = ProactiveContextOrchestratorAgent(
        name="ProactiveContextOrchestrator",
        environmental_monitor=environmental_monitor_agent,
        contextual_precomputation=contextual_precomputation_agent,
        reactive_task_delegator=reactive_task_delegator_agent,
        mcp_toolsets=loaded_mcp_toolsets # Pass toolsets if orchestrator needs to list them in its sub_agents
    )
    logging.info(f"ProactiveContextOrchestratorAgent instance created: {proactive_orchestrator.name}")

    # 4. Wrap the ProactiveContextOrchestratorAgent with AgentTool
    # This makes `proactive_orchestrator` callable as a tool by the `root_agent`.
    # The tool name will be derived from proactive_orchestrator.name.
    # The description for the tool should be clear for the RootAgent.
    proactive_orchestrator_tool = AgentTool(
        agent=proactive_orchestrator,
        # name="ProactiveContextOrchestratorTool", # AgentTool derives name from agent.name
        # description="A powerful orchestrator for visual context analysis, proactive suggestions, and reactive multi-step task execution. Expects 'user_goal' and 'seen_items' in a JSON string via the 'request' argument. Interacts heavily with session state for proactive flows.", # AgentTool derives description from agent.description
        # Optional: skip_summarization=True if the orchestrator's output is always direct and well-formatted.
    )
    logging.info(f"--- Inspecting proactive_orchestrator_tool ---")
    logging.info(f"Type: {type(proactive_orchestrator_tool)}")
    logging.info(f"Attributes: {dir(proactive_orchestrator_tool)}")
    logging.info(f"Has 'name' attribute: {hasattr(proactive_orchestrator_tool, 'name')}, Value: {getattr(proactive_orchestrator_tool, 'name', 'N/A')}")
    logging.info(f"Has 'description' attribute: {hasattr(proactive_orchestrator_tool, 'description')}")
    logging.info(f"Is callable: {callable(proactive_orchestrator_tool)}")
    logging.info(f"Underlying agent: {proactive_orchestrator_tool.agent.name if proactive_orchestrator_tool.agent else 'None'}")
    logging.info(f"--- End Inspecting proactive_orchestrator_tool ---")

    all_root_agent_tools.append(proactive_orchestrator_tool)
    logging.info(f"ProactiveContextOrchestrator wrapped as AgentTool ('{proactive_orchestrator_tool.name}') and added to Root Agent's tools.")

    # 4.5 Wrap ResearchAndVisualizeAgent as a tool for the Root Agent (AVA) to delegate to

    # 4.6 Add the ingest_single_article_data function directly as a tool for the Root Agent

    # --- Patch AgentTool ---
    if hasattr(proactive_orchestrator_tool, 'run_async') and callable(getattr(proactive_orchestrator_tool, 'run_async')):
        proactive_orchestrator_tool.func = proactive_orchestrator_tool.run_async # type: ignore
        logging.info(f"Patched AgentTool '{proactive_orchestrator_tool.name}' with .func attribute pointing to its run_async method.")
    else:
        logging.warning(f"Could not patch AgentTool '{proactive_orchestrator_tool.name}' with .func: 'run_async' not found or not callable.")

    logging.info(f"--- Inspecting proactive_orchestrator_tool AFTER attempting patch ---")
    logging.info(f"Has 'func' attribute after patch: {hasattr(proactive_orchestrator_tool, 'func')}")
    if hasattr(proactive_orchestrator_tool, 'func'):
        logging.info(f"Value of 'func' attribute: {getattr(proactive_orchestrator_tool, 'func')}")
    logging.info(f"--- End Inspecting proactive_orchestrator_tool AFTER attempting patch ---")

    # 5. Create the Root Agent (mcp_streaming_assistant)
    # This is your main, user-facing multimodal agent.
    root_agent = LlmAgent(
        model=MODEL_ID_STREAMING, # Must be a multimodal model
        name="mcp_streaming_assistant", # As defined in your original setup
        instruction=ROOT_AGENT_INSTRUCTION_STREAMING,
        tools=all_root_agent_tools,
        # Add memory callbacks to the root agent as well, in case it needs to
        # make its own LLM calls.
        **shared_callbacks 
    )

    logging.info(f"Root Agent ('{root_agent.name}') created with {len(root_agent.tools or [])} tools.")
    if root_agent.tools:
        tool_names = [getattr(t, 'name', str(type(t))) for t in root_agent.tools]
        logging.info(f"Root Agent tools list: {tool_names}")
    else:
        logging.warning("Root Agent has no tools configured.")

    return root_agent
