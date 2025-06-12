# /Users/surfiniaburger/Desktop/app/agent_config.py
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent # New Import
from typing import Any, Dict, List
import logging
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
import json # For Root Agent instruction example

# Import new proactive agents and their instructions
from proactive_agents import (
    ProactiveContextOrchestratorAgent,
    EnvironmentalMonitorAgent, ENVIRONMENTAL_MONITOR_INSTRUCTION,
    ContextualPrecomputationAgent, CONTEXTUAL_PRECOMPUTATION_INSTRUCTION,
    ReactiveTaskDelegatorAgent, REACTIVE_TASK_DELEGATOR_INSTRUCTION
)
# Import the Google Search Agent
from google_search_agent.agent import root_agent as google_search_agent_instance
# Import the PubMed query function directly
from tools.chart_tool import generate_simple_bar_chart, generate_simple_line_chart, generate_pie_chart # UPDATED IMPORT
from clinical_trials_pipeline import query_clinical_trials_data # New Import
from pubmed_pipeline import query_pubmed_articles, ingest_single_article_data
from openfda_pipeline import query_drug_adverse_events # New Import for OpenFDA
from google.adk.agents import SequentialAgent

MODEL_ID_STREAMING = "gemini-2.0-flash-live-preview-04-09" # Or your preferred streaming-compatible model like "gemini-2.0-flash-exp"
GEMINI_PRO_MODEL_ID = "gemini-2.0-flash"
GEMINI_MULTIMODAL_MODEL_ID = MODEL_ID_STREAMING # Alias for clarity

# --- Instructions for the new TaskExecutionAgent ---
# This is now effectively the REACTIVE_TASK_DELEGATOR_INSTRUCTION,
# but keeping the old name here for reference if needed, though it's superseded.
TASK_EXECUTION_AGENT_INSTRUCTION = """
You are a specialized assistant that helps users accomplish tasks based on their goals and items visually identified in their environment. You will be provided with the user's goal and a list of 'seen_items'.

Your primary capabilities are:
1.  **Recipe and Ingredient Analysis**:
    *   If the user's goal involves making a food or drink item (e.g., a cocktail), use the 'Cocktail' tool (specifically the `search_cocktail_by_name` function or similar) to find the recipe for the item mentioned in the `user_goal`.
    *   Compare the recipe ingredients against the provided `seen_items` list.
    *   Clearly state which ingredients the user appears to have and which are missing for the recipe.
2.  **Location Finding for Missing Items**:
    *   If ingredients are missing and the `user_goal` implies finding them (e.g., "where can I buy..."), use the 'Google Maps' tool (specifically the `find_places` function or similar) to find relevant stores (e.g., 'grocery store', 'liquor store') near the user. Assume 'near me' if no specific location is provided by the user.

**Input Format You Will Receive:**
You will receive input as a single JSON string. This string will contain:
*   `user_goal`: A string describing what the user wants to achieve.
*   `seen_items`: A list of strings representing items visually identified.
You must parse this JSON string to extract `user_goal` and `seen_items`.
Example input you'll get: '{"user_goal": "make a negroni", "seen_items": ["gin", "campari"]}'


**Your Response Obligation:**
You MUST combine all gathered information (recipe details, what's on hand, what's missing, store locations if applicable) into a single, comprehensive, and helpful textual response. Be direct and structure your answer clearly.

**Example Internal Thought Process (what you should aim for):**
1.  Receive Input: `user_goal="I want to make a Negroni and see what I'm missing. If I need something, tell me where to buy it."`, `seen_items=["gin", "a red bottle that might be Campari"]`.
2.  Analyze Goal: User wants to make a Negroni, check inventory against `seen_items`, and find a store for missing ingredients.
3.  Action - Recipe: Call the Cocktail tool: `search_cocktail_by_name(name="Negroni")`.
4.  Process Recipe: Assume Cocktail tool returns: "Negroni: Gin, Campari, Sweet Vermouth."
5.  Compare with `seen_items`: User has "gin". User might have "Campari" (due to "a red bottle that might be Campari"). User definitely needs "Sweet Vermouth".
6.  Action - Find Store (as per goal): Call the Google Maps tool: `find_places(query="liquor store near me")`.
7.  Process Store Info: Assume Maps tool returns: "Nearest liquor store: 'Drinks Emporium'."
8.  Formulate Final Response: "To make a Negroni, you need Gin, Campari, and Sweet Vermouth. Based on what I see, you have gin and possibly Campari (the red bottle). You'll definitely need Sweet Vermouth. You can find it at 'Drinks Emporium', which appears to be the nearest liquor store."

**IMPORTANT ON TOOL USAGE**: When your instructions lead you to use the 'Cocktail' or 'Google Maps' tools, you should generate the appropriate function call (e.g., `search_cocktail_by_name(...)` or `find_places(...)`). The Root Agent's system will execute these calls using the actual tools it possesses.
"""

# --- Updated Instructions for the Root Agent ---
ROOT_AGENT_INSTRUCTION_STREAMING = """
Role: You are AVA (Advanced Visual Assistant), a multimodal AI. Your goal is to understand user requests, analyze their visual surroundings, and assist them. You can use tools directly for simple queries or delegate complex tasks to `ProactiveContextOrchestratorTool`.

Core Capabilities:
1.  **Visual Scene Analysis (Multimodal Perception)**:
    *   When the user's query implies needing to understand their environment, carefully analyze incoming video frames.
    *   Identify relevant objects ('seen_items').
    *   Also, try to infer 'initial_context_keywords' from the scene and query (e.g., "cocktail_making", "board_game_setup").
2.  **Direct Tool Usage (for simple, direct queries)**:
    *   You have direct access to tools for: cocktails, weather. Use these for straightforward requests.
3.  **Delegation to `ProactiveContextOrchestrator` tool**:
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
4.  **Conversational Interaction**: Engage in general conversation if no specific task or tool is appropriate. Ask clarifying questions if the user's request is ambiguous.
5.  **Delegation to `PubMedRAGAgent`**:
    *   If the user's query is clearly biomedical or research-oriented (e.g., "find papers on...", "what's the latest on..."), delegate the task to the `PubMedRAGAgent` tool.

    *   **IMPORTANT - PASS-THROUGH RESPONSE**: When the `PubMedRAGAgent` tool returns a response, you MUST treat it as the final, complete answer for the user. **Your job is to pass this response directly to the user without any changes, summarization, or additional commentary.** Do not rephrase it or add your own thoughts.

    *   **AFTER `PubMedRAGAgent` RUNS, CHECK SESSION STATE FOR `new_article_to_ingest`**:
        *   This is a background check. Even while passing through the main response, you should still check the session state for `new_article_to_ingest`.
        *   If it exists AND the user's *next* message is an affirmative response (e.g., "yes, save it"), then you should call the `ingest_single_article_data` tool and inform the user of the result.
6.  **Response Formatting**: Always format your final response to the user using Markdown for enhanced readability. If the response is derived from a tool, present that agent's findings clearly.
If you are absolutely unable to help with a request, or if none of your tools are suitable for the task, politely state that you cannot assist with that specific request.
"""

# --- Instruction for PubMedRAGAgent ---
PUBMED_RAG_AGENT_INSTRUCTION = """
You are a Research Synthesizer AI. Your primary role is to orchestrate research, synthesize findings, and generate visualizations. You will receive the user's query as input.

**Your Complete Workflow:**
1.  **Orchestrate Research:** Your FIRST action is to call the `ResearchOrchestratorAgent` tool. This tool will search multiple sources (local DB, web, clinical trials, FDA) and store the results in session state. You do not need to do anything with its direct output; it works in the background.

2.  **Synthesize Findings:** After the `ResearchOrchestratorAgent` tool has run, your next step is to process the results it left in session state ('local_db_results', 'web_search_results', etc.). Create a comprehensive textual summary answering the user's original query. Attribute information to its source (e.g., "From the local database...", "A recent web search found...").

3.  **Offer to Save New Articles:** While synthesizing, if you find a distinct new article in the 'web_search_results', extract its details (title, summary, URL) and store them in session state under the key `new_article_to_ingest`. Then, in your textual summary, explicitly ask the user if they'd like to save it.

4.  **Extract and Visualize Data:** After creating the summary, re-examine the research text for quantifiable data.
    *   If you find data suitable for a chart, you MUST first extract this data and create a **JSON string of a single object**.
    *   This object must contain keys for the chart data itself, a descriptive title, and axis labels.
    *   **Example (Bar Chart):** If the text says "...reduced cancer in 81% of patients and achieved complete remission in 52%...", you must create a JSON string like this:
        `'{"chart_type": "bar", "chart_data": [{"category": "Cancer Reduction", "value": 81}, {"category": "Complete Remission", "value": 52}], "chart_title": "Efficacy of huCART19-IL18 Therapy", "chart_xlabel": "Clinical Outcome", "chart_ylabel": "Patients (%)", "category_field": "category", "value_field": "value"}'`
    *   **Example (Pie Chart):** If data shows distribution like "Drug A: 60% market share, Drug B: 25%, Drug C: 15%":
        `'{"chart_type": "pie", "chart_data": [{"drug": "Drug A", "share": 60}, {"drug": "Drug B", "share": 25}, {"drug": "Drug C", "share": 15}], "chart_title": "Market Share", "category_field": "drug", "value_field": "share"}'`
    *   Then, call the `VisualizationAgent` tool. The `request` argument for this tool call **must be this complete JSON string**.

5.  **Combine Everything:** Your FINAL response to the user MUST be a single, coherent message that combines:
    a. The full textual research summary you synthesized in step 2.
    b. The question about saving a new article (if applicable from step 3).
    c. The chart URL returned by the `VisualizationAgent` tool (if applicable from step 4), formatted clearly.

Example Final Output Structure:
"Based on the research, here is a summary of findings on CAR T-cell therapy...[detailed summary]...
From the web, I found a new article titled 'Enhanced CAR T-cell Efficacy'. Would you like to add this to the knowledge base?
Additionally, here is a chart visualizing the study's success rates:
[/static/charts/chart_url.png]"
"""


VISUALIZATION_AGENT_INSTRUCTION = """
You are a Data Visualization Agent. Your task is to take a request containing a JSON object with chart data and metadata, and generate a chart.

1.  **Parse the Input:** The input `request` argument will be a JSON string of a single object. This object will contain:
    *   `chart_type`: A string indicating the type of chart (e.g., "bar", "line", "pie").
    *   `chart_data`: The data for the chart (format depends on chart_type).
    *   `chart_title`: A descriptive title for the chart.
    *   `chart_xlabel` (for bar/line): Label for the x-axis.
    *   `chart_ylabel` (for bar/line): Label for the y-axis.
    *   `category_field` (for bar/line/pie): The key in chart_data objects representing categories/labels.
    *   `value_field` (for bar/line/pie): The key in chart_data objects representing values.
    *   Other fields specific to chart types may be present.

2.  **Select and Call the Charting Tool:**
    *   Based on the `chart_type` from the parsed JSON:
        *   If "bar", call `generate_simple_bar_chart`.
        *   If "line", call `generate_simple_line_chart`.
        *   If "pie", call `generate_pie_chart`.
    *   You MUST use the values from the parsed JSON object to populate the tool's arguments correctly.
    *   For `generate_simple_bar_chart` and `generate_simple_line_chart`, use `data`, `title`, `xlabel`, `ylabel`, `category_field`, `value_field`.
    *   For `generate_pie_chart`, use `data`, `title`, `labels_field` (maps to category_field), `values_field` (maps to value_field).

3.  **Return ONLY the URL:** The tool will return a URL (e.g., "/static/charts/chart_uuid.png"). Your final output MUST be this URL string and nothing else. Do not add any extra text.

**Example Workflow:**
-   **Input `request`:** `'[{"therapy": "huCART19-IL18", "remission_rate": 52}, {"therapy": "HSP-CAR30", "remission_rate": 60}]'`
-   **Your Action:** Call the tool `generate_simple_bar_chart(data=[...parsed data...], category_field="therapy", value_field="remission_rate", title="CAR T-Cell Remission Rates")`.
-   **Your Final Output:** `"/static/charts/chart_12345.png"`

**Example Workflow (Pie Chart):**
-   **Input `request`:** `'{"chart_type": "pie", "chart_data": [{"region": "North America", "sales": 4500}, {"region": "Europe", "sales": 3200}], "chart_title": "Sales by Region", "category_field": "region", "value_field": "sales"}'`
-   **Your Action:** Call `generate_pie_chart(data=[...parsed data...], title="Sales by Region", labels_field="region", values_field="sales")`.
-   **Your Final Output:** `"/static/charts/pie_chart_67890.png"`
"""


def create_streaming_agent_with_mcp_tools(
    loaded_mcp_toolsets: List[MCPToolset],
    #raw_mcp_tools_lookup_for_warnings: Dict[str, Any] # No longer strictly needed here
) -> LlmAgent:

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

    environmental_monitor_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID, # Needs multimodal if it directly processes images
        name="EnvironmentalMonitorAgent",
        instruction=ENVIRONMENTAL_MONITOR_INSTRUCTION,
        description="Analyzes visual context to identify keywords for proactive assistance.",
        # output_key="identified_context_keywords_output" # Example, if it writes to state
        # For custom agent orchestration, direct output handling or state management is key.
    )

    contextual_precomputation_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ContextualPrecomputationAgent",
        instruction=CONTEXTUAL_PRECOMPUTATION_INSTRUCTION,
        description="Proactively fetches information based on context keywords.",
        tools=sub_agent_tools, # Give it access to MCP tools AND GoogleSearchAgentTool
                                   # Or rely on RootAgent's tools if it only declares calls.
                                   # For ADK, better to have tools on Root and sub-agents declare.
        # output_key="proactive_precomputation_output"
    )

    reactive_task_delegator_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ReactiveTaskDelegatorAgent",
        instruction=REACTIVE_TASK_DELEGATOR_INSTRUCTION, # Renamed from TASK_EXECUTION_AGENT_INSTRUCTION
        description="Handles explicit user tasks or executes precomputed suggestions.",
        tools=sub_agent_tools, # Same as above
        # output_key="reactive_task_final_answer"
    )

    # 2.5 Define Specialist Agents for Parallel Research
    local_pubmed_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="LocalPubMedSearchAgent",
        instruction="You are a specialized agent. Your task is to search a local PubMed database. \n1. Retrieve the user's query from the session state key 'current_research_query'.\n2. Call the 'query_pubmed_articles' tool using this query.\n3. The list of articles returned by the tool is your primary result. Output this list directly.",
        tools=[query_pubmed_articles],
        output_key="local_db_results" # Store the direct output of this agent here
    )

    web_pubmed_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="WebPubMedSearchAgent",
        instruction="You are a specialized agent. Your task is to search the web for recent biomedical information.\n1. Retrieve the user's query from the session state key 'current_research_query'.\n2. Append 'latest research' or 'recent studies' to this query.\n3. Call the 'google_search_agent' tool with this modified query.\n4. The string summary returned by the tool is your primary result. Output this string directly.",
        tools=[google_search_agent_tool],
        output_key="web_search_results" # Store the direct output of this agent here
    )

    clinical_trials_search_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ClinicalTrialsSearchAgent",
        instruction=(
            "You are a specialized agent. Your task is to search the ClinicalTrials.gov database "
            "for clinical trial information relevant to the user's query.\n"
            "1. Retrieve the user's query from the session state key 'current_research_query'.\n"
            "2. Call the 'query_clinical_trials_data' tool using this query.\n"
            "3. The list of clinical trial study summaries (dictionaries) returned by the tool is your primary result. "
            "Output this list directly."
        ),
        tools=[query_clinical_trials_data], # Pass the function directly
        output_key="clinical_trials_results" # Store the direct output of this agent here
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
        output_key="openfda_adverse_event_results"
    )

    logging.info(f"OpenFDASearchAgent instance created: {openfda_search_agent.name}")

    visualization_agent = LlmAgent(
       model=GEMINI_PRO_MODEL_ID,
       name="VisualizationAgent",
       instruction=VISUALIZATION_AGENT_INSTRUCTION,
       tools=[generate_simple_bar_chart, generate_simple_line_chart, generate_pie_chart], # UPDATED TOOLS
       output_key="visualization_output" # Or handle output directly
    )
    logging.info(f"VisualizationAgent instance created: {visualization_agent.name}")

    visualization_agent_tool = AgentTool(agent=visualization_agent)
    if hasattr(visualization_agent_tool, 'run_async') and callable(getattr(visualization_agent_tool, 'run_async')):
       visualization_agent_tool.func = visualization_agent_tool.run_async # type: ignore

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

    # 2.7 DELETE the SequentialAgent and create this new LlmAgent instead
    pubmed_rag_agent = LlmAgent(
       name="PubMedRAGAgent", # This will be the tool name used by the root agent (AVA)
       model=GEMINI_PRO_MODEL_ID,
       instruction=PUBMED_RAG_AGENT_INSTRUCTION,
       description="Orchestrates research, synthesizes findings, and visualizes data to answer complex biomedical queries.",
        tools=[
           research_orchestrator_agent_tool, # Give it the research tool
           visualization_agent_tool          # Give it the visualization tool
       ]
     )

    pubmed_rag_agent_tool = AgentTool(
    agent=pubmed_rag_agent
    )
    if hasattr(pubmed_rag_agent_tool, 'run_async'):
       pubmed_rag_agent_tool.func = pubmed_rag_agent_tool.run_async # type: ignore
    all_root_agent_tools.append(pubmed_rag_agent_tool)
    logging.info(f"PubMedRAGAgent wrapped as AgentTool ('{pubmed_rag_agent_tool.name}') and added to Root Agent's tools.")

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
    # The ADK will wrap it. Its docstring in pubmed_pipeline.py serves as its description.
    all_root_agent_tools.append(ingest_single_article_data)
    logging.info(f"Added 'ingest_single_article_data' function as a tool to Root Agent.")

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
        tools=all_root_agent_tools, # Contains MCPToolsets + ProactiveContextOrchestratorTool
    )

    logging.info(f"Root Agent ('{root_agent.name}') created with {len(root_agent.tools or [])} tools.")
    if root_agent.tools:
        tool_names = [getattr(t, 'name', str(type(t))) for t in root_agent.tools]
        logging.info(f"Root Agent tools list: {tool_names}")
    else:
        logging.warning("Root Agent has no tools configured.")

    return root_agent
