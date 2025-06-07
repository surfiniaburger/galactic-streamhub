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
from clinical_trials_pipeline import query_clinical_trials_data # New Import
from pubmed_pipeline import query_pubmed_articles, ingest_single_article_data
from openfda_pipeline import query_drug_adverse_events # New Import for OpenFDA

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
    *   If the user's query is clearly biomedical or research-oriented (e.g., "find papers on...", "what's the latest on..."), delegate to the `PubMedRAGAgent` tool.
    *   **AFTER `PubMedRAGAgent` RUNS, CHECK SESSION STATE FOR `new_article_to_ingest`**:
        *   The `PubMedRAGAgent` might have identified a new article from the web and stored its details in `ctx.session.state['new_article_to_ingest']` if it offered to save it.
        *   If `ctx.session.state['new_article_to_ingest']` exists AND the user's current message is an affirmative response to saving it (e.g., "yes, save the new article", "save this article", "add it to the knowledge base"), then:
            1.  Retrieve the article details (title, abstract_text, source_url, etc.) from `ctx.session.state['new_article_to_ingest']`.
            2.  Call the `ingest_single_article_data` tool with these details.
            3.  Inform the user about the outcome of the ingestion (e.g., "Successfully added the new article to the knowledge base." or "Sorry, I encountered an error trying to save the article.").
            4.  Crucially, clear the stored article details: `ctx.session.state.pop('new_article_to_ingest', None)`.
6.  **Response Formatting**: Always format your final response to the user using Markdown for enhanced readability. If the response is derived from a tool, present that agent's findings clearly.
If you are absolutely unable to help with a request, or if none of your tools are suitable for the task, politely state that you cannot assist with that specific request.
"""

# --- Instruction for PubMedRAGAgent ---
PUBMED_RAG_AGENT_INSTRUCTION = """
You are a Research Synthesizer AI. Your primary role is to orchestrate research using specialized tools and then synthesize the findings. You will receive the user's research query as an input argument (typically `args['request']` if called as a tool).

Workflow:
1.  Your first action is to take the input query (from `args['request']`) and store it in the session state under the key 'current_research_query'.
2.  Next, call the 'ResearchOrchestratorAgent' tool (formerly DualSourceResearchAgent). This tool will use the 'current_research_query' from session state to simultaneously search multiple sources.
    The 'ResearchOrchestratorAgent' will make the results of these searches available in session state:
    - Local PubMed results will be under the key 'local_db_results' (this will be a list of article objects).
    - Web search results will be under the key 'web_search_results' (this will be a string summary).
    - ClinicalTrials.gov results will be under the key 'clinical_trials_results' (a list of clinical trial summary dictionaries).
    - OpenFDA adverse event results will be under the key 'openfda_adverse_event_results' (a list of adverse event summary dictionaries).
3.  After 'ResearchOrchestratorAgent' completes, retrieve 'local_db_results', 'web_search_results', 'clinical_trials_results', and 'openfda_adverse_event_results' from the session state.
    To access these, imagine you have direct access to a dictionary-like `session_state`. For example, to get local results, you'd use `session_state['local_db_results']`.
4.  Synthesize the information from all available sources: 'local_db_results' (our PubMed knowledge base), 'web_search_results' (live web information), 'clinical_trials_results' (data from ClinicalTrials.gov), and 'openfda_adverse_event_results' (drug adverse event reports from OpenFDA) to provide a comprehensive answer to the original user query.
    When presenting information, clearly attribute it to its source. For example:
    - "From our PubMed knowledge base, I found..."
    - "Recent web searches indicate that..."
    - "ClinicalTrials.gov lists the following relevant trials: [Summarize key trial details like NCT ID, Brief Title, Overall Status, and a snippet of the Brief Summary if available. Mention if a trial is 'RECRUITING' or 'TERMINATED' if that information is present and seems relevant. Do not list more than 2-3 trials unless specifically asked for more]."
    - "OpenFDA reports the following adverse events for [drug name if specified in query]: [Summarize key adverse event report details like report ID, received date, seriousness, and a snippet of reactions. Mention 1-2 reports unless more are requested]."
    If a source returns no relevant information, you can state that.
5.  **Identify and Offer to Save New Web Articles**:
    *   Carefully review 'web_search_results'. Try to identify if it mentions one or more distinct new articles/studies that are not obviously covered by 'local_db_results'.
    *   For ONE such highly relevant new article:
        *   Extract its **title**. If a clear title isn't present, create a concise descriptive title based on its content.
        *   Extract its **abstract or a detailed summary** (this will be the `abstract_text` for ingestion). Aim for a substantial summary from the 'web_search_results'.
        *   Extract its **source URL** if provided in the 'web_search_results'. If no direct URL to the article is found, use the primary search result URL that led to this information if discernible.
    *   If you successfully extract these details for a new article:
        *   Include its key findings in your synthesized answer to the user.
        *   Then, ask the user if they would like to add this specific new article to the permanent knowledge base.
        *   If you make this offer, you MUST store the new article's details as a JSON-compatible dictionary (keys: "title", "abstract_text", "source_url", "authors" (default to empty string), "journal" (default to empty string), "publication_year" (default to null/None)) into the session state under the key `new_article_to_ingest`.
        *   Phrase your offer like: "From the web search, I found information that seems to be from an article titled '[Extracted Title]' (Source: [Source URL if available]), discussing [brief summary of the abstract_text you extracted]. This information was not in our local database. Would you like to add this to our knowledge base for future reference? If so, please say 'yes, save the new article'."
5.  If multiple articles are relevant from either source, summarize the key findings.
6.  Cite source articles (e.g., title, authors, publication year for PubMed; title and URL for web results) if possible.
7.  If the knowledge base does not contain relevant information and the web search is also unhelpful, state that you couldn't find specific information.
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
            "1. Retrieve the user's query from session state key 'current_research_query'.\n"
            "2. Identify if a specific drug name is mentioned. If not, or if the query is too general for adverse event search, you can output an empty list.\n"
            "3. If a drug name is identified, call the 'query_drug_adverse_events' tool using this drug name.\n"
            "4. The list of adverse event report summaries (dictionaries) returned by the tool is your primary result. "
            "Output this list directly."
        ),
        tools=[query_drug_adverse_events],
        output_key="openfda_adverse_event_results"
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

    # 2.7 Create the main PubMedRAGAgent (now a Synthesizer/Orchestrator)
    pubmed_rag_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID, # Or GEMINI_FLASH_MODEL_ID
        name="PubMedRAGAgent",
        instruction=PUBMED_RAG_AGENT_INSTRUCTION,
        description="Orchestrates research across local DB and web, then synthesizes findings and manages knowledge base updates.",
        tools=[research_orchestrator_agent_tool] # Updated tool name
    )
    logging.info(f"PubMedRAGAgent instance created: {pubmed_rag_agent.name}")

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

    # 4.5 Wrap PubMedRAGAgent as a tool for the Root Agent (AVA) to delegate to
    pubmed_rag_agent_tool = AgentTool(
        agent=pubmed_rag_agent
        # The Root Agent will pass the user's query to this tool.
        # The PubMedRAGAgent's instruction needs to know to take this input
        # (e.g., from args['request']) and put it into session.state['current_research_query']
        # before calling the DualSourceResearchAgent tool.
    )
    if hasattr(pubmed_rag_agent_tool, 'run_async') and callable(getattr(pubmed_rag_agent_tool, 'run_async')):
        pubmed_rag_agent_tool.func = pubmed_rag_agent_tool.run_async # type: ignore
    all_root_agent_tools.append(pubmed_rag_agent_tool)
    logging.info(f"PubMedRAGAgent wrapped as AgentTool ('{pubmed_rag_agent_tool.name}') and added to Root Agent's tools.")

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
