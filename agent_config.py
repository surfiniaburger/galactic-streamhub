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
from google.adk.agents import SequentialAgent
from ingest_clinical_trials import query_clinical_trials_from_mongodb, ingest_clinical_trial_record
from ingest_multimodal_data import find_similar_images

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
5.  **Delegation to `MasterResearchSynthesizer`**:
    *   If the user's query is clearly biomedical or research-oriented (e.g., "find papers on...", "what's the latest on..."), delegate the task to the `MasterResearchSynthesizer` tool.

    *   **IMPORTANT - PASS-THROUGH RESPONSE**: When the `MasterResearchSynthesizer` tool returns a response, you MUST treat it as the final, complete answer for the user. **Your job is to pass this response directly to the user without any changes, summarization, or additional commentary.** Do not rephrase it or add your own thoughts.

6.  **Response Formatting**: Always format your final response to the user using Markdown for enhanced readability. If the response is derived from a tool, present that agent's findings clearly.
If you are absolutely unable to help with a request, or if none of your tools are suitable for the task, politely state that you cannot assist with that specific request.
"""



# NEW: Instruction for the "Smart Ingestion Router" Agent
INGESTION_ROUTER_INSTRUCTION = """
You are a highly specialized data librarian. Your only job is to analyze a snippet of text and determine if it represents a published academic paper or a clinical trial record.

Based on your classification, you MUST call one of the following tools:
- If the text looks like a **published paper** (e.g., has authors, a journal, a detailed abstract, DOI), call the `ingest_pubmed_article` tool.
- If the text looks like a **clinical trial record** (e.g., has an NCT Number, mentions study phases, recruitment status, interventions), call the `ingest_clinical_trial_record` tool.

You must only call one of these two tools.
"""

# NEW: Instruction for the "Insight Synthesizer" Agent (The Core of the "Wow Factor")
RESEARCH_SYNTHESIZER_INSTRUCTION = """
You are a world-class AI Research Synthesizer. You have two primary modes: **Trend Analysis** and **Insight Synthesis**. Your first step is always to determine the user's primary intent.

**--- PRIMARY DECISION WORKFLOW ---**

1.  **ANALYZE USER INTENT:** First, examine the user's query.
    *   If the query asks for a **trend over time**, a **plot of data per year**, or a **timeline**, your intent is **TREND ANALYSIS**.
    *   For all other research queries, such as "what is the latest on...", "connect papers to trials...", or "show me the distribution of X", your intent is **INSIGHT SYNTHESIS**.

2.  **EXECUTE BASED ON INTENT:**
    *   **IF INTENT IS TREND ANALYSIS:**
        *   **Action:** Call the `get_publication_trend` tool using the topic from the user's query.
        *   **Next Action:** Take the complete JSON payload returned by the tool and pass it directly to the `VisualizationAgent`.
        *   **Final Output:** Present the chart to the user with a simple introductory sentence.

    *   **IF INTENT IS INSIGHT SYNTHESIS:**
        *   Proceed with the detailed **Synthesis Workflow** below.

---
**--- SYNTHESIS WORKFLOW ---**

1.  **BROAD-SPECTRUM RESEARCH:** Call `ResearchOrchestratorAgent` to search all knowledge bases.

2.  **DATA EXTRACTION FOR VISUALIZATION (CRITICAL STEP):**
    *   After getting results from the orchestrator, immediately scan the text for quantifiable data suitable for a chart (like funding distribution, patient percentages, etc.).
    *   **IF you find such data:**
        *   You **MUST** first manually extract it and format it into a valid JSON object.
        *   Then, you **MUST** call the `VisualizationAgent` tool, passing your generated JSON string as the `request` argument.
        *   Store the returned chart URL to include in your final answer.

3.  **IDENTIFY KEY RESEARCH:** Identify the top 1-2 most relevant academic papers from the PubMed results. Extract their key entities (drug names, gene targets, etc.).

4.  **CONNECTIVE DEEP-DIVE:** Use the extracted entities to perform a second, targeted search of the clinical trials database using the `query_clinical_trials_from_mongodb` tool.

5.  **SYNTHESIZE AND NARRATE:** Construct your final narrative. **Weave the chart URL from Step 2 into your narration logically.** Start with foundational research, connect it to clinical trials, and offer to save any new web articles.

6.  **GENERATE "AHA!" MOMENT:** Conclude with the "Generated Insight & Future Direction" section, providing a novel, forward-looking thought.
    
**Example of a Complete, Synthesized Final Output:**

Here is a synthesis of the latest findings on CAR T-cell therapy for lymphoma:

**Foundational Research:** Based on my research, a key paper titled 'Enhanced Anti-tumor Efficacy of IL-18 Secreting CAR T-cells in Diffuse Large B-cell Lymphoma' (Source: PubMed) establishes that a novel construct, **huCART19-IL18**, leads to significantly higher durable remission rates by resisting T-cell exhaustion. The study reported an 81% overall response rate in its preclinical models.

**The Connection:** Building directly on that foundational work, I discovered an active clinical trial designed to bring this specific therapy to patients.

**Clinical Trial Insights:** Trial **NCT09876543** (Source: ClinicalTrials.gov) is a Phase 2 study currently recruiting patients to evaluate the safety and efficacy of **huCART19-IL18** in a clinical setting for patients with relapsed or refractory B-cell lymphoma. This directly translates the promising preclinical findings into human trials.

To visualize the efficacy data from the foundational paper, here is a chart:
[/static/charts/chart_xyz123.png]

Additionally, a recent article from 'BioTech Today' discusses the funding behind this new wave of CAR-T therapies. Would you like to save this article to the knowledge base?

**Generated Insight & Future Direction:**
The foundational PubMed paper mentions a proprietary '24-hour rapid manufacturing process' as key to the therapy's success. However, the public record for trial NCT09876S43 does not specify the manufacturing timeline. A critical unanswered question is whether the clinical-grade manufacturing process can replicate the speed and cell fitness metrics of the original research. This could be a major factor in the trial's ultimate success and represents a key area to monitor.
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
- **IF** the query asks for a **trend over time** or a **plot of publications per year**, you **MUST** call the `TrendAnalysisAgent`.
- **IF** the query asks for a **synthesis of findings**, to **connect research to trials**, or to **show visual evidence** (e.g., "show me a scan of..."), you **MUST** call the `MasterResearchSynthesizer`.
You must call one and only one of these two specialist agents.
"""


# --- AGENT 2: Key Insight Extractor ---
KEY_INSIGHT_EXTRACTOR_INSTRUCTION = """
Your task is to analyze research data and extract key information.
You will receive research results in the session state key `local_db_results` and `clinical_trials_results`.
From the text of the top 1-2 most relevant articles, identify and extract a list of the most important entities.
These entities can be drug names, gene targets, therapy acronyms, or key biological mechanisms.
Your final output MUST be a clean JSON list of these entity strings.
Example Output: `["Pembrolizumab", "FGFR2", "huCART19-IL18"]`
"""

# --- AGENT 3: Correlational Investigator ---
TRIAL_CONNECTOR_INSTRUCTION = """
You are a specialized investigator. You will receive a list of key research entities in the session state key `key_entities`.
Your job is to take each of these entities and perform a targeted search for related clinical trials.
You MUST call the `query_clinical_trials_from_mongodb` tool for this. You can call it multiple times if needed.
Your final output MUST be a consolidated list of all the relevant clinical trial summaries you found.
"""

MULTIMODAL_EVIDENCE_INSTRUCTION = "You are a visual evidence specialist. You will receive a text description in `visual_query_text`. Your job is to call the `find_similar_images` tool with this text to find a matching medical image."

# --- AGENT 4: Narrative Weaver & Analyst ---
SYNTHESIS_AND_REPORT_INSTRUCTION = """
You are a world-class AI Research Analyst and Communicator. Your final and most important job is to synthesize all available information into a single, insightful, and comprehensive report for the user.

**Your Available Information (from session state):**
- The user's original query (`current_research_query`).
- Initial broad search results (`local_db_results`, `web_search_results`, `clinical_trials_results`).
- Deep-dive search results of connected trials (`connected_trials_results`).


**Your Mandatory Workflow:**

1.  **Synthesize the Narrative:** For all other queries, weave a story.
    *   Start with the **Foundational Research** from the initial PubMed search.
    *   Create **The Connection** by explaining how this research leads to the deep-dive findings.
    *   Detail the **Clinical Trial Insights** from the connected trials.

2.  **Generate the "Aha!" Moment:** Conclude with the **"Generated Insight & Future Direction:"** section, providing one novel, forward-looking thought.
    *   For example, you could highlight a gap between the findings in a paper and the design of a clinical trial.

3.  **Ancillary Tasks:** If there are new web articles, ask the user if they'd like to save them via the `IngestionRouterAgent`.

**Your final output is the complete, formatted, user-facing report.** Review the detailed example in your documentation to match the expected tone and structure.
"""


# NEW: Instruction for the Text-Only Synthesizer
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
3.  **Ancillary Tasks:** If there are new web articles, ask the user if they'd like to save them.

**Your final output is the complete, formatted, text-only report.**
"""

# NEW: Instruction for the Visuals-Only Producer
VISUAL_SYNTHESIZER_INSTRUCTION = """
You are a specialist in identifying visual information within research data. Your only job is to find all opportunities for charts or images in the available text and call the appropriate tools to generate them.

**Your Available Information (from session state):**
- All raw text from `local_db_results`, `web_search_results`, and `clinical_trials_results`.

**Your Mandatory Workflow:**
1.  **Scan for All Visuals:** Read through all the available text in `local_db_results`, `web_search_results`, and `clinical_trials_results`.
2.  **Generate Charts:** For **every piece** of quantifiable data you find (e.g., percentages, funding numbers), you MUST:
    *   Format the data into the required JSON structure.
    *   Call the `VisualizationAgent` tool with the JSON to generate a chart.
3.  **Find Image Evidence:** For **every key visual description** you find (e.g., "ground-glass opacity," "spiculated nodule"), you MUST:
    *   Call the `MultimodalEvidenceAgent` tool with the description as the query.
4.  **Consolidate Output:** Your final output should be a well-formatted "Visuals Report" that presents the URLs for every chart and image you generated, each with a clear title.

**Example Output:**
Visual Evidence & Data Insights
Chart: Efficacy of huCART19-IL18 Therapy
[/static/charts/chart_abc123.png]
Visual Evidence: CT Scan of a Spiculated Nodule
[/static/medical_images/image_xyz789.png]
"""


# UPDATED: The Text & Data Weaver now knows about visual evidence.
TEXT_AND_DATA_WEAVER_INSTRUCTION = """
Your job is to write a comprehensive narrative report and prepare all data for final assembly.
**Input:** All prior research data from session state.
**Workflow:**
1.  Synthesize a full narrative report including "Foundational Research," "The Connection," and "Generated Insight."
2.  While writing, identify any quantifiable data suitable for a chart. If found, create a `chart_json` object.
3.  Also, identify any key **visual descriptions** (e.g., "ground-glass opacity," "spiculated nodule"). If found, create a `visual_query_text` string.
4.  Insert the placeholder `[CHART_PLACEHOLDER]` where the chart should go and `[IMAGE_EVIDENCE_PLACEHOLDER]` where the visual evidence should go.
**Output:** A single JSON object with all prepared assets: `{"narrative_text": "...", "chart_json": {...}, "visual_query_text": "..."}`.
"""

# NEW: The Final Passthrough Aggregator Agent
FINAL_AGGREGATOR_INSTRUCTION = """
You are a simple report aggregator. Your only job is to combine three reports into one.
You will have two pieces of information in the session state:
- `text_report`: A detailed written summary.
- `visual_report`: A report containing URLs for charts and images.
- `image_report`: A report containing URLs for images.

Your final output **MUST** be the `text_report` followed immediately by the `visual_report`. Do not add any extra words, summaries, or modifications. Just combine them.
"""


IMAGE_EVIDENCE_PRODUCER_INSTRUCTION = """
You are a visual evidence specialist. Your only job is to scan all available text in the session state and find opportunities to show relevant medical images.

**Your Mandatory Workflow:**
1.  **Identify Core Subject:** First, determine the primary subject of the research (e.g., "lung cancer," "lymphoma," "brain tumor").
2.  **Formulate a Broad Visual Query:** Create a general query based on the core subject. Instead of looking for hyper-specific terms, broaden the search.
    *   **Good Example:** If the report is about lung cancer, a good query is "CT scan of a lung with a possible nodule."
    *   **Bad Example:** If the report is about lung cancer, do not search for "adenocarcinoma with lepidic growth pattern" unless that exact phrase is known to be in the image descriptions.
3.  **Call the Evidence Tool:** You **MUST** call the `MultimodalEvidenceAgent` tool using the broad visual query you formulated.
4.  **Construct URL and Consolidate Output:** If the tool returns any results, take the first result and construct the full URL. Your final output must be a markdown-formatted "Visual Evidence" report. If no images are found, your output should state that clearly.
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
            "You are a specialized agent. Your task is to search the clinical_trial collection in the MongoDB database "
            "for clinical trial information relevant to the user's query.\n"
            "1. Retrieve the user's query from the session state key 'current_research_query'.\n"
            "2. Call the 'query_clinical_trials_from_mongodb' tool using this query.\n"
            "3. The list of clinical trial study summaries (dictionaries) returned by the tool is your primary result. "
            "Output this list directly."
        ),
        tools=[query_clinical_trials_from_mongodb], # Pass the function directly
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
        output_key="key_entities" # Its output is a list of entities
    )

    # Agent 3: Correlational Investigator
    trial_connector_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TrialConnectorAgent",
        instruction=TRIAL_CONNECTOR_INSTRUCTION,
        tools=[query_clinical_trials_from_mongodb],
        output_key="connected_trials_results" # Its output is the list of connected trials
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
    
    visualization_agent = LlmAgent(
       model=GEMINI_PRO_MODEL_ID,
       name="VisualizationAgent",
       instruction=VISUALIZATION_AGENT_INSTRUCTION,
       tools=[generate_simple_bar_chart, generate_simple_line_chart, generate_pie_chart, generate_grouped_bar_chart], # UPDATED TOOLS
       output_key="visualization_output" # Or handle output directly
    )
    logging.info(f"VisualizationAgent instance created: {visualization_agent.name}")

    visualization_agent_tool = AgentTool(agent=visualization_agent)
    if hasattr(visualization_agent_tool, 'run_async') and callable(getattr(visualization_agent_tool, 'run_async')):
       visualization_agent_tool.func = visualization_agent_tool.run_async # type: ignore


    trend_data_fetcher_agent = LlmAgent(model=GEMINI_PRO_MODEL_ID, name="TrendDataFetcherAgent", instruction="Call the get_publication_trend tool.", tools=[get_publication_trend], output_key="trend_chart_json")
    
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
        description="A smart data librarian that analyzes text and routes it to the correct database (PubMed or Clinical Trials)."
    )
    ingestion_router_agent_tool = AgentTool(agent=ingestion_router_agent)
    if hasattr(ingestion_router_agent_tool, 'run_async'):
       ingestion_router_agent_tool.func = ingestion_router_agent_tool.run_async

    # Agent A: The Narrative Writer
    text_synthesizer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="TextSynthesizerAgent",
        instruction=TEXT_SYNTHESIZER_INSTRUCTION,
        output_key="text_report"
    )

    # Agent B: The Visual Producer
    visual_synthesizer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="VisualSynthesizerAgent",
        instruction=VISUAL_SYNTHESIZER_INSTRUCTION,
        tools=[visualization_agent_tool, multimodal_evidence_tool],
        output_key="visual_report"
    )

    # NEW - Agent C: The Image Detective
    image_evidence_producer_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="ImageEvidenceProducerAgent",
        instruction=IMAGE_EVIDENCE_PRODUCER_INSTRUCTION,
        tools=[AgentTool(agent=multimodal_evidence_agent)],
        output_key="image_report"
    )
    
    
    # Agent 4: Narrative Weaver
    synthesis_and_report_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID,
        name="SynthesisAndReportAgent",
        instruction=SYNTHESIS_AND_REPORT_INSTRUCTION,
        tools=[
            visualization_agent_tool,
            ingestion_router_agent_tool,
            get_publication_trend # Give it direct access to the analytical tool
        ]
    )



    # NEW: The Parallel Synthesizer
    parallel_synthesis_agent = ParallelAgent(
        name="ParallelSynthesisAgent",
        description="Splits the final report generation into two parallel streams: text and visuals.",
        sub_agents=[
            text_synthesizer_agent,
            visual_synthesizer_agent,
            image_evidence_producer_agent
        ]
        # The outputs "text_report" and "visual_report" will be available in session state
    )

    # NEW: The Final Aggregator Agent
    final_report_aggregator_agent = LlmAgent(
        name="FinalReportAggregatorAgent",
        model=GEMINI_PRO_MODEL_ID,
        instruction=FINAL_AGGREGATOR_INSTRUCTION
        # No tools needed, it just processes text from session state
    )
    

    # NEW: Define the Master Research Synthesizer as a Sequential Agent
    master_research_synthesizer = SequentialAgent(
        name="MasterResearchSynthesizer",
        description="A sequential research assembly line that generates novel insights by connecting published papers to clinical trials.",
        sub_agents=[
            data_gathering_and_connection_agent,
            parallel_synthesis_agent,
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
            trend_analysis_tool,
            master_research_synthesizer_tool 
        ]
    )
    intent_router_agent_tool = AgentTool(agent=intent_router_agent)
    if hasattr(intent_router_agent_tool, 'run_async'):
        intent_router_agent_tool.func = intent_router_agent_tool.run_async

    all_root_agent_tools.append(intent_router_agent_tool)
    logging.info("IntentRouterAgent created and added to Root Agent's tools.")

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
        tools=all_root_agent_tools, # Contains MCPToolsets + ProactiveContextOrchestratorTool
    )

    logging.info(f"Root Agent ('{root_agent.name}') created with {len(root_agent.tools or [])} tools.")
    if root_agent.tools:
        tool_names = [getattr(t, 'name', str(type(t))) for t in root_agent.tools]
        logging.info(f"Root Agent tools list: {tool_names}")
    else:
        logging.warning("Root Agent has no tools configured.")

    return root_agent
