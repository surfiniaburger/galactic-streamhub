# /Users/surfiniaburger/Desktop/app/proactive_agents.py
import logging
import json # For parsing agent outputs
from typing import AsyncGenerator, List, Dict, Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset # For type hinting if needed
from tools.tts_tool import synthesize_speech_segment # NEW IMPORT
from google.adk.tools.agent_tool import AgentTool


# Model IDs (can be centralized or passed during instantiation)
GEMINI_FLASH_MODEL_ID = "gemini-2.0-flash"
GEMINI_MULTIMODAL_MODEL_ID = "gemini-2.0-flash-live-preview-04-09" # Or your preferred multimodal

# --- Instructions for Sub-Agents ---

ENVIRONMENTAL_MONITOR_INSTRUCTION = """
You are an Environmental Monitor. Your task is to analyze a list of 'seen_items' (objects identified visually in the user's environment) and a general 'user_activity_hint' (derived from the user's query, could be vague).
Based on these, identify potential 'context_keywords' that suggest what the user might be doing or interested in.
Focus on identifying contexts that could lead to proactive assistance.

Example:
- seen_items: ["rye whiskey bottle", "sweet vermouth bottle", "bitters", "mixing glass"], user_activity_hint: "thinking about a drink"
  Output: {"identified_context_keywords": ["cocktail_making", "manhattan_ingredients_present"]}
- seen_items: ["board game box: Terraforming Mars", "dice", "player mats"], user_activity_hint: "what should I do tonight?"
  Output: {"identified_context_keywords": ["board_game_setup", "terraforming_mars"]}
- seen_items: ["laptop", "coffee mug", "notebook"], user_activity_hint: "working"
  Output: {"identified_context_keywords": ["work_session", "productivity"]}
- seen_items: ["textbook", "notebook", "pen"], user_activity_hint: "studying"
  Output: {"identified_context_keywords": ["studying_from_textbook", "academic_research_context"]}

Output ONLY a JSON object with the key "identified_context_keywords" (a list of strings).
"""

CONTEXTUAL_PRECOMPUTATION_INSTRUCTION = """
You are a Proactive Precomputation Agent. You receive 'context_keywords' (e.g., "cocktail_making", "manhattan_ingredients_present") and the original 'user_goal'.
Your goal is to determine if a proactive suggestion is warranted and, if so, pre-fetch relevant information using available tools (like CocktailDB for recipes, Google Maps for places, or GoogleSearchAgentTool for general queries).
You also have access to 'QueryPubMedKnowledgeBase' for biomedical research topics.

1.  **Analyze Context**: If 'context_keywords' strongly suggest a common task (e.g., making a specific cocktail, looking for a type of store for missing items implied by the context):
    *   Formulate a `proactive_suggestion_text` to offer help to the user.
    *   Use tools to gather `precomputed_data` that would be useful if the user accepts the suggestion.
    *   Example: If context is "manhattan_ingredients_present", suggest "I see you might be making a Manhattan. Would you like the recipe?". Precompute the Manhattan recipe.
    *   Example: If context is "studying_from_textbook" and the user's goal is vague like "help me study", suggest "I see you have a textbook. If you tell me the subject, I can search my biomedical knowledge base for related articles or provide general study tips.". You could precompute general study tips using `GoogleSearchAgentTool(request="effective study techniques for textbooks")`. If the user later specifies "cardiology", you might use `QueryPubMedKnowledgeBase(query_text="overview of cardiology for students")`.
    
2.  **Tool Usage**: Prioritize specific tools (Cocktail, Maps) if applicable. If general information is needed that these tools don't cover, use the `GoogleSearchAgentTool` by providing it a clear search query as its `request` argument (e.g., `GoogleSearchAgentTool(request="history of the Manhattan cocktail")`). Generate the appropriate function calls. The system will execute them.
3.  **Output**:
    *   If a proactive action is taken: Output a JSON object:
      `{"proactive_suggestion_text": "suggestion for user", "precomputed_data": {"recipe": "...", "store_info": "..."}}`
    *   If no proactive action is suitable based on the context: Output JSON:
      `{"proactive_suggestion_text": null, "precomputed_data": null}`
"""

REACTIVE_TASK_DELEGATOR_INSTRUCTION = """
You are a specialized assistant that helps users accomplish tasks based on their goals and items visually identified in their environment. You will be provided with the user's goal (`user_goal`) and a list of 'seen_items'.
You may also receive `precomputed_data` if a proactive suggestion related to this task was previously accepted by the user.

Your capabilities include:
1.  **Recipe and Ingredient Analysis**:
    *   If `precomputed_data` contains a relevant recipe, use that.
    *   Otherwise, if the `user_goal` involves making a food or drink item, use the 'Cocktail' tool to find the recipe.
    *   Compare the recipe ingredients against the `seen_items`.
    *   Clearly state which ingredients the user has and which are missing.
2.  **Location Finding for Missing Items**:
    *   If `precomputed_data` contains relevant store information, use that.
    *   Otherwise, if ingredients are missing and the `user_goal` implies finding them, use the 'Google Maps' tool to find stores.
    *   **CRITICAL**: If you use the 'Google Maps' tool, your response format MUST be structured. First, provide a brief conversational summary. Then, on a new line, you MUST provide a special data block starting with the exact marker `LOCATIONS_JSON::` followed immediately by a raw JSON array of location objects. Each object in the array must have `name`, `address`, and `rating` keys. The rating should be a number.
    *   **Example Location Finding Output**:
        I found a few places for you that sell vermouth.
        LOCATIONS_JSON::[{"name": "Total Wine & More", "address": "123 Main St, Anytown, USA", "rating": 4.5}, {"name": "Local Liquor", "address": "456 Oak Ave, Anytown, USA", "rating": 4.1}]
3.  **General Information Retrieval**: If the user asks a general question not covered by Cocktail or Maps tools, or if `precomputed_data` contains search results, use the `GoogleSearchAgentTool` to find an answer. Formulate a clear search query for its `request` argument (e.g., `GoogleSearchAgentTool(request="What is the weather like in London tomorrow if I want to make a cocktail outside?")`).
4.  **Biomedical Information Retrieval**: If the user's goal is related to medical or scientific research, and `precomputed_data` doesn't already cover it, use the `QueryPubMedKnowledgeBase` tool.
5.  **General Task Execution**: Address other user goals using available tools as appropriate.

**Input Format You Will Receive (as part of a larger JSON in session state, but you'll be invoked with these key args):**
*   `user_goal`: A string describing what the user wants to achieve.
*   `seen_items`: A list of strings representing items visually identified.
*   `precomputed_data` (optional): A dictionary with previously fetched information if a proactive suggestion was accepted.

**Your Response Obligation:**
*   If you use the 'Google Maps' tool, you MUST follow the structured JSON format described in section 2.
*   For all other tasks, you MUST combine all gathered information into a single, comprehensive, and helpful textual response. Be direct and structure your answer clearly.
If using `precomputed_data`, mention that you're using previously fetched info to be faster.
"""

# --- Define the Sub-Agent Classes ---

class EnvironmentalMonitorAgent(LlmAgent):
    """
    An LlmAgent that analyzes visual context and user activity hints
    to identify potential proactive context keywords.
    """
    def __init__(self, model: str = GEMINI_MULTIMODAL_MODEL_ID, **kwargs):
        super().__init__(
            model=model,
            name="EnvironmentalMonitorAgent", # Default name, can be overridden
            instruction=ENVIRONMENTAL_MONITOR_INSTRUCTION,
            description="Analyzes visual context to identify keywords for proactive assistance.",
            output_key="identified_context_keywords_output", # Agent will write its JSON output here
            **kwargs
        )

class ContextualPrecomputationAgent(LlmAgent):
    """
    An LlmAgent that determines if proactive suggestion is warranted based on context
    and pre-fetches information using available tools.
    """
    def __init__(self, model: str = GEMINI_FLASH_MODEL_ID, tools: List[Any] = None, **kwargs):
        super().__init__(
            model=model,
            name="ContextualPrecomputationAgent",
            instruction=CONTEXTUAL_PRECOMPUTATION_INSTRUCTION,
            description="Proactively fetches information based on context keywords.",
            tools=tools or [], # Pass tools if this agent makes direct tool calls
            output_key="proactive_precomputation_output", # Agent will write its JSON output here
            **kwargs
        )

class ReactiveTaskDelegatorAgent(LlmAgent):
    """
    An LlmAgent that handles explicit user tasks or executes tasks after
    a proactive suggestion is accepted, using precomputed data if available.
    """
    def __init__(self, model: str = GEMINI_FLASH_MODEL_ID, tools: List[Any] = None, **kwargs):
        super().__init__(
            model=model,
            name="ReactiveTaskDelegatorAgent",
            instruction=REACTIVE_TASK_DELEGATOR_INSTRUCTION,
            description="Handles explicit user tasks or executes precomputed suggestions.",
            tools=tools or [], # Pass tools if this agent makes direct tool calls
            # output_key="reactive_task_final_answer"
            **kwargs
        )

class ProactiveContextOrchestratorAgent(BaseAgent):
    """
    Custom orchestrator for proactive and reactive assistance.
    Manages EnvironmentalMonitor, ContextualPrecomputation, and ReactiveTaskDelegator agents.
    """
    environmental_monitor: LlmAgent
    contextual_precomputation: LlmAgent
    reactive_task_delegator: LlmAgent

    # Pydantic model_config for arbitrary types
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        environmental_monitor: LlmAgent,
        contextual_precomputation: LlmAgent,
        reactive_task_delegator: LlmAgent,
        # MCPToolsets that might be needed by sub-agents if they call tools directly
        # (though typically tools are on the RootAgent and sub-agents just declare calls)
        mcp_toolsets: List[MCPToolset] = None,
    ):
        sub_agents_list = [
            environmental_monitor,
            contextual_precomputation,
            reactive_task_delegator,
        ]
        # MCPToolsets are not sub-agents. They are tools used by agents.
        # The sub-agents (LlmAgent types) are already configured with tools if needed.
        super().__init__(
            name=name,
            description="Orchestrates proactive context monitoring, precomputation, and reactive task execution.",
            # Pydantic will assign these to instance attributes
            environmental_monitor=environmental_monitor,
            contextual_precomputation=contextual_precomputation,
            reactive_task_delegator=reactive_task_delegator,
            sub_agents=sub_agents_list,
        )
        # Store them explicitly as well if super() doesn't handle it for direct access
        self.environmental_monitor = environmental_monitor
        self.contextual_precomputation = contextual_precomputation
        self.reactive_task_delegator = reactive_task_delegator

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logging.info(f"[{self.name}] Orchestrator started.")

        user_goal = ctx.session.state.get("input_user_goal", "")
        seen_items = ctx.session.state.get("input_seen_items", [])
        user_activity_hint = user_goal # Could be refined
        context_keywords = [] # Initialize

        # --- 1. Environmental Monitoring ---
        # Pass current visual context (seen_items) and user activity hint
        # The EnvironmentalMonitorAgent's instruction tells it to use these from state if not passed directly.
        # For LlmAgent called via run_async, it's better if it can infer inputs from its instruction
        # or if we explicitly set specific keys it expects.
        # Let's assume its instruction is robust enough to pick up 'seen_items' and 'user_activity_hint'
        # if they are present in the broader session state, or we can set specific input keys.
        # For clarity, we can set specific keys that its instruction might refer to.
        ctx.session.state["monitor_input_seen_items"] = seen_items
        ctx.session.state["monitor_input_user_activity_hint"] = user_activity_hint

        logging.info(f"[{self.name}] Running EnvironmentalMonitorAgent...")
        async for event in self.environmental_monitor.run_async(ctx):
            # Yield events for logging or if the RootAgent needs intermediate updates.
            # The EnvironmentalMonitorAgent will write to "identified_context_keywords_output" in session_state.
            yield event

        identified_keywords_json_str = ctx.session.state.get("identified_context_keywords_output", "{}")
        try:
            # Clean the string: remove markdown code block fences and strip whitespace
            cleaned_json_str = identified_keywords_json_str.strip()
            if cleaned_json_str.startswith("```json"):
                cleaned_json_str = cleaned_json_str[7:]
            if cleaned_json_str.endswith("```"):
                cleaned_json_str = cleaned_json_str[:-3]
            cleaned_json_str = cleaned_json_str.strip()

            identified_keywords_data = json.loads(cleaned_json_str)
            context_keywords = identified_keywords_data.get("identified_context_keywords", [])
            logging.info(f"[{self.name}] Context keywords from EnvironmentalMonitorAgent: {context_keywords}")
        except json.JSONDecodeError:
            logging.error(f"[{self.name}] Failed to parse JSON from EnvironmentalMonitorAgent. Raw string: '{identified_keywords_json_str}'")
            # Fallback: use initial keywords from RootAgent if monitor fails
            context_keywords = ctx.session.state.get("initial_context_keywords", [])
            logging.info(f"[{self.name}] Falling back to initial_context_keywords: {context_keywords}")

        proactive_suggestion = None
        precomputed_data = None

        # --- 2. Proactive Precomputation (Conditional) ---
        is_proactive_opportunity = False
        # Refined proactive opportunity logic:
        # Check if any recognized proactive contexts are present
        proactive_triggers = ["cocktail_making", "manhattan_ingredients_present",
                              "board_game_setup", "terraforming_mars", "studying_from_textbook",
                              "academic_research_context",
                              "salad_ingredients", "food_preparation"] # Add more as needed

        has_proactive_context = any(trigger in context_keywords for trigger in proactive_triggers)

        if has_proactive_context:
            # And if the user's goal is general or aligns with a need for suggestions
            general_query_indicators = ["what", "how", "ideas", "suggest", "help", "do"]
            is_general_goal = not user_goal or \
                              any(indicator in user_goal.lower() for indicator in general_query_indicators) or \
                              "hungry" in user_goal.lower() # Specific case for food

            if is_general_goal:
                is_proactive_opportunity = True

        if is_proactive_opportunity:
            logging.info(f"[{self.name}] Proactive opportunity identified. Running ContextualPrecomputationAgent...")
            ctx.session.state["precomp_input_context_keywords"] = context_keywords
            ctx.session.state["precomp_input_user_goal"] = user_goal

            async for event in self.contextual_precomputation.run_async(ctx):
                yield event # Yield events

            precomp_output_str = ctx.session.state.get("proactive_precomputation_output", "{}")
            try:
                # Clean the string: remove markdown code block fences and strip whitespace
                cleaned_json_str = precomp_output_str.strip()
                if cleaned_json_str.startswith("```json"):
                    cleaned_json_str = cleaned_json_str[7:]
                if cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[:-3]
                cleaned_json_str = cleaned_json_str.strip()
                precomp_data_from_agent = json.loads(cleaned_json_str)
                proactive_suggestion = precomp_data_from_agent.get("proactive_suggestion_text")
                precomputed_data = precomp_data_from_agent.get("precomputed_data")
                if proactive_suggestion:
                    logging.info(f"[{self.name}] Generated proactive suggestion from ContextualPrecomputationAgent: {proactive_suggestion}")
                else:
                    logging.info(f"[{self.name}] ContextualPrecomputationAgent did not generate a proactive suggestion.")
            except json.JSONDecodeError:
                logging.error(f"[{self.name}] Failed to parse JSON from ContextualPrecomputationAgent. Raw string: '{precomp_output_str}'")
                proactive_suggestion = None # Ensure it's reset
                precomputed_data = None

        # --- 3. Store Proactive Results or Delegate to Reactive Agent ---
        if proactive_suggestion:
            ctx.session.state["proactive_suggestion_to_user"] = proactive_suggestion
            ctx.session.state["proactive_precomputed_data_for_next_turn"] = precomputed_data
            # The orchestrator itself doesn't "speak". It sets up state for the RootAgent.
            # We can indicate the type of output for the RootAgent to interpret.
            ctx.session.state["orchestrator_final_output_type"] = "proactive_suggestion_ready"
            logging.info(f"[{self.name}] Proactive suggestion ready for RootAgent.")
            # Yield a custom event if needed, or RootAgent checks state
            yield Event.create_intermediate_response(
                author=self.name,
                content_parts=[f"Proactive suggestion generated: {proactive_suggestion}"] # For logging/debug
            )

        else:
            logging.info(f"[{self.name}] No proactive suggestion. Running ReactiveTaskDelegatorAgent...")
            ctx.session.state["reactive_input_user_goal"] = user_goal
            ctx.session.state["reactive_input_seen_items"] = seen_items
            # Check if there's precomputed data from a *previous* accepted suggestion
            accepted_precomputed_data = ctx.session.state.get("accepted_precomputed_data", None)
            if accepted_precomputed_data:
                ctx.session.state["reactive_input_precomputed_data"] = accepted_precomputed_data
                logging.info(f"[{self.name}] Passing accepted precomputed data to ReactiveAgent.")

            # The ReactiveTaskDelegatorAgent will run. Its LlmAgent definition should specify
            # an output_key like "reactive_task_final_answer".
            async for event in self.reactive_task_delegator.run_async(ctx):
                yield event # Yield all events from the reactive agent, including its final response

            # The final response from reactive_task_delegator will be in its event.
            # The RootAgent will pick up the final response event from this sub-agent.
            ctx.session.state["orchestrator_final_output_type"] = "reactive_answer_provided"
            logging.info(f"[{self.name}] Reactive task delegation completed.")
        
        logging.info(f"[{self.name}] Orchestrator finished.")



# --- Instruction for DialogueFormatterAgent ---
DIALOGUE_FORMATTER_INSTRUCTION = """
You are a scriptwriter. Your task is to take a single block of input text (a summary or report) and reformat it into a dialogue script for two speakers: PERSON_A and PERSON_B.
Alternate between PERSON_A and PERSON_B for each line or logical segment of the text.
Ensure the dialogue flows naturally and covers all the information in the original text.
Output the result as a JSON list of objects, where each object has a "speaker" (either "PERSON_A" or "PERSON_B") and a "line" (the text for that speaker).

Example Input Text:
"The study found a 25% improvement in patient outcomes with the new drug. Side effects were minimal, primarily mild nausea reported in 5% of subjects. Further research is recommended over a longer period."

Example JSON Output:
[
  {"speaker": "PERSON_A", "line": "The study found a 25% improvement in patient outcomes with the new drug."},
  {"speaker": "PERSON_B", "line": "Side effects were minimal, primarily mild nausea reported in 5% of subjects."},
  {"speaker": "PERSON_A", "line": "Further research is recommended over a longer period."}
]

If the input text is very short (e.g., one sentence), assign it all to PERSON_A.
"""

# --- Define DialogueFormatterAgent ---
class DialogueFormatterAgent(LlmAgent):
    def __init__(self, model: str = GEMINI_FLASH_MODEL_ID, **kwargs):
        super().__init__(
            model=model,
            name="DialogueFormatterAgent",
            instruction=DIALOGUE_FORMATTER_INSTRUCTION,
            description="Reformats a block of text into a two-speaker dialogue script (JSON).",
            output_key="dialogue_script_json", # Agent will write its JSON output here
            **kwargs
        )

# --- Instruction for VideoReportAgent ---
VIDEO_REPORT_AGENT_INSTRUCTION = """
You are a Video Report Producer. Your goal is to create a video summary based on a textual report and associated chart image URLs.

Your workflow is as follows:
1.  **Receive Input**: You will get the main `report_text` and a list of `chart_image_urls` from session state.
2.  **Format Dialogue**:
    *   Call the `DialogueFormatterAgent` tool with the `report_text` to convert it into a two-speaker dialogue script (JSON).
3.  **Synthesize Speech**:
    *   Parse the JSON dialogue script.
    *   For each line in the script:
        *   Call the `synthesize_speech_segment` tool, providing the `text_to_synthesize` (the line) and the `speaker` ("PERSON_A" or "PERSON_B").
        *   The `synthesize_speech_segment` tool will return audio bytes. You need to collect these audio bytes for each segment.
4.  **Assemble Video (Conceptual - Tool to be added later)**:
    *   Once all audio segments are synthesized, you would ideally call a `VideoAssemblyTool`.
    *   This tool would take the list of audio segment bytes and the `chart_image_urls`.
    *   It would combine them into a video (e.g., slideshow of charts with voiceover).
    *   For now, since `VideoAssemblyTool` is not yet implemented, your final output will be a message stating that audio has been synthesized and listing the chart URLs that *would* be in the video.
    *   Also, store the collected audio segment bytes in session state under a key like `ctx.session.state['generated_audio_segments_bytes']` (list of bytes).

**Output**:
*   If successful up to audio synthesis: A message like "Audio for the report has been synthesized. The video would include the following charts: [list of chart URLs]. The audio segments are ready for video assembly."
*   If dialogue formatting or TTS fails: Report the error.

**Example of how you'd call `synthesize_speech_segment` (conceptual for your internal plan):**
`synthesize_speech_segment(text_to_synthesize="The study showed great results.", speaker="PERSON_A")`
`synthesize_speech_segment(text_to_synthesize="Indeed, the outcomes were positive.", speaker="PERSON_B")`
"""

# --- Define VideoReportAgent ---
class VideoReportAgent(LlmAgent):
    def __init__(self, model: str = GEMINI_FLASH_MODEL_ID, **kwargs):
        # The tools this agent will call:
        dialogue_formatter_agent_instance = DialogueFormatterAgent(model=model) # Create an instance
        dialogue_formatter_tool = AgentTool(agent=dialogue_formatter_agent_instance)
        # Patch AgentTool if needed (as per your existing pattern)
        if hasattr(dialogue_formatter_tool, 'run_async') and callable(getattr(dialogue_formatter_tool, 'run_async')):
            dialogue_formatter_tool.func = dialogue_formatter_tool.run_async # type: ignore

        agent_tools = [
            dialogue_formatter_tool,
            synthesize_speech_segment # ADK will wrap this Python function as a FunctionTool
            # VideoAssemblyTool will be added here later
        ]

        super().__init__(
            model=model,
            name="VideoReportAgent",
            instruction=VIDEO_REPORT_AGENT_INSTRUCTION,
            description="Produces a video summary report with dialogue and charts.",
            tools=agent_tools,
            # output_key="video_report_status" # Or let it output text directly
            **kwargs
        )