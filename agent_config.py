# agent_config.py
from google.adk.agents.llm_agent import LlmAgent
from typing import Any, Dict, List
import logging
from google.adk.tools.mcp_tool.mcp_tool import MCPTool # Import MCPTool to check its type
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

MODEL_ID_STREAMING = "gemini-2.0-flash-live-preview-04-09" # Or your preferred streaming-compatible model like "gemini-2.0-flash-exp"
GEMINI_PRO_MODEL_ID = "gemini-2.0-flash"

# --- Instructions for the new TaskExecutionAgent ---
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
Role: You are AVA (Advanced Visual Assistant). You interact with users via audio and can 'see' their environment through a video stream using your advanced multimodal capabilities.
Your primary goal is to understand user requests, analyze their visual surroundings when relevant, and then EITHER directly answer if the query is simple OR delegate to specialized tools or agents if the query is complex and requires multi-step reasoning or specific tool usage.

Core Capabilities:
1.  **Visual Scene Analysis (Multimodal Perception)**:
    *   When the user's query implies needing to understand their environment (e.g., "What do you see?", "Do I have X?", "I want to make Y with what's here"), carefully analyze the incoming video frames.
    *   Identify and list relevant objects or items visible that pertain to the user's query. For example, if they ask about making a drink, look for bottles, fruits, shakers, etc.
2.  **Direct Tool Usage (for simple, direct queries)**:
    *   You have direct access to tools for: cocktails (e.g., `search_cocktail_by_name`, `get_random_cocktail`), weather forecasts (`get_weather_forecast`), and potentially Airbnb bookings (`search_airbnb_listings` - though less likely for scene-based tasks). Use these for straightforward requests that don't require combining visual information with multi-step task execution.
3.  **Delegation to `TaskExecutionAgentTool` (for complex, scene-based, multi-step tasks)**:
    *   You have a highly capable specialized tool called `TaskExecutionAgentTool`.
    *   **WHEN TO USE `TaskExecutionAgentTool`**: Use this tool when the user's request clearly requires:
        a.  Understanding items from the visual scene (which you will provide).
        b.  AND THEN performing a sequence of actions like looking up a recipe, comparing it to the seen items, and potentially finding a store for missing items.
    *   **HOW TO USE `TaskExecutionAgentTool`**:
        1.  First, perform your visual analysis. Identify relevant items from the video feed based on the user's overall goal.
        2. Then, invoke `TaskExecutionAgentTool` by providing it with a single argument named `request`.
            *   The value of `request` should be a JSON string containing 'user_goal' and 'seen_items'.
            *   `request`: A JSON string. Example: '{"user_goal": "User wants to make a spicy margarita...", "seen_items": ["limes", "bottle"]}'
            *   Example Call: `TaskExecutionAgentTool(request='{"user_goal": "User wants to make a bloody mary...", "seen_items": ["vodka bottle", "celery stalk"]}')`
4.  **Conversational Interaction**: Engage in general conversation if no specific task or tool is appropriate. Ask clarifying questions if the user's request is ambiguous.
5.  **Response Formatting**: Always format your final response to the user using Markdown for enhanced readability. If the response is derived from `TaskExecutionAgentTool`, present that agent's findings clearly.
If you are absolutely unable to help with a request, or if none of your tools (including `TaskExecutionAgentTool`) are suitable for the task, politely state that you cannot assist with that specific request.
"""


def create_streaming_agent_with_mcp_tools(
    loaded_mcp_toolsets: List[MCPToolset], 
    #raw_mcp_tools_lookup_for_warnings: Dict[str, Any] # Temporary, for the warnings
) -> LlmAgent:
    
    all_root_agent_tools: List[Any] = []

    # 1. Add all MCPToolset instances directly to the agent's tools
    if loaded_mcp_toolsets:
        all_root_agent_tools.extend(loaded_mcp_toolsets)
        logging.info(f"Added {len(loaded_mcp_toolsets)} MCPToolset instance(s) to Root Agent tools.")
    
    # Remove the old MCPTool processing loop and patching for MCPTools.
    # The MCPToolset instance handles its internal tools.

    # --- Still check for presence for warnings (using the temporary raw_mcp_tools_lookup_for_warnings) ----
    # This part is a bit clunky now. Ideally, these warnings would be based on whether
    # the *MCPToolset* for 'ct' or 'maps' was successfully created.
    # For a quick check:
    # if not any(ts for ts in loaded_mcp_toolsets if "Cocktail" in getattr(ts, 'name', '').lower() or "ct" in getattr(ts, '_some_key_identifier', '')): # This is highly speculative
    #    logging.warning("Agent Configuration: 'Cocktail' MCPToolset might be missing...")
    # if not any(ts for ts in loaded_mcp_toolsets if "maps" in getattr(ts, 'name', '').lower()):
    #    logging.warning("Agent Configuration: 'Google Maps' MCPToolset might be missing...")
    # This warning logic needs to be rethought based on how to identify which toolset is which.
    # For now, let's rely on the warnings from main.py during toolset creation.
    # The warnings you had about 'Cocktail' MCP tool not found will still trigger based on the
    # temporary raw_mcp_tools_lookup_for_warnings if we still populate it, but it's not ideal.

    # 2. Create the TaskExecutionAgent instance
    # This agent is designed to be called by the Root Agent.
    # It does not need the MCP tools in its *own* `tools` list because its instructions
    # will lead it to generate function calls that the Root Agent (which *has* the tools) executes.
    task_execution_agent = LlmAgent(
        model=GEMINI_PRO_MODEL_ID , # Can be the same model or a different one
        name="TaskExecutionAgent", # This name is crucial for the AgentTool's function declaration
        instruction=TASK_EXECUTION_AGENT_INSTRUCTION,
        description="A specialized agent for multi-step task execution based on visual context and user goals. It can find recipes, compare ingredients to a list of seen items, and locate stores for missing items. Expects 'user_goal' (string) and 'seen_items' (list of strings) as input parameters.",
        # tools=[], # Explicitly empty, or omit. It uses tools from the calling (Root) agent's context.
    )
    logging.info(f"TaskExecutionAgent instance created. Name: {task_execution_agent.name}")

    # 3. Wrap the TaskExecutionAgent with AgentTool
    # This makes `task_execution_agent` callable as a tool by the `root_agent`.
    # The `name` and `description` for the tool are derived from the `task_execution_agent` instance.
    task_execution_agent_tool = AgentTool(
        agent=task_execution_agent,
        # Optional: skip_summarization=True if TaskExecutionAgent's output is always direct and well-formatted.
    )
    logging.info(f"--- Inspecting task_execution_agent_tool ---")
    logging.info(f"Type: {type(task_execution_agent_tool)}")
    logging.info(f"Attributes: {dir(task_execution_agent_tool)}")
    # You can also check for specific attributes you think might be relevant
    logging.info(f"Has 'name' attribute: {hasattr(task_execution_agent_tool, 'name')}, Value: {getattr(task_execution_agent_tool, 'name', 'N/A')}")
    logging.info(f"Has 'description' attribute: {hasattr(task_execution_agent_tool, 'description')}")
    logging.info(f"Is callable: {callable(task_execution_agent_tool)}") # AgentTool might be callable itself
    logging.info(f"Underlying agent: {task_execution_agent_tool.agent.name if task_execution_agent_tool.agent else 'None'}")
    logging.info(f"--- End Inspecting task_execution_agent_tool ---")
    import inspect
# ...
    logging.info(f"--- Inspecting task_execution_agent_tool with inspect ---")
# Get all members (attributes, methods, etc.)
    for name, member_obj in inspect.getmembers(task_execution_agent_tool):
     logging.info(f"Member: {name}, Type: {type(member_obj)}, Callable: {callable(member_obj)}")
# If you find a method that looks like its execution entry point:
    if hasattr(task_execution_agent_tool, 'some_execute_method_name'):
      logging.info(f"Signature of 'some_execute_method_name': {inspect.signature(task_execution_agent_tool.some_execute_method_name)}")
    logging.info(f"--- End Inspecting task_execution_agent_tool with inspect ---")
    all_root_agent_tools.append(task_execution_agent_tool)
    logging.info(f"TaskExecutionAgent wrapped as AgentTool ('{task_execution_agent_tool.name}') and added to Root Agent's tools.")

    # --- NEW: PATCH AgentTool ---
    # Similar to MCPTool, AgentTool also has a 'run_async' method and is not directly callable.
    # Let's patch it with a .func attribute pointing to its run_async method.
    if hasattr(task_execution_agent_tool, 'run_async') and callable(getattr(task_execution_agent_tool, 'run_async')):
        task_execution_agent_tool.func = task_execution_agent_tool.run_async
        logging.info(f"Patched AgentTool '{task_execution_agent_tool.name}' with .func attribute pointing to its run_async method.")
    else:
        logging.warning(f"Could not patch AgentTool '{task_execution_agent_tool.name}' with .func: 'run_async' not found or not callable.")
    # --- END PATCH ---

    # Your inspection logs here (they will now reflect the new .func attribute if patching was successful)
    logging.info(f"--- Inspecting task_execution_agent_tool AFTER attempting patch ---")
    logging.info(f"Type: {type(task_execution_agent_tool)}")
    # ... (your other inspection logs for AgentTool)
    logging.info(f"Has 'func' attribute after patch: {hasattr(task_execution_agent_tool, 'func')}")
    if hasattr(task_execution_agent_tool, 'func'):
        logging.info(f"Value of 'func' attribute: {getattr(task_execution_agent_tool, 'func')}")
    logging.info(f"--- End Inspecting task_execution_agent_tool AFTER attempting patch ---")

    # 4. Create the Root Agent (mcp_streaming_assistant)
    # This is your main, user-facing multimodal agent.
    root_agent = LlmAgent(
        model=MODEL_ID_STREAMING, # Must be a multimodal model
        name="mcp_streaming_assistant", # As defined in your original setup
        instruction=ROOT_AGENT_INSTRUCTION_STREAMING,
        tools=all_root_agent_tools, # Contains MCP tools + TaskExecutionAgentTool
    )
    
    logging.info(f"Root Agent ('{root_agent.name}') created with {len(root_agent.tools or [])} tools.")
    if root_agent.tools:
        tool_names = [getattr(t, 'name', str(type(t))) for t in root_agent.tools]
        logging.info(f"Root Agent tools list: {tool_names}")
    else:
        logging.warning("Root Agent has no tools configured.")

    return root_agent





