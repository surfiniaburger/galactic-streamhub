import logging
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, Union
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext # For tool callbacks
from google.adk.tools.base_tool import BaseTool # For tool callbacks
from google.adk.models import LlmRequest, LlmResponse # For model callbacks
from google.genai import types as genai_types # Alias for clarity
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field

# --- Constants ---
APP_NAME = "story_app_with_callbacks"
USER_ID = "callback_user"
SESSION_ID = "callback_session"
GEMINI_2_FLASH = "gemini-2.0-flash" # or your preferred model

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Callback Function Definitions ---

async def my_before_agent_callback(callback_context: CallbackContext) -> Optional[genai_types.Content]:
    logger.info(f"[CALLBACK] === BEFORE AGENT: {callback_context.agent_name} (Inv: {callback_context.invocation_id}) ===")
    logger.info(f"[CALLBACK] Before Agent - Current State: {callback_context.state.to_dict()}")
    # Example: To skip agent execution, return genai_types.Content(...)
    # if callback_context.state.get("skip_agent_" + callback_context.agent_name):
    #     logger.info(f"[CALLBACK] Skipping agent {callback_context.agent_name} based on state.")
    #     return genai_types.Content(parts=[genai_types.Part(text=f"Agent {callback_context.agent_name} skipped by before_agent_callback.")], role="model")
    return None

async def my_after_agent_callback(callback_context: CallbackContext) -> Optional[genai_types.Content]:
    logger.info(f"[CALLBACK] === AFTER AGENT: {callback_context.agent_name} (Inv: {callback_context.invocation_id}) ===")
    logger.info(f"[CALLBACK] After Agent - Current State: {callback_context.state.to_dict()}")
    # Example: To replace agent's output, return genai_types.Content(...)
    # if callback_context.agent_output and callback_context.state.get("modify_output_" + callback_context.agent_name):
    #     logger.info(f"[CALLBACK] Modifying output of agent {callback_context.agent_name}.")
    #     original_text = callback_context.agent_output.parts[0].text if callback_context.agent_output.parts else ""
    #     return genai_types.Content(parts=[genai_types.Part(text=f"Modified: {original_text}")], role="model")
    return None

async def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    logger.info(f"[CALLBACK] --- BEFORE MODEL ({callback_context.agent_name}) ---")
    logger.info(f"[CALLBACK] Before Model - LLM Request Config: {llm_request.config}")
    logger.info(f"[CALLBACK] Before Model - LLM Request Contents: {llm_request.contents}")
    # Example: To skip LLM call and return a canned response:
    # if "canned_response_trigger" in llm_request.contents[-1].parts[0].text:
    #     logger.info(f"[CALLBACK] Skipping LLM call for {callback_context.agent_name}, providing canned response.")
    #     return LlmResponse(content=genai_types.Content(parts=[genai_types.Part(text="Canned LLM response.")], role="model"))
    return None

async def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    logger.info(f"[CALLBACK] --- AFTER MODEL ({callback_context.agent_name}) ---")
    logger.info(f"[CALLBACK] After Model - LLM Response: {llm_response.content}")
    # Example: To modify LLM response:
    # if llm_response.content and llm_response.content.parts:
    #     original_text = llm_response.content.parts[0].text
    #     modified_text = f"[POST-PROCESSED BY CALLBACK] {original_text}"
    #     llm_response.content.parts[0].text = modified_text
    #     logger.info(f"[CALLBACK] Modified LLM response for {callback_context.agent_name}.")
    #     return llm_response # Return the modified LlmResponse
    return None

async def my_before_tool_callback(tool: BaseTool, args: Dict[str, Any], callback_context: ToolContext) -> Optional[Dict[str, Any]]:
    logger.info(f"[CALLBACK] >>> BEFORE TOOL: {tool.name} (Agent: {callback_context.agent_name}) >>>")
    logger.info(f"[CALLBACK] Before Tool - Args: {args}")
    # Example: To skip tool execution and return a mock result:
    # if tool.name == "SomeSpecificTool" and args.get("mock_me"):
    #     logger.info(f"[CALLBACK] Mocking result for tool {tool.name}.")
    #     return {"mocked_data": "This is a mock result"}
    return None

async def my_after_tool_callback(tool: BaseTool, args: Dict[str, Any], callback_context: ToolContext, tool_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    logger.info(f"[CALLBACK] <<< AFTER TOOL: {tool.name} (Agent: {callback_context.agent_name}) <<<")
    logger.info(f"[CALLBACK] After Tool - Original Response: {tool_response}")
    # Example: To modify tool response:
    # if "original_key" in tool_response:
    #     tool_response["modified_key"] = "added by after_tool_callback"
    #     logger.info(f"[CALLBACK] Modified tool response for {tool.name}.")
    #     return tool_response
    return None

# --- Custom Orchestrator Agent ---
class StoryFlowAgent(BaseAgent):
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent
    loop_agent: LoopAgent
    sequential_agent: SequentialAgent

    model_config = {"arbitrary_types_allowed": True}

# Inside class StoryFlowAgent:
    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
        before_agent_callback: Optional[callable] = None,
        after_agent_callback: Optional[callable] = None,
    ):
        # Create internal agents *before* calling super().__init__
        loop_agent = LoopAgent(
            name="CriticReviserLoop",
            sub_agents=[critic, reviser],
            max_iterations=2,
            before_agent_callback=my_before_agent_callback, # Agent lifecycle callbacks are fine
            after_agent_callback=my_after_agent_callback,
        )
        sequential_agent = SequentialAgent(
            name="PostProcessingSequential",
            sub_agents=[grammar_check, tone_check],
            before_agent_callback=my_before_agent_callback, # Agent lifecycle callbacks are fine
            after_agent_callback=my_after_agent_callback,
        )

        sub_agents_list = [
            story_generator,
            loop_agent,
            sequential_agent,
        ]

        super().__init__(
            name=name,
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            loop_agent=loop_agent,
            sequential_agent=sequential_agent,
            sub_agents=sub_agents_list,
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting story generation workflow.")

        # Helper to create a tool context for manual callback invocation
        # ToolContext is initialized directly from the InvocationContext
        callback_context = ToolContext(ctx)

        # 1. Initial Story Generation
        logger.info(f"[{self.name}] Running StoryGenerator ({self.story_generator.name})...")
        await my_before_tool_callback(tool=self.story_generator, args={"topic": ctx.session.state.get("topic")}, callback_context=callback_context)
        async for event in self.story_generator.run_async(ctx):
            logger.debug(f"[{self.name}] Event from StoryGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        await my_after_tool_callback(tool=self.story_generator, args={"topic": ctx.session.state.get("topic")}, callback_context=callback_context, tool_response={"output": ctx.session.state.get("current_story")})

        if "current_story" not in ctx.session.state or not ctx.session.state["current_story"]:
             logger.error(f"[{self.name}] Failed to generate initial story. Aborting workflow.")
             return

        logger.info(f"[{self.name}] Story state after generator: '{ctx.session.state.get('current_story')[:50]}...'")

        # 2. Critic-Reviser Loop
        logger.info(f"[{self.name}] Running CriticReviserLoop ({self.loop_agent.name})...")
        await my_before_tool_callback(tool=self.loop_agent, args={}, callback_context=callback_context)
        async for event in self.loop_agent.run_async(ctx):
            logger.debug(f"[{self.name}] Event from CriticReviserLoop: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        await my_after_tool_callback(tool=self.loop_agent, args={}, callback_context=callback_context, tool_response={"output": ctx.session.state.get("current_story")})

        logger.info(f"[{self.name}] Story state after loop: '{ctx.session.state.get('current_story')[:50]}...'")

        # 3. Sequential Post-Processing (Grammar and Tone Check)
        logger.info(f"[{self.name}] Running PostProcessingSequential ({self.sequential_agent.name})...")
        await my_before_tool_callback(tool=self.sequential_agent, args={}, callback_context=callback_context)
        async for event in self.sequential_agent.run_async(ctx):
            logger.debug(f"[{self.name}] Event from PostProcessingSequential: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        await my_after_tool_callback(tool=self.sequential_agent, args={}, callback_context=callback_context, tool_response={"output": ctx.session.state.get("tone_check_result")})

        # 4. Tone-Based Conditional Logic
        tone_check_result = ctx.session.state.get("tone_check_result")
        logger.info(f"[{self.name}] Tone check result: {tone_check_result}")

        if tone_check_result == "negative":
            logger.info(f"[{self.name}] Tone is negative. Regenerating story with {self.story_generator.name}...")
            await my_before_tool_callback(tool=self.story_generator, args={"topic": ctx.session.state.get("topic")}, callback_context=callback_context)
            async for event in self.story_generator.run_async(ctx): # StoryGenerator is treated as a tool here by StoryFlowAgent
                logger.debug(f"[{self.name}] Event from StoryGenerator (Regen): {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
            await my_after_tool_callback(tool=self.story_generator, args={"topic": ctx.session.state.get("topic")}, callback_context=callback_context, tool_response={"output": ctx.session.state.get("current_story")})
        else:
            logger.info(f"[{self.name}] Tone is not negative. Keeping current story.")

        logger.info(f"[{self.name}] Workflow finished.")
        # StoryFlowAgent's final response is implicitly the last event yielded from its sub-agents,
        # or it could explicitly yield a final Event.ADK.InvocationResponse here if needed.

# --- Define the individual LLM agents with all callbacks ---
common_agent_callbacks = {
    "before_agent_callback": my_before_agent_callback,
    "after_agent_callback": my_after_agent_callback,
}
common_llm_callbacks = {
    "before_model_callback": my_before_model_callback,
    "after_model_callback": my_after_model_callback,
}
# Tool callbacks are usually for agents that *use* tools, or orchestrators that treat sub-agents as tools.
# LlmAgents don't directly have before/after_tool_callbacks for their own execution,
# but if they *use* ADK Tools (like FunctionTool), those tools can have these callbacks,
# or the LlmAgent itself can specify `before_tool_callback` for when *it* calls a tool.

story_generator = LlmAgent(
    name="StoryGenerator",
    model=GEMINI_2_FLASH,
    instruction="You are a story writer. Write a short story (around 100 words) about a cat, based on the topic provided in session state with key 'topic'",
    output_key="current_story",
    **common_agent_callbacks,
    **common_llm_callbacks,
)

critic = LlmAgent(
    name="Critic",
    model=GEMINI_2_FLASH,
    instruction="You are a story critic. Review the story provided in session state with key 'current_story'. Provide 1-2 sentences of constructive criticism on how to improve it. Focus on plot or character.",
    output_key="criticism",
    **common_agent_callbacks,
    **common_llm_callbacks,
)

reviser = LlmAgent(
    name="Reviser",
    model=GEMINI_2_FLASH,
    instruction="You are a story reviser. Revise the story provided in session state with key 'current_story', based on the criticism in session state with key 'criticism'. Output only the revised story.",
    output_key="current_story",
    **common_agent_callbacks,
    **common_llm_callbacks,
)

grammar_check = LlmAgent(
    name="GrammarCheck",
    model=GEMINI_2_FLASH,
    instruction="You are a grammar checker. Check the grammar of the story provided in session state with key 'current_story'. Output only the suggested corrections as a list, or output 'Grammar is good!' if there are no errors.",
    output_key="grammar_suggestions",
    **common_agent_callbacks,
    **common_llm_callbacks,
)

tone_check = LlmAgent(
    name="ToneCheck",
    model=GEMINI_2_FLASH,
    instruction="You are a tone analyzer. Analyze the tone of the story provided in session state with key 'current_story'. Output only one word: 'positive' if the tone is generally positive, 'negative' if the tone is generally negative, or 'neutral' otherwise.",
    output_key="tone_check_result",
    **common_agent_callbacks,
    **common_llm_callbacks,
)

# --- Create the custom agent instance with its own callbacks ---
story_flow_agent = StoryFlowAgent(
    name="StoryFlowAgentOrchestrator",
    story_generator=story_generator,
    critic=critic,
    reviser=reviser,
    grammar_check=grammar_check,
    tone_check=tone_check,
    before_agent_callback=my_before_agent_callback, # For StoryFlowAgentOrchestrator itself
    after_agent_callback=my_after_agent_callback,
)

async def main():
    session_service = InMemorySessionService()
    initial_topic = "a brave kitten exploring a haunted house"
    initial_state = {"topic": initial_topic}
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )
    logger.info(f"Initial session state: {session.state}")

    runner = Runner(
        agent=story_flow_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    user_input_topic = "a lonely robot finding a friend in a junkyard"
    current_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    if not current_session:
        logger.error("Session not found!")
        return

    current_session.state["topic"] = user_input_topic
    logger.info(f"Updated session state topic to: {user_input_topic}")
    
    # Create a user message for the runner
    user_message_content = genai_types.Content(role='user', parts=[genai_types.Part(text=f"Let's write a story about: {user_input_topic}")])

    logger.info(f"\n--- Running Agent Workflow for topic: '{user_input_topic}' ---")
    events_iterable = runner.run(
        user_id=USER_ID, session_id=SESSION_ID, new_message=user_message_content
    )
    
    final_response_text = "No final response captured from agent events."
    if events_iterable:
        for event in events_iterable:
            if event and event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                logger.info(f"Captured Final Response from [{event.author}]: {final_response_text}")
            elif event and event.is_error():
                logger.error(f"Error event during run: {event.error_details}")
            # You can log other event types too if needed
            # logger.debug(f"Event received: {event.model_dump_json(indent=2, exclude_none=True)}")
    else:
        logger.warning("Runner.run did not return an iterable of events.")
    
    print("\n--- Agent Interaction Result ---")
    print("Agent Final Response Text: ", final_response_text) # This might not be meaningful depending on StoryFlowAgent's last action
    
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    if final_session:
        print("Final Session State:")
        import json
        print(json.dumps(final_session.state, indent=2))
    else:
        print("Could not retrieve final session state.")
    print("-------------------------------\n")

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY environment variable is set if not using Vertex AI auth
    # or Application Default Credentials (ADC) are configured.
    # import os
    # if "GOOGLE_API_KEY" not in os.environ:
    #     print("ERROR: GOOGLE_API_KEY environment variable not set.")
    #     print("Please set it before running the script, e.g., os.environ['GOOGLE_API_KEY'] = 'YOUR_KEY'")
    # else:
    #     asyncio.run(main())

    # For testing without needing API key for every run if structure is the focus:
    # You might need to mock LLM calls or expect errors if API key is not valid.
    # For now, let's assume it's set or using ADC.
    asyncio.run(main())