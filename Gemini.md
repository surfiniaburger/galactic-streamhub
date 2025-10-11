Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackVertex AI Agent Engine Memory Bank overview

bookmark_border
Release Notes
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

Vertex AI Agent Engine Memory Bank lets you dynamically generate long-term memories based on users' conversations with your agent. Long-term memories are personalized information that can be accessed across multiple sessions for a particular user. The agent can use the memories to personalize responses to the user and create cross-session continuity.

Overview
Memory Bank helps you manage memories, so that you can personalize how your agent interacts with users and manage the context window. For each scope, Memory Bank maintains an isolated collection of memories. Each memory is an independent, self-contained piece of information that can be used to expand the context available to your agent. For example:



{
  "name": "projects/.../locations/.../reasoningEngines/.../memories/...",
  "scope": {
    "agent_name": "My agent"
  },
  "fact": "I use Memory Bank to manage my memories."
}
Memory Bank includes the following features:

Memory generation: Create, refine, and manage memories using a large language model (LLM).

Memory Extraction: Extract only the most meaningful information from source data to persist as memories.

Memory Consolidation: Consolidate newly extracted information with existing memories, allowing memories to evolve as new information is ingested. You can also consolidate pre-extracted memories (like information that your agent or a human-in-the-loop considers meaningful) with existing memories.

Asynchronous Generation: Memories can be generated in the background, so that your agent doesn't have to wait for memory generation to complete.

Customizable Extraction: Configure what information Memory Bank considers meaningful by providing specific topics and few-shot examples.

Multimodal Understanding: Process multimodal information to generate and persist textual insights.

Managed Storage and Retrieval: Benefit from a fully managed, persistent, and accessible memory store.

Data isolation across identities: Memory consolidation and retrieval is isolated to a specific identity.

Persistent and Accessible Storage: Store memories that can be accessed from multiple environments, including Vertex AI Agent Engine Runtime, your local environment, or other deployment options.

Similarity Search: Retrieve memories using similarity search that is scoped to a specific identity.

Automatic Expiration: Set a time to live (TTL) on memories to ensure stale information is automatically deleted. Configure your Memory Bank instance so that a TTL is automatically applied to inserted or generated memories.

Agent integration: Connect Memory Bank to your agent, so that it can orchestrate calls to generate and retrieve memories.

Agent Development Kit (ADK) Integration: Orchestrate calls from your ADK-based agent using built-in ADK tools and the VertexAiMemoryBankService to read from and write to Memory Bank.

Other frameworks: Wrap your Memory Bank code in tools and callbacks to orchestrate memory generation and retrieval.

Use cases
You can use Memory Bank to transform stateless agent interactions into stateful, contextual experiences where the agent remembers, learns, and adapts over time. Memory Bank is ideal for applications that require:

Long-Term Personalization: Build experiences that are tailored to individual users. Memory Bank scopes memories to a specific identity, allowing an agent to remember a user's preferences, history, and key details across multiple sessions.

Example: A customer service agent that remembers key information from a user's past support tickets and product preferences without needing to ask again.
LLM-driven Knowledge Extraction: Use when you need to automatically identify and persist the most important information from conversations or multimodal content without manual intervention.

Example: A research agent that reads a series of technical papers and builds a consolidated memory of key findings, methodologies, and conclusions.
Dynamic & Evolving Context: Use Memory Bank when you need a knowledge source that isn't static. Memory Bank is designed to continuously integrate new information from your agent, refining and updating stored memories as new data becomes available. This ensures the context your agent relies on is always current and accurate. Whereas RAG has a static, external knowledge base, Memory Bank can evolve based on context provided by the agent.

Example usage
Vertex AI Agent Engine Memory Bank conceptual overview

You can use Memory Bank with Vertex AI Agent Engine Sessions to generate memories from stored sessions using the following process:

(Sessions) CreateSession: At the start of each conversation, create a new session. The conversation history used by the agent is scoped to this session. A session contains the chronological sequence of messages and actions (SessionEvents) for an interaction between a user and your agent. All sessions must have a user ID; the extracted memories (see GenerateMemories) for this session are mapped to this user.

(Sessions) AppendEvent: As the user interacts with the agent, events (such as user messages, agent responses, tool actions) are uploaded to Sessions. The events persist conversation history and create a record of the conversation that can be used to generate memories.

(Sessions) ListEvents: As the user interacts with the agent, the agent retrieves the conversation history.

(Memory Bank) Generate or create memories:

GenerateMemories: At a specified interval (such as the end of every session or the end of every turn), the agent can trigger memories to be generated using conversation history. Facts about the user are automatically extracted from the conversation history so that they're available for current or future sessions.

CreateMemory: Your agent can write memories directly to Memory Bank. For example, the agent can decide when a memory should be written and what information should be saved (memory-as-a-tool). Use CreateMemory when you want your agent to have more control over what facts are extracted.

(Memory Bank) RetrieveMemories: As the user interacts with your agent, the agent can retrieve memories saved about that user. You can either retrieve all memories (simple retrieval) or only the most relevant memories to the current conversation (similarity search retrieval). Then you can insert the retrieved memories into your prompt.

Quickstarts
Get started with Memory Bank using the following quickstarts:

Quickstart using REST API: Follow the REST API quickstart to make API calls directly to Vertex AI Agent Engine Sessions and Memory Bank.

Quickstart using Agent Development Kit (ADK): Follow the Agent Development Kit (ADK) quickstart if you want your ADK agent to orchestrate calls to Vertex AI Agent Engine Sessions and Memory Bank for you.

Security risks of prompt injection
In addition to the security responsibilities outlined in Vertex AI shared responsibility, consider the risk of prompt injection and memory poisoning that can affect your agent when using long-term memories. Memory poisoning occurs when false information is stored in Memory Bank. The agent may then operate on this false or malicious information in future sessions.

To mitigate the risk of memory poisoning, you can do the following:

Model Armor: Use Model Armor to inspect prompts being sent to Memory Bank or from your agent.

Adversarial testing: Proactively test your LLM application for prompt injection vulnerabilities by simulating attacks. This is typically known as "red teaming."

Sandbox execution: If the agent has the ability to execute or interact with external or critical systems, these actions should be performed in a sandboxed environment with strict access control and human review.

For more information, see Google's Approach for Secure AI Agents.

What's next
Set up Memory Bank.
Quickstart with Agent Development Kit.
Quickstart with Vertex AI Agent Engine SDK.
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded..


Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackSet up Memory Bank

bookmark_border
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

To use Vertex AI Agent Engine Memory Bank, you need an Vertex AI Agent Engine instance. This page demonstrates how to set up your environment and create an Vertex AI Agent Engine instance.

Getting started
Before you work with Vertex AI Agent Engine Memory Bank, you must set up your environment.

Set up your Google Cloud project
Note: A project is optional if you're using Vertex AI in express mode.
Every project can be identified in two ways: the project number or the project ID. The PROJECT_NUMBER is automatically created when you create the project, whereas the PROJECT_ID is created by you, or whoever created the project. To set up a project:

In the Google Cloud console, on the project selector page, select or create a Google Cloud project.

Roles required to select or create a project

Note: If you don't plan to keep the resources that you create in this procedure, create a project instead of selecting an existing project. After you finish these steps, you can delete the project, removing all resources associated with the project.
Go to project selector

Verify that billing is enabled for your Google Cloud project.

Enable the Vertex AI API.

Roles required to enable APIs

Enable the API

Note: To enable APIs, you need the serviceusage.services.enable permission. If you don't have this permission, ask your administrator to grant you the Service Usage Admin (roles/serviceusage.serviceUsageAdmin) role.
Get the required roles
To get the permissions that you need to use Vertex AI Agent Engine, ask your administrator to grant you the Vertex AI User (roles/aiplatform.user) IAM role on your project. For more information about granting roles, see Manage access to projects, folders, and organizations.

You might also be able to get the required permissions through custom roles or other predefined roles.

If you're making requests to Memory Bank from an agent deployed on Google Kubernetes Engine or Cloud Run, make sure that your service account has the necessary permissions. The Reasoning Engine Service Agent already has the necessary permissions to read and write memories, so outbound requests from Agent Engine Runtime should already have permission to access Memory Bank.

Install libraries
This section assumes that you have set up a Python development environment, or are using a runtime with a Python development environment (such as Colab).

Install the Vertex AI SDK:

  pip install google-cloud-aiplatform>=1.111.0
Authentication
Authentication instructions depend on whether you're using Vertex AI in express mode:

If you're not using Vertex AI in express mode, follow the instructions at Authenticate to Vertex AI.

If you're using Vertex AI in express mode, set up authentication by setting the API key in the environment:

  os.environ["API_KEY"] = "API_KEY"
Note: Vertex AI in express mode cannot be used for deploying an agent to Agent Engine Runtime. If you want to deploy your agent to Agent Engine Runtime, you must graduate from express mode.
Set up a Vertex AI SDK client
Run the following code to set up a Vertex AI SDK client:

import vertexai

client = vertexai.Client(
    project="PROJECT_ID",
    location="LOCATION",
)
where

PROJECT_ID is your project ID.
LOCATION is one of the supported regions for Memory Bank.
Create or update an Agent Engine instance
To get started with Memory Bank, you first need an Agent Engine instance. If you don't already have an instance, you can create it using the default configuration:

agent_engine = client.agent_engines.create()
If you want to customize the configuration of your new or existing Memory Bank instance's behavior, refer to Configure your Agent Engine instance for Memory Bank. For example, you can specify what information Memory Bank considers meaningful to persist.

Your Agent Engine instance supports Vertex AI Agent Engine Sessions and Memory Bank out-of-the-box. No agent is deployed when you create the instance. To use Vertex AI Agent Engine Runtime, you must provide the agent that should be deployed when creating or updating your Agent Engine instance.

Once you have an Agent Engine instance, you can use the name of the instance to read or write memories. For example:

# Generate memories using your Memory Bank instance.
client.agent_engines.memories.generate(
  # `name` should have the format `projects/.../locations/.../reasoningEngines/...`.
  name=agent_engine.api_resource.name,
  ...
)
Use with Vertex AI Agent Engine Runtime
Although Memory Bank can be used in any runtime, you can also use Memory Bank with Agent Engine Runtime to read and write memories from your deployed agent.

To deploy an agent with Memory Bank on Vertex AI Agent Engine Runtime, first set up your environment for Agent Engine runtime. Then, prepare your agent to be deployed on Agent Engine Runtime with memory integration. Your deployed agent should make calls to read and write memories as needed.

AdkApp
Custom agent
If you're using the Agent Engine Agent Development Kit template, the agent uses the VertexAiMemoryBankService by default when deployed to Agent Engine Runtime. This means that the ADK Memory tools read memories from Memory Bank.

from google.adk.agents import Agent
from vertexai.preview.reasoning_engines import AdkApp

# Develop an agent using the ADK template.
agent = Agent(...)

adk_app = AdkApp(
      agent=adk_agent,
      ...
)

# Deploy the agent to Agent Engine Runtime.
agent_engine = client.agent_engines.create(
      agent_engine=adk_app,
      config={
            "staging_bucket": "STAGING_BUCKET",
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]"],
            # Optional.
            **context_spec
      }
)

# Update an existing Agent Engine to add or modify the Runtime.
agent_engine = client.agent_engines.update(
      name=agent_engine.api_resource.name,
      agent=adk_app,
      config={
            "staging_bucket": "STAGING_BUCKET",
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]"],
            # Optional.
            **context_spec
      }
)
Replace the following:

STAGING_BUCKET: Your Cloud Storage bucket to use for staging your Agent Engine Runtime.
For more information about using Memory Bank with ADK, refer to the Quickstart with Agent Development Kit.

If you're using the default service agent for your agent on Vertex AI Agent Engine Runtime, your agent already has permission to read and write memories. If you're using a customer service account, you need to grant permissions to your service account to read and write memories. The required permissions depend on what operations your agent should be able to perform. If you only want your agent to retrieve and generate memories, aiplatform.memories.generate and aiplatform.memories.retrieve are sufficient.

Use in all other runtimes
If you want to use Memory Bank in a different environment, like Cloud Run or Colab, create an Agent Engine without providing an agent. Creating a new Agent Engine without a Runtime should only take a few seconds. If you don't provide a configuration, Memory Bank is created with the default settings for managing memory generation and retrieval:

agent_engine = client.agent_engines.create()
If you want to configure behavior, provide a Memory Bank configuration:

Create
Update
agent_engine = client.agent_engines.create(
  config={
    "context_spec": {
      "memory_bank_config": ...
    }
  }
)
You can use Memory Bank in any environment that has permission to read and write memories. For example, to use Memory Bank with Cloud Run, grant permissions to the Cloud Run service identity to read and write memories. The required permissions depend on what operations your agent should be able to perform. If you only want your agent to retrieve and generate memories, aiplatform.memories.generate and aiplatform.memories.retrieve are sufficient.

Configure your Agent Engine instance for Memory Bank
Optionally, you can configure your Memory Bank to customize how memories are generated and managed. If the configuration is not provided, then Memory Bank uses the default settings for each type of configuration.

The Memory Bank configuration is set when creating or updating your Agent Engine instance:

client.agent_engines.create(
      ...,
      config={
            "context_spec": {
                  "memory_bank_config": memory_bank_config
            }
      }
)

# Alternatively, update an existing Agent Engine's Memory Bank config.
agent_engine = client.agent_engines.update(
      name=agent_engine.api_resource.name,
      config={
          "context_spec": {
                "memory_bank_config": memory_bank_config
          }
      }
)
You can configure the following settings for your instance:

Customization configuration: Configures how memories should be extracted from source data.
Similarity search configuration: Configures which embedding model is used for similarity search. Defaults to text-embedding-005.
Generation configuration: Configures which LLM is used for memory generation. Defaults to gemini-2.5-flash.
TTL configuration: Configures how TTL is automatically set for created or updated memories. Defaults to no TTL.
Customization configuration
If you want to customize how memories are extracted from your source data, you can configure the memory extraction behavior when setting up your Agent Engine instance. There are two levers that you can use for customization:

Configuring memory topics: Define the type of information that Memory Bank should consider meaningful to persist. Only information that fits one of these memory topics will be persisted by Memory Bank.
Providing few-shot examples: Demonstrate expected behavior for memory extraction to Memory Bank.
You can think of customizing your Memory Bank's extraction behavior in two steps: Telling and Showing. Memory Topics tell Memory Bank what information to persist. Few-shots show Memory Bank what kind of information should result in a specific memory, helping it learn the patterns, nuance, and phrasing that you expect it to understand.

You can optionally configure different behavior for different scope-levels. For example, the topics that are meaningful for session-level memories may not be meaningful for user-level memories (across multiple sessions). To configure behavior for a certain subset of memories, set the scope keys of the customization configuration. Only GenerateMemories requests that include those scope keys will use that configuration. You can also configure default behavior (applying to all sets of scope keys) by omitting the scope_key field. This configuration will apply to all requests that don't have a configuration that exactly match the scope keys for another customization configuration.

For example, the user_level_config would only apply to GenerateMemories requests that exactly use the scope key user_id (i.e. scope={"user_id": "123"} with no additional keys). default_config would apply to other requests:

Dictionary
Class-based

user_level_config = {
  "scope_keys": ["user_id"],
  "memory_topics": [...],
  "generate_memories_examples": [...]
}

default_config = {
  "memory_topics": [...],
  "generate_memories_examples": [...]
}

memory_bank_config = {
  "customization_configs": [
    user_level_config,
    default_config
  ]
}
Configuring memory topics
"Memory topics" identify what information Memory Bank considers to be meaningful and should thus be persisted as generated memories. Memory Bank supports two types of memory topics:

Managed topics: Label and instructions are defined by Memory Bank. You only need to provide the name of the managed topic. For example,

Dictionary
Class-based
memory_topic = {
  "managed_memory_topic": {
    "managed_topic_enum": "USER_PERSONAL_INFO"
  }
}
The following managed topics are supported by Memory Bank:

Personal information (USER_PERSONAL_INFO): Significant personal information about the user, like names, relationships, hobbies, and important dates. For example, "I work at Google" or "My wedding anniversary is on December 31".
User preferences (USER_PREFERENCES): Stated or implied likes, dislikes, preferred styles, or patterns. For example, "I prefer the middle seat."
Key conversation events and task outcomes (KEY_CONVERSATION_DETAILS): Important milestones or conclusions within the dialogue. For example, "I booked plane tickets for a round trip between JFK and SFO. I leave on June 1, 2025 and return on June 7, 2025."
Explicit remember / forget instructions (EXPLICIT_INSTRUCTIONS): Information that the user explicitly asks the agent to remember or forget. For example, if the user says "Remember that I primarily use Python," Memory Bank generates a memory such as "I primarily use Python."
Custom topics: Label and instructions are defined by you when setting up your Memory Bank instance. They will be used in the prompt for Memory Bank's extraction step. For example,

Dictionary
Class-based
memory_topic = {
  "custom_memory_topic": {
    "label": "business_feedback",
    "description": """Specific user feedback about their experience at
the coffee shop. This includes opinions on drinks, food, pastries, ambiance,
staff friendliness, service speed, cleanliness, and any suggestions for
improvement."""
  }
}
When using custom topics, it's recommended to also provide few-shot examples demonstrating how memories should be extracted from your conversation.

With customization, you can use any combination of memory topics. For example, you can use a subset of the available managed memory topics:

Dictionary
Class-based
customization_config = {
  "memory_topics": [
    { "managed_memory_topic": { "managed_topic_enum": "USER_PERSONAL_INFO" } },
    { "managed_memory_topic": { "managed_topic_enum": "USER_PREFERENCES" } }
  ]
}
You can also use a combination of managed and custom topics (or only use custom topics):

Dictionary
Class-based
customization_config = {
  "memory_topics": [
    { "managed_memory_topic": { "managed_topic_enum": "USER_PERSONAL_INFO" } },
    {
      "custom_memory_topic": {
        "label": "business_feedback",
        "description": """Specific user feedback about their experience at
the coffee shop. This includes opinions on drinks, food, pastries, ambiance,
staff friendliness, service speed, cleanliness, and any suggestions for
improvement."""
        }
    }
  ]
}
Few-shot examples
Few-shot examples allow you to demonstrate expected memory extraction behavior to Memory Bank. For example, you can provide a sample input conversation and the memories that are expected to be extracted from that conversation.

We recommend always using few-shots with custom topics so that Memory Bank can learn the intended behavior. Few-shots are optional when using managed topics since Memory Bank defines examples for each topic. Demonstrate conversations that are not expected to result in memories by providing an empty generated_memories list.

For example, you can provide few-shot examples that demonstrate how to extract feedback about your business from customer messages:

Dictionary
Class-based
example = {
    "conversationSource": {
      "events": [
        {
          "content": {
            "role": "model",
            "parts": [{ "text": "Welcome back to The Daily Grind! We'd love to hear your feedback on your visit." }] }
        },
        {
          "content": {
            "role": "user",
            "parts": [{ "text": "Hey. The drip coffee was a bit lukewarm today, which was a bummer. Also, the music was way too loud, I could barely hear my friend." }] }
        }
      ]
    },
    "generatedMemories": [
      {
        "fact": "The user reported that the drip coffee was lukewarm."
      },
      {
        "fact": "The user felt the music in the shop was too loud."
      }
    ]
}
You can also provide examples of conversations that shouldn't result in any generated memories by providing an empty list for the expected output (generated_memories):

Dictionary
Class-based
example = {
    "conversationSource": {
        "events": [
          {
              "content": {
                  "role": "model",
                  "parts": [{ "text": "Good morning! What can I get for you at The Daily Grind?" }] }
          },
          {
              "content": {
                  "role": "user",
                  "parts": [{ "text": "Thanks for the coffee." }] }
          }
        ]
    },
    "generatedMemories": []
}
Similarity search configuration
The similarity search configuration controls which embedding model is used by your instance for similarity search. Similarity search is used for identifying which memories should be candidates for consolidation and for similarity search-based memory retrieval. If this configuration is not provided, Memory Bank uses text-embedding-005 as the default model.

If you expect user conversations to be in non-English languages, use a model that supports multiple languages, such as gemini-embedding-001 or text-multilingual-embedding-002, to improve retrieval quality.

Dictionary
Class-based
memory_bank_config = {
    "similarity_search_config": {
        "embedding_model": "EMBEDDING_MODEL",
    }
}
Replace the following:

EMBEDDING_MODEL: The Google text embedding model to use for similarity search, in the format projects/{project}/locations/{location}/publishers/google/models/{model}.
Generation configuration
The generation configuration controls which LLM is used for generating memories, including extracting memories and consolidating new memories with existing memories.

Memory Bank uses gemini-2.5-flash as the default model. For regions that don't have regional Gemini availability, the global endpoint is used.

Dictionary
Class-based
memory_bank_config = {
      "generation_config": {
            "model": "LLM_MODEL",
      }
}
Replace the following:

LLM_MODEL: The Google LLM model to use for extracting and consolidating memories, in the format projects/{project}/locations/{location}/publishers/google/models/{model}.
Time to live (TTL) configuration
The TTL configuration controls how Memory Bank should dynamically set memories' expiration time. After their expiration time elapses, memories won't be available for retrieval and will be deleted.

If the configuration is not provided, expiration time won't be dynamically set for created or updated memories, so memories won't expire unless their expiration time is manually set.

There are two options for the TTL configuration:

Default TTL: The TTL will be applied to all operations that create or update a memory, including UpdateMemory, CreateMemory, and GenerateMemories.

Dictionary
Class-based
memory_bank_config = {
    "ttl_config": {
        "default_ttl": f"TTLs"
    }
}
Replace the following:

TTL: The duration in seconds for the TTL. For updated memories, the newly calculated expiration time (now + TTL) will overwrite the Memory's previous expiration time.
Granular (per-operation) TTL: The TTL is calculated based on which operation created or updated the Memory. If not set for a given operation, then the operation won't update the Memory's expiration time.

Dictionary
Class-based
memory_bank_config = {
    "ttl_config": {
        "granular_ttl": {
            "create_ttl": f"CREATE_TTLs",
            "generate_created_ttl": f"GENERATE_CREATED_TTLs",
            "generate_updated_ttl": f"GENERATE_UPDATED_TTLs"
        }
    }
}
Replace the following:

CREATE_TTL: The duration in seconds for the TTL for memories created using CreateMemory.
GENERATE_CREATED_TTL: The duration in seconds for the TTL for memories created using GeneratedMemories.
GENERATE_UPDATED_TTL: The duration in seconds for the TTL for memories updated using GeneratedMemories. The newly calculated expiration time (now + TTL) will overwrite the Memory's previous expiration time.
What's next
Quickstart with Vertex AI Agent Engine SDK.
Quickstart with Agent Development Kit.
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded.

Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackQuickstart with Vertex AI Agent Engine SDK

bookmark_border
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

This tutorial demonstrates how to make API calls directly to Vertex AI Agent Engine Sessions and Memory Bank using the Vertex AI Agent Engine SDK. Use the Vertex AI Agent Engine SDK if you don't want an agent framework to orchestrate calls for you, or you want to integrate Sessions and Memory Bank with agent frameworks other than Agent Development Kit (ADK).

For the quickstart using ADK, see Quickstart with Agent Development Kit.

This tutorial uses the following steps:

Create memories using the following options:
Generate memories using Vertex AI Agent Engine Memory Bank: Write sessions and events to Vertex AI Agent Engine Sessions as sources for Vertex AI Agent Engine Memory Bank to generate memories.
Upload memories directly: Write your own memories or have your agent build memories if you want full control over what information is persisted.
Retrieve memories.
Clean up.
To see an example of using Memory Bank with the Agent Engine SDK, run the "Get started with Memory Bank" notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench | View on GitHub

To see an example of using Memory Bank with LangGraph, run the "Get started with Memory Bank - LangGraph" notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench | View on GitHub

Before you begin
To complete the steps demonstrated in this tutorial, you must first follow the steps in Set up for Memory Bank.

Generate memories with Vertex AI Agent Engine Sessions
After setting up Vertex AI Agent Engine Sessions and Memory Bank, you can create sessions and append events to them. Memories are generated as facts from the user's conversation with the agent so that they're available for future user interactions. For more information, see Generate memories and Fetch memories.

Note: You can also provide the conversation history directly in the GenerateMemories payload using direct contents source.
Create a session with an opaque user ID. Any memories generated from this session are automatically keyed by the scope {"user_id": "USER_ID"} unless you explicitly provide a scope when generating memories.

import vertexai

client = vertexai.Client(
  project="PROJECT_ID",
  location="LOCATION"
)

session = client.agent_engines.sessions.create(
  name=AGENT_ENGINE_NAME,
  user_id="USER_ID"
)
Replace the following:

PROJECT_ID: Your project ID.

LOCATION: Your region. See the supported regions for Memory Bank.

AGENT_ENGINE_NAME: The name of the Vertex AI Agent Engine instance that you created or an existing Vertex AI Agent Engine instance. The name should be in the following format: projects/{your project}/locations/{your location}/reasoningEngine/{your reasoning engine}.

USER_ID: An identifier for your user. Any memories generated from this session are automatically keyed by the scope {"user_id": "USER_ID"} unless you explicitly provide a scope when generating memories.

Iteratively upload events to your session. Events can include any interactions between your user, agent, and tools. The ordered list of events represents your session's conversation history. This conversation history is used as the source material for generating memories for that particular user.

import datetime

client.agent_engines.sessions.events.append(
  name=session.response.name,
  author="user",  # Required by Sessions.
  invocation_id="1",  # Required by Sessions.
  timestamp=datetime.datetime.now(tz=datetime.timezone.utc),  # Required by Sessions.
  config={
    "content": {
      "role": "user",
      "parts": [{"text": "hello"}]
    }
  }
)
To generate memories from your conversation history, trigger a memory generation request for the session:

client.agent_engines.memories.generate(
  name=agent_engine.api_resource.name,
  vertex_session_source={
    # `session` should have the format "projects/.../locations/.../reasoningEngines/.../sessions/...".
    "session": session.response.name
  },
  # Optional when using Agent Engine Sessions. Defaults to {"user_id": session.user_id}.
  scope=SCOPE
)
Replace the following:

(Optional) SCOPE: A dictionary representing the scope of the generated memories, with a maximum of 5 key value pairs and no * characters. For example, {"session_id": "MY_SESSION"}. Only memories with the same scope are considered for consolidation. If not provided, {"user_id": session.user_id} is used.
Upload memories
As an alternative to generating memories using raw dialogue, you can upload memories or have your agents add them directly using GenerateMemories with pre-extracted facts. Rather than Memory Bank extracting information from your content, you provide the facts that should be stored about your user directly. We recommended that you write facts about users in first person (for example, I am a software engineer).

client.agent_engines.memories.generate(
    name=agent_engine.api_resource.name,
    direct_memories_source={"direct_memories": [{"fact": "FACT"}]},
    scope=SCOPE
)
Replace the following:

FACT: The pre-extracted fact that should be consolidated with existing memories. You can provide up to 5 pre-extracted facts in a list like the following:

{"direct_memories": [{"fact": "fact 1"}, {"fact": "fact 2"}]}
SCOPE: A dictionary, representing the scope of the generated memories. For example, {"session_id": "MY_SESSION"}. Only memories with the same scope are considered for consolidation.

Alternatively, you can use CreateMemory to upload memories without using Memory Bank for either memory extraction or consolidation.

Caution: Memories uploaded using CreateMemory won't be consolidated with existing memories, so you can end up with duplicated memories for the same scope. Created memories are available for similarity search and can be consolidated for future requests when generating memories.
memory = client.agent_engines.memories.create(
    name=agent_engine.api_resource.name,
    fact="This is a fact.",
    scope={"user_id": "123"}
)

"""
Returns an AgentEngineMemoryOperation containing the created Memory like:

AgentEngineMemoryOperation(
  done=True,
  metadata={
    "@type': 'type.googleapis.com/google.cloud.aiplatform.v1beta1.CreateMemoryOperationMetadata",
    "genericMetadata": {
      "createTime": '2025-06-26T01:15:29.027360Z',
      "updateTime": '2025-06-26T01:15:29.027360Z'
    }
  },
  name="projects/.../locations/us-central1/reasoningEngines/.../memories/.../operations/...",
  response=Memory(
    create_time=datetime.datetime(2025, 6, 26, 1, 15, 29, 27360, tzinfo=TzInfo(UTC)),
    fact="This is a fact.",
    name="projects/.../locations/us-central1/reasoningEngines/.../memories/...",
    scope={
      "user_id": "123"
    },
    update_time=datetime.datetime(2025, 6, 26, 1, 15, 29, 27360, tzinfo=TzInfo(UTC))
  )
)
"""
Retrieve and use memories
You can retrieve memories for your user and include them in your system instructions to give the LLM access to your personalized context.

For more information about retrieving memories using a scope-based method, see Fetch memories.

# Retrieve all memories for User ID 123.
retrieved_memories = list(
    client.agent_engines.memories.retrieve(
        name=agent_engine.api_resource.name,
        scope={"user_id": "123"}
    )
)
You can use jinja to convert your structured memories into a prompt:


from jinja2 import Template

template = Template("""
<MEMORIES>
Here is some information about the user:
{% for retrieved_memory in data %}* {{ retrieved_memory.memory.fact }}
{% endfor %}</MEMORIES>
""")

prompt = template.render(data=retrieved_memories)

"""
Output:

<MEMORIES>
Here is some information about the user:
* This is a fact
</MEMORIES>
"""

Delete memories
You can delete a specific memory using its resource name:

client.agent_engines.memories.delete(name=MEMORY_NAME)
Replace the following:

MEMORY_NAME: The name of the Memory to delete. The name should be in the following format: projects/{your project}/locations/{your location}/reasoningEngine/{your reasoning engine}/memories/{your memory}. You can find the Memory name by fetching memories.
Clean up
To clean up all resources used in this project, you can delete the Google Cloud project you used for the quickstart.

Otherwise, you can delete the individual resources you created in this tutorial, as follows:

Use the following code sample to delete the Vertex AI Agent Engine instance, which also deletes any sessions or memories associated with the Vertex AI Agent Engine instance.

agent_engine.delete(force=True)
Delete any locally created files.

What's next
Quickstart with Agent Development Kit.
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded..


Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackQuickstart with Agent Development Kit

bookmark_border
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

After you configure your Agent Development Kit (ADK) agent to use Memory Bank, your agent orchestrates calls to Memory Bank to manage long-term memories for you.

This tutorial demonstrates how you can use Memory Bank with the ADK to manage long-term memories:

Create your local ADK agent and runner.

Interact with your agent to dynamically generate long-term memories that are accessible across sessions.

Clean up.

To make calls directly to Memory Bank without ADK orchestration, see Quickstart with Agent Engine SDK. Using the Agent Engine SDK is helpful for understanding how Memory Bank generates memories or for inspecting the contents of Memory Bank.

To see an example of using Memory Bank with ADK, run the "Get started with Memory Bank on ADK" notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench | View on GitHub

Before you begin
To complete the steps demonstrated in this tutorial, you must first follow the steps in Set up for Memory Bank.

Set environment variables
To use the ADK, set your environment variables:

import os

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] = "PROJECT_ID"
os.environ["GOOGLE_CLOUD_LOCATION"] = "LOCATION"
Replace the following:

PROJECT_ID: Your project ID.
LOCATION: Your region. See the supported regions for Memory Bank.
Create your ADK agent
When developing your ADK agent, include a Memory tool that controls when the agent retrieves memories and how memories are included in the prompt. The example agent uses the PreloadMemoryTool, which always retrieves memories at the start of each turn and includes the memories in the system instruction:

from google import adk

agent = adk.Agent(
    model="gemini-2.0-flash",
    name='stateful_agent',
    instruction="""You are a Vehicle Voice Agent, designed to assist users with information and in-vehicle actions.

1.  **Direct Action:** If a user requests a specific vehicle function (e.g., "turn on the AC"), execute it immediately using the corresponding tool. You don't have the outcome of the actual tool execution, so provide a hypothetical tool execution outcome.
2.  **Information Retrieval:** Respond concisely to general information requests with your own knowledge (e.g., restaurant recommendation).
3.  **Clarity:** When necessary, try to seek clarification to better understand the user's needs and preference before taking an action.
4.  **Brevity:** Limit responses to under 30 words.
""",
    tools=[adk.tools.preload_memory_tool.PreloadMemoryTool()]
)
Create a VertexAiMemoryBankService memory service, which the ADK runner uses for retrieving memories. This is optional if you're using the Agent Engine ADK template instead of defining your own ADK runtime.

from google.adk.memory import VertexAiMemoryBankService

agent_engine_id = agent_engine.api_resource.name.split("/")[-1]

memory_service = VertexAiMemoryBankService(
    project="PROJECT_ID",
    location="LOCATION",
    agent_engine_id=agent_engine_id
)
VertexAiMemoryBankService is an ADK wrapper around Memory Bank that is defined by ADK's BaseMemoryServiceand uses a different interface than the Agent Engine SDK. You can use the Agent Engine SDK to make direct API calls to Memory Bank. The VertexAiMemoryBankService interface includes:

memory_service.add_session_to_memory which triggers a GenerateMemories request to Memory Bank using the provided adk.Session as the source content. Calls to this method are not orchestrated by the ADK runner. If you want to automate memory generation with ADK, you need to define your own callback functions.

memory_service.search_memory which triggers a RetrieveMemories request to Memory Bank to fetch relevant memories for the current user_id and app_name. Calls to this method are orchestrated by the ADK runner when you provide a Memory tool to your agent.

Create an ADK Runtime, which orchestrates the execution of your agents, tools, and callbacks. The ADK Runner setup depends on which deployment environment you're using:

adk.Runner
Agent Engine ADK template
adk.Runner is generally used in a local environment, like Colab. Most deployment options, like Agent Engine Runtime, offer their own runtime for ADK.

from google.adk.sessions import VertexAiSessionService
from google.genai import types

# You can use any ADK session service.
session_service = VertexAiSessionService(
    project="PROJECT_ID",
    location="LOCATION",
    agent_engine_id=agent_engine_id
)

app_name="APP_NAME"

runner = adk.Runner(
    agent=agent,
    app_name=app_name,
    session_service=session_service,
    memory_service=memory_service
)

def call_agent(query, session, user_id):
  content = types.Content(role='user', parts=[types.Part(text=query)])
  events = runner.run(user_id=user_id, session_id=session, new_message=content)

  for event in events:
      if event.is_final_response():
          final_response = event.content.parts[0].text
          print("Agent Response: ", final_response)
Replace the following:

APP_NAME: The name of your ADK app.
Interact with your agent
After defining your agent and setting up Memory Bank, you can interact with your agent.

Create your first session. Since there are no available memories during the first session with a user, the agent doesn't know any user preferences, such as their preferred temperature:

Note: There may be some variance in the agent and Memory Bank's responses depending on what models are used.
adk.Runner
Agent Engine ADK template
When using adk.Runner, you can call your ADK memory and session services directly.

session = await session_service.create_session(
    app_name="APP_NAME",
    user_id="USER_ID"
)

call_agent(
    "Can you update the temperature to my preferred temperature?",
    session.id,
    "USER_ID"
)

# Agent response: "What is your preferred temperature?"
call_agent("I like it at 71 degrees", session.id, "USER_ID")
# Agent Response:  Setting the temperature to 71 degrees Fahrenheit.
# Temperature successfully changed.
Replace the following:

APP_NAME: App name for your runner.
USER_ID: An identifier for your user. Memories generated from this session are keyed by this opaque identifier. The generated memories' scope is stored as {"user_id": "USER_ID"}.
Generate memories for your current session. If Memory Bank extracts memories from the conversation, they are stored under the scope {"user_id": USER_ID, "app_name": APP_NAME}.

adk.Runner
Agent Engine ADK template
session = await session_service.get_session(
    app_name=app_name,
    user_id="USER_ID",
    session_id=session.id
)
memory_service.add_session_to_memory(session)
Create your second session. If you used the PreloadMemoryTool, the agent retrieves memories at the beginning of each turn to access preferences the user previously communicated to the agent.

adk.Runner
Agent Engine ADK template
session = await session_service.create_session(
    app_name=app_name,
    user_id="USER_ID"
)

call_agent("Fix the temperature!", session.id, "USER_ID")
# Agent Response:  Setting temperature to 71 degrees.  Is that correct?
You can also use memory_service.search_memory to retrieve memories directly:

await memory_service.search_memory(
    app_name="APP_NAME",
    user_id="USER_ID",
    query="Fix the temperature!",
)
Clean up
To clean up all resources used in this project, you can delete the Google Cloud project you used for the quickstart.

Otherwise, you can delete the individual resources you created in this tutorial, as follows:

Use the following code sample to delete the Vertex AI Agent Engine instance, which also deletes any Sessions or Memories belonging to that Vertex AI Agent Engine.

agent_engine.delete(force=True)
Delete any locally created files.

What's next
Quickstart with Agent Engine SDK.
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded.

Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackGenerate memories

bookmark_border
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

Memory Bank lets you construct long-term memories from conversations between the user and your agent. This page describes how memory generation works, how you can customize how memories are extracted, and how to trigger memory generation.

To complete the steps demonstrated in this guide, you must first follow the steps in Set up for Memory Bank.

Understanding memory generation
Memory Bank extracts memories from source data and self-curates memories for a specific collection of memories (defined by a scope) by adding, updating, and removing memories over time.

When you trigger memory generation, Memory Bank performs the following operations:

Extraction: Extracts information about the user from their conversations with the agent. Only information that matches at least one of your instance's memory topics will be persisted.

Consolidation: Identifies if existing memories with the same scope should be deleted or updated based on the extracted information. Memory Bank checks that new memories are not duplicative or contradictory before merging them with existing memories. If existing memories don't overlap with the new information, a new memory will be created.

Important: Not all user-agent interactions result in memories being created or updated. Memory Bank only persists information that is judged to be valuable for future interactions.
Caution: While Memory Bank has been instructed to exclude sensitive or personal data, the system is not infallible. Therefore, you should assume that some sensitive or personal information may still be stored and shouldn't rely solely on this feature to filter out such data.
Memory topics
"Memory topics" identify what information Memory Bank considers to be meaningful and should thus be persisted as generated memories. Memory Bank supports two types of memory topics:

Managed topics: Label and instructions are defined by Memory Bank. You only need to provide the name of the managed topic. For example:

Dictionary
Class-based
memory_topic = {
    "managed_memory_topic": {
        "managed_topic_enum": "USER_PERSONAL_INFO"
    }
}
Custom topics: Label and instructions are defined by you when setting up your Memory Bank instance. They will be used in the prompt for Memory Bank's extraction step. For example:

Dictionary
Class-based
memory_topic = {
    "custom_memory_topic": {
        "label": "business_feedback",
        "description": """Specific user feedback about their experience at
the coffee shop. This includes opinions on drinks, food, pastries, ambiance,
staff friendliness, service speed, cleanliness, and any suggestions for
improvement."""
    }
}
When using custom topics, it's recommended to also provide few-shot examples to demonstrate how memories should be extracted from your conversation.

By default, Memory Bank persists all of the following managed topics:

Personal information (USER_PERSONAL_INFO): Significant personal information about the user, like names, relationships, hobbies, and important dates. For example, "I work at Google" or "My wedding anniversary is on December 31".
User preferences (USER_PREFERENCES): Stated or implied likes, dislikes, preferred styles, or patterns. For example, "I prefer the middle seat."
Key conversation events and task outcomes (KEY_CONVERSATION_DETAILS): Important milestones or conclusions within the dialogue. For example, "I booked plane tickets for a round trip between JFK and SFO. I leave on June 1, 2025 and return on June 7, 2025."
Explicit remember / forget instructions (EXPLICIT_INSTRUCTIONS): Information that the user explicitly asks the agent to remember or forget. For example, if the user says "Remember that I primarily use Python," Memory Bank generates a memory such as "I primarily use Python."
This is equivalent to using the following set of managed memory topics:

Dictionary
Class-based
  memory_topics = [
      {"managed_memory_topic": {"managed_topic_enum": "USER_PERSONAL_INFO"}},
      {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
      {"managed_memory_topic": {"managed_topic_enum": "KEY_CONVERSATION_DETAILS"}},
      {"managed_memory_topic": {"managed_topic_enum": "EXPLICIT_INSTRUCTIONS"}},
  ]
If you want to customize what topics Memory Bank persists, set the memory topics in your customization configuration when setting up Memory Bank.

Triggering memory generation
You can trigger memory generation using GenerateMemories at the end of a session or at regular intervals within a session. Memory generation extracts key context from source conversations and combines it with existing memories for the same scope. For example, you can create session-level memories by using a scope such as {"user_id": "123", "session_id": "456"}. Memories with the same scope can be consolidated and retrieved together.

GenerateMemories is a long-running operation. Once the operation is done, the AgentEngineGenerateMemoriesOperation contains a list of generated memories, if any are generated:

AgentEngineGenerateMemoriesOperation(
  name="projects/.../locations/.../reasoningEngines/.../operations/...",
  done=True,
  response=GenerateMemoriesResponse(
    generatedMemories=[
      GenerateMemoriesResponseGeneratedMemory(
        memory=Memory(
          "name": "projects/.../locations/.../reasoningEngines/.../memories/..."
        ),
        action="CREATED",
      ),
      GenerateMemoriesResponseGeneratedMemory(
        memory=Memory(
          "name": "projects/.../locations/.../reasoningEngines/.../memories/..."
        ),
        action="UPDATED",
      ),
      GenerateMemoriesResponseGeneratedMemory(
        memory=Memory(
          "name": "projects/.../locations/.../reasoningEngines/.../memories/..."
        ),
        action="DELETED",
      ),
    ]
  )
)
Each generated memory includes the action that was performed on that memory:

CREATED: Indicates that a new memory was added, representing a novel concept that wasn't captured by existing memories.
UPDATED: Indicates that an existing memory was updated, which happens if the memory covered similar concepts as the newly extracted information. The memory's fact may be updated with new information or remain the same.
DELETED: Indicates that the existing memory was deleted, because its information was contradictory to new information extracted from the conversation.
Note: If Memory Bank doesn't find meaningful information in the user-agent conversation, no memories are generated. See the troubleshooting guide if you expected memories to be generated.
For CREATED or UPDATED memories, you can use GetMemories to retrieve the full content of the memory. Retrieving DELETED memories results in a 404 error.

Generating memories in the background
GenerateMemories is a long-running operation. By default, client.agent_engines.generate_memories is a blocking function that polls the operation until the operation completes. Executing memory generation as a blocking operation is helpful when you want to manually inspect generated memories or notify end users about what memories were generated.

However, for production agents, you generally want to run memory generation in the background as an asynchronous process. In most cases, the client doesn't need to use the output for the current run, so it's unnecessary to incur additional latency waiting for a response. If you want memory generation to execute in the background, set wait_for_completion to False:

client.agent_engines.memories.generate(
    ...,
    config={
        "wait_for_completion": False
    }
)
Data sources
There are multiple way to provide source data for memory generation:

Provide events directly in the payload.

Provide events using Vertex AI Agent Engine Sessions.

Provide pre-extracted facts to consolidate them with existing memories for the same scope.

When you provide events directly in the payload or use Vertex AI Agent Engine Sessions, information is extracted from the conversation and consolidated with existing memories. If you only want to extract information from these data sources, you can disable consolidation:

client.agent_engines.memories.generate(
    ...
    config={
        "disable_consolidation": True
    }
)
Using events in payload as the data source
Use direct_contents_source when you want to generate memories using events provided directly in the payload. Meaningful information is extracted from these events and consolidated with existing information for the same scope. This approach can be used if you're using a different session storage from Vertex AI Agent Engine Sessions.

Dictionary
Class-based
The events should include Content dictionaries.

events =  [
  {
    "content": {
      "role": "user",
      "parts": [
        {"text": "I work with LLM agents!"}
      ]
    }
  }
]

client.agent_engines.memories.generate(
    name=agent_engine.api_resource.name,
    direct_contents_source={
      "events": EVENTS
    },
    # For example, `scope={"user_id": "123"}`.
    scope=SCOPE,
    config={
        "wait_for_completion": True
    }
)
Replace the following:

SCOPE: A dictionary, representing the scope of the generated memories. For example, {"session_id": "MY_SESSION"}. Only memories with the same scope are considered for consolidation.
Using Vertex AI Agent Engine Sessions as the data source
With Agent Engine Sessions, Memory Bank uses session events as the source conversation for memory generation.

To scope the generated memories, Memory Bank extracts and uses the user ID from the session by default. For example, the memories' scope is stored as {"user_id": "123"} if the session's user_id is "123". You can also provide a scope directly, which overrides using the session's user_id as the scope.

Dictionary
Class-based
client.agent_engines.memories.generate(
  name=agent_engine.api_resource.name,
  vertex_session_source={
      # For example, projects/.../locations/.../reasoningEngines/.../sessions/...
      "session": "SESSION_NAME"
  },
  # Optional when using Agent Engine Sessions. Defaults to {"user_id": session.user_id}.
  scope=SCOPE,
  config={
      "wait_for_completion": True
  }
)
Replace the following:

SESSION_NAME: The fully-qualified session name.

(Optional) SCOPE: A dictionary, representing the scope of the generated memories. For example, {"session_id": "MY_SESSION"}. Only memories with the same scope are considered for consolidation. If not provided, {"user_id": session.user_id} is used.

Optionally, you can provide a time range indicating which events in the Session should be included. If not provided, all events in the Session are included.

Dictionary
Class-based
import datetime

client.agent_engines.memories.generate(
  name=agent_engine.api_resource.name,
  vertex_session_source={
      "session": "SESSION_NAME",
      # Extract memories from the last hour of events.
      "start_time": datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=24 * 60),
      "end_time": datetime.datetime.now(tz=datetime.timezone.utc)
  },
  scope=SCOPE
)
Consolidating pre-extracted memories
As an alternative to using Memory Bank's automatic extraction process, you can directly provide pre-extracted memories. Direct source memories will be consolidated with existing memories for the same scope. This can be useful for when you want your agent or a human-in-the-loop to be responsible for extracting memories, but you still want to take advantage of Memory Bank's consolidation to ensure there are no duplicate or contradictory memories.

client.agent_engines.memories.generate(
    name=agent_engine.api_resource.name,
    direct_memories_source={"direct_memories": [{"fact": "FACT"}]},
    scope=SCOPE
)
Replace the following:

FACT: The pre-extracted fact that should be consolidated with existing memories. You can provide up to 5 pre-extracted facts in a list like the following:

{"direct_memories": [{"fact": "fact 1"}, {"fact": "fact 2"}]}
SCOPE: A dictionary, representing the scope of the generated memories. For example, {"session_id": "MY_SESSION"}. Only memories with the same scope are considered for consolidation.

Using multimodal input
To see an example of generating memories with multimodal input, run the "Building a Multimodal Trip Planner with ADK on Memory Bank" notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench | View on GitHub

You can extract memories from multimodal input. However, memories are only extracted from text, inline files, and file data in the source content. All other content, including function calls and responses, are ignored when generating memories.

Memories can be extracted from images, video, and audio provided by the user. If the context provided by the multimodal input is judged by Memory Bank to be meaningful for future interactions, then a textual memory may be created including information extracted from the input. For example, if the user provides an image of a golden retriever with the text "This is my dog" then Memory Bank generates a memory such as "My dog is a golden retriever."

For example, you can provide an image and context for the image in the payload:

Dictionary
Class-based
with open(file_name, "rb") as f:
    inline_data = f.read()

events =  [
  {
    "content": {
      "role": "user",
      "parts": [
        {"text": "This is my dog"},
        {
          "inline_data": {
            "mime_type": "image/jpeg",
            "data": inline_data
          }
        },
        {
          "file_data": {
            "file_uri": "gs://cloud-samples-data/generative-ai/image/dog.jpg",
            "mime_type": "image/jpeg"
          }
        },
      ]
    }
  }
]
When using Vertex AI Agent Engine Sessions as the data source, the multimodal content is provided directly in the Session's events.

What's next
Fetch generated memories
Customize memory extraction behavior
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded..



Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackFetch memories

bookmark_border
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

This page describes how to fetch generated and uploaded memories from Memory Bank. For the entire workflow of configuring, generating, and using Memory Bank, see the Quickstart with REST API.

Note: If you configured an Agent Development Kit agent to use Memory Bank, the agent orchestrates calls to RetrieveMemories for you, so you don't need to make direct calls to retrieve memories.
You have the following options to fetch generated memories:

Get memory: Get the full content of a single memory.

List memories: List memories

Retrieve memories: Retrieve memories using scope-based memory retrieval. Retrieve memories using similarity search or all memories within the scope.

Before you begin
To complete the steps demonstrated in this page, you must first follow the steps in Set up for Memory Bank.

Get memory
Use GetMemories to get the full content of a single memory:

memory = client.agent_engines.memories.get(
    name="MEMORY_NAME")
Replace the following:

MEMORY_NAME: A fully-qualified memory name in the format "projects/.../locations/.../reasoningEngines/.../memories...".
List memories
Use ListMemories to fetch all memories in your Memory Bank.

pager = client.agent_engines.memories.list(name=agent_engine.api_resource.name)
for page in pager:
  print(page)
Fetch memories using scope-based retrieval
You can use RetrieveMemories to retrieve memories for a particular scope. Only memories that have the exact same scope (independent of order) as the retrieval request are returned. For example, you can retrieve all memories that are scoped to a particular user by using {"user_id": "123"}. If no memories are returned, Memory Bank doesn't have any memories for the provided scope.

A memory's scope is defined when the memory is generated or created and is immutable.

You can use RetrieveMemories to perform the following operations for a particular scope:

Retrieve memories using similarity search
Retrieve all memories
Retrieve memories using similarity search
For cases where you have many memories for a particular scope, you can use similarity search to retrieve only the most similar memories by providing similarity search parameters. Memory Bank only considers memories that have exactly the same scope as the request when performing similarity search. Similarity search compares the embedding vectors between memories' facts and the request's search query.

Returned memories are sorted from most similar (shortest Euclidean distance) to least similar (greatest Euclidean distance):

results = client.agent_engines.memories.retrieve(
    name=agent_engine.api_resource.name,
    scope=SCOPE,
    similarity_search_params={
        "search_query": "QUERY",
        # Optional. Defaults to 3.
        "top_k": 3
    }
)
# RetrieveMemories returns a pager. You can use `list` to retrieve all memories.
list(results)

"""
Returns:

[
    RetrieveMemoriesResponseRetrievedMemory(
      memory=Memory(
        name="projects/.../locations/.../reasoningEngines/.../memories/...",
        ...
        fact="This is a fact."
      },
      distance=0.5
    ),
    RetrieveMemoriesResponseRetrievedMemory(
      memory=Memory(
        name="projects/.../locations/.../reasoningEngines/.../memories/...",
        ...
        fact="This is another fact."
      },
      distance=0.7
    ),
]
"""
Replace the following:

QUERY: The query for which to perform similarity search. For example, you can use the last user turn of the conversation as the query.

SCOPE: A dictionary, representing the scope for the similarity search. For example, {"user_id": "123"}. Only memories with the same scope as the request are considered.

Retrieve all memories
If no similarity search parameters are provided, RetrieveMemories returns all memories that have the provided scope, regardless of their similarity with the current conversation.

results = client.agent_engines.memories.retrieve(
    name=agent_engine.api_resource.name,
    scope=SCOPE
)
# RetrieveMemories returns a pager. You can use `list` to retrieve all pages' memories.
list(results)

"""
Returns:

[
    RetrieveMemoriesResponseRetrievedMemory(
      memory=Memory(
        name="projects/.../locations/.../reasoningEngines/.../memories/...",
        ...
        fact="This is a fact."
      }
    ),
    RetrieveMemoriesResponseRetrievedMemory(
      memory=Memory(
        name="projects/.../locations/.../reasoningEngines/.../memories/...",
        ...
        fact="This is another fact."
      }
    ),
]
"""
Replace the following:

SCOPE: A dictionary representing the scope for retrieval. For example, {"user_id": "123"}. Only memories with the same scope as the request are returned.
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded.



Skip to main content
Google Cloud
Documentation
Technology areas

Cross-product tools

Related sites

Search
/


English
Console

Generative AI on Vertex AI
Guides
API reference
Vertex AI Cookbook
Prompt gallery
Resources
FAQ
Contact Us
Filter

Generative AI on Vertex AI 
Documentation
Was this helpful?

Send feedbackTroubleshooting for Memory Bank

bookmark_border
Preview

This feature is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the Service Specific Terms. Pre-GA features are available "as is" and might have limited support. For more information, see the launch stage descriptions.

This document shows you how to resolve common issues when using Vertex AI Agent Engine Memory Bank.

No memories were generated
The memory generation process includes a crucial step: determining if information in the source content is meaningful enough to persist. An empty response indicates the process ran successfully but found no information that met the criteria for being saved. If you were expecting memories to be generated, this guide can help you identify a potential error or misconfiguration.

To troubleshoot why memories are not being generated, follow these steps in order:

Check if memory generation was triggered
First, confirm that the memory generation process was actually initiated. Memory generation is initiated by invoking GenerateMemories (client.agent_engines.memories.generate(...)).

If you're using ADK's VertexAiMemoryBankService, memory generation is not automatically triggered. You must ensure your agent or application has explicitly called the add_session_to_memory method to trigger the process.

add_session_to_memory takes a Session object as input and uses the session's events as the data source for memory generation. It only calls your Memory Bank instance if there are events populated in the session object. If your ADK application or agent is invoking add_session_to_memory but memory generation was not triggered, the Session object's events may not be populated. This is possible even if you have interacted with the session, especially if you're using adk.Runner. To address this issue, fetch the session and its events to the environment where you're invoking add_session_to_memory:

session = await session_service.get_session(
    app_name=app_name,
    user_id=user_id,
    session_id=session.id
)
# Confirm that events are populated.
print(session.events)
memory_service.add_session_to_memory(session)
Verify that the memory generation LRO is complete
Memory generation is a long-running operation (LRO) and can take a few seconds to complete. The exact latency depends on the length of the input conversation and the complexity of the information being processed.

When using the Agent Engine SDK, memory generation is a blocking operation by default. So, client.generate_memories(...) blocks your code's execution until the memory generation LRO is complete.

When using ADK's VertexAiMemoryBankService, add_session_to_memory is a non-blocking operation. It only triggers memory generation and does not wait for the LRO to complete.

Look for errors in the operation response
The LRO response may an error message similar that indicates that memory generation was unsuccessful. For example:

RuntimeError: Failed to generate memory: {'code': 3, 'message': 'Failed to extract memories: Please use a valid role: user, model.'}
Common errors include:

Resource exhausted errors for Gemini when you use pay-as-you-go. With dynamic shared quota (DSQ), there are no predefined quota limits on your usage. To help ensure high availability for Memory Bank and to get predictable service levels for your production workloads, see Provisioned Throughput.

Invalid source data, like using roles other than model and user in your Content.

Determine if the conversation was meaningful
If the process was triggered, completed successfully, and produced no errors, it's likely that Memory Bank determined that no information in the source conversation was meaningful enough to persist.

Memory Bank uses "memory topics" to identify what information is meaningful. If the content of your conversation doesn't align with any configured topics, no memories are generated.

If you believe information should have been persisted, you can customize your Memory Bank instance's configuration to better align with your expectations.

Configuring memory topics lets you define what information should be persisted. Configuring few-shot examples helps your Memory Bank instance adapt to your expectations by teaching it the nuances in what information should be persisted and with what phrasing. You can think of customizing your Memory Bank in two steps: Telling and Showing. Memory Topics tell Memory Bank what information to persist. Few-shots show Memory Bank what kind of information should result in a specific memory, helping it learn the patterns, nuance, and phrasing that you expect it to understand.

Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-10-10 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Community forums
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
The new page has loaded..