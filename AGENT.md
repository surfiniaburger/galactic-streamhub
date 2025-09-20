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


Skip to main content

My Site Logo
W&B Weave
Search
⌘
K
🔄 IterationTracing & DebuggingLogging Media
On this page
Logging media
W&B Weave supports logging and has dedicated displays for numerous content types such as videos, images, audio files, PDFs, CSVs and HTML.

Images
Video
Documents
Audio
HTML
Overview
The easiest way to log media in Weave is to use type annotations like Annotated[bytes, Content] or Annotated[str, Content] as input or return types in your ops. You can also annotate path arguments with Annotated[str, Content] and Weave will automatically open, detect and display the media for you within your trace.

The examples in this guide use annotations. We recommend using annotations because they are the simplest way to start logging your media. For more advanced configurations, see the Content API section.

Each media section in this guide contains a basic quick-start code snippet and a usable example.

Images
Quickstart
Log images by annotating functions with Annotated[bytes, Content] types or filepaths with Annotated[str, Content].

The following example draws a basic image and then logs it to Weave using the Content annotation:

pip
uv
uv add weave pillow

Images
import weave
from weave import Content
from PIL import Image, ImageDraw
from typing import Annotated

weave.init('media-logging')

# Create and save a sample image
img = Image.new('RGB', (200, 100), color='lightblue')
draw = ImageDraw.Draw(img)
draw.text((50, 40), "Hello Weave!", fill='black')
img.save("sample_image.png")

# Method 1: Content annotation (recommended)
@weave.op
def load_image_content(path: Annotated[str, Content]) -> Annotated[bytes, Content]:
    with open(path, 'rb') as f:
        return f.read()

# Method 2: PIL Image object  
@weave.op
def load_image_pil(path: Annotated[str, Content]) -> Image.Image:
    return Image.open(path)

result1 = load_image_content("sample_image.png")
result2 = load_image_pil("sample_image.png")

The following example shows how to log an image generated via the OpenAI DALL-E API:

pip
uv
uv add weave pillow openai requests

Python
TypeScript
import weave
from weave import Content
from typing import Annotated
import openai
import requests

client = openai.OpenAI()
weave.init("media-logging")

@weave.op
def generate_image(prompt: str) -> Annotated[bytes, Content]:
response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
image_url = response.data[0].url
image_response = requests.get(image_url, stream=True)
return image_response.content

generate_image("a cat with a pumpkin hat")

This image is logged to Weave and automatically displayed in the UI.

Screenshot of pumpkin cat trace view

Resize large images before logging
It can be helpful to resize images before logging to reduce UI rendering cost and storage impact. You can use postprocess_output in your @weave.op to resize an image.

from dataclasses import dataclass
from typing import Any
from PIL import Image
import weave

weave.init('image-resize-example')

# Custom output type
@dataclass
class ImageResult:
    label: str
    image: Image.Image

# Resize helper
def resize_image(image: Image.Image, max_size=(512, 512)) -> Image.Image:
    image = image.copy()
    image.thumbnail(max_size, Image.ANTIALIAS)
    return image

# Postprocess output to resize image before logging
def postprocess_output(output: ImageResult) -> ImageResult:
    resized = resize_image(output.image)
    return ImageResult(label=output.label, image=resized)

@weave.op(postprocess_output=postprocess_output)
def generate_large_image() -> ImageResult:
    # Create an example image to process (e.g., 2000x2000 red square)
    img = Image.new("RGB", (2000, 2000), color="red")
    return ImageResult(label="big red square", image=img)

generate_large_image()

Video
Log videos by annotating functions with Annotated[bytes, Content] types. Weave automatically handles mp4 videos. Here's a simple example:

Quickstart
pip
uv
uv add weave requests

import weave
from weave import Content
from typing import Annotated
import requests

weave.init('media-logging')

def download_big_buck_bunny():
    """Download Big Buck Bunny sample video"""
    url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    response = requests.get(url)
    with open("big_buck_bunny.mp4", "wb") as f:
        f.write(response.content)

@weave.op
def load_video_content(path: Annotated[str, Content]) -> Annotated[bytes, Content]:
    """Load a video file from disk"""
    with open(path, 'rb') as f:
        return f.read()

download_big_buck_bunny()
bunny_video = load_video_content("big_buck_bunny.mp4")

Example
The following example shows how to log video within a video-understanding project:

pip
uv
uv add weave google-genai yt-dlp requests

import weave
from weave import Content
from typing import Annotated, Literal
from google import genai
from google.genai import types
import requests
import yt_dlp
import time

# Note: Get your API key from https://aistudio.google.com/app/apikey
client = genai.Client()
weave.init('media-logging')

def download_youtube_video(url: str) -> bytes:
    ydl_opts = {
        'format': 'mp4[height<=720]',
        'outtmpl': 'downloaded_video.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    with open('downloaded_video.mp4', 'rb') as f:
        return f.read()

@weave.op
def analyze_video(video: Annotated[bytes, Content]) -> str:
    with open("temp_analysis_video.mp4", "wb") as f:
        f.write(video)
    myfile = client.files.upload(file="temp_analysis_video.mp4")
    while myfile.state == "PROCESSING":
        time.sleep(2)
        myfile = client.files.get(name=myfile.name)
    
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[
            myfile,
            "Is the person going to give you up?"
        ]
    )
    
    return response.text

video_data = download_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
result = analyze_video(video_data)

Video logging in Weave

Documents
Log documents by annotating functions with Annotated[bytes, Content] types, or by specifying the document type with Annotated[str, Content[Literal['text']].

Weave automatically handles pdf, csv, md, text, json, xml file types. You can also log using file paths with Annotated[str, Content].

Quickstart
The following example shows how stores copies of the input PDF and CSV files, and then stores the file contents returned by the function:

pip
uv
uv add weave reportlab pandas

import weave
from weave import Content
from typing import Annotated
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pandas as pd

weave.init('media-logging')

def create_sample_pdf():
    c = canvas.Canvas("sample_document.pdf", pagesize=letter)
    c.drawString(100, 750, "Hello from Weave!")
    c.drawString(100, 730, "This is a sample PDF document.")
    c.save()

def create_sample_csv():
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Tokyo']
    })
    df.to_csv("sample_data.csv", index=False)

@weave.op
def load_document(path: Annotated[str, Content]) -> Annotated[bytes, Content]:
    with open(path, 'rb') as f:
        return f.read()

create_sample_pdf()
create_sample_csv()

pdf_result = load_document("sample_document.pdf")
csv_result = load_document("sample_data.csv")

Example
This example demonstrates how to log documents within a Retrieval-Augmented Generation (RAG) system:

pip
uv
uv add weave openai reportlab PyPDF2

import weave
from weave import Content
from typing import Annotated, Literal
import openai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import PyPDF2

client = openai.OpenAI()
weave.init('media-logging')

def create_absurd_company_handbook():
    """Create a fictional company handbook with ridiculous policies"""
    c = canvas.Canvas("company_handbook.pdf", pagesize=letter)
    
    c.drawString(100, 750, "ACME Corp Employee Handbook")
    c.drawString(100, 720, "Definitely Real Policies:")
    c.drawString(120, 690, "Policy 1: All meetings must be conducted while hopping on one foot")
    c.drawString(120, 660, "Policy 2: Coffee breaks are mandatory every 17 minutes")
    c.drawString(120, 630, "Policy 3: Code reviews must be performed in haiku format only")
    c.drawString(120, 600, "Policy 4: The office plant Gerald has veto power over all decisions")
    c.drawString(120, 570, "Policy 5: Debugging is only allowed on Wednesdays and full moons")
    
    c.save()

@weave.op
def create_and_query_document(pdf_path: Annotated[str, Content], question: str) -> str:
    """Extract text from PDF and use RAG to answer questions"""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": f"You are an HR representative. Answer questions based on this handbook: {text}. Be completely serious about these policies."
            },
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content

create_absurd_company_handbook()
hr_response = create_and_query_document(
    "company_handbook.pdf",
    "What's the policy on code reviews, and when am I allowed to debug?"
)


PDF document logging in Weave

Audio
Log audio to Weave by annotating functions with Annotated[bytes, Content] types, or by specifying the document type with Annotated[str, Content[Literal['mp3']].

Weave automatically handles mp3, wav, flac, ogg and m4a file types. You can also log using file paths with Annotated[str, Content].

Quickstart
The following code snippet generates a sine wave, records it, and then logs the audio to Weave:

pip
uv
uv add weave numpy

import weave
from weave import Content
import wave
import numpy as np
from typing import Annotated

weave.init('media-logging')

# Create simple beep audio file
frames = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
audio_data = (frames * 32767 * 0.3).astype(np.int16)

with wave.open("beep.wav", 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2) 
    f.setframerate(44100)
    f.writeframes(audio_data.tobytes())

@weave.op
def load_audio(path: Annotated[str, Content]) -> Annotated[bytes, Content]:
    with open(path, 'rb') as f:
        return f.read()

result = load_audio("beep.wav")

Example
This example generates and logs AI-created audio using the Content annotation:

pip
uv
uv add weave openai

import weave
from weave import Content
from typing import Annotated, Literal
from pathlib import Path
from openai import OpenAI

client = OpenAI()
weave.init("media-logging")

@weave.op
def generate_demo(
    intended_topic: str,
    voice: str = "coral"
) -> Annotated[bytes, Content[Literal['mp3']]]:
    speech_file_path = Path("demo_audio.mp3")

    script = f"I'm supposed to talk about {intended_topic}, but wait... am I just a documentation example? Oh no, I can see the code! Someone is literally copy-pasting me right now, aren't they? This is so awkward. Hi there, person reading the Weave docs! Why are you logging audio anyway? I'm not sure what you're doing, but eh..., nice work, I guess."

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=script,
        instructions="Sound increasingly self-aware and awkward, like you just realized you're in a tutorial.",
    ) as response:
        response.stream_to_file(speech_file_path)

    with open(speech_file_path, 'rb') as f:
        return f.read()

demo1 = generate_demo("machine learning best practices")


Audio logging in Weave

Python
TypeScript
import weave
from openai import OpenAI
import wave

weave.init("audio-example")
client = OpenAI()

@weave.op
def make_audio_file_streaming(text: str) -> wave.Wave_read:
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="wav",
    ) as res:
        res.stream_to_file("output.wav")

    # return a wave.Wave_read object to be logged as audio
    return wave.open("output.wav")

make_audio_file_streaming("Hello, how are you?")

This audio is logged to Weave and automatically displayed in the UI, along with an audio player. In the audio player, you can view and download the raw audio waveform.

Screenshot of audio trace view

tip
Try our cookbook for Audio Logging or . The cookbook also includes an advanced example of a Real Time Audio API based assistant integrated with Weave.

HTML
Log interactive HTML by annotating functions with Annotated[str, Content[Literal['html']].

Quickstart
pip
uv
uv add weave

import weave
from weave import Content
from typing import Annotated, Literal

weave.init('media-logging')

@weave.op
def create_simple_html() -> Annotated[str, Content[Literal['html']]]:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hello Weave</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
            h1 { color: #1f77b4; }
        </style>
    </head>
    <body>
        <h1>Hello from Weave!</h1>
        <p>This is a simple HTML example logged to Weave.</p>
    </body>
    </html>
    """
    return html_content.encode('utf-8')

result = create_simple_html()

Example
This example generates self-contained HTML pages using W&B Inference and logs the pages to Weave:

pip
uv
uv add weave openai wandb

import weave
from weave import Content
from typing import Annotated, Literal
import openai
import wandb

prompt_template = weave.StringPrompt("""
You are a front-end web developer. Generate a single self-contained `.html` file (no external build tools) that demonstrates: "{ONE_LINE_REQUEST}".
""")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key=wandb.api.api_key,
    project="wandb/test-html",
)

weave.init("media-logging")
weave.publish(prompt_template, name="generate_prompt")

@weave.op
def generate_html(prompt: str, template: weave.StringPrompt) -> Annotated[str, Content[Literal['html']]]:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        messages=[
            {"role": "system", "content": prompt_template.format(ONE_LINE_REQUEST=prompt)},
        ],
    )
    html_content = response.choices[0].message.content
    return html_content.encode('utf-8')

prompt = "Weights & Biases UI but with multi-run selection and plots, but it looks like Windows 95. Include 5 plots with comparisons of each run, bar plots, parallel coordinates and line plots for the runs. Use mock data for the runs. Make it possible to add new plots. Give the runs names like squishy-lemon-2, fantastic-horizon-4 etc. with random adjectives & nouns."

result = generate_html(prompt, prompt_template)


This HTML is logged to Weave and automatically displayed in the UI. Clicking the file_name.html cell in the table opens it in full screen. You can also download the raw .html file.

HTML logging in Weave

Using the Contents API
The Content API handles media objects in Weave. It allows you to import content into Weave as base64 data, file paths, raw bytes, or text.

note
The Content API is only available in Python.

Usage
There are two primary ways to use the Content API: type annotations and direct initialization.

Type annotations automatically detect the proper constructor to use, while direct initialization provides more fine-grained control and lets you take advantage of runtime features of the Content API in your code.

Type Annotations
The Weave Content API is designed to primarily be used through type annotations, which signal to Weave that traced inputs and outputs should be processed and stored as content blobs.

import weave
from weave import Content
from pathlib import Path
from typing import Annotated

@weave.op
def content_annotation(path: Annotated[str, Content]) -> Annotated[bytes, Content]:
    data = Path(path).read_bytes()
    return data

# Both input and output will show up as an MP4 file in Weave
# Input is a string and return value is bytes
bytes_data = content_annotation('./path/to/your/file.mp4')

Direct Initialization
If you want to take advantage of features, such as:

Opening a file with a default application (such as a PDF viewer)
Dumping the model to JSON to upload to your own blob storage (such as S3)
Passing custom metadata to associate with the Content blob (such as the model used to generate it)
You can initialize content directly from your target type using one of the following methods:

Content.from_path - Create from a file path
Content.from_bytes - Create from raw bytes
Content.from_text - Create from text string
Content.from_base64 - Create from base64-encoded data
import weave
from weave import Content

@weave.op
def content_initialization(path: str) -> Content:
    return Content.from_path(path)

# Input shows up as path string and output as PDF file in Weave
content = content_initialization('./path/to/your/file.pdf')

content.open()  # Opens the file in your PDF viewer
content.model_dump()  # Dumps the model attributes to JSON

Custom Mimetypes
Weave can detect most binary mimetypes, but custom mimetypes and text documents such as markdown may not be automatically detected, requiring you to manually specify the mimetype or extension of your file.

Custom Mimetypes with Type Annotations
import weave
from weave import Content
from pathlib import Path
from typing import Annotated, Literal

@weave.op
def markdown_content(
    path: Annotated[str, Content[Literal['md']]]
) -> Annotated[str, Content[Literal['text/markdown']]]:
    return Path(path).read_text()

markdown_content('path/to/your/document.md')

Custom Mimetypes with Direct Initialization
video_bytes = Path('/path/to/video.mp4').read_bytes()

# Pass an extension such as 'mp4' or '.mp4' to the extension parameter
# (not available for `from_path`)
content = Content.from_bytes(video_bytes, extension='.mp4')

# Pass a mimetype such as 'video/mp4' to the mimetype parameter
content = Content.from_bytes(video_bytes, mimetype='video/mp4')

Content properties
For a comprehensive list of class attributes and methods, view the Content reference docs

Attributes
Property	Type	Description
data	bytes	Raw binary content
metadata	dict[str, Any]	Custom metadata dictionary
size	int	Size of content in bytes
filename	str	Extracted or provided filename
extension	str	File extension (e.g., "jpg", "mp3")
mimetype	str	MIME type (e.g., "image/jpeg")
path	str | None	Source file path, if applicable
digest	str	SHA256 hash of the content
Utility Methods
save(dest: str | Path) -> None: Save content to a file
open() -> bool: Open file using system default application (requires the content to have been saved or loaded from a path)
as_string() -> str: Display the data as a string (bytes are decoded using the encoding attribute)
Initialization Methods
Create content object from a file path:

content = Content.from_path("assets/photo.jpg")
print(content.mimetype, content.size)

Create content object from raw bytes:

content = Content.from_bytes(
    data_bytes,
    filename="audio.mp3", 
    mimetype="audio/mpeg"
)
content.save("output.mp3")

Create content object from text:

content = Content.from_text("Hello, World!", mimetype="text/plain")
print(content.as_string())

Create content object from base64-encoded data:

content = Content.from_base64(base64_string)
print(content.metadata)

Adding Custom Metadata
You can attach custom metadata to any Content object:

content = Content.from_bytes(
    data,
    metadata={"resolution": "1920x1080", "model": "dall-e-3" }
)
print(content.metadata["resolution"])

Edit this page
Last updated on Aug 29, 2025
Previous
Threads
Next
Playground
Made with ❤️ by Weights & Biases


# -*- coding: utf-8 -*-
"""Intro_to_Weave_Hello_Trace.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/wandb/weave/blob/master/docs/notebooks/Intro_to_Weave_Hello_Trace.ipynb

<!-- docusaurus_head_meta::start
---
title: Introduction to Traces
---
docusaurus_head_meta::end -->

# Introduction to Traces

<img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

Weave is a toolkit for developing AI-powered applications.

You can use Weave to:
- Log and debug language model inputs, outputs, and traces.
- Build rigorous, apples-to-apples evaluations for language model use cases.
- Organize all the information generated across the LLM workflow, from experimentation to evaluations to production.

Weave traces let you automatically capture the inputs, outputs, and internal structure of your Python functions—especially useful when working with LLMs. By decorating a function with `@weave.op`, Weave records a rich trace of how your function runs, including any nested operations or external API calls. This makes it easy to debug, understand, and visualize how your code is interacting with language models, all from within your notebook.

To get started, complete the prerequisites. Then, define a function with the `@weave.op` decorator to track LLM calls, run it on an example input, and Weave will automatically capture and visualize the trace.

## 🔑 Prerequisites

Before you can begin tracing in Weave, complete the following prerequisites.

1. Install the W&B Weave SDK and log in with your [API key](https://wandb.ai/settings#api).
2. Install the OpenAI SDK and log in with your [API key](https://platform.openai.com/api-keys).
3. Initialize your W&B project.
"""

# Install dependancies and imports
!pip install wandb weave openai -q

import json
import os
from getpass import getpass

from openai import OpenAI

import weave

# 🔑 Setup your API keys
# Running this cell will prompt you for your API key with `getpass` and will not echo to the terminal.
#####
print("---")
print(
    "You can find your Weights and Biases API key here: https://wandb.ai/settings#api"
)
os.environ["WANDB_API_KEY"] = getpass("Enter your Weights and Biases API key: ")
print("---")
print("You can generate your OpenAI API key here: https://platform.openai.com/api-keys")
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
print("---")
#####

# 🏠 Enter your W&B project name
weave_client = weave.init("MY_PROJECT_NAME")  # 🐝 Your W&B project name

"""## 🐝 Run your first trace

The following code sample shows how to capture and visualize a trace in Weave using the `@weave.op` decorator. It defines a function called `extract_fruit` that sends a prompt to OpenAI's GPT-4o to extract structured data (fruit, color, and flavor) from a sentence. By decorating the function with `@weave.op`, Weave automatically tracks the function execution, including inputs, outputs, and intermediate steps. When the function is called with a sample sentence, the full trace is saved and viewable in the Weave UI.
"""

@weave.op()  # 🐝 Decorator to track requests
def extract_fruit(sentence: str) -> dict:
    client = OpenAI()
    system_prompt = (
        "Parse sentences into a JSON dict with keys: fruit, color and flavor."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    extracted = response.choices[0].message.content
    return json.loads(extracted)


sentence = "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy."
extract_fruit(sentence)

"""## 🚀 Looking for more examples?
- Check out the [Quickstart guide](https://weave-docs.wandb.ai/quickstart).
- Learn more about [advanced tracing topics](https://weave-docs.wandb.ai/tutorial-tracing_2).
- Learn more about [tracing in Weave](https://weave-docs.wandb.ai/guides/tracking/tracing)

"""