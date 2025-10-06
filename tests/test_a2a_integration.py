import multiprocessing
import time
import requests
import pytest
import uvicorn
import os
import uuid
from main_agent.agent import a2a_app as main_a2a_app
from evaluator_agent.agent import a2a_app as evaluator_a2a_app
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.generativeai.protos import Candidate, Content, Part
from unittest.mock import patch

# This is the mock function that will replace the real API call
async def mock_generate_content_async(*args, **kwargs):
    """Mocks the async LLM call to avoid needing credentials."""
    return genai.types.GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[Part(text="This is a mock response from the LLM.")],
                    role="model"
                ),
            )
        ]
    )

def run_server_with_mock(app, host, port):
    """Helper function that mocks the LLM and then runs a uvicorn server."""
    # We patch the class method within the scope of this child process
    with patch(
        "google.generativeai.GenerativeModel.generate_content_async",
        new=mock_generate_content_async
    ):
        uvicorn.run(app, host=host, port=port)

def test_a2a_integration():
    """
    Integration test to verify A2A communication between the main and evaluator agents.
    """
    os.environ["IS_TESTING"] = "true"
    os.environ["GOOGLE_API_KEY"] = "test-key"  # Dummy key is still needed for initialization

    # Start servers in background processes using the new helper
    main_proc = multiprocessing.Process(target=run_server_with_mock, args=(main_a2a_app, "127.0.0.1", 8001))
    eval_proc = multiprocessing.Process(target=run_server_with_mock, args=(evaluator_a2a_app, "127.0.0.1", 8002))
    main_proc.start()
    eval_proc.start()

    # Give the servers time to start and check if they are alive
    time.sleep(15) # Increased sleep time
    if not main_proc.is_alive() or not eval_proc.is_alive():
        main_proc.terminate()
        eval_proc.terminate()
        pytest.fail("One of the agent servers failed to start.")

    try:
        # Send a request to the main agent's A2A endpoint using the correct message/send method.
        response = requests.post(
            'http://localhost:8001/',  # The endpoint is the root URL
            json={
                "jsonrpc": "2.0",
                "method": "message/send", # Correct A2A method
                "params": {
                    "message": {
                        "messageId": str(uuid.uuid4()),
                        "role": "user",
                        "parts": [
                            {"text": "What is a good cocktail to make with gin?"}
                        ]
                    }
                },
                "id": "1",
            },
            timeout=15
        )
        assert response.status_code == 200
        response_json = response.json()
        task_id = response_json.get('result', {}).get('task', {}).get('id')
        assert task_id is not None, f"Could not get task_id from response: {response_json}"

        # Poll for the result.
        final_status = None
        for i in range(45):  # Poll for up to 45 seconds
            time.sleep(1)
            task_response = requests.post(
                'http://localhost:8001/',
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {"id": task_id},
                    "id": str(i + 2) # Unique ID for each request
                },
                timeout=5
            )

            if task_response.status_code != 200:
                print(f"Polling request failed with status: {task_response.status_code}")
                continue

            task_json = task_response.json()
            task_data = task_json.get('result', {}).get('task', {})
            final_status = task_data.get('status', {}).get('state')

            if final_status == 'completed':
                artifacts = task_data.get('artifacts', [])
                assert len(artifacts) > 0, "Task completed but no artifacts found."
                final_response_part = artifacts[0].get('parts', [{}])[0]
                assert 'text' in final_response_part, "Artifact does not contain a text part."
                assert "This is a mock response" in final_response_part['text']
                print(f"Final Agent Response: {final_response_part['text']}")
                break
            elif final_status == 'failed':
                pytest.fail(f"Task failed with message: {task_data.get('status', {}).get('message')}")

        assert final_status == 'completed', f"Task did not complete in time. Final status: {final_status}"

    finally:
        # Clean up the background processes
        if main_proc.is_alive():
            main_proc.terminate()
            main_proc.join()
        if eval_proc.is_alive():
            eval_proc.terminate()
            eval_proc.join()
