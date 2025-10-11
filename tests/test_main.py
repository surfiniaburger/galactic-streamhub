import sys
import os
from fastapi.testclient import TestClient

# Set env var to disable MCP servers before importing the app
os.environ["DISABLE_MCP_SERVERS"] = "true"

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

def test_read_main():
    """
    Test that the root endpoint returns a 200 OK status code and the content type is html.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
