import vertexai
from vertexai import rag

# --- CONFIGURATION ---
PROJECT_ID = "silver-455021"  # Your project ID
LOCATION = "us-central1"      # The location of your corpus
# This is the full resource name you got from your create_corpus script
CORPUS_NAME = "projects/140457946058/locations/us-central1/ragCorpora/2305843009213693952" 
# --- END CONFIGURATION ---

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Delete the RagCorpus
# The 'force=True' flag is necessary to delete a non-empty corpus
print(f"Attempting to delete RAG Corpus: {CORPUS_NAME}")
rag.delete_corpus(name=CORPUS_NAME)