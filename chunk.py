# import os
# from google.cloud import storage

# my_text = "It's a true multi-agent, multi-stream, multimodal synthesis engine that is robust, efficient, and architecturally elegant. "



# def chunk_text(text, chunk_size=150):  # Chunk size in characters
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
#     return chunks

# def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(source_file_name)
#     print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# # 1. Chunk the text
# chunks = chunk_text(my_text)

# # 2.  Define GCS bucket
# BUCKET_NAME = "galactic-streamhub" # replace with your bucket

# # 3. Save chunks to files and upload
# for i, chunk in enumerate(chunks):
#     file_name = f"chunk_{i}.txt"
#     with open(file_name, "w") as f:
#         f.write(chunk)
#     gcs_blob_name = f"my_rag_data/chunk_{i}.txt" # a folder in your bucket
#     upload_to_gcs(BUCKET_NAME, file_name, gcs_blob_name)
#     os.remove(file_name) # Clean up the local file


#https://storage.googleapis.com/galactic-streamhub/my_rag_data/chunk_0.txt
# gs://galactic-streamhub/my_rag_data/chunk_0.txt



# from vertexai import rag
# import vertexai

#  # TODO(developer): Update below lines
# PROJECT_ID = "silver-455021" # Replace with your actual project ID
# LOCATION = "us-central1"  # Or your desired region
# DISPLAY_NAME = "galactic-streamhub"  # A user-friendly name for your corpus
# DESCRIPTION = "A description of your corpus"  # Optional description

#  # Initialize Vertex AI API
# vertexai.init(project=PROJECT_ID, location=LOCATION)

#  # Configure the embedding model
# embedding_model_config = rag.RagEmbeddingModelConfig(
#      vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
#          publisher_model="publishers/google/models/text-embedding-005"  # Or another embedding model
#      )
#  )

#  # Create the RAG corpus
# rag_corpus = rag.create_corpus(
#      display_name=DISPLAY_NAME,
#      description=DESCRIPTION,
#      backend_config=rag.RagVectorDbConfig(
#          rag_embedding_model_config=embedding_model_config
#      ),
#  )

# print(rag_corpus)
# print(f"RAG Corpus created with name: {rag_corpus.name}")  # IMPORTANT: You'll need this name for your agent_config.py
# projects/140457946058/locations/us-central1/ragCorpora/2305843009213693952




from vertexai import rag
import vertexai

# TODO(developer): Update below lines
PROJECT_ID = "silver-455021"  # Replace with your actual project ID
LOCATION = "us-central1"  # Or your desired region
CORPUS_NAME = "projects/140457946058/locations/us-central1/ragCorpora/2305843009213693952"  # Replace with the full resource name of your corpus
FILE_PATHS = ["gs://galactic-streamhub/my_rag_data/chunk_0.txt"]  # Replace with the GCS path to your text file

# Initialize Vertex AI API
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Import files into the RAG corpus
response = rag.import_files(
    corpus_name=CORPUS_NAME,
    paths=FILE_PATHS,
    transformation_config=rag.TransformationConfig(  # Optional: Configure chunking (if you didn't chunk manually)
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,  # Adjust as needed
            chunk_overlap=100,  # Adjust as needed
        ),
    ),
    max_embedding_requests_per_min=1000,  # Optional: Adjust rate limit
)

print(f"Imported {response.imported_rag_files_count} files.")
