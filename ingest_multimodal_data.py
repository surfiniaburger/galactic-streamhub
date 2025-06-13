import os
import logging
import pydicom
from PIL import Image
import numpy as np
import vertexai
from vertexai.vision_models import Image as VertexImage
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import secretmanager
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
from pymongo.operations import SearchIndexModel


# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = "us-central1"

# The path where you downloaded the sample TCIA data
TCIA_DATA_PATH = os.path.join('data', 'tcia_lidc_idri')

# MongoDB Configuration
#MONGODB_URI = os.environ.get("MULTIMODAL_MONGODB_URI") # Make sure this is set in your environment
MONGODB_SECRET_ID = "MULTIMODAL_MONGODB_URI"
MONGODB_DATABASE_NAME = "biomed_research"
MONGODB_MULTIMODAL_COLLECTION_NAME = "medical_images"
MONGODB_MULTIMODAL_INDEX_NAME = "multimodal_image_index" 

# --- Setup Logging and Vertex AI ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    # Load the pre-trained multimodal embedding model
    embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logging.info("Successfully initialized Vertex AI and loaded the multimodal embedding model.")
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
    raise

def get_mongodb_uri():
    """
    Retrieves the MongoDB URI from an environment variable or Secret Manager.
    Prefers the environment variable if it is set.
    """
    mongodb_uri = os.environ.get("MONGODB_URI") # Name of environment variable
    if mongodb_uri:
        logging.info("Using MongoDB URI from environment variable.")
        return mongodb_uri
    else:
        logging.info("MongoDB URI not found in environment variable.  Retrieving from Secret Manager.")
        return get_secret(MONGODB_SECRET_ID)


def get_secret(secret_id, version_id="latest"):
    """Retrieves a secret from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.error(f"Failed to retrieve secret {secret_id}: {e}", exc_info=True)
        raise

def process_dicom_file(filepath: str) -> np.ndarray | None:
    """Reads a DICOM file and returns its pixel data as a NumPy array."""
    try:
        dicom_data = pydicom.dcmread(filepath)
        # Normalize pixel values to a standard 8-bit range (0-255) for image conversion
        pixels = dicom_data.pixel_array.astype(float)
        pixels_scaled = (np.maximum(pixels, 0) / pixels.max()) * 255.0
        return pixels_scaled.astype(np.uint8)
    except Exception as e:
        logging.error(f"Could not read or process DICOM file {filepath}: {e}")
        return None

def generate_multimodal_embedding(image_bytes: bytes, text_description: str) -> list | None:
    """
    Generates a multimodal embedding from image bytes and a text description.

    Args:
        image_bytes (bytes): The image file content as bytes.
        text_description (str): A description of the image content.

    Returns:
        A list representing the 1408-dimension embedding vector, or None on error.
    """
    try:
        # Load the image from the raw bytes
        image = VertexImage(image_bytes=image_bytes)

        # Generate the embeddings
        embeddings = embedding_model.get_embeddings(
            image=image,
            contextual_text=text_description,
            dimension=1408,  # Specify the full dimension for highest quality
        )
        # The result contains both an image and a text embedding. For multimodal search,
        # the image embedding is typically the one to store and use.
        return embeddings.image_embedding
    except Exception as e:
        logging.error(f"Failed to generate multimodal embedding: {e}", exc_info=True)
        return None



# --- NEW: Function to Create the Vector Search Index ---
def create_multimodal_vector_index(mongo_client: MongoClient):
    """
    Creates the Atlas Vector Search index on the medical_images collection if it doesn't exist.
    """
    db = mongo_client[MONGODB_DATABASE_NAME]
    collection = db[MONGODB_MULTIMODAL_COLLECTION_NAME]
    index_name = MONGODB_MULTIMODAL_INDEX_NAME

    try:
        # Check if the index already exists
        for index in collection.list_search_indexes():
            if index['name'] == index_name:
                logging.info(f"Vector search index '{index_name}' already exists.")
                return
    except Exception as e:
        logging.warning(f"Could not list existing search indexes for {collection.name}: {e}. Attempting to create index anyway.")

    # Define the vector search index
    index_definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "multimodal_embedding": {
                    "type": "knnVector",
                    "dimensions": 1408,  # Must match the embedding dimension
                    "similarity": "cosine"
                }
            }
        }
    }
    search_index_model = SearchIndexModel(definition=index_definition, name=index_name)
    try:
        logging.info(f"Creating vector search index '{index_name}'. This may take a few minutes...")
        collection.create_search_index(model=search_index_model)
        logging.info("Successfully initiated creation of vector search index.")
    except Exception as e:
        logging.error(f"Error creating vector search index: {e}", exc_info=True)

# --- NEW: Function to Perform a Sample Query ---
def find_similar_images(query_text: str, limit: int = 3):
    """
    Performs a multimodal vector search to find images similar to a text description.
    """
    logging.info(f"\n--- Running Sample Query: '{query_text}' ---")
    try:
        uri = get_mongodb_uri() 
        # Explicitly use certifi's CA bundle for TLS connections
        client = MongoClient(
            uri,
            server_api=ServerApi('1'),
            tlsCAFile=certifi.where()
        )
        client.admin.command('ping')
        logging.info("Successfully connected to MongoDB!")
        
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        raise

    try:
        db = client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_MULTIMODAL_COLLECTION_NAME]
        
        # Generate an embedding for the text query *without an image*
        # This allows us to search images using only text
        query_embeddings = embedding_model.get_embeddings(
            contextual_text=query_text,
            dimension=1408,
        )
        query_vector = query_embeddings.text_embedding

        # Define the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": MONGODB_MULTIMODAL_INDEX_NAME,
                    "path": "multimodal_embedding",
                    "queryVector": query_vector,
                    "numCandidates": 50,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "image_id": 1,
                    "text_description": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        
        if results:
            logging.info(f"Found {len(results)} similar images:")
            for result in results:
                logging.info(f"  - Image ID: {result['image_id']}, Score: {result['score']:.4f}")
                logging.info(f"    Description: {result['text_description']}")
        else:
            logging.warning("No similar images found for the query.")

    except Exception as e:
        logging.error(f"An error occurred during vector search: {e}", exc_info=True)
    finally:
        client.close()

def run_multimodal_ingestion_pipeline(sample_limit: int = 3):
    """
    Main pipeline to find, process, embed, and ingest a sample of DICOM series.

    Args:
        sample_limit (int): The number of patient series to process from the download folder.
    """
    logging.info("Starting multimodal ingestion pipeline...")
    try:
        uri = get_mongodb_uri() 
        # Explicitly use certifi's CA bundle for TLS connections
        client = MongoClient(
            uri,
            server_api=ServerApi('1'),
            tlsCAFile=certifi.where()
        )
        client.admin.command('ping')
        logging.info("Successfully connected to MongoDB!")
        
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        raise

    db = client[MONGODB_DATABASE_NAME]
    collection = db[MONGODB_MULTIMODAL_COLLECTION_NAME]
    
    # Get a list of patient series directories from the downloaded data
    series_dirs = [d for d in os.listdir(TCIA_DATA_PATH) if os.path.isdir(os.path.join(TCIA_DATA_PATH, d))]
    
    if not series_dirs:
        logging.error(f"No series directories found in {TCIA_DATA_PATH}. Did the download script run correctly?")
        return

    processed_count = 0
    for series_dir_name in series_dirs:
        if processed_count >= sample_limit:
            logging.info(f"Reached sample limit of {sample_limit}. Stopping.")
            break

        series_path = os.path.join(TCIA_DATA_PATH, series_dir_name)
        logging.info(f"--- Processing Series: {series_dir_name} ---")

        # Find the DICOM file roughly in the middle of the series for a representative slice
        dicom_files = sorted([f for f in os.listdir(series_path) if f.endswith('.dcm')])
        if not dicom_files:
            logging.warning(f"No DICOM files found in {series_path}. Skipping.")
            continue
        
        middle_slice_path = os.path.join(series_path, dicom_files[len(dicom_files) // 2])
        
        # --- 1. Process the DICOM image ---
        pixel_array = process_dicom_file(middle_slice_path)
        if pixel_array is None:
            continue
        
        # Convert NumPy array to a PIL Image, then save to a byte buffer as a PNG
        pil_image = Image.fromarray(pixel_array)
        with open('temp_image.png', 'wb') as f:
            pil_image.save(f, format='PNG')
        with open('temp_image.png', 'rb') as f:
            image_bytes = f.read()

        # --- 2. Create Text Description & Generate Embedding ---
        # In a real scenario, you'd parse XML annotations here.
        # For the hackathon, we'll create a descriptive placeholder text.
        text_description = f"A CT scan of a lung from the LIDC-IDRI dataset, patient series {series_dir_name}. This scan may contain nodules or signs of cancer."
        
        embedding_vector = generate_multimodal_embedding(image_bytes, text_description)
        if embedding_vector is None:
            continue

        # --- 3. Prepare and Ingest the Document into MongoDB ---
        document = {
            "image_id": f"{series_dir_name}_slice_{len(dicom_files)//2}",
            "source_dataset": "LIDC-IDRI",
            "patient_series_uid": series_dir_name,
            "text_description": text_description,
            "multimodal_embedding": embedding_vector
        }

        try:
            collection.insert_one(document)
            logging.info(f"Successfully ingested document for {document['image_id']} into MongoDB.")
            processed_count += 1
            # After ingestion, create the index
            create_multimodal_vector_index(client)
        except Exception as e:
            logging.error(f"Failed to insert document into MongoDB: {e}")

    # --- 4. Clean up and close connections ---
    if os.path.exists('temp_image.png'):
        os.remove('temp_image.png')
    client.close()
    logging.info("Multimodal ingestion pipeline finished.")
    # IMPORTANT: After running this, you need to go to the Atlas UI and create a Vector Search Index
    # on the 'medical_images' collection, targeting the 'multimodal_embedding' field.

if __name__ == '__main__':
    #run_multimodal_ingestion_pipeline(sample_limit=5) # Process 5 patient series for the demo
    # After ingestion, run a sample query to test the index
    find_similar_images(query_text="A CT scan showing a possible cancerous lung nodule", limit=3)