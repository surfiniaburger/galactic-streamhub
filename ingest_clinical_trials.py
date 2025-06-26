import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager, bigquery
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
from pymongo.operations import SearchIndexModel
from vertexai.language_models import TextEmbeddingModel
from pydantic import BaseModel, Field

# --- Configuration ---
# GCP Project ID for Vertex AI, Secret Manager, BigQuery
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# MongoDB Configuration - SPECIFIC TO CLINICAL TRIALS
MONGODB_SECRET_ID = "mongodb-uri"
MONGODB_DATABASE_NAME = "biomed_research"
MONGODB_COLLECTION_NAME = "clinical_trials"  # New Collection
MONGODB_VECTOR_INDEX_NAME = "clinical_trials_vector_index" # New Index

# BigQuery Configuration - SPECIFIC TO CLINICAL TRIALS
BIGQUERY_DATASET_ID = "pubmed_data" # Can use the same dataset
BIGQUERY_TABLE_ID = "clinical_trials_archive" # New Table

# Clinical Trials Data Source
CLINICAL_TRIALS_CSV_PATH = "data/ctg-studies.csv"

# Embedding Model
EMBEDDING_MODEL_NAME = "text-embedding-005"

# --- Setup (Logging, Clients) - Reusable ---

def setup_logging():
    """Sets up Google Cloud Logging."""
    # Renaming logger to avoid conflicts if imported elsewhere
    logger = logging.getLogger('clinical_trials_pipeline')
    if logger.handlers:
        return logger, logger.handlers[0], None # Assume already configured

    gcp_logging_client = google.cloud.logging.Client(project=GCP_PROJECT_ID)
    handler = CloudLoggingHandler(gcp_logging_client)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger, handler, gcp_logging_client

logger, gcp_log_handler, gcp_log_client_instance = setup_logging()

try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to initialize BigQuery client or Embedding Model: {e}", exc_info=True)
    raise

class ClinicalTrialStudy(BaseModel):
    """Pydantic model for a clinical trial study record."""
    nct_number: str = Field(alias="NCT Number")
    study_title: Optional[str] = Field(None, alias="Study Title")
    study_status: Optional[str] = Field(None, alias="Study Status")
    conditions: Optional[str] = Field(None, alias="Conditions")
    interventions: Optional[str] = Field(None, alias="Interventions")
    sponsor: Optional[str] = Field(None, alias="Sponsor")
    study_url: Optional[str] = Field(None, alias="Study URL")
    score: Optional[float] = None


def get_mongodb_uri():
    """Retrieves the MongoDB URI from an environment variable or Secret Manager."""
    mongodb_uri = os.environ.get("MONGODB_URI")
    if mongodb_uri:
        logger.info("Using MongoDB URI from environment variable.")
        return mongodb_uri
    else:
        logger.info("MongoDB URI not found in environment variable. Retrieving from Secret Manager.")
        return get_secret(MONGODB_SECRET_ID)

def get_secret(secret_id, version_id="latest"):
    """Retrieves a secret from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_id}: {e}", exc_info=True)
        raise

def connect_to_mongodb():
    """Connects to MongoDB using URI from Secret Manager."""
    try:
        uri = get_mongodb_uri()
        client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        return client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        raise

# --- BigQuery Schema and Table Creation for Clinical Trials ---
CLINICAL_TRIALS_BQ_SCHEMA = [
    bigquery.SchemaField("nct_number", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("study_title", "STRING"),
    bigquery.SchemaField("study_status", "STRING"),
    bigquery.SchemaField("conditions", "STRING"),
    bigquery.SchemaField("interventions", "STRING"),
    bigquery.SchemaField("sponsor", "STRING"),
    bigquery.SchemaField("collaborators", "STRING"),
    bigquery.SchemaField("study_type", "STRING"),
    bigquery.SchemaField("study_url", "STRING"),
    bigquery.SchemaField("rich_text_for_embedding", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED")
]

def create_bq_clinical_trials_table_if_not_exists():
    """Creates the BigQuery table for clinical trials if it doesn't exist."""
    dataset_ref = bq_client.dataset(BIGQUERY_DATASET_ID)
    bq_client.create_dataset(dataset_ref, exists_ok=True)
    table_ref = dataset_ref.table(BIGQUERY_TABLE_ID)
    table = bigquery.Table(table_ref, schema=CLINICAL_TRIALS_BQ_SCHEMA)
    bq_client.create_table(table, exists_ok=True)
    logger.info(f"BigQuery table '{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}' ensured.")


# --- Data Processing and Embedding for Clinical Trials ---
def process_clinical_trials_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the Clinical Trials DataFrame to create rich_text for embedding, and generates embeddings.
    """
    logger.info(f"Processing {len(df)} clinical trial records...")
    print(f"Processing {len(df)} clinical trial records...")

    # Fill NaN values with empty strings to prevent errors during concatenation
    df_filled = df.fillna('')

    # Create the 'rich_text_for_embedding' field
    df_filled['rich_text_for_embedding'] = (
        "Trial Title: " + df_filled['Study Title'] + ". " +
        "Conditions Treated: " + df_filled['Conditions'] + ". " +
        "Interventions Used: " + df_filled['Interventions']
    )

    logger.info("Generating embeddings for clinical trials...")
    embeddings_list = []
    batch_size = 50
    for i in range(0, len(df_filled), batch_size):
        batch_texts = df_filled['rich_text_for_embedding'][i:i+batch_size].tolist()
        try:
            batch_embeddings = embedding_model.get_embeddings(batch_texts)
            embeddings_list.extend([emb.values for emb in batch_embeddings])
        except Exception as e:
            logger.error(f"Error generating embeddings for batch starting at index {i}: {e}", exc_info=True)
            embeddings_list.extend([[] for _ in range(len(batch_texts))])

    df_filled['embedding'] = embeddings_list
    df_filled['ingested_at'] = datetime.now(timezone.utc)

    # Filter out rows where embedding failed
    processed_df = df_filled[df_filled['embedding'].apply(lambda x: bool(x))].copy()

    # Rename columns to match BigQuery schema (snake_case)
    processed_df.rename(columns={
        'NCT Number': 'nct_number',
        'Study Title': 'study_title',
        'Study Status': 'study_status',
        'Conditions': 'conditions',
        'Interventions': 'interventions',
        'Sponsor': 'sponsor',
        'Collaborators': 'collaborators',
        'Study Type': 'study_type',
        'Study URL': 'study_url'
    }, inplace=True)

    logger.info(f"Finished processing. {len(processed_df)} clinical trials have embeddings.")
    return processed_df

def upload_to_mongodb(df: pd.DataFrame, mongo_client: MongoClient, collection_name: str):
    """Uploads DataFrame to a specified MongoDB collection."""
    if df.empty:
        logger.info(f"No data to upload to MongoDB collection '{collection_name}'.")
        return

    db = mongo_client[MONGODB_DATABASE_NAME]
    collection = db[collection_name]

    # Select columns that exist in the DataFrame to avoid KeyErrors
    columns_to_upload = [col for col in CLINICAL_TRIALS_BQ_SCHEMA if col.name in df.columns]
    column_names = [col.name for col in columns_to_upload]
    records = df[column_names].to_dict("records")

    try:
        collection.insert_many(records)
        logger.info(f"Inserted {len(records)} records into MongoDB collection '{collection_name}'.")
        print(f"Inserted {len(records)} records into MongoDB collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to insert data into MongoDB: {e}", exc_info=True)


def create_mongodb_vector_index(mongo_client: MongoClient, collection_name: str, index_name: str):
    """Creates the Atlas Vector Search index on a specified collection if it doesn't exist."""
    db = mongo_client[MONGODB_DATABASE_NAME]
    collection = db[collection_name]

    try:
        existing_indexes = list(collection.list_search_indexes())
        if any(idx.get('name') == index_name for idx in existing_indexes):
            logger.info(f"MongoDB vector index '{index_name}' already exists.")
            print(f"MongoDB vector index '{index_name}' already exists.")
            return
    except Exception as e:
        logger.warning(f"Could not list existing search indexes for {collection_name}: {e}. Attempting to create index.")

    index_definition = {
        "mappings": {
            "dynamic": True,  # More flexible for varied document structures
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": 768,
                    "similarity": "cosine"
                }
            }
        }
    }
    search_index_model = SearchIndexModel(definition=index_definition, name=index_name)
    try:
        result = collection.create_search_indexes([search_index_model])
        logger.info(f"Successfully initiated creation of MongoDB vector index '{index_name}'. It may take a few minutes to become active. Result: {result}")
        print(f"Successfully initiated creation of MongoDB vector index '{index_name}'. It may take a few minutes to become active.")
    except Exception as e:
        logger.error(f"Error creating MongoDB vector index: {e}", exc_info=True)


def query_clinical_trials_from_mongodb(query_text: str, limit: int = 5) -> List[ClinicalTrialStudy]:
    """
    Performs a vector search on the MongoDB 'clinical_trials' collection.
    """
    mongo_client = None
    results_to_return: List[ClinicalTrialStudy] = []
    try:
        mongo_client = connect_to_mongodb()
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_COLLECTION_NAME]

        query_embedding_response = embedding_model.get_embeddings([query_text])
        if not query_embedding_response:
             logger.error(f"Embedding model returned no response for query: '{query_text}'")
             return []
        query_embedding = query_embedding_response[0].values

        pipeline = [
            {
                "$vectorSearch": {
                    "index": MONGODB_VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "NCT Number": "$nct_number",
                    "Study Title": "$study_title",
                    "Study Status": "$study_status",
                    "Conditions": "$conditions",
                    "Interventions": "$interventions",
                    "Sponsor": "$sponsor",
                    "Study URL": "$study_url",
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        logger.info(f"Found {len(results)} clinical trials in MongoDB for query: '{query_text[:50]}...'")
        print(f"Found {len(results)} clinical trials in MongoDB for query: '{query_text[:50]}...'")

        for res_dict in results:
            try:
                # Use by_alias=True to map from the projected field names to the Pydantic model fields
                study = ClinicalTrialStudy.model_validate(res_dict, from_attributes=True)
                results_to_return.append(study)
            except Exception as e_pydantic:
                logger.error(f"Pydantic validation error for clinical trial data {res_dict}: {e_pydantic}")

        return results_to_return
    except Exception as e:
        logger.error(f"Error querying clinical trials from MongoDB: {e}", exc_info=True)
        return []
    finally:
        if mongo_client:
            mongo_client.close()


# Place this function within your ingest_clinical_trials.py file,
# perhaps after the query_clinical_trials_from_mongodb function.

def ingest_clinical_trial_record(
    nct_number: str,
    study_title: str,
    study_status: str,
    conditions: str,
    interventions: str,
    sponsor: str,
    study_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processes and ingests a single new clinical trial record into the MongoDB database.
    This tool is designed to be called by an agent that has identified a new trial,
    for example from a web search result.

    Args:
        nct_number (str): The unique identifier for the trial (e.g., "NCT01234567").
        study_title (str): The title of the clinical trial.
        study_status (str): The current status of the trial (e.g., "Recruiting", "Completed").
        conditions (str): The health conditions or diseases being studied.
        interventions (str): The interventions being tested (e.g., drug names, procedures).
        sponsor (str): The primary sponsor of the trial.
        study_url (Optional[str]): The URL where the trial information was found.

    Returns:
        Dict[str, Any]: A dictionary indicating the status of the ingestion,
                        e.g., {"status": "success", "nct_number": "..."}
                        or {"status": "error", "message": "..."}.
    """
    logger.info(f"Attempting to ingest new clinical trial: {nct_number} - {study_title[:50]}...")

    # 1. Construct the rich text for embedding, just like in the bulk process
    rich_text = (
        f"Trial Title: {study_title}. "
        f"Conditions Treated: {conditions}. "
        f"Interventions Used: {interventions}."
    )

    # 2. Generate the embedding for the new record
    try:
        embedding_response = embedding_model.get_embeddings([rich_text])
        if not embedding_response:
            raise ValueError("Embedding model returned no response.")
        embedding_values = embedding_response[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding for new trial '{nct_number}': {e}", exc_info=True)
        return {"status": "error", "message": "Embedding generation failed."}

    # 3. Create the document to be inserted into MongoDB
    # Ensure keys match the snake_case schema used in the bulk ingestion
    trial_doc = {
        "nct_number": nct_number,
        "study_title": study_title,
        "study_status": study_status,
        "conditions": conditions,
        "interventions": interventions,
        "sponsor": sponsor,
        "study_url": study_url,
        "rich_text_for_embedding": rich_text,
        "embedding": embedding_values,
        "ingested_at": datetime.now(timezone.utc)
        # You can add other fields like 'collaborators' or 'study_type' as None or empty strings
        # if they aren't provided by the agent call, to maintain a consistent schema.
    }

    # 4. Connect to MongoDB and insert the document
    mongo_client = None
    try:
        mongo_client = connect_to_mongodb()
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_COLLECTION_NAME] # Targets the 'clinical_trials' collection

        collection.insert_one(trial_doc)

        logger.info(f"Successfully ingested new clinical trial '{nct_number}' into MongoDB.")
        return {"status": "success", "nct_number": nct_number, "message": "Clinical trial ingested successfully."}
    except Exception as e:
        logger.error(f"Failed to upload new clinical trial '{nct_number}' to MongoDB: {e}", exc_info=True)
        return {"status": "error", "message": "MongoDB upload failed."}
    finally:
        if mongo_client:
            mongo_client.close()


# --- Main Ingestion Pipeline Runner ---
def run_clinical_trials_ingestion_pipeline():
    """Runs the full data ingestion pipeline for Clinical Trials data."""
    logger.info("Starting Clinical Trials data ingestion pipeline...")
    print("Starting Clinical Trials data ingestion pipeline...")

    # 1. Load data from CSV
    try:
        df = pd.read_csv(CLINICAL_TRIALS_CSV_PATH)
        # Taking a smaller sample for faster dev/test runs
        df = df.sample(n=400, random_state=42)
        print(f"Loaded {len(df)} rows from {CLINICAL_TRIALS_CSV_PATH} for processing.")
        logger.info(f"Loaded {len(df)} rows from {CLINICAL_TRIALS_CSV_PATH}")
    except FileNotFoundError:
        print(f"Clinical Trials CSV file not found at {CLINICAL_TRIALS_CSV_PATH}.")
        logger.error(f"Clinical Trials CSV file not found at {CLINICAL_TRIALS_CSV_PATH}.")
        return
    except Exception as e:
        logger.error(f"Error loading Clinical Trials CSV: {e}", exc_info=True)
        return

    # 2. Process data (create rich_text, generate embeddings)
    processed_df = process_clinical_trials_data(df)

    if processed_df.empty:
        logger.warning("No data to upload after processing.")
        return

    # 3. Connect to MongoDB
    mongo_client = None
    try:
        mongo_client = connect_to_mongodb()

        # 4. Upload to MongoDB
        upload_to_mongodb(processed_df, mongo_client, MONGODB_COLLECTION_NAME)

        # 5. Ensure MongoDB vector index exists
        create_mongodb_vector_index(mongo_client, MONGODB_COLLECTION_NAME, MONGODB_VECTOR_INDEX_NAME)

    finally:
        if mongo_client:
            mongo_client.close()

    logger.info("Clinical Trials data ingestion pipeline completed.")


if __name__ == "__main__":
    # To run the ingestion pipeline:
    # Ensure all necessary environment variables and auth are set up.
    #run_clinical_trials_ingestion_pipeline()

    # --- Example query after ingestion ---
    print("\n--- Running example query after ingestion ---")
    test_query = "What are the latest clinical trials for lymphoma using CAR T-cell therapy?"
    results = query_clinical_trials_from_mongodb(test_query, limit=3)
    if results:
        for res in results:
            print(f"Title: {res.study_title}\nStatus: {res.study_status}\nScore: {res.score}\nURL: {res.study_url}\n---")
    else:
         print(f"No results found for query: {test_query}")

    # Gracefully close GCP logging resources
    if gcp_log_handler and isinstance(gcp_log_handler, CloudLoggingHandler):
        gcp_log_handler.close()
    if gcp_log_client_instance:
        # No explicit close method on the client itself, handler closure is key
        pass
    print("Clinical trials pipeline script finished.")