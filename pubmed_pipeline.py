import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Any

import pandas as pd
import uuid # For generating unique IDs for new articles
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager, bigquery
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi # <-- Add this import
from pymongo.operations import SearchIndexModel
from vertexai.language_models import TextEmbeddingModel # type: ignore
from pydantic import BaseModel, Field # HttpUrl removed

# --- Configuration ---
# GCP Project ID for Vertex AI, Secret Manager, BigQuery
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "silver-455021")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# MongoDB Configuration
MONGODB_SECRET_ID = "mongodb-uri" # Assumes your MongoDB URI is stored here
MONGODB_DATABASE_NAME = "biomed_research"
MONGODB_COLLECTION_NAME = "pubmed_articles"
MONGODB_VECTOR_INDEX_NAME = "pubmed_vector_index"

# BigQuery Configuration
BIGQUERY_DATASET_ID = "pubmed_data"
BIGQUERY_TABLE_ID = "articles_archive"

# PubMed Data Source (adjust path as needed)
PUBMED_CSV_PATH = "data/pubmed_rct_200k.csv" # Example path

# Embedding Model
EMBEDDING_MODEL_NAME = "text-embedding-005"

# --- Setup (Logging, Clients) ---

def setup_logging():
    """Sets up Google Cloud Logging."""
    gcp_logging_client = google.cloud.logging.Client(project=GCP_PROJECT_ID)
    handler = CloudLoggingHandler(gcp_logging_client)
    logger = logging.getLogger('pubmed_pipeline')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers(): # Avoid adding multiple handlers
        logger.addHandler(handler)
    return logger, handler, gcp_logging_client

logger, gcp_log_handler, gcp_log_client_instance = setup_logging()

try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to initialize BigQuery client or Embedding Model: {e}", exc_info=True)
    raise

class PubMedArticle(BaseModel):
    article_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[str] = None # Or List[str] if you parse it
    journal: Optional[str] = None
    source_url: Optional[str] = None # Changed from HttpUrl to str
    publication_year: Optional[int] = None
    # rich_text_for_embedding: Optional[str] = None # Usually not needed by LLM directly
    score: Optional[float] = None


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
        uri = get_secret(MONGODB_SECRET_ID)
        # Explicitly use certifi's CA bundle for TLS connections
        client = MongoClient(
            uri,
            server_api=ServerApi('1'),
            tlsCAFile=certifi.where()
        )
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        return client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        raise

# --- BigQuery Schema and Table Creation ---
PUBMED_BQ_SCHEMA = [
    bigquery.SchemaField("article_id", "STRING", mode="REQUIRED"), # Unique ID for the article
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("abstract", "STRING"),
    bigquery.SchemaField("authors", "STRING"), # Could be a REPEATED field if parsed
    bigquery.SchemaField("journal", "STRING"),
    bigquery.SchemaField("source_url", "STRING"), # New field for web articles
    bigquery.SchemaField("publication_year", "INTEGER"),
    bigquery.SchemaField("rich_text_for_embedding", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED")
]

def create_bq_pubmed_table_if_not_exists():
    """Creates the BigQuery table for PubMed articles if it doesn't exist."""
    dataset_ref = bq_client.dataset(BIGQUERY_DATASET_ID)
    try:
        bq_client.create_dataset(dataset_ref, exists_ok=True)
        logger.info(f"BigQuery dataset '{BIGQUERY_DATASET_ID}' ensured.")
    except Exception as e:
        logger.error(f"Failed to create/ensure BigQuery dataset '{BIGQUERY_DATASET_ID}': {e}", exc_info=True)
        raise

    table_ref = dataset_ref.table(BIGQUERY_TABLE_ID)
    table = bigquery.Table(table_ref, schema=PUBMED_BQ_SCHEMA)
    try:
        bq_client.create_table(table, exists_ok=True)
        logger.info(f"BigQuery table '{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}' ensured.")
    except Exception as e:
        logger.error(f"Failed to create/ensure BigQuery table '{BIGQUERY_TABLE_ID}': {e}", exc_info=True)
        raise

# --- Data Processing and Embedding ---
def process_pubmed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes PubMed DataFrame (expected to be sentence-per-row with 'abstract_id', 
    'abstract_text', 'line_number', 'target') to reconstruct full abstracts, 
    create rich_text for embedding, and generate embeddings.
    """
    logger.info(f"Processing {df['abstract_id'].nunique()} unique PubMed abstracts from {len(df)} lines...")

    # Group by abstract_id and reconstruct abstracts
    reconstructed_articles = []
    for name, group in df.groupby('abstract_id'):
        sorted_lines = group.sort_values('line_number')
        
        full_abstract_parts = []
        structured_abstract_parts = []

        # Attempt to find a title - often the first line of BACKGROUND or OBJECTIVE
        # This is heuristic and might need refinement based on dataset specifics.
        title_candidate = ""
        if not sorted_lines.empty:
            first_line_text = sorted_lines.iloc[0]['abstract_text']
            # A simple heuristic: if it's short and doesn't end with a period, it might be a title.
            # Or, if the dataset has a convention for titles within the abstract text.
            # For "PubMed 200k RCT", titles are not explicitly in this CSV.
            # We will use the first sentence of the background as a proxy or leave it more generic.
            title_candidate = f"Research Abstract ID: {name}" # Default title
            # You could try to find the first sentence of BACKGROUND if it exists
            background_lines = sorted_lines[sorted_lines['target'] == 'BACKGROUND']
            if not background_lines.empty:
                title_candidate = background_lines.iloc[0]['abstract_text']

        for _, row in sorted_lines.iterrows():
            full_abstract_parts.append(row['abstract_text'])
            structured_abstract_parts.append(f"{row['target']}: {row['abstract_text']}")
            
        reconstructed_articles.append({
            'article_id': str(name),
            'title': title_candidate, # Heuristic title
            'abstract': " ".join(full_abstract_parts),
            'rich_text_for_embedding': " ".join(structured_abstract_parts),
            'authors': "", # Not available in this CSV format
            'journal': "", # Not available in this CSV format
            'source_url': None, # Not applicable for CSV ingested data
            'publication_year': 0 # Not available in this CSV format
        })

    if not reconstructed_articles:
        logger.warning("No articles were reconstructed. Check CSV format and content.")
        return pd.DataFrame()

    processed_df = pd.DataFrame(reconstructed_articles)

    logger.info("Generating embeddings...")
    # Batch embeddings for efficiency
    embeddings_list = []
    batch_size = 50 # Vertex AI embedding limits can be around 250, but smaller batches are safer
    for i in range(0, len(processed_df), batch_size):
        batch_texts = processed_df['rich_text_for_embedding'][i:i+batch_size].tolist()
        try:
            batch_embeddings = embedding_model.get_embeddings(batch_texts)
            embeddings_list.extend([emb.values for emb in batch_embeddings])
        except Exception as e:
            logger.error(f"Error generating embeddings for batch starting at index {i}: {e}", exc_info=True)
            # Add empty embeddings for failed batch to maintain DataFrame structure
            embeddings_list.extend([[] for _ in range(len(batch_texts))])

    processed_df['embedding'] = embeddings_list
    processed_df['ingested_at'] = datetime.now(timezone.utc)
    
    # Filter out rows where embedding failed
    processed_df = processed_df[processed_df['embedding'].apply(lambda x: bool(x))]
    logger.info(f"Finished processing. {len(processed_df)} articles have embeddings.")
    return processed_df

def upload_to_bigquery(df: pd.DataFrame):
    """Uploads DataFrame to BigQuery."""
    if df.empty:
        logger.info("No data to upload to BigQuery.")
        return
    
    # Ensure schema alignment (select only columns in BQ_SCHEMA)
    bq_column_names = [field.name for field in PUBMED_BQ_SCHEMA]
    df_for_bq = df[[col for col in bq_column_names if col in df.columns]].copy()

    job_config = bigquery.LoadJobConfig(schema=PUBMED_BQ_SCHEMA, write_disposition="WRITE_APPEND")
    try:
        job = bq_client.load_table_from_dataframe(df_for_bq, f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}", job_config=job_config)
        job.result()
        logger.info(f"Uploaded {len(df_for_bq)} rows to BigQuery table '{BIGQUERY_TABLE_ID}'.")
    except Exception as e:
        logger.error(f"Failed to upload data to BigQuery: {e}", exc_info=True)

def upload_to_mongodb(df: pd.DataFrame, mongo_client: MongoClient):
    """Uploads DataFrame to MongoDB."""
    if df.empty:
        logger.info("No data to upload to MongoDB.")
        return

    db = mongo_client[MONGODB_DATABASE_NAME]
    collection = db[MONGODB_COLLECTION_NAME]
    
    # Prepare records for MongoDB (ensure embedding is a list)
    records = df[['article_id', 'title', 'abstract', 'authors', 'journal', 'source_url', 'publication_year', 'rich_text_for_embedding', 'embedding', 'ingested_at']].to_dict("records")
    
    try:
        collection.insert_many(records)
        logger.info(f"Inserted {len(records)} records into MongoDB collection '{MONGODB_COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to insert data into MongoDB: {e}", exc_info=True)

def create_mongodb_vector_index(mongo_client: MongoClient):
    """Creates the Atlas Vector Search index if it doesn't exist."""
    db = mongo_client[MONGODB_DATABASE_NAME]
    collection = db[MONGODB_COLLECTION_NAME]
    
    try:
        existing_indexes = list(collection.list_search_indexes()) # list() to exhaust cursor
        if any(idx.get('name') == MONGODB_VECTOR_INDEX_NAME for idx in existing_indexes):
            logger.info(f"MongoDB vector index '{MONGODB_VECTOR_INDEX_NAME}' already exists.")
            return
    except Exception as e:
        logger.warning(f"Could not list existing search indexes for {MONGODB_COLLECTION_NAME}: {e}. Attempting to create index anyway.")

    index_definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": 768, # Dimension of text-embedding-005
                    "similarity": "cosine" # Or dotProduct / euclidean
                }
            }
        }
    }
    search_index_model = SearchIndexModel(definition=index_definition, name=MONGODB_VECTOR_INDEX_NAME)
    try:
        collection.create_search_indexes([search_index_model])
        logger.info(f"Successfully created MongoDB vector index '{MONGODB_VECTOR_INDEX_NAME}'. It may take a few minutes to become active.")
    except Exception as e:
        logger.error(f"Error creating MongoDB vector index: {e}", exc_info=True)

# --- RAG Query Function ---
def query_pubmed_articles(query_text: str, limit: int = 5) -> List[PubMedArticle]:
    """
    Queries a specialized knowledge base of PubMed scientific abstracts for information
    related to biomedical research questions and returns the most relevant articles.

    Args:
        query_text (str): The user's biomedical question or search query.
        limit (int, optional): The maximum number of articles to return. Defaults to 5.

    Returns:
        List[PubMedArticle]: A list of PubMedArticle objects, where each object
                             represents a retrieved PubMed article.
                             Returns an empty list if no relevant articles are found or an error occurs.
    """
    mongo_client = None
    results_to_return: List[PubMedArticle] = []

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
                    "numCandidates": 100, # Number of candidates to consider
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 0, # Exclude MongoDB's default _id
                    "article_id": 1,
                    "title": 1,
                    "abstract": 1,
                    "authors": 1,
                    "journal": 1,
                    "source_url": 1,
                    "publication_year": 1,
                    #"rich_text_for_embedding": 1, # For context if needed
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        logger.info(f"Found {len(results)} articles for query: '{query_text[:50]}...'")
        for res_dict in results:
            try:
                article = PubMedArticle(**res_dict)
                results_to_return.append(article)
            except Exception as e_pydantic:
                logger.error(f"Pydantic validation error for article data {res_dict}: {e_pydantic}")
        return results_to_return
    except Exception as e:
        logger.error(f"Error querying PubMed articles from MongoDB: {e}", exc_info=True)
        return []
    finally:
        if mongo_client:
            mongo_client.close()

# --- Function to Ingest a Single New Article (e.g., from web search) ---
def ingest_single_article_data(
    title: str, 
    abstract_text: str, 
    source_url: Optional[str] = None,
    authors: Optional[str] = "", # Default to empty string
    journal: Optional[str] = "", # Default to empty string
    publication_year: Optional[int] = None # Default to None, could try to parse from text
) -> Dict[str, Any]:
    """
    Processes and ingests a single new article into BigQuery and MongoDB.
    This is typically used for articles found via web search that are not yet in the local database.

    Args:
        title (str): The title of the new article.
        abstract_text (str): The full abstract or main text content of the new article.
        source_url (Optional[str]): The URL where the new article was found. Defaults to None.
        authors (Optional[str]): A string listing the authors of the article. Defaults to an empty string.
        journal (Optional[str]): The name of the journal or publication source. Defaults to an empty string.
        publication_year (Optional[int]): The year the article was published. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary indicating the status of the ingestion,
                        e.g., {"status": "success", "article_id": "...", "message": "..."}
                        or {"status": "error", "message": "..."}.
    """
    logger.info(f"Attempting to ingest new article: {title[:50]}...")
    article_id = str(uuid.uuid4()) # Generate a unique ID

    # Construct rich_text for embedding
    # For web articles, 'target' labels aren't available, so we create a simpler rich_text
    rich_text = f"Title: {title}. Abstract: {abstract_text}."
    if authors:
        rich_text += f" Authors: {authors}."
    if journal:
        rich_text += f" Journal: {journal}."
    if source_url:
        rich_text += f" Source: {source_url}."

    try:
        embedding_values = embedding_model.get_embeddings([rich_text])[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding for new article '{title[:50]}...': {e}", exc_info=True)
        return {"status": "error", "message": "Embedding generation failed."}

    article_doc = {
        "article_id": article_id,
        "title": title,
        "abstract": abstract_text,
        "authors": authors or "",
        "journal": journal or "",
        "source_url": source_url,
        "publication_year": publication_year, # Might be None
        "rich_text_for_embedding": rich_text,
        "embedding": embedding_values,
        "ingested_at": datetime.now(timezone.utc)
    }

    # Create a single-row DataFrame for uploading
    df_single = pd.DataFrame([article_doc])

    # Upload to BigQuery
    try:
        create_bq_pubmed_table_if_not_exists() # Ensure table exists
        upload_to_bigquery(df_single)
    except Exception as e:
        logger.error(f"Failed to upload new article '{title[:50]}...' to BigQuery: {e}", exc_info=True)
        # Continue to try MongoDB upload

    # Upload to MongoDB
    mongo_client = None
    try:
        mongo_client = connect_to_mongodb()
        upload_to_mongodb(df_single, mongo_client)
        create_mongodb_vector_index(mongo_client) # Ensure index exists
        logger.info(f"Successfully ingested new article '{title[:50]}...' into MongoDB.")
        return {"status": "success", "article_id": article_id, "message": "Article ingested successfully."}
    except Exception as e:
        logger.error(f"Failed to upload new article '{title[:50]}...' to MongoDB: {e}", exc_info=True)
        return {"status": "error", "message": "MongoDB upload failed."}
    finally:
        if mongo_client:
            mongo_client.close()

# --- Main Ingestion Pipeline ---
def run_ingestion_pipeline():
    """Runs the full data ingestion pipeline for PubMed data."""
    logger.info("Starting PubMed data ingestion pipeline...")
    
    # 1. Ensure BigQuery table exists
    create_bq_pubmed_table_if_not_exists()
    
    # 2. Load data from CSV
    try:
        df = pd.read_csv(PUBMED_CSV_PATH)
        # Optional: Select a subset for testing
        df = df.sample(n=1000, random_state=42) 
        logger.info(f"Loaded {len(df)} rows from {PUBMED_CSV_PATH}")
    except FileNotFoundError:
        logger.error(f"PubMed CSV file not found at {PUBMED_CSV_PATH}. Please check the path.")
        return
    except Exception as e:
        logger.error(f"Error loading PubMed CSV: {e}", exc_info=True)
        return
        
    # 3. Process data (create rich_text, generate embeddings)
    processed_df = process_pubmed_data(df)
    
    if processed_df.empty:
        logger.warning("No data to upload after processing.")
        return

    # 4. Upload to BigQuery
    upload_to_bigquery(processed_df)
    
    # 5. Connect to MongoDB and upload
    mongo_client = None
    try:
        mongo_client = connect_to_mongodb()
        upload_to_mongodb(processed_df, mongo_client)
        
        # 6. Ensure MongoDB vector index exists
        create_mongodb_vector_index(mongo_client)
    finally:
        if mongo_client:
            mongo_client.close()
            
    logger.info("PubMed data ingestion pipeline completed.")

if __name__ == "__main__":
    # To run the ingestion:
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set for GCP auth.
    # Ensure your .env file has GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.
    # Ensure 'mongodb-uri' secret is set up in Secret Manager.
    # Ensure 'data/pubmed_rct_200k.csv' (or your CSV path) exists.
    # run_ingestion_pipeline()

    # Example query after ingestion:
    test_query = "What are the latest treatments for Alzheimer's disease based on clinical trials?"
    results = query_pubmed_articles(test_query, limit=3)
    # print(f"\n--- Query Results for: '{test_query}' ---")
    if results:
        for res in results:
            # Pydantic models are iterated directly
            print(f"Title: {res.title}\nScore: {res.score}\nAbstract: {res.abstract[:200] if res.abstract else ''}...\n---")
    else:
         print(f"No results found for query: {test_query}")
    
    # Example of ingesting a single new article (for testing the new function)
    # new_article_ingestion_result = ingest_single_article_data(
    #     title="A Fictional Breakthrough in AI Wellness",
    #     abstract_text="This paper discusses a novel approach to AI-driven wellness programs, showing remarkable improvements in simulated digital beings. Further research is needed to apply these findings to humans.",
    #     source_url="http://example.com/ai-wellness-breakthrough"
    # )
    # print(f"Ingestion result for new article: {new_article_ingestion_result}")

    # Gracefully close GCP logging resources
    if gcp_log_handler:
        gcp_log_handler.close()
    if gcp_log_client_instance:
        gcp_log_client_instance.close()
    print("PubMed pipeline script finished.")