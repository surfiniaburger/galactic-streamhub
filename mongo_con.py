import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def setup_logging():
    """
    Set up Google Cloud Logging
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Instantiate the Cloud Logging client
    gcp_logging_client = google.cloud.logging.Client()
    
    # Configure the Cloud Logging handler
    handler = CloudLoggingHandler(gcp_logging_client)
    
    # Set up the logger
    logger = logging.getLogger('mongodb_connection')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger, handler, gcp_logging_client

def get_secret(project_id, secret_id, version_id="latest", logger=None):
    """
    Retrieve secret from Google Cloud Secret Manager.
    
    Args:
        project_id (str): Google Cloud project ID
        secret_id (str): Name of the secret
        version_id (str): Version of the secret (default: "latest")
        logger (logging.Logger): Logger instance
    
    Returns:
        str: Secret value
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        
        if logger:
            logger.info(f"Attempting to access secret: {secret_id}")
            
        response = client.access_secret_version(request={"name": name})
        
        if logger:
            logger.info(f"Successfully retrieved secret: {secret_id}")
            
        return response.payload.data.decode("UTF-8")
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to retrieve secret {secret_id}: {str(e)}")
        raise

def connect_to_mongodb():
    """
    Connect to MongoDB using URI from Secret Manager
    
    Returns:
        MongoClient: MongoDB client instance
    """
    # Set up logging and get handler and client for proper shutdown
    logger, cloud_handler, gcp_log_client = setup_logging()
    
    try:
        # Replace with your Google Cloud project ID
        project_id = "silver-455021"
        # Replace with your secret name in Secret Manager
        secret_id = "mongodb-uri"
        
        logger.info("Starting MongoDB connection process")
        
        # Get MongoDB URI from Secret Manager
        uri = get_secret(project_id, secret_id, logger=logger)
        
        logger.info("Retrieved MongoDB URI from Secret Manager")
        
        # Create MongoDB client
        client = MongoClient(uri, server_api=ServerApi('1'))
        
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        print("Successfully connected to MongoDB!")

        return client, logger, cloud_handler, gcp_log_client
        
    except Exception as e:
        # Logger might be available here to log the error to GCP
        error_message = f"Error connecting to MongoDB: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Connect to MongoDB
        mongo_client, main_logger, main_cloud_handler, main_gcp_logging_client = connect_to_mongodb()
        # You can use mongo_client here if needed for further operations
        if main_logger:
            main_logger.info("MongoDB connection process completed in __main__.")
            
    except Exception as e:
        print(f"Failed to connect: {e}")
        # Ensure logging components are still cleaned up if partially initialized
        # and an error occurs later in the try block (though here it's mostly connect_to_mongodb)
        # This specific assignment might not be necessary if connect_to_mongodb fails early,
        # as they would be None.
        main_logger = getattr(e, '__self__', {}).get('logger', None) if hasattr(e, '__self__') else None # Heuristic
        main_cloud_handler = None # Difficult to retrieve if connect_to_mongodb fails during setup_logging
        main_gcp_logging_client = None

    finally:
        if 'main_logger' in locals() and main_logger and 'main_cloud_handler' in locals() and main_cloud_handler:
            main_logger.info("Application shutting down. Closing Google Cloud logging resources.")
        
        if 'main_cloud_handler' in locals() and main_cloud_handler:
            main_cloud_handler.close() # This will also flush the transport
            
        if 'main_gcp_logging_client' in locals() and main_gcp_logging_client:
            main_gcp_logging_client.close()

        print("Script finished.")