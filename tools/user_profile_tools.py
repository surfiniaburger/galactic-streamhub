# In tools/user_profile_tools.py
from pymongo.mongo_client import MongoClient
from pubmed_pipeline import connect_to_mongodb # Re-use your existing connection logic

MONGODB_DATABASE_NAME = "biomed_research"
MONGODB_PROFILES_COLLECTION = "user_profiles"

def read_user_profile(user_id: str) -> dict | None:
    """
    Reads a user's profile from MongoDB based on their user_id.
    
    Args:
        user_id: The unique identifier for the user (e.g., Firebase UID).
        
    Returns:
        A dictionary representing the user's profile, or None if not found.
    """
    mongo_client = None
    try:
        mongo_client = connect_to_mongodb()
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_PROFILES_COLLECTION]
        profile = collection.find_one({"user_id": user_id})
        if profile:
            # MongoDB's _id is not JSON serializable, so we remove it
            profile.pop('_id', None)
        return profile
    finally:
        if mongo_client:
            mongo_client.close()

def write_user_profile(user_id: str, profile_data: dict):
    """
    Writes (updates or inserts) a user's profile to MongoDB.
    
    Args:
        user_id: The unique identifier for the user.
        profile_data: A dictionary containing the user's profile data.
    """
    mongo_client = None
    try:
        mongo_client = connect_to_mongodb()
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_PROFILES_COLLECTION]
        
        # Use update_one with upsert=True to create the document if it doesn't exist
        collection.update_one(
            {"user_id": user_id},
            {"$set": profile_data},
            upsert=True
        )
    finally:
        if mongo_client:
            mongo_client.close()