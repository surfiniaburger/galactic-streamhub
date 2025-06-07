# /Users/surfiniaburger/Desktop/app/clinical_trials_pipeline.py
import httpx
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field # For potential internal validation/parsing

# Setup logger (can be more sophisticated if needed)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

CLINICAL_TRIALS_API_BASE_URL = "https://clinicaltrials.gov/api/v2/" # Official API
# CLINICAL_TRIALS_API_BASE_URL = "https://beta-ut.clinicaltrials.gov/api/v2/" # Beta API from your example

# --- Pydantic Models for structuring parts of the ClinicalTrial Data (Optional but good practice) ---
# These help in understanding and validating the structure if you parse deeply.
# For direct use by an LLM, often a summarized dictionary or string is better.

class IdentificationModule(BaseModel):
    nctId: Optional[str] = Field(None, alias="nctId")
    briefTitle: Optional[str] = Field(None, alias="briefTitle")
    officialTitle: Optional[str] = Field(None, alias="officialTitle")

class StatusModule(BaseModel):
    overallStatus: Optional[str] = Field(None, alias="overallStatus")
    startDateStruct: Optional[Dict[str, Any]] = Field(None, alias="startDateStruct")
    lastUpdatePostDateStruct: Optional[Dict[str, Any]] = Field(None, alias="lastUpdatePostDateStruct")

class ConditionsModule(BaseModel):
    conditions: Optional[List[str]] = None

class DescriptionModule(BaseModel):
    briefSummary: Optional[str] = Field(None, alias="briefSummary")

class ProtocolSection(BaseModel):
    identificationModule: Optional[IdentificationModule] = Field(None, alias="identificationModule")
    statusModule: Optional[StatusModule] = Field(None, alias="statusModule")
    conditionsModule: Optional[ConditionsModule] = Field(None, alias="conditionsModule")
    descriptionModule: Optional[DescriptionModule] = Field(None, alias="descriptionModule")

class ClinicalTrialAPIResponseStudy(BaseModel):
    """Represents a single clinical trial study as structured in the API response."""
    protocolSection: Optional[ProtocolSection] = Field(None, alias="protocolSection")
    hasResults: Optional[bool] = Field(None, alias="hasResults")


async def query_clinical_trials_data(query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Queries the ClinicalTrials.gov API for studies related to the query_text.
    Returns a list of dictionaries, where each dictionary is a summary of a clinical trial.
    This format is often easier for LLMs to consume directly.

    Args:
        query_text (str): The user's research query (e.g., "lung cancer treatment", "diabetes type 2 new drugs").
        limit (int, optional): Max number of studies to return. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each summarizing a clinical trial.
                              Keys typically include "id", "title", "status", "summary", "conditions", "last_updated".
    """
    # Construct query parameters.
    # Using 'query.term' for a general search. 'query.cond' could be used if the text is specifically a condition.
    # The API documentation you provided shows many specific query fields (query.cond, query.intr, etc.)
    # For a general purpose tool, 'query.term' or a combination might be best.
    # An LLM could also be prompted to break down a complex query into these specific fields if needed.
    params = {
        "query.term": query_text,
        "format": "json",
        "pageSize": limit,
        "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,BriefSummary,Condition,StartDate,LastUpdatePostDate", # Key fields
        "sort": "LastUpdatePostDate:desc" # Get most recently updated
    }
    logger.info(f"Querying ClinicalTrials.gov with params: {params}")
    
    summaries = []
    try:
        async with httpx.AsyncClient(base_url=CLINICAL_TRIALS_API_BASE_URL, timeout=20.0) as client:
            response = await client.get("studies", params=params)
            response.raise_for_status() 
            data = response.json()
            
            studies_data = data.get("studies", [])
            logger.info(f"Received {len(studies_data)} studies from ClinicalTrials.gov API for query: '{query_text}'.")

            for study_json in studies_data:
                try:
                    # Parse into Pydantic model for validation and easier access
                    study_api_obj = ClinicalTrialAPIResponseStudy.model_validate(study_json)
                    
                    ps = study_api_obj.protocolSection
                    summary_dict = {
                        "id": ps.identificationModule.nctId if ps and ps.identificationModule else "N/A",
                        "title": (ps.identificationModule.briefTitle or 
                                  ps.identificationModule.officialTitle if ps and ps.identificationModule else "N/A"),
                        "status": ps.statusModule.overallStatus if ps and ps.statusModule else "N/A",
                        "summary": (ps.descriptionModule.briefSummary[:500] + "..." 
                                    if ps and ps.descriptionModule and ps.descriptionModule.briefSummary and len(ps.descriptionModule.briefSummary) > 500 
                                    else (ps.descriptionModule.briefSummary if ps and ps.descriptionModule else "N/A")),
                        "conditions": (", ".join(ps.conditionsModule.conditions)
                                       if ps and ps.conditionsModule and ps.conditionsModule.conditions 
                                       else "N/A"),
                        "last_updated": (ps.statusModule.lastUpdatePostDateStruct.get('date')
                                         if ps and ps.statusModule and ps.statusModule.lastUpdatePostDateStruct 
                                         else "N/A")
                    }
                    summaries.append(summary_dict)
                except Exception as e_parse:
                    nct_id = study_json.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "UNKNOWN_ID")
                    logger.error(f"Error parsing study {nct_id}: {e_parse}")
            return summaries
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from ClinicalTrials.gov API: {e.response.status_code} - {e.response.text}")
        return []
    except httpx.TimeoutException:
        logger.error(f"Timeout error querying ClinicalTrials.gov API for query: {query_text}")
        return []
    except Exception as e:
        logger.error(f"General error querying ClinicalTrials.gov: {e}", exc_info=True)
        return []

# --- Main Ingestion Pipeline (for ClinicalTrials.gov, this would be different) ---
# Unlike PubMed, we are not bulk-downloading and embedding ClinicalTrials.gov data.
# We are querying its API live. So, a "run_ingestion_pipeline" for this source
# isn't applicable in the same way. We might have a function to test the API connection
# or to fetch a sample if needed for development.

async def test_clinical_trials_api_connection():
    """Tests the connection and a sample query to the ClinicalTrials.gov API."""
    logger.info("Testing ClinicalTrials.gov API connection...")
    sample_query = "diabetes"
    results = await query_clinical_trials_data(sample_query, limit=1)
    if results:
        logger.info(f"Successfully fetched {len(results)} trial(s) for query '{sample_query}'. Sample: {results[0]}")
        return True
    else:
        logger.warning(f"Could not fetch trials for query '{sample_query}'. API might be down or query returned no results.")
        return False

if __name__ == "__main__":
    # To test this module directly:
    # asyncio.run(test_clinical_trials_api_connection())
    pass
