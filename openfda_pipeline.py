# /Users/surfiniaburger/Desktop/app/openfda_pipeline.py
import httpx
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

OPENFDA_API_BASE_URL = "https://api.fda.gov/"

# --- Pydantic Models (Optional, but good for structure) ---
class OpenFDAReaction(BaseModel):
    reactionmeddrapt: Optional[str] = None

class OpenFDAPatient(BaseModel):
    patientsex: Optional[str] = None
    reaction: Optional[List[OpenFDAReaction]] = None
    drug: Optional[List[Dict[str, Any]]] = None # Simplified for this example

class AdverseEventReport(BaseModel):
    report_id: Optional[str] = Field(None, alias="safetyreportid")
    receive_date: Optional[str] = Field(None, alias="receivedate")
    seriousness_details: Optional[str] = Field(None, alias="serious") # e.g., "1" for serious
    patient: Optional[OpenFDAPatient] = None
    # We'll construct a summary string for the LLM
    summary_for_llm: Optional[str] = None


async def query_drug_adverse_events(drug_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Queries the OpenFDA API for adverse event reports related to a specific drug name.

    Args:
        drug_name (str): The brand or generic name of the drug.
        limit (int, optional): Max number of adverse event reports to return. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each summarizing an adverse event report.
    """
    # Search for the drug name in patient.drug.medicinalproduct or patient.drug.activesubstance.activesubstancename
    # Using a more general search for simplicity here:
    search_query = f'(patient.drug.medicinalproduct:"{drug_name}" OR patient.drug.openfda.generic_name:"{drug_name}" OR patient.drug.openfda.brand_name:"{drug_name}")'
    
    params = {
        "search": search_query,
        "limit": limit,
        "sort": "receivedate:desc" # Get most recent reports
    }
    endpoint = "drug/event.json"
    logger.info(f"Querying OpenFDA Adverse Events with params: {params}")

    report_summaries = []
    try:
        async with httpx.AsyncClient(base_url=OPENFDA_API_BASE_URL, timeout=20.0) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            logger.info(f"Received {len(results)} adverse event reports from OpenFDA for drug: '{drug_name}'.")

            for report_json in results:
                try:
                    patient_data = report_json.get("patient", {})
                    reactions = [r.get("reactionmeddrapt", "Unknown reaction") for r in patient_data.get("reaction", [])[:2]] # Max 2 reactions for summary
                    drugs_involved_raw = patient_data.get("drug", [])
                    # Simplified drug listing for summary
                    drugs_involved = [d.get("medicinalproduct", "Unknown drug") for d in drugs_involved_raw[:2]]

                    summary = (
                        f"Report ID: {report_json.get('safetyreportid', 'N/A')}, "
                        f"Received: {report_json.get('receivedate', 'N/A')}, "
                        f"Serious: {'Yes' if report_json.get('serious') == '1' else 'No'}. "
                        f"Patient Sex: {patient_data.get('patientsex', 'Unknown')}. "
                        f"Reactions: {', '.join(reactions) if reactions else 'N/A'}. "
                        f"Drugs (first few): {', '.join(drugs_involved) if drugs_involved else 'N/A'}."
                    )
                    report_summaries.append({"report_summary": summary})
                except Exception as e_parse:
                    logger.error(f"Error parsing OpenFDA report {report_json.get('safetyreportid', 'UNKNOWN_ID')}: {e_parse}")
            return report_summaries
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from OpenFDA API: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"General error querying OpenFDA: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    async def main_test():
        # test_drug = "LISINOPRIL"
        test_drug = "Ozempic"
        print(f"Testing OpenFDA adverse event query for: {test_drug}")
        events = await query_drug_adverse_events(test_drug, limit=2)
        if events:
            for event_summary in events:
                print(event_summary)
        else:
            print(f"No adverse events found for {test_drug} or error occurred.")

    asyncio.run(main_test())