import os
import requests
import logging
import csv
from io import StringIO
from tqdm import tqdm
import zipfile

# --- Configuration ---
COLLECTION_NAME = 'LIDC-IDRI'
DOWNLOAD_PATH = os.path.join('data', 'tcia_lidc_idri')

# NEW: Control the number of image series to download for the demo
SAMPLE_SIZE = 10  # Let's download just 10 series to demonstrate the flow.

TCIA_API_BASE_URL = 'https://services.cancerimagingarchive.net/nbia-api/services/v1'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_tcia_sample(collection_name: str, download_path: str, sample_size: int):
    """
    Downloads a SMALL SAMPLE of an image collection from The Cancer Imaging Archive (TCIA).

    This function is designed for demonstration purposes to avoid downloading massive datasets.
    It fetches a manifest, then downloads and unzips only the specified number of image series.

    Args:
        collection_name (str): The official name of the TCIA collection.
        download_path (str): The local directory to save the downloaded files.
        sample_size (int): The number of image series to download.
    """
    logging.info(f"Starting SAMPLE download for TCIA collection: '{collection_name}'")
    logging.info(f"Target sample size: {sample_size} series.")

    # --- 1. Create Destination Directory ---
    os.makedirs(download_path, exist_ok=True)
    logging.info(f"Data will be saved in: {os.path.abspath(download_path)}")

    # --- 2. Fetch the Manifest of All Image Series ---
    manifest_file = os.path.join(download_path, 'manifest.csv')
    if not os.path.exists(manifest_file):
        logging.info("Manifest file not found. Fetching from TCIA...")
        try:
            params = {'Collection': collection_name, 'format': 'csv'}
            response = requests.get(f'{TCIA_API_BASE_URL}/getSeries', params=params)
            response.raise_for_status()
            with open(manifest_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logging.info(f"Successfully downloaded and saved manifest to {manifest_file}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch manifest from TCIA: {e}")
            return
    else:
        logging.info("Found existing manifest file.")

    # --- 3. Parse the Manifest and Get the Sample UIDs ---
    try:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Take only the first 'sample_size' UIDs from the manifest
            series_to_download = [row['SeriesInstanceUID'] for i, row in enumerate(reader) if i < sample_size]
    except (IOError, KeyError) as e:
        logging.error(f"Could not read or parse the manifest file. Error: {e}")
        return

    total_series = len(series_to_download)
    if total_series == 0:
        logging.error("No series found in the manifest file. Cannot proceed.")
        return
        
    logging.info(f"Will download a sample of {total_series} image series.")

    # --- 4. Download and Unzip Each Image Series in the Sample ---
    for i, series_uid in enumerate(series_to_download):
        zip_filename = f"{series_uid}.zip"
        zip_filepath = os.path.join(download_path, zip_filename)
        
        logging.info(f"--- Processing Series {i+1}/{total_series} (UID: {series_uid}) ---")
        
        # Download step
        if not os.path.exists(zip_filepath):
            try:
                params = {'SeriesInstanceUID': series_uid}
                response = requests.get(f'{TCIA_API_BASE_URL}/getImage', params=params, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                with open(zip_filepath, 'wb') as f, tqdm(
                    desc=zip_filename, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
                logging.info(f"Successfully downloaded '{zip_filename}'.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download series {series_uid}. Error: {e}. Skipping.")
                continue
        else:
            logging.info(f"Zip file '{zip_filename}' already exists. Skipping download.")

        # Unzip step
        try:
            logging.info(f"Extracting '{zip_filename}'...")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # Create a sub-directory for each series to keep images organized
                extract_path = os.path.join(download_path, series_uid)
                os.makedirs(extract_path, exist_ok=True)
                zip_ref.extractall(extract_path)
            logging.info(f"Successfully extracted to '{extract_path}'.")
            
            # Optional: Clean up zip file after extraction
            os.remove(zip_filepath)
            logging.info(f"Removed zip archive '{zip_filename}'.")

        except zipfile.BadZipFile:
            logging.error(f"Error: '{zip_filename}' is not a valid zip file. It may be corrupted.")
        except Exception as e:
            logging.error(f"An error occurred during extraction of '{zip_filename}': {e}")

    logging.info("Sample dataset acquisition process finished.")

if __name__ == '__main__':
    download_tcia_sample(
        collection_name=COLLECTION_NAME,
        download_path=DOWNLOAD_PATH,
        sample_size=SAMPLE_SIZE
    )