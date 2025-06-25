import polars as pl
import json
from pathlib import Path
import logging
import multiprocessing
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

JSON_DIR = Path("processed/jsons_raw/raw/")
OUTPUT_PATH = Path("processed/json_features.parquet")

def extract_features_from_json(file_path: Path) -> list:
    """Extracts relevant features from a single JSON file."""
    features = []
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            ranker_id = data.get("ranker_id")
            if not ranker_id:
                return []
            
            data_section = data.get("data", {})
            flight_options = data_section.get("$values", []) if isinstance(data_section, dict) else data_section

            for flight_option in flight_options:
                if not isinstance(flight_option, dict):
                    logger.warning(f"Skipping non-dict flight_option in {file_path}")
                    continue
                
                flight_id = flight_option.get("id")
                for pricing in flight_option.get("pricings", []):
                    if not isinstance(pricing, dict):
                        logger.warning(f"Skipping non-dict pricing in {file_path}")
                        continue

                    pricing_id = pricing.get("id")
                    total_price_json = pricing.get("totalPrice")
                    
                    pricing_info = pricing.get("pricingInfo", [{}])
                    if not pricing_info: continue
                    
                    fares_info = pricing_info[0].get("faresInfo", [{}])
                    if not fares_info: continue

                    fare_family_key = fares_info[0].get("fareFamilyKey")
                    baggage_quantity = fares_info[0].get("baggageAllowance", {}).get("quantity")
                    
                    cancellation_fee = None
                    for rule in pricing.get("miniRules", []):
                        if isinstance(rule, dict) and rule.get("category") == 31: # Cancellation fee
                            cancellation_fee = rule.get("monetaryAmount")
                            break

                    features.append({
                        "ranker_id": ranker_id,
                        "flight_id": flight_id,
                        "pricing_id": pricing_id,
                        "total_price_json": total_price_json,
                        "fare_family_key": fare_family_key,
                        "cancellation_fee": cancellation_fee,
                        "baggage_quantity": baggage_quantity,
                    })
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    return features

def extract_with_progress(args):
    """Wrapper function for multiprocessing with progress tracking."""
    file_path, file_index, total_files = args
    
    # Log progress every 100 files or for the first/last few files
    if file_index % 100 == 0 or file_index < 5 or file_index >= total_files - 5:
        logger.info(f"Processing file {file_index + 1}/{total_files}: {file_path.name}")
    
    return extract_features_from_json(file_path)

def main():
    """
    Main function to iterate through JSONs, extract features, and save to Parquet.
    """
    start_time = time.time()
    logger.info("Starting JSON feature extraction...")
    
    json_files = list(JSON_DIR.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {JSON_DIR}")
        return

    total_files = len(json_files)
    logger.info(f"Found {total_files} JSON files to process")

    # Use all available CPUs
    num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} processes for extraction")

    # Prepare arguments for multiprocessing with progress tracking
    process_args = [(file_path, i, total_files) for i, file_path in enumerate(json_files)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        logger.info("Starting parallel processing...")
        results = pool.map(extract_with_progress, process_args)

    # Flatten the list of lists
    all_features = [item for sublist in results for item in sublist]

    if not all_features:
        logger.warning("No features were extracted.")
        return

    logger.info(f"Creating DataFrame from {len(all_features)} extracted features...")
    df = pl.DataFrame(all_features)
    
    logger.info(f"Writing features to {OUTPUT_PATH}...")
    df.write_parquet(OUTPUT_PATH)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Successfully extracted {len(df)} features from {total_files} files")
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Average time per file: {elapsed_time/total_files:.3f} seconds")
    logger.info(f"Features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()