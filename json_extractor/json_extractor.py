import argparse
import logging
import multiprocessing
import sqlite3
import tarfile
from functools import partial
from pathlib import Path

import dpath
import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
error_logger = logging.getLogger("error_logger")
error_logger.addHandler(logging.FileHandler("failed_extraction.log"))

# --- Database Functions for Resumability ---


def setup_database(db_path):
    """Initializes the SQLite database and creates the processed_files table."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_files (
                filename TEXT PRIMARY KEY
            )
            """
        )
        conn.commit()


def get_processed_files(db_path):
    """Retrieves the set of already processed filenames from the database."""
    if not Path(db_path).exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM processed_files")
        return {row[0] for row in cursor.fetchall()}


def add_processed_files(db_path, filenames):
    """Adds a list of filenames to the processed_files table."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO processed_files (filename) VALUES (?)",
            [(name,) for name in filenames],
        )
        conn.commit()


# --- Feature Extraction Logic ---


def get_json_feature_map(map_file="json_feature_map.yaml"):
    """Loads the feature map from a YAML file."""
    with open(map_file, "r") as f:
        return yaml.safe_load(f)


def infer_schema(feature_map):
    """Infers the Arrow schema from the feature map."""
    dtype_map = {
        "str": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
    }
    fields = [pa.field("ranker_id", pa.string())]
    for feature in feature_map["features"]:
        fields.append(
            pa.field(feature["name"], dtype_map.get(feature["dtype"], pa.string()))
        )
    return pa.schema(fields)


def extract_worker(member_names, tar_path, feature_map, schema):
    """
    Worker function to extract features from a chunk of JSON files within a tar archive.

    Args:
        member_names (list[str]): A list of tar member names to process.
        tar_path (str): Path to the .tar.gz archive.
        feature_map (dict): The feature extraction mapping.
        schema (pa.Schema): The Arrow schema for the output RecordBatch.

    Returns:
        list[pa.RecordBatch]: A list of Arrow RecordBatches for each successfully processed file.
    """
    worker_id = multiprocessing.current_process().pid
    logging.info(f"[Worker {worker_id}] Starting processing of {len(member_names)} files.")
    
    batches = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member_name in member_names:
            try:
                member_file = tar.extractfile(member_name)
                if not member_file:
                    continue

                json_data = orjson.loads(member_file.read())
                ranker_id = Path(member_name).stem

                feature_values = {"ranker_id": [ranker_id]}
                for feature in feature_map["features"]:
                    try:
                        # Use dpath to get nested values, supporting wildcards
                        values = list(dpath.values(json_data, feature["path"]))
                        # Join list values into a single string; could be handled differently
                        value = ",".join(map(str, values)) if values else None
                        feature_values[feature["name"]] = [value]
                    except (KeyError, TypeError):
                        feature_values[feature["name"]] = [None]

                # Create a RecordBatch from the extracted features
                arrays = []
                for field in schema:
                    col_data = feature_values.get(field.name, [None])
                    try:
                        # Ensure data type consistency before creating array
                        typed_data = pa.array(col_data, type=field.type, safe=False)
                        arrays.append(typed_data)
                    except (pa.ArrowInvalid, TypeError, ValueError):
                        # Handle type conversion errors by creating a null array
                        arrays.append(pa.nulls(len(col_data), type=field.type))

                batches.append(pa.RecordBatch.from_arrays(arrays, schema=schema))

            except (orjson.JSONDecodeError, KeyError, tarfile.ReadError, EOFError) as e:
                error_logger.error(f"Failed to process {member_name}: {e}")
                continue
    
    logging.info(f"[Worker {worker_id}] Finished processing chunk.")
    return batches


# --- Main Execution Block ---


def main():
    """Main function to run the parallel JSON extraction."""
    parser = argparse.ArgumentParser(
        description="""
        Extracts features from JSON files in a .tar.gz archive in parallel
        and stores them in a partitioned Parquet dataset.
        """
    )
    parser.add_argument(
        "--tar-path",
        type=str,
        default="data/jsons_raw.tar.kaggle",
        help="Path to the input .tar.gz file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="processed/features_raw",
        help="Output directory for Parquet files.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="processed/processed_files.sqlite",
        help="Path to the SQLite database for resumability.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--buffer-rows",
        type=int,
        default=100000,
        help="Number of rows to buffer in memory before writing to Parquet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing, skipping already processed files.",
    )
    args = parser.parse_args()

    # Setup output directory and database
    output_path = Path(args.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_database(args.db_path)

    # Load feature map and infer schema
    feature_map = get_json_feature_map()
    schema = infer_schema(feature_map)

    # Get the queue of files to process
    logging.info(f"Scanning tar archive: {args.tar_path}")
    with tarfile.open(args.tar_path, "r:gz") as tar:
        member_names = [m.name for m in tar.getmembers() if m.isfile()]

    if args.resume:
        logging.info("Resuming job. Checking for already processed files...")
        processed_files = get_processed_files(args.db_path)
        files_to_process = [
            name for name in member_names if name not in processed_files
        ]
        logging.info(
            f"Found {len(processed_files)} processed files. "
            f"Skipping them. {len(files_to_process)} files remaining."
        )
    else:
        files_to_process = member_names
        logging.info("Starting a new job. All files will be processed.")

    if not files_to_process:
        logging.info("No files to process. Exiting.")
        return

    # Split files into chunks for each worker
    num_workers = args.workers
    chunks = np.array_split(files_to_process, num_workers)
    # Filter out empty chunks
    chunks = [chunk.tolist() for chunk in chunks if chunk.size > 0]

    logging.info(
        f"Split {len(files_to_process)} files into {len(chunks)} chunks for {num_workers} workers."
    )

    # Configure the worker function with fixed arguments
    worker_func = partial(
        extract_worker,
        tar_path=args.tar_path,
        feature_map=feature_map,
        schema=schema,
    )

    # Main processing loop
    batch_buffer = []
    processed_filenames_buffer = []
    total_rows_in_buffer = 0

    logging.info(f"Starting extraction with {num_workers} workers...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            # pool.map returns a list of lists of RecordBatches
            results_nested = pool.map(worker_func, chunks)
            
            # Flatten the list of lists and process results
            for result_list in results_nested:
                for result in result_list:
                    pbar.update(1)
                    if result:
                        batch_buffer.append(result)
                        # The filename is the ranker_id + .json extension
                        ranker_id = result.column("ranker_id")[0].as_py()
                        original_filename = f"{ranker_id}.json"
                        processed_filenames_buffer.append(original_filename)
                        total_rows_in_buffer += result.num_rows

                    if total_rows_in_buffer >= args.buffer_rows:
                        logging.info(f"Buffer reached {total_rows_in_buffer} rows. Writing to Parquet...")
                        table = pa.Table.from_batches(batch_buffer)
                        pq.write_to_dataset(
                            table,
                            root_path=str(output_path),
                            compression="snappy",
                            row_group_size=64000,
                        )
                        add_processed_files(args.db_path, processed_filenames_buffer)
                        
                        # Reset buffers
                        batch_buffer.clear()
                        processed_filenames_buffer.clear()
                        total_rows_in_buffer = 0
                        logging.info("Write complete. Continuing extraction.")

    # Write any remaining data in the buffer
    if batch_buffer:
        logging.info(f"Writing remaining {total_rows_in_buffer} rows to Parquet...")
        table = pa.Table.from_batches(batch_buffer)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            compression="snappy",
            row_group_size=64000,
        )
        add_processed_files(args.db_path, processed_filenames_buffer)
        logging.info("Final write complete.")

    logging.info("JSON extraction has finished successfully.")


if __name__ == "__main__":
    main()