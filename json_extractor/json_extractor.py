import argparse
import logging
import multiprocessing
import tarfile
import time
from pathlib import Path
import signal

import dpath
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_logger.addHandler(logging.FileHandler('failed.log'))

# --- Helper Functions ---

def get_json_feature_map(map_file='json_feature_map.yaml'):
    """Loads the feature map from a YAML file."""
    with open(map_file, 'r') as f:
        return yaml.safe_load(f)

def infer_schema(feature_map):
    """Infers the Arrow schema from the feature map."""
    dtype_map = {
        'str': pa.string(),
        'int': pa.int64(),
        'float': pa.float64(),
        'bool': pa.bool_()
    }
    fields = [pa.field('ranker_id', pa.string())]
    for feature in feature_map['features']:
        fields.append(pa.field(feature['name'], dtype_map.get(feature['dtype'], pa.string())))
    return pa.schema(fields)

def extract_worker(member_name, tar_path, feature_map, schema):
    """Worker function to extract features from a single JSON file."""
    try:
        with tarfile.open(tar_path, 'r|') as tar:
            member_file = tar.extractfile(member_name)
            if not member_file:
                return None
            
            data = orjson.loads(member_file.read())
            ranker_id = Path(member_name).stem
            
            features = {'ranker_id': ranker_id}
            for feature in feature_map['features']:
                try:
                    # Use dpath to get nested values, handling wildcards
                    values = [v for v in dpath.values(data, feature['path'])]
                    # For simplicity, we'll join list values; this could be more sophisticated
                    features[feature['name']] = ','.join(map(str, values)) if values else None
                except (KeyError, TypeError):
                    features[feature['name']] = None

            # Create a RecordBatch
            arrays = []
            for field in schema:
                val = features.get(field.name)
                # Ensure data type consistency
                try:
                    if val is None:
                        arrays.append(pa.nulls(1, type=field.type))
                    else:
                        if field.type == pa.string():
                           arrays.append(pa.array([str(val)], type=field.type))
                        elif field.type == pa.int64():
                           arrays.append(pa.array([int(val)], type=field.type))
                        elif field.type == pa.float64():
                           arrays.append(pa.array([float(val)], type=field.type))
                        elif field.type == pa.bool_():
                           arrays.append(pa.array([bool(val)], type=field.type))
                        else:
                           arrays.append(pa.array([val], type=field.type))

                except (ValueError, TypeError):
                     arrays.append(pa.nulls(1, type=field.type))


            return pa.RecordBatch.from_arrays(arrays, schema=schema)

    except (orjson.JSONDecodeError, KeyError, tarfile.ReadError) as e:
        error_logger.error(f"Failed to process {member_name}: {e}")
    return None

def main():
    """Main function to run the parallel extraction."""
    parser = argparse.ArgumentParser(description="Parallel JSON Extractor")
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes.')
    parser.add_argument('--buffer-rows', type=int, default=100000, help='Number of rows to buffer before writing.')
    parser.add_argument('--tar-path', type=str, default='data/jsons_raw.tar.kaggle', help='Path to the tar file.')
    parser.add_argument('--out-dir', type=str, default='processed/features_raw', help='Output directory for Parquet files.')
    parser.add_argument('--resume', action='store_true', help='Resume processing, skipping existing partitions.')
    args = parser.parse_args()

    feature_map = get_json_feature_map()
    schema = infer_schema(feature_map)
    
    output_path = Path(args.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Graceful Shutdown ---
    stop_event = multiprocessing.Event()
    def signal_handler(sig, frame):
        logging.info("Shutdown signal received. Telling workers to stop.")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with tarfile.open(args.tar_path, 'r|') as tar, \
         multiprocessing.Pool(processes=args.workers) as pool:

        manager = multiprocessing.Manager()
        queue = manager.Queue(maxsize=args.workers * 2)

        # --- Producer ---
        def producer():
            for member in tar:
                if stop_event.is_set():
                    break
                if member.isfile():
                    queue.put(member.name)
            for _ in range(args.workers):
                queue.put(None) # Sentinel values
        
        producer_proc = multiprocessing.Process(target=producer)
        producer_proc.start()

        # --- Consumers ---
        results = []
        total_files = sum(1 for m in tar if m.isfile()) # This can be slow for large tars
        
        with tqdm(total=total_files, desc="Processing files") as pbar:
            while True:
                if stop_event.is_set():
                    break
                
                member_name = queue.get()
                if member_name is None:
                    break

                result = pool.apply_async(extract_worker, (member_name, args.tar_path, feature_map, schema))
                results.append(result)
                pbar.update(1)

                if len(results) >= args.buffer_rows:
                    # Write batch
                    batches = [r.get() for r in results if r.get() is not None]
                    if batches:
                        table = pa.Table.from_batches(batches)
                        pq.write_to_dataset(table, root_path=str(output_path), compression='snappy')
                    results = []
        
        # Write remaining results
        if results:
            batches = [r.get() for r in results if r.get() is not None]
            if batches:
                table = pa.Table.from_batches(batches)
                pq.write_to_dataset(table, root_path=str(output_path), compression='snappy')

        producer_proc.join()

    logging.info("JSON extraction complete.")

if __name__ == '__main__':
    main()