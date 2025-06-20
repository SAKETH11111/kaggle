import tarfile
import orjson
import csv
import os
from collections import Counter
from itertools import islice

def get_key_paths(source, path=''):
    """
    Recursively finds all key paths in a nested dictionary or list.
    """
    if isinstance(source, dict):
        for key, value in source.items():
            new_path = f"{path}.{key}" if path else key
            yield new_path
            yield from get_key_paths(value, new_path)
    elif isinstance(source, list):
        for item in source:
            yield from get_key_paths(item, path)

def main():
    """
    Analyzes a sample of JSON files from a .tar.gz archive to understand their structure.
    """
    archive_path = 'data/jsons_raw.tar.kaggle'
    output_path = 'processed/json_schema_sample.csv'
    sample_size = 1000
    
    key_counts = Counter()
    files_processed = 0

    print(f"Starting reconnaissance on '{archive_path}'...")

    try:
        with tarfile.open(archive_path, "r|gz") as tar:
            # Use islice to efficiently get the first N members
            members = islice(tar, sample_size)
            
            for member in members:
                if not member.isfile():
                    continue

                try:
                    # Extract file object in memory
                    file_obj = tar.extractfile(member)
                    if not file_obj:
                        continue
                    
                    content = file_obj.read()
                    data = orjson.loads(content)
                    
                    # Use a set to get unique paths for this file before updating the main counter
                    unique_paths_in_file = set(get_key_paths(data))
                    key_counts.update(unique_paths_in_file)
                    
                    files_processed += 1
                    if files_processed % 100 == 0:
                        print(f"  ... processed {files_processed}/{sample_size} files.")

                except orjson.JSONDecodeError:
                    print(f"  - Warning: Skipping invalid JSON file: {member.name}")
                    continue
                except Exception as e:
                    print(f"  - Warning: Could not process file {member.name}: {e}")
                    continue

    except FileNotFoundError:
        print(f"FATAL: Archive not found at '{archive_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"FATAL: An error occurred: {e}")
        return

    if files_processed == 0:
        print("No JSON files were processed. Exiting.")
        return

    print(f"\nReconnaissance complete. Analyzed {files_processed} files.")
    print(f"Found {len(key_counts)} unique key paths.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key_path', 'percentage_present'])
        
        # Sort by most common
        for path, count in key_counts.most_common():
            percentage = (count / files_processed) * 100
            writer.writerow([path, f"{percentage:.2f}"])

    print(f"Results saved to '{output_path}'.")

if __name__ == '__main__':
    main()