import polars as pl
import time
import psutil
import os

def get_memory_usage():
    """Gets the resident set size of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def main():
    """
    Runs memory and performance benchmarks on the training dataset and
    generates a report.
    """
    dataset_path = "dataset_full/train/"
    report_path = "benchmark.md"

    # --- Memory Benchmark ---
    mem_before = get_memory_usage()
    
    # Load the dataset. Polars automatically discovers and reads partitioned
    # parquet files when given a directory path.
    df = pl.read_parquet(f"{dataset_path}/**/*.parquet")
    
    mem_after = get_memory_usage()
    memory_usage_mb = mem_after - mem_before

    # --- Performance Benchmark ---
    start_time = time.time()
    
    # Perform a scan and count operation. Using lazy evaluation is key for this.
    count = df.lazy().select(pl.count()).collect().item()
    
    end_time = time.time()
    performance_time_s = end_time - start_time

    # --- Report Generation ---
    report_content = f"""# Dataset Benchmark Report

*   **Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`

## Memory Usage
*   **RSS increase after loading:** `{memory_usage_mb:.2f} MB`

## Performance
*   **Time for `scan` and `count`:** `{performance_time_s:.4f} seconds`
*   **Total rows:** `{count:,}`
"""

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"Benchmark complete. Report saved to `{report_path}`.")

if __name__ == "__main__":
    main()