# Parallel JSON Extraction Architecture

This document outlines the architecture for a parallel JSON extraction system.

## Pattern

The system uses a producer-consumer pattern with a `multiprocessing.Pool` of 8-12 worker processes to parallelize the extraction of data from JSON files within a tar archive.

- **Producer:** The main process acts as the producer. It iterates through the members of the tar archive and adds the name of each member (JSON file) to a shared queue.
- **Consumers:** The worker processes in the pool act as consumers. They retrieve member names from the queue and process them in parallel.

## Communication

A shared `multiprocessing.Queue` is used for inter-process communication. The main process (producer) populates the queue with tar member names, and the worker processes (consumers) consume from this queue. This ensures that work is distributed efficiently among the available workers.

## Worker Function

The core of the parallel processing is the `extract_worker(member)` function. Each worker process executes this function with the following responsibilities:

1.  **Receive Member:** Accepts a tar member name from the queue.
2.  **Extract Data:** Opens the corresponding JSON file from the tar archive.
3.  **Feature Selection:** Parses the JSON and extracts specific fields as defined in the `json_feature_map.yaml` configuration file.
4.  **Create RecordBatch:** The extracted features, along with the `ranker_id` (derived from the filename), are packaged into an Apache Arrow `RecordBatch`. This is an efficient, columnar in-memory format.
5.  **Return Value:** The function returns the created `RecordBatch`.

## Buffering and Output

The `RecordBatch` objects returned by the worker processes are handled as follows:

1.  **Buffering:** The main process collects the `RecordBatch` objects from the workers. They are buffered in memory to accumulate a sufficient amount of data before writing to disk.
2.  **Parquet Output:** Once the buffer reaches a certain size, or all members have been processed, the collected data is written to a partitioned Parquet dataset.
    -   **Path:** `features_raw/feature=.../part-*.parquet`
    -   **Compression:** Snappy compression is used for a good balance of speed and compression ratio.
    - **Row Group Size:** A row group size of 64k is used to optimize for read performance.
    
    ## Architectural Enhancements
    
    Based on feedback, the following enhancements will be incorporated into the design:
    
    | Area                | Suggestion                                                                                                                                                                                                 |
    | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | **Graceful Shutdown** | Trap `KeyboardInterrupt` in the main process. A sentinel value (e.g., `None`) will be placed on the queue for each worker to signal them to exit cleanly.                                                   |
    | **Back-pressure**     | The `multiprocessing.Queue` will be initialized with a `maxsize` (e.g., `2 * num_workers`) to prevent the producer from getting too far ahead of the consumers, which could lead to high memory usage.      |
    | **Chunk Batching**    | Instead of writing every `RecordBatch` to disk individually, they will be accumulated in a list. Once the list contains a sufficient number of rows (e.g., ~100k), the batches will be concatenated using `pyarrow.concat_tables` and then written, improving I/O efficiency. |
    | **Schema Stability**  | A stable `pyarrow.schema` will be inferred from a sample of the first 1,000 JSON files. This schema will be passed to every `RecordBatch` constructor to ensure metadata consistency in the final Parquet file. |
    | **Error Logging**     | The JSON parsing logic within the worker will be wrapped in a `try...except` block to catch `orjson.JSONDecodeError` and `KeyError`. If a file fails to parse, its name will be logged to a `failed.log` file, and the worker will continue processing, preventing a single bad file from halting the entire run. |
    | **CLI Flags**         | The script will accept command-line arguments for configuration: `--workers`, `--buffer-rows`, `--tar-path`, `--out-dir`, and `--resume` (to skip already-written partitions). |
    | **Performance Metrics**| The script will use `tqdm` to display real-time performance metrics, including processed files/sec and MB/sec, to monitor progress and identify potential bottlenecks. |
    
    ---
    
    ## `json_extractor.py` Skeleton

The following is the planned skeleton for the `json_extractor.py` file. It will be created in the implementation phase.

```python
"""
This module will contain the implementation for a parallel JSON extraction system.

The architecture is based on a producer-consumer pattern using a multiprocessing pool.
A shared multiprocessing.Queue will be used to distribute tar member names to worker
processes.

The main components are:
- A main process that walks the tar file and puts member names into a queue.
- A pool of worker processes (8-12 workers).
- A worker function `extract_worker(member)` that:
    - Receives a tar member from the queue.
    - Extracts selected fields based on `json_feature_map.yaml`.
    - Returns an Arrow RecordBatch with features and `ranker_id`.
- A process that collects RecordBatch objects from the workers, buffers them,
  and writes them to disk in a partitioned Parquet format.
"""