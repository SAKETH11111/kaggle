import polars as pl
from diskcache import Cache
from typing import Optional, List

# Initialize a cache for storing LazyFrames
# The cache is stored on disk in the 'cache_directory'
cache = Cache('cache_directory')

def get_lazy(split: str, cols: Optional[List[str]] = None) -> pl.LazyFrame:
    """
    Retrieves a Polars LazyFrame for a given data split, with optional column selection.

    This function provides a high-level API to access the feature store. It uses
    a caching mechanism to avoid re-loading data from disk, which can be slow.
    If a previously loaded LazyFrame for the given split and columns is found in
    the cache, it is returned immediately. Otherwise, the data is loaded from
    the Parquet files, stored in the cache, and then returned.

    Args:
        split: The name of the data split to load. Must be one of "train" or "test".
        cols: An optional list of column names to select from the dataset. If None,
              all columns are loaded.

    Returns:
        A Polars LazyFrame representing the requested data.
    """
    # Create a unique cache key based on the split and selected columns
    cache_key = f"{split}_{'_'.join(cols) if cols else 'all'}"

    # Check if the LazyFrame is already in the cache
    if cache_key in cache:
        return cache.get(cache_key)

    # If not in cache, load the dataset from the appropriate directory
    # pl.scan_parquet is used for lazy loading, which is memory-efficient
    path = f"dataset_full/{split}/**/*.parquet"
    lazy_frame = pl.scan_parquet(path)

    # If a list of columns is provided, project the LazyFrame
    if cols:
        lazy_frame = lazy_frame.select(cols)

    # Store the new LazyFrame in the cache before returning
    cache.set(cache_key, lazy_frame)

    return lazy_frame