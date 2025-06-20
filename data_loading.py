import polars as pl
from pathlib import Path

DATA_PATH = Path("data")


def load_dataset(split: str, cols: list[str] = None) -> pl.LazyFrame:
    """
    Loads a dataset split ('train' or 'test') from a parquet file in lazy mode.

    Args:
        split: The name of the split to load, either "train" or "test".
        cols: An optional list of column names to select from the dataset.
              If None, all columns are selected.

    Returns:
        A Polars LazyFrame representing the dataset.

    Raises:
        ValueError: If the split name is not 'train' or 'test'.
    """
    if split not in ["train", "test"]:
        raise ValueError("split must be one of 'train' or 'test'")

    file_path = DATA_PATH / f"{split}.parquet"
    lf = pl.scan_parquet(file_path)

    if cols:
        lf = lf.select(cols)

    return lf