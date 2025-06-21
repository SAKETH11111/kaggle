import polars as pl
import pyarrow.parquet as pq
from pathlib import Path
from data_loading import load_dataset

FEATURE_BLOCKS_PATH = Path("feature_blocks")
OUTPUT_DATASET_PATH = Path("dataset_full")

def main():
    """
    Merges the original structured data with the post-processed JSON feature blocks.
    """
    # Load the base train and test datasets
    train_df = load_dataset("train")
    test_df = load_dataset("test")

    # Iterate through all the feature blocks
    feature_files = list(FEATURE_BLOCKS_PATH.glob("*.parquet"))
    print(f"Found {len(feature_files)} feature blocks to merge.")

    for feature_file in feature_files:
        print(f"Merging {feature_file.name}...")
        feature_df = pl.scan_parquet(feature_file).with_columns(pl.col("ranker_id").cast(pl.Utf8))
        train_df = train_df.join(feature_df, on="ranker_id", how="left")
        test_df = test_df.join(feature_df, on="ranker_id", how="left")

    # Add the partitioning column
    train_df = train_df.with_columns(
        (pl.col("ranker_id").hash() % 100).alias("ranker_id_mod_100")
    )
    test_df = test_df.with_columns(
        (pl.col("ranker_id").hash() % 100).alias("ranker_id_mod_100")
    )

    # Ensure the output directory exists
    OUTPUT_DATASET_PATH.mkdir(exist_ok=True)
    
    # Write the final datasets
    print("Writing final train dataset...")
    pq.write_to_dataset(
        train_df.collect().to_arrow(),
        root_path=OUTPUT_DATASET_PATH / "train",
        partition_cols=["ranker_id_mod_100"],
    )

    print("Writing final test dataset...")
    pq.write_to_dataset(
        test_df.collect().to_arrow(),
        root_path=OUTPUT_DATASET_PATH / "test",
        partition_cols=["ranker_id_mod_100"],
    )

    print("Feature merging complete.")

if __name__ == "__main__":
    main()