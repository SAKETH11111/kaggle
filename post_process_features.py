import polars as pl
import yaml
from pathlib import Path

def post_process_features():
    """
    Loads raw features, processes them, and saves them as feature blocks.
    """
    # Load feature map
    with open("json_feature_map.yaml", "r") as f:
        feature_map = yaml.safe_load(f)["features"]

    # Create feature_blocks directory if it doesn't exist
    feature_blocks_dir = Path("feature_blocks")
    feature_blocks_dir.mkdir(exist_ok=True)

    # Define type mapping from YAML dtype to polars dtype
    dtype_map = {
        "str": pl.Utf8,
        "int": pl.Int64,
        "float": pl.Float64,
    }

    for feature_def in feature_map:
        feature_name = feature_def["name"]
        raw_feature_path = Path("features_raw") / f"{feature_name}.parquet"
        
        if not raw_feature_path.exists():
            print(f"Warning: Raw feature file not found for {feature_name}, skipping.")
            continue

        # Load raw feature data
        lazy_df = pl.scan_parquet(raw_feature_path)

        # Data Type Casting
        col_name = feature_name
        if col_name in lazy_df.columns:
            target_dtype = dtype_map.get(feature_def["dtype"])
            if target_dtype:
                lazy_df = lazy_df.with_columns(pl.col(col_name).cast(target_dtype))

        # Feature Derivation: flex_score
        if feature_name == "cancellation_penalties_max_penalty_pct":
            lazy_df = lazy_df.with_columns(
                (1 - (pl.col("cancellation_penalties_max_penalty_pct") / 100)).alias("flex_score")
            )

        # Cardinality Reduction for string columns
        if feature_def["dtype"] == "str" and col_name in lazy_df.columns:
            # This requires computation, so we collect here.
            # For large datasets, this might need to be optimized.
            df = lazy_df.collect()
            value_counts = df[col_name].value_counts()
            rare_levels = value_counts.filter(pl.col("count") < 50)[col_name]
            
            if len(rare_levels) > 0:
                df = df.with_columns(
                    pl.when(pl.col(col_name).is_in(rare_levels))
                    .then(pl.lit("other"))
                    .otherwise(pl.col(col_name))
                    .alias(col_name)
                )
            lazy_df = df.lazy()


        # Write to Delta Table (Parquet file)
        output_path = feature_blocks_dir / f"{feature_name}.parquet"
        lazy_df.collect().write_parquet(output_path)
        print(f"Successfully processed and wrote {feature_name} to {output_path}")


if __name__ == "__main__":
    # Note: This script assumes that the raw features have been extracted to 'features_raw'
    # and that each feature from the yaml map corresponds to a parquet file.
    # e.g., features_raw/ticket_restrictions.parquet
    # You might need to run a prior script to generate these raw files.
    
    # For demonstration, let's create some dummy raw feature files.
    raw_features_dir = Path("features_raw")
    raw_features_dir.mkdir(exist_ok=True)

    # Dummy data for cancellation_penalties_max_penalty_pct
    df_cancel = pl.DataFrame({
        "ranker_id": range(100),
        "cancellation_penalties_max_penalty_pct": [i for i in range(100)]
    })
    df_cancel.write_parquet(raw_features_dir / "cancellation_penalties_max_penalty_pct.parquet")

    # Dummy data for ticket_restrictions (string type for cardinality check)
    categories = [f"cat_{i}" for i in range(10)] + ["rare_cat_1"] * 40 + ["rare_cat_2"] * 10
    df_tickets = pl.DataFrame({
        "ranker_id": range(len(categories)),
        "ticket_restrictions": categories
    })
    df_tickets.write_parquet(raw_features_dir / "ticket_restrictions.parquet")
    
    print("Created dummy raw feature files for demonstration.")

    post_process_features()