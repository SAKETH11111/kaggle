import polars as pl
import logging
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path("data")
PROCESSED_PATH = Path("processed")

def duration_to_minutes(duration_col):
    duration_struct = (
        duration_col
        .str.extract_groups(r"(\d+):(\d+):(\d+)")
        .struct.rename_fields(["hours", "minutes", "seconds"])
    )
    return (
        duration_struct.struct.field("hours").cast(pl.Int32) * 60 +
        duration_struct.struct.field("minutes").cast(pl.Int32)
    )

def run_final_analysis():
    """
    Performs the final, in-depth analysis by merging the main dataset
    with the extracted JSON features.
    """
    logger.info("Starting final analysis...")

    try:
        # --- 1. Load and Merge Data ---
        logger.info("Loading datasets...")
        train_df = pl.read_parquet(DATA_PATH / "train.parquet")
        json_features_df = pl.read_parquet(PROCESSED_PATH / "json_features.parquet")

        # --- 2. Centralized Feature Engineering ---
        logger.info("Performing feature engineering...")
        
        # Join the data
        merged_df = train_df.with_columns(
            pl.col("Id").cast(pl.Utf8)
        ).join(
            json_features_df.with_columns(pl.col("total_price_json").cast(pl.Float64)),
            left_on=["ranker_id", "Id", "totalPrice"],
            right_on=["ranker_id", "flight_id", "total_price_json"],
            how="left"
        )
        
        # Clean and create features
        analysis_df = merged_df.with_columns([
            duration_to_minutes(pl.col("legs0_duration")).alias("duration_minutes"),
            pl.col("cancellation_fee").fill_null(0),
            pl.col("baggage_quantity").fill_null(0),
            (pl.col("legs0_departureAt").str.to_datetime() - pl.col("requestDate")).dt.total_days().alias("booking_window_days")
        ])

        logger.info(f"Successfully merged and engineered features. New shape: {analysis_df.shape}")
        
        selected_flights_df = analysis_df.filter(pl.col("selected") == 1)

        # --- 11. Feature Interaction Discovery ---
        logger.info("\n--- Feature Interaction Discovery ---")
        price_duration_tradeoff = selected_flights_df.select(["totalPrice", "duration_minutes"]).describe()
        logger.info(f"Price vs. Duration Tradeoff:\n{price_duration_tradeoff}")

        # --- 12. Extreme Case Analysis ---
        logger.info("\n--- Extreme Case Analysis ---")
        most_expensive = analysis_df.filter(
            (pl.col("selected") == 1) &
            (pl.col("totalPrice") == pl.col("totalPrice").max().over("ranker_id"))
        )
        logger.info(f"Found {len(most_expensive)} instances where the most expensive flight was chosen.")
        if len(most_expensive) > 0:
            logger.info(f"Details of one such choice:\n{most_expensive.head(1).select(['Id', 'totalPrice', 'duration_minutes', 'cancellation_fee'])}")

        # --- 14. Temporal Patterns (Booking Window) ---
        logger.info("\n--- Temporal Patterns (Booking Window) ---")
        logger.info(f"Booking Window Analysis:\n{selected_flights_df.select('booking_window_days').describe()}")

        # --- 15. Statistical Tests ---
        logger.info("\n--- Statistical Tests ---")
        
        # Inspect variance of JSON features
        logger.info("Inspecting variance of key JSON features for selected flights:")
        feature_variance = selected_flights_df.select([
            "cancellation_fee",
            "baggage_quantity"
        ]).describe()
        logger.info(f"\n{feature_variance}")

        corr_matrix = selected_flights_df.select([
            "totalPrice",
            "duration_minutes",
            "cancellation_fee",
            "baggage_quantity"
        ]).corr()
        logger.info(f"Correlation Matrix:\n{corr_matrix}")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    run_final_analysis()