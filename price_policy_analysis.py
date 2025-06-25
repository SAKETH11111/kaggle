import polars as pl
import numpy as np
import logging
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path("data")

def run_all_analyses():
    """
    Performs a comprehensive, memory-efficient analysis of all requested metrics.
    """
    logger.info("Starting final, comprehensive data analysis...")

    try:
        # --- 1. Load Data and Perform Feature Engineering ---
        def duration_to_minutes(duration_col):
            duration_struct = (
                pl.col(duration_col)
                .str.extract_groups(r"(\d+):(\d+):(\d+)")
                .struct.rename_fields(["hours", "minutes", "seconds"])
            )
            return (
                duration_struct.struct.field("hours").cast(pl.Int32) * 60 +
                duration_struct.struct.field("minutes").cast(pl.Int32)
            )

        lf = pl.scan_parquet(DATA_PATH / "train.parquet").with_columns([
            pl.col("legs0_departureAt").str.to_datetime().dt.hour().alias("departure_hour"),
            pl.col("legs0_departureAt").str.to_datetime().dt.weekday().alias("departure_weekday"),
            (pl.col("totalPrice") - pl.col("totalPrice").min().over("ranker_id")).alias("price_premium"),
            pl.col("legs0_segments1_departureFrom_airport_iata").is_null().alias("is_direct_flight"),
            duration_to_minutes("legs0_duration").alias("total_duration_minutes"),
            pl.col("miniRules0_monetaryAmount").fill_null(0).alias("cancellation_fee")
        ])
        
        df = lf.collect()
        selected_flights_df = df.filter(pl.col("selected") == 1)
        total_selections = selected_flights_df.shape[0]

        # --- 7. Route Complexity Tolerance ---
        logger.info("\n--- Route Complexity Tolerance ---")
        direct_flight_premium = selected_flights_df.group_by("is_direct_flight").agg(
            pl.mean("price_premium").alias("avg_premium")
        )
        logger.info(f"Direct Flight Premium:\n{direct_flight_premium}")

        # --- 8. Missing Data Patterns ---
        logger.info("\n--- Missing Data Patterns ---")
        missing_data = df.select(pl.all().is_null().mean() * 100).to_pandas().transpose()
        missing_data.columns = ["percentage_missing"]
        logger.info(f"Missing Data:\n{missing_data[missing_data['percentage_missing'] > 0]}")

        # --- 9. Airline/Aircraft Analysis ---
        logger.info("\n--- Airline/Aircraft Analysis ---")
        airline_selection_rates = selected_flights_df["legs0_segments0_marketingCarrier_code"].value_counts()
        logger.info(f"Airline Selection Rates:\n{airline_selection_rates.head(10)}")

        # --- 10. Flexibility Analysis ---
        logger.info("\n--- Flexibility Analysis ---")
        flexibility_analysis = selected_flights_df.group_by("cancellation_fee").agg(
            pl.count().alias("num_selections")
        ).sort("cancellation_fee")
        logger.info(f"Flexibility Analysis (Selections by Cancellation Fee):\n{flexibility_analysis}")

    except FileNotFoundError:
        logger.error(f"Data not found at {DATA_PATH / 'train.parquet'}.")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    run_all_analyses()