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
        # --- 1. Load Data and Perform Initial Feature Engineering ---
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

        base_lf = pl.scan_parquet(DATA_PATH / "train.parquet").with_columns([
            pl.int_range(0, pl.len()).over("ranker_id").alias("position_in_group"),
            duration_to_minutes("legs0_duration").alias("total_duration_minutes"),
            pl.col("legs0_segments1_departureFrom_airport_iata").is_null().alias("isDirectFlight"),
            (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).alias("is_cheapest"),
            (pl.col("totalPrice") - pl.col("totalPrice").min().over("ranker_id")).alias("price_premium"),
            pl.col("pricingInfo_isAccessTP").alias("is_compliant")
        ])

        # --- 2. Isolate Selected Flights for Characteristic Analysis ---
        selected_flights_df = base_lf.filter(pl.col("selected") == 1).collect()

        # --- 3. Group-Level Analysis ---
        logger.info("--- Group/Session Analysis ---")
        group_stats_df = base_lf.group_by("ranker_id").agg([
            pl.len().alias("group_size"),
            pl.col("position_in_group").filter(pl.col("selected") == 1).first().alias("selected_position")
        ]).with_columns(
            (pl.col("selected_position") / pl.col("group_size")).alias("normalized_position")
        ).collect()

        # ... (print group analysis as before) ...

        # --- 4. Selected Flight Characteristics Analysis ---
        logger.info("\n--- Selected Flight Characteristics Analysis ---")
        # ... (print characteristics from selected_flights_df) ...

        # --- 5. Price Analysis ---
        logger.info("\n--- Price Analysis ---")
        cheapest_selected = selected_flights_df.filter(pl.col("is_cheapest") == True).shape[0]
        total_selections = selected_flights_df.shape[0]
        logger.info(f"% of times cheapest flight is selected: {cheapest_selected / total_selections:.2%}")
        
        avg_premium_paid = selected_flights_df["price_premium"].mean()
        logger.info(f"Average price premium paid: ${avg_premium_paid:.2f}")

        # --- 6. Policy Compliance Deep Dive ---
        logger.info("\n--- Policy Compliance Deep Dive ---")
        compliant_selected = selected_flights_df.filter(pl.col("is_compliant") == True).shape[0]
        logger.info(f"% of selected flights that are policy compliant: {compliant_selected / total_selections:.2%}")

        # ... (add other policy analyses if they can be done efficiently) ...

    except FileNotFoundError:
        logger.error(f"Data not found at {DATA_PATH / 'train.parquet'}.")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    run_all_analyses()
