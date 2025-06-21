import polars as pl
import logging
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    """
    Runs a quick functional test on the feature engineering pipeline.
    """
    logger.info("Starting feature engineering smoke test...")

    try:
        # 1. Load a small sample of the data
        logger.info("Loading a sample of training data...")
        pipeline = DataPipeline()
        # Load only a few profiles to keep it fast
        profile_ids = list(range(10)) 
        sample_df = pipeline.prepare_training_data(profile_ids=profile_ids)

        # 2. Apply the full feature engineering pipeline
        logger.info("Applying feature engineering...")
        feature_engineer = FeatureEngineer()
        enriched_df = feature_engineer.engineer_all_features(sample_df)

        # 3. Collect the result into a DataFrame for testing
        logger.info("Collecting LazyFrame to DataFrame...")
        df_test = enriched_df.collect()

        # 4. Run assertions
        logger.info("Running assertions...")

        # Check for NaNs in a key ratio feature
        assert df_test.filter(pl.col("rel_price_vs_min").is_nan()).height == 0, \
            "Found NaN values in 'rel_price_vs_min'"
        logger.info("Assertion PASSED: No NaN values in 'rel_price_vs_min'.")

        # Check for invalid values in another key ratio feature
        # duration_vs_min should always be >= 1, but we test for <= 0 to catch serious errors.
        assert df_test.filter(pl.col("duration_vs_min") <= 0).is_empty(), \
            "Found non-positive values in 'duration_vs_min'"
        logger.info("Assertion PASSED: No non-positive values in 'duration_vs_min'.")

        # 5. Print a sample for visual inspection
        logger.info("Displaying head of the final DataFrame for visual inspection:")
        print(df_test.head())

        logger.info("Smoke test completed successfully!")

    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_test()