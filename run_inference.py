import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb

from data_pipeline import DataPipeline, DataConfig
from feature_engineering import FeatureEngineer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

N_SPLITS = 5
MODEL_DIR = Path("processed/models")
PROCESSED_DIR = Path("processed")

# --- Helper Functions ---
def load_and_prepare_test_data(pipeline: DataPipeline, feature_engineer: FeatureEngineer) -> pl.DataFrame:
    """Loads and prepares the test data for inference."""
    logger.info("Loading and preparing test data...")
    
    test_df_lazy = pipeline.loader.load_structured_data("test")
    json_features_lazy = pl.scan_parquet(PROCESSED_DIR / "json_features_aggregated.parquet")
    
    # Join with JSON features before engineering
    test_df_lazy = test_df_lazy.join(json_features_lazy, on="ranker_id", how="left")
    
    test_featured_lazy = feature_engineer.engineer_all_features(test_df_lazy)
    test_df = test_featured_lazy.collect()
    
    logger.info(f"Test data shape: {test_df.shape}")
    return test_df

def generate_predictions(test_df: pl.DataFrame) -> np.ndarray:
    """Generates and ensembles predictions from all fold models."""
    all_preds = []
    for i in range(1, N_SPLITS + 1):
        model_path = MODEL_DIR / f"lgbm_ranker_fold_{i}.txt"
        logger.info(f"Loading model from {model_path}...")
        booster = lgb.Booster(model_file=str(model_path))
        
        feature_cols = booster.feature_name()
        X_test = test_df[feature_cols].to_pandas()
# Ensure only numeric and boolean columns are used for prediction
        numeric_features = X_test.select_dtypes(include=np.number).columns.tolist()
        boolean_features = X_test.select_dtypes(include=np.bool_).columns.tolist()
        X_test = X_test[numeric_features + boolean_features]

        # Convert object columns to categorical for LightGBM
        for col in X_test.select_dtypes(include=['object']).columns:
            X_test[col] = X_test[col].astype('category')
        
        preds = booster.predict(X_test)
        all_preds.append(preds)
    
    logger.info("Ensembling predictions...")
    return np.mean(all_preds, axis=0)

def create_submission_file(test_df: pl.DataFrame, predictions: np.ndarray):
    """Creates the final submission file in the correct format."""
    logger.info("Creating submission file...")
    
    # Add the predictions to the test dataframe
    submission_df = test_df.with_columns(
        pl.Series("predicted_score", predictions)
    )

    # Rank the predictions within each group
    submission_df = submission_df.with_columns(
        pl.col("predicted_score").rank(method="ordinal", descending=True).over("ranker_id").alias("selected")
    )

    # Select only the required columns for submission
    submission_df = submission_df.select(["Id", "selected"])
    
    # Ensure the submission is sorted by the original Id to maintain row order
    submission_df = submission_df.sort("Id")
    
    submission_path = PROCESSED_DIR / "submission.csv"
    submission_df.write_csv(submission_path)
    logger.info(f"Submission file created at {submission_path}")

# --- Main Execution Block ---
def main():
    """
    Main function to run the inference pipeline.
    """
    logger.info("Starting inference pipeline...")
    
    config = DataConfig()
    pipeline = DataPipeline(config)
    feature_engineer = FeatureEngineer()
    
    test_df = load_and_prepare_test_data(pipeline, feature_engineer)
    ensembled_preds = generate_predictions(test_df)
    create_submission_file(test_df, ensembled_preds)

if __name__ == "__main__":
    main()