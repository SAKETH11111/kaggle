import polars as pl
import numpy as np
import logging
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path

from data_pipeline import DataPipeline, DataConfig

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "processed/models/lgbm_ranker_fold_1.txt"
N_SHAP_SAMPLES = 10000  # Number of samples to use for SHAP analysis

# --- Main Execution Block ---
def main():
    """
    Main function to run the model analysis.
    """
    logger.info("Starting model analysis...")

    # --- 1. Load Model and Data ---
    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        booster = lgb.Booster(model_file=MODEL_PATH)
    except lgb.basic.LightGBMError:
        logger.error(f"Could not find model file at {MODEL_PATH}. Please run run_training.py to generate the final model.")
        return
    
    logger.info("Loading data for analysis...")
    config = DataConfig()
    pipeline = DataPipeline(config)
    full_df = pipeline.prepare_training_data().collect()

    feature_cols = booster.feature_name()
    X = full_df[feature_cols].to_pandas()

    # Convert object columns to categorical for SHAP
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    # Convert datetime columns to numeric int64 (nanoseconds since epoch) to avoid dtype promotion issues
    datetime_cols = X.select_dtypes(include=['datetime', 'datetimetz']).columns
    for col in datetime_cols:
        # Preserve missing values while converting to int64
        X[col] = X[col].view('int64')

    # --- 2. Feature Importance Analysis ---
    logger.info("Generating feature importance plot...")
    lgb.plot_importance(booster, max_num_features=30, figsize=(10, 15))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("processed/feature_importance.png")
    plt.close()
    logger.info("Feature importance plot saved to processed/feature_importance.png")

    # --- 3. SHAP Analysis ---
    logger.info(f"Generating SHAP analysis on {N_SHAP_SAMPLES} samples...")
    
    # Sample the data for SHAP analysis to manage memory and compute time
    X_sample = X.sample(n=N_SHAP_SAMPLES, random_state=42)
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Summary Plot (Bar)")
    plt.tight_layout()
    plt.savefig("processed/shap_summary_bar.png")
    plt.close()

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot (Dot)")
    plt.tight_layout()
    plt.savefig("processed/shap_summary_dot.png")
    plt.close()
    
    logger.info("SHAP analysis plots saved to processed/")

if __name__ == "__main__":
    main()