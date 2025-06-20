import polars as pl
import numpy as np
import logging
from pathlib import Path
import argparse

from data_pipeline import DataPipeline, DataConfig
from ranker_trainer import RankerTrainer
from utils import CrossValidationUtils, ValidationUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the end-to-end training pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument(
        "--start-fold",
        type=int,
        default=1,
        help="The fold number to start training from (1-based).",
    )
    args = parser.parse_args()

    logger.info("Starting end-to-end training pipeline...")

    # --- 1. Configuration ---
    config = DataConfig()
    N_SPLITS = 5
    RANDOM_SEED = 42
    MODEL_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 20,
        'random_state': RANDOM_SEED,
        'device': 'cpu',
        'n_jobs': 32,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
    }

    # --- 2. Data Preparation ---
    logger.info("Preparing training data...")
    pipeline = DataPipeline(config)
    full_df_lazy = pipeline.prepare_training_data()
    
    # Collect the full dataframe into memory for CV splitting
    full_df = full_df_lazy.collect()
    logger.info(f"Full training data shape: {full_df.shape}")

    # Define feature columns, excluding identifiers and raw categorical strings
    feature_cols = [
        col for col in full_df.columns 
        if col not in [
            'Id', 'ranker_id', 'selected', 'profileId', 'search_route', 
            'price_bin', 'departure_time_category', 'cabin_class_name', 
            'origin_airport', 'destination_airport', 'marketing_carrier', 
            'operating_carrier', 'aircraft_code'
        ]
    ]
    logger.info(f"Using {len(feature_cols)} features for training.")

    # --- 3. Cross-Validation Setup ---
    logger.info(f"Setting up {N_SPLITS}-fold StratifiedGroupKFold cross-validation...")
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=full_df,
        group_col='profileId',
        stratify_col='selected',
        n_splits=N_SPLITS,
        seed=RANDOM_SEED
    )

    oof_preds = []
    fold_scores = []

    # --- 4. Training Loop ---
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        # The training loop is 0-indexed, but user input is 1-indexed.
        current_fold_num = fold + 1
        if current_fold_num < args.start_fold:
            logger.info(f"--- Skipping Fold {current_fold_num}/{N_SPLITS} ---")
            continue

        logger.info(f"--- Starting Fold {current_fold_num}/{N_SPLITS} ---")

        # Split data and sort by ranker_id for contiguous groups
        train_fold_df = full_df[train_idx].sort('ranker_id')
        val_fold_df = full_df[val_idx].sort('ranker_id')

        X_train = train_fold_df[feature_cols].to_pandas()
        y_train = train_fold_df['selected'].to_pandas()
        # Calculate group sizes for LightGBM (rows per ranker_id in original order)
        group_train = (
            train_fold_df
            .group_by('ranker_id', maintain_order=True)
            .agg(pl.count())
            .get_column('count')
            .to_numpy()
        )

        X_val = val_fold_df[feature_cols].to_pandas()
        y_val = val_fold_df['selected'].to_pandas()
        group_val = (
            val_fold_df
            .group_by('ranker_id', maintain_order=True)
            .agg(pl.count())
            .get_column('count')
            .to_numpy()
        )
        
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # Initialize and train the model
        trainer = RankerTrainer(model_params=MODEL_PARAMS)
        trainer.train(X_train, y_train, group_train, X_val, y_val, group_val)

        # Generate predictions on the validation set
        val_preds = trainer.predict(X_val)
        
        val_results_df = val_fold_df.select(['ranker_id', 'selected']).with_columns(
            pl.Series("predicted_score", val_preds)
        )

        # Calculate and log HitRate@3
        hit_rate = ValidationUtils.calculate_hitrate_at_k(val_results_df, k=3)
        fold_scores.append(hit_rate)
        logger.info(f"Fold {current_fold_num} HitRate@3: {hit_rate:.4f}")

        # Save the model for this fold
        trainer.save_model(f"lgbm_ranker_fold_{current_fold_num}.txt")
        
        oof_preds.append(val_results_df)

    # --- 5. Final Evaluation ---
    logger.info("--- Cross-Validation Complete ---")
    avg_hit_rate = np.mean(fold_scores)
    logger.info(f"Average HitRate@3 across {N_SPLITS} folds: {avg_hit_rate:.4f}")

    # Save out-of-fold predictions
    oof_df = pl.concat(oof_preds)
    oof_df.write_parquet("processed/oof_predictions.parquet")
    logger.info("Out-of-fold predictions saved to processed/oof_predictions.parquet")

if __name__ == "__main__":
    main()