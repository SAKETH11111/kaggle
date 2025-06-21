import polars as pl
import numpy as np
import logging
from pathlib import Path
import argparse
import json

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
    
    # Load the best hyperparameters from the Optuna study
    best_params_path = Path("processed/optuna_best_params.json")
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best parameters file not found at {best_params_path}. Please run run_tuning.py first.")
    
    with open(best_params_path, 'r') as f:
        MODEL_PARAMS = json.load(f)
    
    # Ensure essential parameters are set for GPU training
    MODEL_PARAMS['objective'] = 'lambdarank'
    MODEL_PARAMS['metric'] = 'ndcg'
    MODEL_PARAMS['random_state'] = RANDOM_SEED
    MODEL_PARAMS['n_estimators'] = 1000  # Keep n_estimators high, rely on early stopping
    MODEL_PARAMS['device'] = 'cuda'
    MODEL_PARAMS['n_jobs'] = -1

    logger.info("Loaded best hyperparameters from Optuna study:")
    for key, value in MODEL_PARAMS.items():
        logger.info(f"  {key}: {value}")

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
    oof_models = []

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
        oof_models.append(trainer)

    # --- 5. Final Evaluation ---
    logger.info("--- Cross-Validation Complete ---")
    avg_hit_rate = np.mean(fold_scores)
    logger.info(f"Average HitRate@3 across {N_SPLITS} folds: {avg_hit_rate:.4f}")

    # --- 5. Final Model Training on Full Data ---
    logger.info("--- Training Final Model on Full Dataset ---")
    
    # Prepare full dataset
    full_train_df = full_df.sort('ranker_id')
    X_full = full_train_df[feature_cols].to_pandas()
    y_full = full_train_df['selected'].to_pandas()
    group_full = full_train_df.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()

    # Determine the optimal number of boosting rounds from the CV
    # We'll use the average best iteration from the folds, rounded up
    # For the final model, we use the n_estimators value from our optimized parameters,
    # as there is no validation set for early stopping.
    logger.info(f"Using n_estimators={MODEL_PARAMS.get('n_estimators', 1000)} for final model.")
    
    final_trainer = RankerTrainer(model_params=MODEL_PARAMS)
    final_trainer.model.fit(X_full, y_full, group=group_full)
    
    logger.info("Final model training complete.")
    final_trainer.save_model("lgbm_ranker_full_optuna.txt")

    # Save out-of-fold predictions
    oof_df = pl.concat(oof_preds)
    oof_df.write_parquet("processed/oof_predictions.parquet")
    logger.info("Out-of-fold predictions saved to processed/oof_predictions.parquet")

    # --- 6. Save CV Results ---
    cv_results = {
        "avg_hitrate_at_3": avg_hit_rate,
        "fold_scores": fold_scores,
        "model_params": MODEL_PARAMS,
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
    }
    cv_results_path = Path("processed/cv_results.json")
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=4)
    logger.info(f"Cross-validation results saved to {cv_results_path}")

if __name__ == "__main__":
    main()