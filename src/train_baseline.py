import polars as pl
import numpy as np
import logging
from pathlib import Path
import argparse
import json
from sklearn.model_selection import StratifiedGroupKFold

from data_pipeline import DataPipeline, DataConfig
from ranker_trainer import RankerTrainer
from utils import ValidationUtils

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
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation.",
    )
    args = parser.parse_args()

    logger.info("Starting end-to-end training pipeline...")

    # --- 1. Configuration ---
    config = DataConfig()
    N_SPLITS = args.folds
    RANDOM_SEED = 42
    
    # Load the best hyperparameters from the Optuna study
    best_params_path = Path("processed/optuna_best_params.json")
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best parameters file not found at {best_params_path}. Please run run_tuning.py first.")
    
    with open(best_params_path, 'r') as f:
        MODEL_PARAMS = json.load(f)
    
    MODEL_PARAMS['objective'] = 'lambdarank'
    MODEL_PARAMS['metric'] = 'ndcg'
    MODEL_PARAMS['random_state'] = RANDOM_SEED
    MODEL_PARAMS['n_estimators'] = 1000  # Keep n_estimators high, rely on early stopping
    MODEL_PARAMS['device'] = 'cpu'
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

    # Define the baseline feature set based on The_research.md
    # This list is based on the core value drivers mentioned in the research document
    # (price, duration, directness, policy, position, etc.).
    baseline_feature_cols = [
        # Position/Rank Features
        'position', 'price_rank', 'duration_rank', 'price_percentile', 'duration_percentile',
        # Price Features
        'totalPrice', 'price_per_segment', 'price_vs_avg_search_price', 'price_diff_from_min',
        'price_diff_from_mean', 'group_price_std', 'group_price_range', 'is_cheapest',
        'is_pricier_than_average',
        # Duration/Time Features
        'total_duration_minutes', 'duration_per_segment', 'departure_hour', 'departure_weekday',
        'is_weekend', 'is_red_eye', 'booking_lead_days',
        # Route/Flight Features
        'num_stops', 'is_direct', 'is_roundtrip', 'num_segments', 'distance_km',
        # Policy/Flexibility Features
        'pricingInfo_isAccessTP', 'can_cancel', 'can_exchange', 'cancellation_fee_rate',
        'exchange_fee_rate', 'has_corporate_tariff', 'corporate_policy_interaction',
        'group_policy_compliance_rate',
        # User/Context Features
        'isVip', 'books_by_self', 'group_size', 'unique_carriers_in_group',
        # Interaction features mentioned in research
        'price_x_duration', 'vip_x_price',
        # JSON-derived features mentioned in research
        'avg_total_price', 'total_seats_available', 'avg_policy_compliant',
    ]

    # Filter to only available columns, as not all may be generated in this run
    available_cols = full_df.columns
    feature_cols = [col for col in baseline_feature_cols if col in available_cols]
    
    # Verify that core drivers are present
    core_drivers = ['position', 'price_rank', 'duration_rank', 'pricingInfo_isAccessTP']
    missing_drivers = [driver for driver in core_drivers if driver not in feature_cols]
    if missing_drivers:
        logger.warning(f"Core driver features missing from data: {missing_drivers}")

    logger.info(f"Using {len(feature_cols)} baseline features for training.")
    logger.info(f"Selected features: {feature_cols}")

    # --- 3. Cross-Validation Setup ---
    logger.info(f"Setting up {N_SPLITS}-fold StratifiedGroupKFold cross-validation...")
    y = full_df['selected']
    groups = full_df['profileId']

    # StratifiedGroupKFold requires X for shape info, but doesn't use its values for splitting.
    # A placeholder is sufficient and memory-efficient.
    X_placeholder = np.zeros((len(full_df), 1))

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = skf.split(X_placeholder, y.to_numpy(), groups.to_numpy())

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
    group_full = full_train_df.group_by('ranker_id', maintain_order=True).agg(pl.count()).get_column('count').to_numpy()

    # Determine the optimal number of boosting rounds from the CV
    # We'll use the average best iteration from the folds, rounded up
    # For the final model, we use the n_estimators value from our optimized parameters,
    # as there is no validation set for early stopping.
    logger.info(f"Using n_estimators={MODEL_PARAMS.get('n_estimators', 1000)} for final model.")
    
    final_trainer = RankerTrainer(model_params=MODEL_PARAMS)
    final_trainer.model.fit(X_full, y_full, group=group_full)
    
    logger.info("Final model training complete.")
    final_trainer.save_model("lgbm_ranker_full_optuna.txt")

    # --- 6. Feature Importance Plot ---
    logger.info("Generating and saving feature importance plot...")
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        feature_importances = final_trainer.model.feature_importances_
        
        if hasattr(final_trainer.model, 'booster_') and final_trainer.model.booster_:
            feature_names = final_trainer.model.booster_.feature_name()
        else:
            feature_names = X_full.columns

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        top_30_features = importance_df.head(30)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 12))
        
        ax.barh(top_30_features['feature'], top_30_features['importance'], color='c')
        
        ax.set_xlabel("Feature Importance (Gain)", fontsize=14, labelpad=10)
        ax.set_ylabel("Feature", fontsize=14, labelpad=10)
        ax.set_title("Top 30 Feature Importances for Baseline LGBMRanker", fontsize=18, pad=20)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        for i, v in enumerate(top_30_features['importance']):
            ax.text(v + 1, i + .25, str(round(v, 2)), color='gray', fontweight='bold')

        fig.tight_layout()
        
        output_path = Path("processed/baseline_feature_importance.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Feature importance plot saved to {output_path}")

        # Save importance dataframe to JSON for verification
        importance_json_path = Path("processed/baseline_feature_importance.json")
        importance_df.to_json(importance_json_path, orient='records', indent=4)
        logger.info(f"Feature importance data saved to {importance_json_path}")

    except ImportError:
        logger.warning("Matplotlib not found. Skipping feature importance plot generation.")
    except Exception as e:
        logger.error(f"An error occurred during feature importance plot generation: {e}")

    # --- 7. Save OOF and CV Results ---
    # Save out-of-fold predictions only if CV was run
    if oof_preds:
        oof_df = pl.concat(oof_preds)
        oof_df.write_parquet("processed/oof_predictions.parquet")
        logger.info("Out-of-fold predictions saved to processed/oof_predictions.parquet")
    else:
        logger.info("No out-of-fold predictions to save (CV was skipped).")

    # --- 8. Save CV Results ---
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