import polars as pl
import numpy as np
import logging
from pathlib import Path
import argparse
import json

from data_pipeline import DataPipeline, DataConfig
from feature_engineering import FeatureEngineer
from ranker_trainer import RankerTrainer
from utils import CrossValidationUtils, ValidationUtils, TargetEncodingUtils
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the end-to-end training pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--fold", type=int, default=0, help="Run a single fold (1-based). 0 to run all.")
    parser.add_argument("--features", type=str, default="week1", choices=["baseline", "week1"], help="Feature set to use.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Fraction of data to use for a quick run.")
    args = parser.parse_args()

    logger.info(f"Starting training pipeline with feature set: {args.features}")

    # --- 1. Configuration ---
    config = DataConfig()
    N_SPLITS = 5
    RANDOM_SEED = 42
    
    best_params_path = Path("processed/optuna_best_params.json")
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best parameters file not found at {best_params_path}. Run tuning first.")
    
    with open(best_params_path, 'r') as f:
        MODEL_PARAMS = json.load(f)
    
    MODEL_PARAMS.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'random_state': RANDOM_SEED,
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.05,
        'lambda_l2': 0.05,
        'num_leaves': 1024,
        'min_child_samples': 20,
        'min_gain_to_split': 0.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': RANDOM_SEED,
        'boosting_type': 'gbdt',
    })
    logger.info(f"Using model parameters: {MODEL_PARAMS}")

    # --- 2. Data Preparation ---
    logger.info("Preparing training data...")
    pipeline = DataPipeline(config)
    
    # New feature engineering pipeline for Week 1
    full_df_lazy = pipeline.loader.load_structured_data("train")
    
    if args.sample_frac:
        logger.info(f"Sampling data to {args.sample_frac} fraction.")
        unique_ranker_ids = full_df_lazy.select("ranker_id").unique()
        sampled_ranker_ids = unique_ranker_ids.sample(fraction=args.sample_frac, seed=RANDOM_SEED)
        full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")

    if args.features == "baseline":
        # Legacy data prep for baseline
        # Note: This path might need updates if baseline features are different
        feature_engineer = None # Or a baseline-specific feature engineer
        full_df_lazy = pipeline.prepare_training_data() # This might need adjustment
    else:
        feature_engineer = FeatureEngineer()
        full_df_lazy = feature_engineer.engineer_features(full_df_lazy)

    full_df = full_df_lazy.collect()
    logger.info(f"Full training data shape: {full_df.shape}")

    # Filter to groups >10 items to align with evaluation metric
    group_sizes = full_df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    full_df = full_df.join(group_sizes, on='ranker_id', how='inner')
    full_df = full_df.filter(pl.col('group_size') > 10).drop('group_size')

    # --- 3. Cross-Validation Setup (before feature selection to enable target encoding) ---
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=full_df, group_col='profileId', stratify_col='selected', n_splits=N_SPLITS, seed=RANDOM_SEED
    )

    # --- 4. Target Encoding for high-cardinality features ---
    target_encode_cols = []
    for col in ['searchRoute', 'legs0_segments0_operatingCarrier_code', 'companyID']:
        if col in full_df.columns:
            target_encode_cols.append(col)
    
    if target_encode_cols and args.features == "week1":
        logger.info(f"Applying target encoding to: {target_encode_cols}")
        full_df = TargetEncodingUtils.kfold_target_encode(
            df=full_df, 
            cat_cols=target_encode_cols, 
            target='selected', 
            folds=cv_splits, 
            noise=0.01
        )
        # Remove frequency-encoded searchRoute if it exists (superseded by target encoding)
        if 'searchRoute_freq' in full_df.columns:
            full_df = full_df.drop('searchRoute_freq')
        logger.info("Target encoding complete")

    # --- 5. Feature Selection ---
    if feature_engineer:
        feature_cols = feature_engineer.get_feature_names(full_df)
    else: # Baseline features
        feature_cols = [col for col in full_df.columns if col not in ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute']]
    
    # Add target-encoded features to feature list (avoiding duplicates)
    te_feature_cols = [f"{col}_te" for col in target_encode_cols if f"{col}_te" in full_df.columns]
    for te_col in te_feature_cols:
        if te_col not in feature_cols:
            feature_cols.append(te_col)
    
    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    
    categorical_cols = [col for col in feature_cols if full_df[col].dtype in [pl.Categorical, pl.Utf8] or col.endswith("_le")]
    
    logger.info(f"Using {len(feature_cols)} features (including {len(te_feature_cols)} target-encoded). Categorical features: {categorical_cols}")

    oof_preds, fold_scores = [], []
    folds_to_run = [args.fold - 1] if args.fold > 0 else range(N_SPLITS)

    # --- 6. Training Loop ---
    for fold in folds_to_run:
        train_idx, val_idx = cv_splits[fold]
        logger.info(f"--- Starting Fold {fold + 1}/{N_SPLITS} ---")

        train_fold_df = full_df[train_idx].sort('ranker_id')
        val_fold_df = full_df[val_idx].sort('ranker_id')

        X_train = train_fold_df.select(feature_cols).to_pandas()
        y_train = train_fold_df['selected'].to_numpy()
        group_train = train_fold_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()

        X_val = val_fold_df.select(feature_cols).to_pandas()
        y_val = val_fold_df['selected'].to_numpy()
        group_val = val_fold_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
        
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        trainer = RankerTrainer(model_params=MODEL_PARAMS)
        callbacks = [lgb.early_stopping(100, verbose=False)]
        
        trainer.train(X_train, y_train, group_train, X_val, y_val, group_val, categorical_features=categorical_cols, callbacks=callbacks)

        val_preds_scores = trainer.predict(X_val)
        val_results_df = val_fold_df.select(['ranker_id', 'selected']).with_columns(pl.Series("predicted_score", val_preds_scores))

        hit_rate = ValidationUtils.calculate_hitrate_at_k(val_results_df, k=3)
        fold_scores.append(hit_rate)
        logger.info(f"Fold {fold + 1} HitRate@3: {hit_rate:.5f}")

        trainer.save_model(f"lgbm_ranker_{args.features}_fold_{fold + 1}.txt")
        oof_preds.append(val_results_df)

    # --- 7. Final Evaluation ---
    if not folds_to_run:
        logger.warning("No folds were run. Exiting.")
        return

    avg_hit_rate = np.mean(fold_scores)
    logger.info(f"--- CV Complete ---")
    logger.info(f"Average HitRate@3 across {len(folds_to_run)} folds: {avg_hit_rate:.5f}")

    oof_df = pl.concat(oof_preds)
    oof_df.write_parquet(f"processed/oof_predictions_{args.features}.parquet")
    logger.info(f"OOF predictions saved for feature set '{args.features}'.")

    # --- 8. Save CV Results ---
    cv_results = {
        "feature_set": args.features,
        "avg_hitrate_at_3": avg_hit_rate,
        "fold_scores": fold_scores,
        "model_params": MODEL_PARAMS,
    }
    cv_results_path = Path(f"processed/cv_results_{args.features}.json")
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=4)
    logger.info(f"CV results saved to {cv_results_path}")

if __name__ == "__main__":
    main()