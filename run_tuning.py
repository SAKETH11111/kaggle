import polars as pl
import numpy as np
import logging
from pathlib import Path
import argparse
import json
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline, DataConfig
from feature_engineering import FeatureEngineer
from ranker_trainer import RankerTrainer
from utils import CrossValidationUtils, ValidationUtils, TargetEncodingUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, group_train, X_val, y_val, group_val, categorical_features):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Suggest hyperparameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 256, 4096),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
        'min_gain_to_split': 0.0,
        'n_estimators': 1000,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, categorical_feature=categorical_features, reference=train_data)
    
    callbacks = [lgb.early_stopping(200, verbose=False)]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks
    )
    
    # Predict and calculate HitRate@3
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Create results DataFrame for HitRate calculation
    val_results_df = pl.DataFrame({
        'predicted_score': val_preds,
        'selected': y_val
    })
    
    # Add ranker_id by reconstructing groups
    ranker_ids = []
    group_idx = 0
    for i, group_size in enumerate(group_val):
        ranker_ids.extend([f"group_{i}"] * group_size)
        group_idx += group_size
    
    val_results_df = val_results_df.with_columns(pl.Series("ranker_id", ranker_ids))
    
    hit_rate = ValidationUtils.calculate_hitrate_at_k(val_results_df, k=3, min_group_size=1)
    
    return hit_rate

def main():
    """
    Run hyperparameter tuning using Optuna.
    """
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning.")
    parser.add_argument("--features", type=str, default="week1", choices=["baseline", "week1"], help="Feature set to use.")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="Fraction of data to use for tuning.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 1 hour).")
    args = parser.parse_args()

    logger.info(f"Starting hyperparameter tuning with {args.trials} trials on {args.sample_ratio*100}% of data")

    # --- 1. Configuration ---
    config = DataConfig()
    RANDOM_SEED = 42
    
    # --- 2. Data Preparation ---
    logger.info("Preparing tuning data...")
    pipeline = DataPipeline(config)
    
    # Load and sample data
    full_df_lazy = pipeline.loader.load_structured_data("train")
    
    logger.info(f"Sampling data to {args.sample_ratio} fraction.")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=args.sample_ratio, seed=RANDOM_SEED)
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")

    # Feature engineering
    if args.features == "week1":
        feature_engineer = FeatureEngineer()
        full_df_lazy = feature_engineer.engineer_features(full_df_lazy)
    else:
        feature_engineer = None

    full_df = full_df_lazy.collect()
    logger.info(f"Tuning data shape: {full_df.shape}")

    # Filter to groups >10 items
    group_sizes = full_df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    full_df = full_df.join(group_sizes, on='ranker_id', how='inner')
    full_df = full_df.filter(pl.col('group_size') > 10).drop('group_size')

    # --- 3. Create single validation split for tuning ---
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=full_df, group_col='profileId', stratify_col='selected', n_splits=5, seed=RANDOM_SEED
    )
    
    # Use first fold for tuning
    train_idx, val_idx = cv_splits[0]

    # --- 4. Target Encoding (if applicable) ---
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
        # Remove frequency-encoded searchRoute if it exists
        if 'searchRoute_freq' in full_df.columns:
            full_df = full_df.drop('searchRoute_freq')

    # --- 5. Feature Selection ---
    if feature_engineer:
        feature_cols = feature_engineer.get_feature_names(full_df)
    else:
        feature_cols = [col for col in full_df.columns if col not in ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute']]
    
    # Add target-encoded features (avoiding duplicates)
    te_feature_cols = [f"{col}_te" for col in target_encode_cols if f"{col}_te" in full_df.columns]
    for te_col in te_feature_cols:
        if te_col not in feature_cols:
            feature_cols.append(te_col)
    
    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    
    categorical_cols = [col for col in feature_cols if full_df[col].dtype in [pl.Categorical, pl.Utf8] or col.endswith("_le")]
    
    logger.info(f"Using {len(feature_cols)} features for tuning")

    # --- 6. Prepare train/val data ---
    train_fold_df = full_df[train_idx].sort('ranker_id')
    val_fold_df = full_df[val_idx].sort('ranker_id')

    X_train = train_fold_df.select(feature_cols).to_pandas()
    y_train = train_fold_df['selected'].to_numpy()
    group_train = train_fold_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()

    X_val = val_fold_df.select(feature_cols).to_pandas()
    y_val = val_fold_df['selected'].to_numpy()
    group_val = val_fold_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # --- 7. Run Optuna optimization ---
    logger.info("Starting Optuna optimization...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name='lgbm_ranker_optimization'
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, group_train, X_val, y_val, group_val, categorical_cols),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True
    )

    # --- 8. Save results ---
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best HitRate@3: {best_score:.5f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Save best parameters
    output_path = Path("processed/optuna_best_params.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Best parameters saved to {output_path}")
    
    # Save study results
    study_path = Path("processed/optuna_study_results.json")
    study_results = {
        "best_score": best_score,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "feature_set": args.features,
        "sample_ratio": args.sample_ratio
    }
    
    with open(study_path, 'w') as f:
        json.dump(study_results, f, indent=4)
    
    logger.info(f"Study results saved to {study_path}")

if __name__ == "__main__":
    main()