import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb

from data_pipeline import DataPipeline, DataConfig
from aggressive_features import AggressiveFeatureEngineer
from ranker_trainer import RankerTrainer
from utils import CrossValidationUtils, ValidationUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test aggressive feature engineering approach"""
    logger.info("Testing ultra-aggressive feature engineering...")
    
    # --- 1. Load data ---
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    # Use a sample for faster testing
    full_df_lazy = pipeline.loader.load_structured_data("train")
    
    # Sample 5% of data for faster testing
    logger.info("Sampling 5% of data for testing...")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.05, seed=42)
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    full_df = full_df_lazy.collect()
    logger.info(f"Sample data shape: {full_df.shape}")
    
    # Filter to groups >10 items
    group_sizes = full_df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    full_df = full_df.join(group_sizes, on='ranker_id', how='inner')
    full_df = full_df.filter(pl.col('group_size') > 10).drop('group_size')
    logger.info(f"Filtered data shape: {full_df.shape}")
    
    # --- 2. CV splits for target encoding ---
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=full_df, group_col='profileId', stratify_col='selected', n_splits=5, seed=42
    )
    
    # --- 3. Apply aggressive feature engineering ---
    engineer = AggressiveFeatureEngineer()
    full_df = engineer.engineer_aggressive_features(full_df, cv_splits)
    
    # --- 4. Feature selection ---
    exclude_cols = ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute',
                    'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration']
    
    # Get all target-encoded columns
    te_cols = [col for col in full_df.columns if col.endswith('_te')]
    
    # Get other numeric columns
    numeric_cols = []
    for col in full_df.columns:
        if col not in exclude_cols and not col.endswith('_te'):
            if full_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Int16]:
                numeric_cols.append(col)
    
    feature_cols = te_cols + numeric_cols
    logger.info(f"Using {len(feature_cols)} features ({len(te_cols)} target-encoded)")
    
    # --- 5. Train on single fold ---
    train_idx, val_idx = cv_splits[0]
    
    train_fold_df = full_df[train_idx].sort('ranker_id')
    val_fold_df = full_df[val_idx].sort('ranker_id')

    X_train = train_fold_df.select(feature_cols).to_pandas()
    y_train = train_fold_df['selected'].to_numpy()
    group_train = train_fold_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()

    X_val = val_fold_df.select(feature_cols).to_pandas()
    y_val = val_fold_df['selected'].to_numpy()
    group_val = val_fold_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # --- 6. Train with optimized parameters ---
    with open("processed/optuna_best_params.json", 'r') as f:
        import json
        best_params = json.load(f)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'n_estimators': 2000,  # Reduced for faster testing
        **best_params
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(100, verbose=False)]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks
    )
    
    # --- 7. Evaluate ---
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    val_results_df = val_fold_df.select(['ranker_id', 'selected']).with_columns(
        pl.Series("predicted_score", val_preds)
    )
    
    hit_rate = ValidationUtils.calculate_hitrate_at_k(val_results_df, k=3)
    logger.info(f"AGGRESSIVE APPROACH HitRate@3: {hit_rate:.5f}")
    
    # Show feature importance
    importance_df = pl.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance('gain')
    }).sort('importance', descending=True)
    
    logger.info("Top 10 most important features:")
    for row in importance_df.head(10).iter_rows():
        logger.info(f"  {row[0]}: {row[1]}")

if __name__ == "__main__":
    main()