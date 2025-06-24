"""
Minimal solution using only the most powerful features identified
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb

from data_pipeline import DataPipeline, DataConfig
from utils import CrossValidationUtils, ValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_minimal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create only the most powerful features"""
    
    logger.info("Creating minimal high-power features...")
    
    # 1. POSITION features (9,704 importance)
    df = df.with_row_count('global_position')
    df = df.with_columns([
        # Position within session (most important positional feature)
        pl.col('global_position').rank().over('ranker_id').alias('position_in_session'),
        
        # Position percentile within session
        ((pl.col('global_position').rank().over('ranker_id') - 1) / 
         (pl.count().over('ranker_id') - 1)).alias('position_percentile'),
        
        # Is first option? (binary version)
        (pl.col('global_position').rank().over('ranker_id') == 1).cast(pl.Int8).alias('is_first_option')
    ])
    
    # 2. PRICE RATIO features (31,583 and 7,006 importance)
    session_price_stats = df.group_by('ranker_id').agg([
        pl.col('totalPrice').min().alias('session_min_price'),
        pl.col('totalPrice').median().alias('session_median_price'),
        pl.col('totalPrice').std().alias('session_price_std')
    ])
    
    df = df.join(session_price_stats, on='ranker_id')
    
    df = df.with_columns([
        # The two most important price features
        (pl.col('totalPrice') / pl.col('session_median_price')).alias('price_vs_median_ratio'),
        (pl.col('totalPrice') / pl.col('session_min_price')).alias('price_vs_cheapest_ratio'),
        
        # Additional price features that might help
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank'),
        (pl.col('totalPrice') == pl.col('session_min_price')).cast(pl.Int8).alias('is_cheapest'),
        
        # Price z-score
        ((pl.col('totalPrice') - pl.col('session_median_price')) / pl.col('session_price_std')).alias('price_z_score')
    ])
    
    # 3. POLICY feature (if available) - simple version
    if 'pricingInfo_isAccessTP' in df.columns:
        df = df.with_columns([
            pl.col('pricingInfo_isAccessTP').cast(pl.Int8).alias('is_policy_compliant')
        ])
    
    # 4. ULTIMATE SIMPLE SCORE
    # Combine the most powerful signals into one score
    if 'is_policy_compliant' in df.columns:
        policy_boost = pl.col('is_policy_compliant') * 2  # Policy boost
    else:
        policy_boost = pl.lit(0)
    
    df = df.with_columns([
        # Simple scoring: favor first position + cheap price + policy
        (pl.col('is_first_option') * 10 +          # First position is powerful
         pl.col('is_cheapest') * 5 +              # Cheapest is powerful
         policy_boost +                           # Policy compliance
         (1 / pl.col('price_vs_median_ratio'))    # Lower price ratio is better
        ).alias('simple_score')
    ])
    
    return df

def train_minimal_model(df: pl.DataFrame, cv_splits: list) -> dict:
    """Train with minimal features"""
    
    logger.info("Training minimal model...")
    
    # Use only the most powerful features
    minimal_features = [
        'position_percentile',      # 9,704 importance
        'price_vs_median_ratio',    # 31,583 importance  
        'price_vs_cheapest_ratio',  # 7,006 importance
        'position_in_session',      # Direct position
        'price_rank',               # Price ranking
        'is_first_option',          # First position flag
        'is_cheapest',              # Cheapest flag
        'simple_score',             # Combined score
        'price_z_score'             # Price standardization
    ]
    
    # Add policy if available
    if 'is_policy_compliant' in df.columns:
        minimal_features.append('is_policy_compliant')
    
    logger.info(f"Using {len(minimal_features)} minimal features: {minimal_features}")
    
    # Single fold test
    train_idx, val_idx = cv_splits[0]
    
    train_df = df[train_idx].sort('ranker_id')
    val_df = df[val_idx].sort('ranker_id')
    
    X_train = train_df.select(minimal_features).fill_null(0).to_pandas()
    y_train = train_df['selected'].to_numpy()
    group_train = train_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    X_val = val_df.select(minimal_features).fill_null(0).to_pandas()
    y_val = val_df['selected'].to_numpy()
    group_val = val_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Optimized parameters for minimal features
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,        # Higher learning rate for simple model
        'num_leaves': 256,           # Moderate complexity
        'min_child_samples': 20,     
        'feature_fraction': 1.0,     # Use all features (there are few)
        'bagging_fraction': 0.9,
        'lambda_l1': 0.01,           # Light regularization
        'lambda_l2': 0.01,
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 2000,
        'n_jobs': -1
    }
    
    # Train
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    
    # Evaluate
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_results_df = val_df.select(['ranker_id', 'selected']).with_columns(
        pl.Series('predicted_score', val_preds)
    )
    
    hit_rate = ValidationUtils.calculate_hitrate_at_k(val_results_df, k=3)
    logger.info(f"MINIMAL MODEL HitRate@3: {hit_rate:.5f}")
    
    return {
        'model': model,
        'features': minimal_features,
        'hit_rate': hit_rate
    }

def test_simple_rules(df: pl.DataFrame) -> None:
    """Test ultra-simple rule-based approaches"""
    
    logger.info("Testing simple rules...")
    
    # Rule 1: Always pick the first option
    df_rule1 = df.with_columns([
        (pl.col('global_position').rank().over('ranker_id') == 1).cast(pl.Float64).alias('rule1_score')
    ])
    
    eval_df1 = df_rule1.select(['ranker_id', 'selected', 'rule1_score'])
    hit_rate1 = ValidationUtils.calculate_hitrate_at_k(eval_df1.rename({'rule1_score': 'predicted_score'}), k=3)
    logger.info(f"Rule 1 (first option): {hit_rate1:.5f}")
    
    # Rule 2: Always pick the cheapest
    df_rule2 = df.with_columns([
        (pl.col('totalPrice') == pl.col('totalPrice').min().over('ranker_id')).cast(pl.Float64).alias('rule2_score')
    ])
    
    eval_df2 = df_rule2.select(['ranker_id', 'selected', 'rule2_score'])
    hit_rate2 = ValidationUtils.calculate_hitrate_at_k(eval_df2.rename({'rule2_score': 'predicted_score'}), k=3)
    logger.info(f"Rule 2 (cheapest): {hit_rate2:.5f}")
    
    # Rule 3: First + cheapest combination
    df_rule3 = df.with_columns([
        ((pl.col('global_position').rank().over('ranker_id') == 1).cast(pl.Int8) * 2 +
         (pl.col('totalPrice') == pl.col('totalPrice').min().over('ranker_id')).cast(pl.Int8)).cast(pl.Float64).alias('rule3_score')
    ])
    
    eval_df3 = df_rule3.select(['ranker_id', 'selected', 'rule3_score'])
    hit_rate3 = ValidationUtils.calculate_hitrate_at_k(eval_df3.rename({'rule3_score': 'predicted_score'}), k=3)
    logger.info(f"Rule 3 (first+cheapest): {hit_rate3:.5f}")

def main():
    """Run the minimal solution"""
    
    logger.info("MINIMAL SOLUTION - USING ONLY THE MOST POWERFUL FEATURES")
    
    # Load data
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    # Sample for testing
    full_df_lazy = pipeline.loader.load_structured_data("train")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.05, seed=42)  # 5% sample
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    full_df = full_df_lazy.collect()
    logger.info(f"Sample data shape: {full_df.shape}")
    
    # Filter to large groups
    group_sizes = full_df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    full_df = full_df.join(group_sizes, on='ranker_id', how='inner')
    full_df = full_df.filter(pl.col('group_size') > 10).drop('group_size')
    logger.info(f"Large groups: {full_df.shape}")
    
    # CV splits
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=full_df, group_col='profileId', stratify_col='selected', n_splits=5, seed=42
    )
    
    # Create minimal features
    full_df = create_minimal_features(full_df)
    logger.info(f"With minimal features: {full_df.shape}")
    
    # Test simple rules first
    test_simple_rules(full_df)
    
    # Train minimal model
    results = train_minimal_model(full_df, cv_splits)
    
    # Show feature importance
    importance = results['model'].feature_importance('gain')
    logger.info("Feature importance:")
    for feat, imp in zip(results['features'], importance):
        logger.info(f"  {feat}: {imp}")

if __name__ == "__main__":
    main()