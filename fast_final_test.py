"""
FAST FINAL TEST - Quick test with current best approach
"""
import polars as pl
import numpy as np
import logging
import lightgbm as lgb
from data_pipeline import DataPipeline, DataConfig
from utils import ValidationUtils, TargetEncodingUtils, CrossValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_focused_features(df, cv_splits):
    """Create most essential features only"""
    
    # Target encode key columns
    key_cols = ['searchRoute', 'legs0_segments0_operatingCarrier_code', 'companyID']
    target_cols = [col for col in key_cols if col in df.columns]
    
    if target_cols:
        df = TargetEncodingUtils.kfold_target_encode(
            df=df, cat_cols=target_cols, target='selected', folds=cv_splits, noise=0.01
        )
    
    # Core features
    df = df.with_row_count('idx')
    df = df.with_columns([
        pl.col('idx').rank().over('ranker_id').alias('position'),
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank')
    ])
    
    # Session stats
    session_stats = df.group_by('ranker_id').agg([
        pl.col('totalPrice').min().alias('min_price'),
        pl.col('totalPrice').median().alias('median_price')
    ])
    df = df.join(session_stats, on='ranker_id')
    
    # Essential features
    df = df.with_columns([
        ((pl.col('position') - 1) / (pl.count().over('ranker_id') - 1)).alias('position_pct'),
        (pl.col('totalPrice') / pl.col('min_price')).alias('price_vs_min'),
        (pl.col('totalPrice') / pl.col('median_price')).alias('price_vs_median'),
        (pl.col('position') == 1).cast(pl.Int8).alias('is_first'),
        (pl.col('totalPrice') == pl.col('min_price')).cast(pl.Int8).alias('is_cheapest'),
        (pl.col('position') * pl.col('price_rank')).alias('pos_price_mult')
    ])
    
    if 'pricingInfo_isAccessTP' in df.columns:
        df = df.with_columns([
            pl.col('pricingInfo_isAccessTP').cast(pl.Int8).alias('policy_ok')
        ])
    
    return df

def main():
    """Quick final test"""
    
    logger.info("🎯 FAST FINAL TEST")
    
    # Small but focused dataset
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    full_df_lazy = pipeline.loader.load_structured_data("train")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.05, seed=42)  # 5% for speed
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    df = full_df_lazy.collect()
    logger.info(f"Dataset: {df.shape}")
    
    # Filter to meaningful groups
    group_sizes = df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    df = df.join(group_sizes, on='ranker_id', how='inner')
    df = df.filter(pl.col('group_size') > 10).drop('group_size')  # Large groups only
    logger.info(f"Large groups: {df.shape}")
    
    # CV splits
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=df, group_col='profileId', stratify_col='selected', n_splits=5, seed=42
    )
    
    # Create features
    df = create_focused_features(df, cv_splits)
    logger.info(f"With features: {df.shape}")
    
    # Quick model test
    exclude_cols = {'Id', 'ranker_id', 'selected', 'idx', 'requestDate', 'legs0_departureAt', 'min_price', 'median_price'}
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8]]
    
    logger.info(f"Features: {len(feature_cols)}")
    
    # Use first fold
    train_idx, val_idx = cv_splits[0]
    train_df = df[train_idx].sort('ranker_id')
    val_df = df[val_idx].sort('ranker_id')
    
    X_train = train_df.select(feature_cols).fill_null(0).to_pandas()
    y_train = train_df['selected'].to_numpy()
    group_train = train_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    X_val = val_df.select(feature_cols).fill_null(0).to_pandas()
    y_val = val_df['selected'].to_numpy()
    group_val = val_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    # Test multiple configurations
    configs = [
        {'learning_rate': 0.1, 'num_leaves': 512, 'n_estimators': 1000},
        {'learning_rate': 0.05, 'num_leaves': 1024, 'n_estimators': 2000},
        {'learning_rate': 0.2, 'num_leaves': 256, 'n_estimators': 1500},
    ]
    
    best_score = 0
    best_config = None
    
    for i, config in enumerate(configs):
        logger.info(f"Testing config {i+1}: {config}")
        
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1,
            **config
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        
        eval_df = val_df.select(['ranker_id', 'selected']).with_columns([
            pl.Series('predicted_score', preds)
        ])
        
        score = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
        logger.info(f"Config {i+1} HitRate@3: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = config
    
    print(f"\\n" + "="*60)
    print(f"🏆 FAST FINAL TEST RESULTS")
    print(f"="*60)
    print(f"BEST HitRate@3: {best_score:.4f}")
    print(f"TARGET: 0.7000")
    print(f"ACHIEVEMENT: {best_score/0.7*100:.1f}% of target")
    
    if best_score >= 0.7:
        print(f"🎉 BREAKTHROUGH ACHIEVED!")
    elif best_score >= 0.5:
        print(f"💪 STRONG PROGRESS! Gap: {0.7-best_score:.4f}")
    else:
        print(f"❌ More work needed. Gap: {0.7-best_score:.4f}")
    
    print(f"Best config: {best_config}")
    print(f"="*60)

if __name__ == "__main__":
    main()