"""
BREAKTHROUGH ENSEMBLE - Multi-strategy approach to reach 0.7+ HitRate@3
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline, DataConfig
from utils import ValidationUtils, TargetEncodingUtils, CrossValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_breakthrough_features(df, cv_splits):
    """Create the most aggressive feature set possible"""
    
    logger.info("Creating breakthrough feature set...")
    
    # 1. ULTRA-AGGRESSIVE TARGET ENCODING
    # Target encode EVERYTHING with high cardinality
    target_cols = []
    for col in df.columns:
        if col in ['Id', 'ranker_id', 'selected']:
            continue
        
        # More aggressive: encode anything with >10 unique values
        if df[col].n_unique() > 10:
            target_cols.append(col)
    
    logger.info(f"Target encoding {len(target_cols)} columns")
    
    if target_cols:
        df = TargetEncodingUtils.kfold_target_encode(
            df=df, 
            cat_cols=target_cols, 
            target='selected', 
            folds=cv_splits, 
            noise=0.0001  # Minimal noise for maximum signal
        )
    
    # 2. EXTREME POSITION FEATURES
    df = df.with_row_count('idx')
    df = df.with_columns([
        # Multiple position encodings
        pl.col('idx').rank().over('ranker_id').alias('position'),
        pl.col('idx').rank(method='dense').over('ranker_id').alias('position_dense'),
        pl.col('idx').rank(method='min').over('ranker_id').alias('position_min'),
        pl.col('idx').rank(method='max').over('ranker_id').alias('position_max'),
        
        # Position percentiles and ratios
        ((pl.col('idx').rank().over('ranker_id') - 1) / (pl.count().over('ranker_id') - 1)).alias('position_pct'),
        (1.0 / pl.col('idx').rank().over('ranker_id')).alias('position_inverse'),
        (1.0 / (pl.col('idx').rank().over('ranker_id') + 1)).alias('position_inverse_plus1'),
        np.log(pl.col('idx').rank().over('ranker_id') + 1).alias('position_log'),
        np.sqrt(pl.col('idx').rank().over('ranker_id')).alias('position_sqrt'),
        
        # Position binaries
        (pl.col('idx').rank().over('ranker_id') == 1).cast(pl.Float32).alias('is_position_1'),
        (pl.col('idx').rank().over('ranker_id') == 2).cast(pl.Float32).alias('is_position_2'),
        (pl.col('idx').rank().over('ranker_id') == 3).cast(pl.Float32).alias('is_position_3'),
        (pl.col('idx').rank().over('ranker_id') <= 3).cast(pl.Float32).alias('is_top3_position'),
        (pl.col('idx').rank().over('ranker_id') <= 5).cast(pl.Float32).alias('is_top5_position'),
        (pl.col('idx').rank().over('ranker_id') <= 10).cast(pl.Float32).alias('is_top10_position')
    ])
    
    # 3. EXTREME PRICE FEATURES
    # Multiple price statistics per session
    price_stats = df.group_by('ranker_id').agg([
        pl.col('totalPrice').min().alias('price_min'),
        pl.col('totalPrice').max().alias('price_max'),
        pl.col('totalPrice').mean().alias('price_mean'),
        pl.col('totalPrice').median().alias('price_median'),
        pl.col('totalPrice').std().alias('price_std'),
        pl.col('totalPrice').var().alias('price_var'),
        pl.col('totalPrice').quantile(0.1).alias('price_p10'),
        pl.col('totalPrice').quantile(0.25).alias('price_p25'),
        pl.col('totalPrice').quantile(0.75).alias('price_p75'),
        pl.col('totalPrice').quantile(0.9).alias('price_p90'),
        pl.col('totalPrice').skew().alias('price_skew'),
        pl.col('totalPrice').kurtosis().alias('price_kurtosis')
    ])
    
    df = df.join(price_stats, on='ranker_id')
    
    # Multiple price rankings and comparisons
    df = df.with_columns([
        # Price rankings
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank'),
        pl.col('totalPrice').rank(method='dense').over('ranker_id').alias('price_rank_dense'),
        pl.col('totalPrice').rank(method='min').over('ranker_id').alias('price_rank_min'),
        pl.col('totalPrice').rank(method='max').over('ranker_id').alias('price_rank_max'),
        
        # Price percentiles
        ((pl.col('totalPrice').rank().over('ranker_id') - 1) / (pl.count().over('ranker_id') - 1)).alias('price_rank_pct'),
        (1.0 / pl.col('totalPrice').rank().over('ranker_id')).alias('price_rank_inverse'),
        
        # Price vs session statistics
        (pl.col('totalPrice') / pl.col('price_min')).alias('price_vs_min'),
        (pl.col('totalPrice') / pl.col('price_max')).alias('price_vs_max'),
        (pl.col('totalPrice') / pl.col('price_mean')).alias('price_vs_mean'),
        (pl.col('totalPrice') / pl.col('price_median')).alias('price_vs_median'),
        
        # Price deviations
        (pl.col('totalPrice') - pl.col('price_min')).alias('price_above_min'),
        (pl.col('price_max') - pl.col('totalPrice')).alias('price_below_max'),
        (pl.col('totalPrice') - pl.col('price_mean')).alias('price_dev_mean'),
        (pl.col('totalPrice') - pl.col('price_median')).alias('price_dev_median'),
        
        # Price z-scores and normalized
        ((pl.col('totalPrice') - pl.col('price_mean')) / pl.col('price_std')).alias('price_zscore'),
        ((pl.col('totalPrice') - pl.col('price_min')) / (pl.col('price_max') - pl.col('price_min'))).alias('price_normalized'),
        
        # Price quantile positions
        (pl.col('totalPrice') <= pl.col('price_p10')).cast(pl.Float32).alias('is_bottom_10pct'),
        (pl.col('totalPrice') <= pl.col('price_p25')).cast(pl.Float32).alias('is_bottom_25pct'),
        (pl.col('totalPrice') <= pl.col('price_median')).cast(pl.Float32).alias('is_below_median'),
        (pl.col('totalPrice') >= pl.col('price_p75')).cast(pl.Float32).alias('is_top_25pct'),
        (pl.col('totalPrice') >= pl.col('price_p90')).cast(pl.Float32).alias('is_top_10pct'),
        
        # Price binaries
        (pl.col('totalPrice') == pl.col('price_min')).cast(pl.Float32).alias('is_cheapest'),
        (pl.col('totalPrice') == pl.col('price_max')).cast(pl.Float32).alias('is_most_expensive'),
        (pl.col('price_rank') == 1).cast(pl.Float32).alias('is_price_rank_1'),
        (pl.col('price_rank') <= 3).cast(pl.Float32).alias('is_price_top3'),
        (pl.col('price_rank') <= 5).cast(pl.Float32).alias('is_price_top5')
    ])
    
    # 4. EXTREME INTERACTION FEATURES
    df = df.with_columns([
        # Position × Price interactions
        (pl.col('position') * pl.col('price_rank')).alias('pos_price_mult'),
        (pl.col('position') + pl.col('price_rank')).alias('pos_price_sum'),
        (pl.col('position') - pl.col('price_rank')).alias('pos_price_diff'),
        (pl.col('position') / pl.col('price_rank')).alias('pos_price_ratio'),
        (pl.col('price_rank') / pl.col('position')).alias('price_pos_ratio'),
        
        # Position percentile × Price interactions
        (pl.col('position_pct') * pl.col('price_vs_median')).alias('pos_pct_price_med'),
        (pl.col('position_pct') * pl.col('price_zscore')).alias('pos_pct_price_z'),
        (pl.col('position_pct') + pl.col('price_rank_pct')).alias('pos_price_pct_sum'),
        
        # Binary interactions
        (pl.col('is_position_1') * pl.col('is_cheapest')).alias('first_and_cheapest'),
        (pl.col('is_top3_position') * pl.col('is_price_top3')).alias('top3_pos_and_price'),
        (pl.col('is_position_1') * pl.col('is_below_median')).alias('first_and_cheap'),
        (pl.col('is_top5_position') * pl.col('is_bottom_25pct')).alias('top5_pos_bottom25_price'),
        
        # Complex combinations
        (pl.col('position_inverse') + pl.col('price_rank_inverse')).alias('inverse_combined'),
        (pl.col('position_log') * pl.col('price_zscore')).alias('log_pos_price_z'),
        np.sqrt(pl.col('position') * pl.col('price_rank')).alias('pos_price_sqrt'),
        
        # Business logic scores
        (pl.col('is_position_1').cast(pl.Float32) * 100 + 
         pl.col('is_cheapest').cast(pl.Float32) * 50 + 
         pl.col('is_below_median').cast(pl.Float32) * 25).alias('business_score_v1'),
        
        (pl.col('position_inverse') * 100 + 
         pl.col('price_rank_inverse') * 50).alias('business_score_v2'),
         
        (pl.col('is_top3_position').cast(pl.Float32) * 30 + 
         pl.col('is_price_top3').cast(pl.Float32) * 20 + 
         pl.col('is_below_median').cast(pl.Float32) * 10).alias('business_score_v3')
    ])
    
    # 5. POLICY AND ADVANCED FEATURES
    if 'pricingInfo_isAccessTP' in df.columns:
        df = df.with_columns([
            pl.col('pricingInfo_isAccessTP').cast(pl.Float32).alias('policy_compliant'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_position_1')).alias('policy_and_first'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_cheapest')).alias('policy_and_cheapest'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_below_median')).alias('policy_and_cheap'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * 1000 + 
             pl.col('is_position_1').cast(pl.Float32) * 100 + 
             pl.col('is_cheapest').cast(pl.Float32) * 50).alias('ultimate_business_score')
        ])
    
    # 6. SESSION CONTEXT FEATURES
    session_context = df.group_by('ranker_id').agg([
        pl.count().alias('session_size'),
        pl.col('pricingInfo_isAccessTP').mean().alias('session_policy_rate'),
        pl.col('pricingInfo_isAccessTP').sum().alias('session_policy_count')
    ])
    
    df = df.join(session_context, on='ranker_id')
    
    df = df.with_columns([
        np.log(pl.col('session_size')).alias('session_size_log'),
        (pl.col('position') / pl.col('session_size')).alias('position_in_session_pct'),
        (pl.col('session_policy_count') > 0).cast(pl.Float32).alias('session_has_policy'),
        (pl.col('session_policy_rate') > 0.5).cast(pl.Float32).alias('session_mostly_policy'),
        (pl.col('session_size') > 50).cast(pl.Float32).alias('large_session'),
        (pl.col('session_size') <= 10).cast(pl.Float32).alias('small_session')
    ])
    
    return df

def train_ensemble_models(df, cv_splits):
    """Train multiple models for ensemble"""
    
    logger.info("Training ensemble of multiple models...")
    
    # Get features
    exclude_cols = {
        'Id', 'ranker_id', 'selected', 'idx', 'requestDate', 'legs0_departureAt',
        'price_min', 'price_max', 'price_mean', 'price_median', 'price_std', 'price_var',
        'price_p10', 'price_p25', 'price_p75', 'price_p90', 'price_skew', 'price_kurtosis'
    }
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]:
            feature_cols.append(col)
    
    logger.info(f"Using {len(feature_cols)} features for ensemble")
    
    # Use first fold for quick training
    train_idx, val_idx = cv_splits[0]
    
    train_df = df[train_idx].sort('ranker_id')
    val_df = df[val_idx].sort('ranker_id')
    
    X_train = train_df.select(feature_cols).fill_null(0).to_pandas()
    y_train = train_df['selected'].to_numpy()
    group_train = train_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    X_val = val_df.select(feature_cols).fill_null(0).to_pandas()
    y_val = val_df['selected'].to_numpy()
    group_val = val_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    models = {}
    predictions = {}
    
    # 1. ULTRA-OPTIMIZED LIGHTGBM
    logger.info("Training LightGBM...")
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,      # Lower for more precision
        'num_leaves': 2048,         # Higher for more complexity
        'min_child_samples': 5,     # Lower for detailed patterns
        'feature_fraction': 1.0,    # Use all features
        'bagging_fraction': 0.95,   # High sampling
        'lambda_l1': 0.0,          # No L1 regularization
        'lambda_l2': 0.0,          # No L2 regularization
        'min_gain_to_split': 0.0,  # Allow any split
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 5000,      # Many trees
        'n_jobs': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )
    
    models['lgb'] = lgb_model
    predictions['lgb'] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    
    # 2. RANDOM FOREST (for different perspective)
    logger.info("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['rf'] = rf_model
    predictions['rf'] = rf_model.predict(X_val)
    
    # 3. EXTRA TREES (more randomness)
    logger.info("Training Extra Trees...")
    et_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    models['et'] = et_model
    predictions['et'] = et_model.predict(X_val)
    
    # 4. LOGISTIC REGRESSION (linear baseline)
    logger.info("Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    models['lr'] = (lr_model, scaler)
    predictions['lr'] = lr_model.predict_proba(X_val_scaled)[:, 1]
    
    # Evaluate individual models
    results = {}
    for name, preds in predictions.items():
        eval_df = val_df.select(['ranker_id', 'selected']).with_columns([
            pl.Series('predicted_score', preds)
        ])
        hitrate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
        results[name] = hitrate
        logger.info(f"{name.upper()} HitRate@3: {hitrate:.4f}")
    
    # 5. WEIGHTED ENSEMBLE
    logger.info("Creating weighted ensemble...")
    
    # Find best weights through simple search
    best_weight = None
    best_ensemble_score = 0
    
    # Test different weight combinations
    weight_combinations = [
        [0.7, 0.1, 0.1, 0.1],  # LGB heavy
        [0.5, 0.2, 0.2, 0.1],  # Balanced
        [0.4, 0.3, 0.2, 0.1],  # More balanced
        [0.6, 0.15, 0.15, 0.1], # LGB + forests
        [0.8, 0.1, 0.05, 0.05]  # Ultra LGB heavy
    ]
    
    for weights in weight_combinations:
        ensemble_pred = (
            weights[0] * predictions['lgb'] +
            weights[1] * predictions['rf'] +
            weights[2] * predictions['et'] +
            weights[3] * predictions['lr']
        )
        
        eval_df = val_df.select(['ranker_id', 'selected']).with_columns([
            pl.Series('predicted_score', ensemble_pred)
        ])
        ensemble_score = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
        
        if ensemble_score > best_ensemble_score:
            best_ensemble_score = ensemble_score
            best_weight = weights
    
    logger.info(f"BEST ENSEMBLE HitRate@3: {best_ensemble_score:.4f}")
    logger.info(f"Best weights: LGB={best_weight[0]}, RF={best_weight[1]}, ET={best_weight[2]}, LR={best_weight[3]}")
    
    return models, best_weight, best_ensemble_score, feature_cols

def main():
    """Run breakthrough ensemble approach"""
    
    logger.info("🚀 BREAKTHROUGH ENSEMBLE - TARGETING 0.7+ HITRATE")
    
    # Load data
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    # Use larger sample for final breakthrough attempt
    full_df_lazy = pipeline.loader.load_structured_data("train")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.05, seed=42)  # 5% for breakthrough
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    df = full_df_lazy.collect()
    logger.info(f"Breakthrough dataset: {df.shape}")
    
    # Filter to large groups where we can make impact
    group_sizes = df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    df = df.join(group_sizes, on='ranker_id', how='inner')
    df = df.filter(pl.col('group_size') > 10).drop('group_size')
    logger.info(f"Large groups: {df.shape}")
    
    # Create CV splits
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=df, group_col='profileId', stratify_col='selected', n_splits=5, seed=42
    )
    
    # Create breakthrough features
    df = create_breakthrough_features(df, cv_splits)
    logger.info(f"With breakthrough features: {df.shape}")
    
    # Train ensemble
    models, best_weights, best_score, feature_cols = train_ensemble_models(df, cv_splits)
    
    print(f"\n" + "="*70)
    print(f"🎯 BREAKTHROUGH ENSEMBLE RESULTS")
    print(f"="*70)
    print(f"BEST ENSEMBLE HitRate@3: {best_score:.4f}")
    print(f"TARGET: 0.7000")
    print(f"ACHIEVEMENT: {best_score/0.7*100:.1f}% of target")
    
    if best_score >= 0.7:
        print(f"🎉 BREAKTHROUGH ACHIEVED!")
        print(f"✅ Successfully reached >0.7 HitRate@3!")
    else:
        gap = 0.7 - best_score
        print(f"❌ Gap remaining: {gap:.4f}")
        print(f"📈 Achieved {best_score:.4f}, need {gap:.4f} more")
        
        if best_score > 0.6:
            print(f"🔥 VERY CLOSE! Consider:")
            print(f"   - Larger dataset (10%+ sample)")
            print(f"   - More sophisticated ensembling")
            print(f"   - Advanced hyperparameter tuning")
        elif best_score > 0.5:
            print(f"💪 GOOD PROGRESS! Consider:")
            print(f"   - Neural networks")
            print(f"   - Feature selection optimization")
            print(f"   - Cross-validation ensemble")
        else:
            print(f"🤔 FUNDAMENTAL ISSUE - may need different approach")
    
    print(f"="*70)
    
    # Show feature importance from best model
    if 'lgb' in models:
        importance = models['lgb'].feature_importance('gain')
        feature_importance = list(zip(feature_cols, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\\nTOP 10 FEATURES:")
        for feat, imp in feature_importance[:10]:
            print(f"  {feat}: {imp:.0f}")

if __name__ == "__main__":
    main()