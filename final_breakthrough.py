"""
FINAL BREAKTHROUGH - Ultra-focused approach to reach 0.7+ HitRate@3
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline, DataConfig
from utils import ValidationUtils, TargetEncodingUtils, CrossValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ultra_focused_features(df, cv_splits):
    """Create only the most powerful features based on our analysis"""
    
    logger.info("Creating ultra-focused features...")
    
    # 1. AGGRESSIVE TARGET ENCODING of key columns only
    key_target_cols = []
    for col in ['searchRoute', 'legs0_segments0_operatingCarrier_code', 'companyID', 'legs0_segments0_fareFamily']:
        if col in df.columns:
            key_target_cols.append(col)
    
    if key_target_cols:
        logger.info(f"Target encoding {len(key_target_cols)} key columns")
        df = TargetEncodingUtils.kfold_target_encode(
            df=df, 
            cat_cols=key_target_cols, 
            target='selected', 
            folds=cv_splits, 
            noise=0.001
        )
    
    # 2. CORE POSITION FEATURES
    df = df.with_row_count('idx')
    
    # Create position features step by step to avoid dependency issues
    df = df.with_columns([
        pl.col('idx').rank().over('ranker_id').alias('position')
    ])
    
    df = df.with_columns([
        ((pl.col('position') - 1) / (pl.count().over('ranker_id') - 1)).alias('position_pct'),
        (1.0 / pl.col('position')).alias('position_inverse'),
        (pl.col('position') == 1).cast(pl.Float32).alias('is_first'),
        (pl.col('position') <= 3).cast(pl.Float32).alias('is_top3'),
        (pl.col('position') <= 5).cast(pl.Float32).alias('is_top5')
    ])
    
    # 3. CORE PRICE FEATURES
    # Session price statistics
    price_stats = df.group_by('ranker_id').agg([
        pl.col('totalPrice').min().alias('session_min_price'),
        pl.col('totalPrice').max().alias('session_max_price'),
        pl.col('totalPrice').mean().alias('session_avg_price'),
        pl.col('totalPrice').median().alias('session_median_price'),
        pl.col('totalPrice').std().alias('session_price_std')
    ])
    
    df = df.join(price_stats, on='ranker_id')
    
    # Price rankings
    df = df.with_columns([
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank')
    ])
    
    # Price ratios and features
    df = df.with_columns([
        (pl.col('totalPrice') / pl.col('session_min_price')).alias('price_vs_min'),
        (pl.col('totalPrice') / pl.col('session_median_price')).alias('price_vs_median'),
        ((pl.col('totalPrice') - pl.col('session_avg_price')) / pl.col('session_price_std')).alias('price_zscore'),
        (pl.col('totalPrice') == pl.col('session_min_price')).cast(pl.Float32).alias('is_cheapest'),
        (pl.col('price_rank') <= 3).cast(pl.Float32).alias('is_price_top3')
    ])
    
    # 4. CRITICAL INTERACTIONS
    df = df.with_columns([
        # The most important interactions based on our analysis
        (pl.col('position') * pl.col('price_rank')).alias('pos_price_mult'),
        (pl.col('is_first') * pl.col('is_cheapest')).alias('first_and_cheapest'),
        (pl.col('is_top3') * pl.col('is_price_top3')).alias('top3_pos_and_price'),
        (pl.col('position_inverse') + (1.0 / pl.col('price_rank'))).alias('inverse_combined')
    ])
    
    # 5. POLICY FEATURES
    if 'pricingInfo_isAccessTP' in df.columns:
        df = df.with_columns([
            pl.col('pricingInfo_isAccessTP').cast(pl.Float32).alias('policy_compliant'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_first')).alias('policy_and_first'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_cheapest')).alias('policy_and_cheapest')
        ])
    
    # 6. ULTIMATE BUSINESS SCORE
    policy_weight = pl.col('policy_compliant') * 1000 if 'policy_compliant' in df.columns else pl.lit(0)
    
    df = df.with_columns([
        (policy_weight + 
         pl.col('is_first') * 500 + 
         pl.col('is_cheapest') * 300 + 
         pl.col('is_top3') * 100 + 
         pl.col('is_price_top3') * 50).alias('ultimate_score')
    ])
    
    return df

def ultra_optimize_lgb(X_train, y_train, group_train, X_val, y_val, group_val, n_trials=100):
    """Ultra-aggressive LightGBM hyperparameter optimization"""
    
    logger.info(f"Ultra-optimizing LightGBM with {n_trials} trials...")
    
    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
            
            # Aggressive parameter ranges
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 128, 4096),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.1),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.1),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 8000)
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        # Predict and calculate HitRate@3
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Create temporary DataFrame for evaluation
        temp_df = pl.DataFrame({
            'ranker_id': [f'temp_{i//10}' for i in range(len(val_preds))],  # Fake ranker_ids
            'selected': y_val,
            'predicted_score': val_preds
        })
        
        # Quick HitRate calculation
        try:
            hitrate = ValidationUtils.calculate_hitrate_at_k(temp_df, k=3)
            return hitrate
        except:
            return 0.0
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour max
    
    logger.info(f"Best HitRate@3: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params, study.best_value

def train_breakthrough_model(df, cv_splits):
    """Train the final breakthrough model"""
    
    logger.info("Training breakthrough model...")
    
    # Get focused feature set
    exclude_cols = {
        'Id', 'ranker_id', 'selected', 'idx', 'requestDate', 'legs0_departureAt',
        'session_min_price', 'session_max_price', 'session_avg_price', 
        'session_median_price', 'session_price_std'
    }
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]:
            feature_cols.append(col)
    
    logger.info(f"Using {len(feature_cols)} breakthrough features")
    
    # Use first fold for optimization
    train_idx, val_idx = cv_splits[0]
    
    train_df = df[train_idx].sort('ranker_id')
    val_df = df[val_idx].sort('ranker_id')
    
    X_train = train_df.select(feature_cols).fill_null(0).to_pandas()
    y_train = train_df['selected'].to_numpy()
    group_train = train_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    X_val = val_df.select(feature_cols).fill_null(0).to_pandas()
    y_val = val_df['selected'].to_numpy()
    group_val = val_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Ultra-optimize hyperparameters
    best_params, best_score = ultra_optimize_lgb(
        X_train, y_train, group_train, X_val, y_val, group_val, n_trials=50
    )
    
    # Train final model with best parameters
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )
    
    # Final evaluation
    final_preds = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    
    eval_df = val_df.select(['ranker_id', 'selected']).with_columns([
        pl.Series('predicted_score', final_preds)
    ])
    
    final_hitrate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
    
    logger.info(f"FINAL BREAKTHROUGH HitRate@3: {final_hitrate:.4f}")
    
    return final_model, final_hitrate, feature_cols, best_params

def test_ensemble_strategies(df, cv_splits):
    """Test different ensemble strategies for breakthrough"""
    
    logger.info("Testing ensemble strategies...")
    
    # Prepare data
    exclude_cols = {
        'Id', 'ranker_id', 'selected', 'idx', 'requestDate', 'legs0_departureAt',
        'session_min_price', 'session_max_price', 'session_avg_price', 
        'session_median_price', 'session_price_std'
    }
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]:
            feature_cols.append(col)
    
    train_idx, val_idx = cv_splits[0]
    train_df = df[train_idx].sort('ranker_id')
    val_df = df[val_idx].sort('ranker_id')
    
    X_train = train_df.select(feature_cols).fill_null(0).to_pandas()
    y_train = train_df['selected'].to_numpy()
    group_train = train_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    X_val = val_df.select(feature_cols).fill_null(0).to_pandas()
    y_val = val_df['selected'].to_numpy()
    
    # Strategy 1: Multiple LightGBM with different seeds
    logger.info("Testing multi-seed LightGBM ensemble...")
    
    lgb_predictions = []
    for seed in [42, 123, 456, 789, 999]:
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': 0.05,
            'num_leaves': 1024,
            'random_state': seed,
            'verbose': -1,
            'n_estimators': 3000
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_df.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy(), reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        lgb_predictions.append(preds)
    
    # Average ensemble
    ensemble_pred = np.mean(lgb_predictions, axis=0)
    
    eval_df = val_df.select(['ranker_id', 'selected']).with_columns([
        pl.Series('predicted_score', ensemble_pred)
    ])
    
    ensemble_hitrate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
    logger.info(f"Multi-seed LightGBM ensemble HitRate@3: {ensemble_hitrate:.4f}")
    
    # Strategy 2: Rule + ML hybrid
    logger.info("Testing rule + ML hybrid...")
    
    # Create rule-based score
    rule_df = val_df.with_columns([
        # Ultimate business rule
        (pl.col('is_first').cast(pl.Float32) * 1000 + 
         pl.col('is_cheapest').cast(pl.Float32) * 500 + 
         pl.col('policy_compliant').cast(pl.Float32) * 300 if 'policy_compliant' in val_df.columns 
         else pl.col('is_first').cast(pl.Float32) * 1000 + pl.col('is_cheapest').cast(pl.Float32) * 500).alias('rule_score')
    ])
    
    # Combine rule and ML
    hybrid_pred = 0.3 * rule_df['rule_score'].to_numpy() + 0.7 * ensemble_pred
    
    hybrid_eval_df = val_df.select(['ranker_id', 'selected']).with_columns([
        pl.Series('predicted_score', hybrid_pred)
    ])
    
    hybrid_hitrate = ValidationUtils.calculate_hitrate_at_k(hybrid_eval_df, k=3)
    logger.info(f"Rule + ML hybrid HitRate@3: {hybrid_hitrate:.4f}")
    
    return max(ensemble_hitrate, hybrid_hitrate)

def main():
    """Run final breakthrough attempt"""
    
    logger.info("🎯 FINAL BREAKTHROUGH ATTEMPT - ULTRA-FOCUSED FOR 0.7+")
    
    # Load data with maximum sample size
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    full_df_lazy = pipeline.loader.load_structured_data("train")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.08, seed=42)  # 8% for final attempt
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    df = full_df_lazy.collect()
    logger.info(f"Final dataset: {df.shape}")
    
    # Focus only on large groups where we can make impact
    group_sizes = df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    df = df.join(group_sizes, on='ranker_id', how='inner')
    df = df.filter(pl.col('group_size') > 20).drop('group_size')  # Only very large groups
    logger.info(f"Large groups only: {df.shape}")
    
    # Create CV splits
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=df, group_col='profileId', stratify_col='selected', n_splits=5, seed=42
    )
    
    # Create ultra-focused features
    df = create_ultra_focused_features(df, cv_splits)
    logger.info(f"With focused features: {df.shape}")
    
    # Train breakthrough model
    model, hitrate, features, best_params = train_breakthrough_model(df, cv_splits)
    
    # Test ensemble strategies
    ensemble_hitrate = test_ensemble_strategies(df, cv_splits)
    
    # Final results
    best_result = max(hitrate, ensemble_hitrate)
    
    print(f"\\n" + "="*80)
    print(f"🏆 FINAL BREAKTHROUGH RESULTS")
    print(f"="*80)
    print(f"Single model HitRate@3: {hitrate:.4f}")
    print(f"Ensemble HitRate@3: {ensemble_hitrate:.4f}")
    print(f"BEST RESULT: {best_result:.4f}")
    print(f"TARGET: 0.7000")
    print(f"ACHIEVEMENT: {best_result/0.7*100:.1f}% of target")
    
    if best_result >= 0.7:
        print(f"🎉🎉🎉 BREAKTHROUGH ACHIEVED! 🎉🎉🎉")
        print(f"✅ Successfully reached 0.7+ HitRate@3!")
        print(f"🚀 Mission accomplished!")
    else:
        gap = 0.7 - best_result
        print(f"❌ Gap remaining: {gap:.4f}")
        
        if best_result > 0.65:
            print(f"🔥 EXTREMELY CLOSE! Consider:")
            print(f"   - Increase sample size to 15-20%")
            print(f"   - More sophisticated target encoding")
            print(f"   - Advanced ensemble techniques")
        elif best_result > 0.55:
            print(f"💪 SIGNIFICANT PROGRESS! The path forward:")
            print(f"   - Feature selection optimization")
            print(f"   - Multi-objective optimization")
            print(f"   - Stacking ensembles")
        else:
            print(f"🤔 May require fundamental approach change")
            print(f"   - External data sources")
            print(f"   - Domain expert insights")
            print(f"   - Different problem formulation")
    
    print(f"="*80)
    
    # Show final insights
    if model:
        importance = model.feature_importance('gain')
        feature_importance = list(zip(features, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\\nFINAL TOP FEATURES:")
        for feat, imp in feature_importance[:15]:
            print(f"  {feat}: {imp:.0f}")

if __name__ == "__main__":
    main()