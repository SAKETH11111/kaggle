"""
Binary classification approach based on forensic insights
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from data_pipeline import DataPipeline, DataConfig
from utils import CrossValidationUtils, ValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_session_level_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create session-level features for binary classification"""
    
    # Session-level aggregations
    session_features = df.group_by('ranker_id').agg([
        # Basic stats
        pl.count().alias('session_size'),
        pl.col('totalPrice').min().alias('min_price'),
        pl.col('totalPrice').max().alias('max_price'),
        pl.col('totalPrice').mean().alias('avg_price'),
        pl.col('totalPrice').std().alias('price_std'),
        
        # Policy compliance
        pl.col('pricingInfo_isAccessTP').mean().alias('policy_compliance_rate'),
        pl.col('pricingInfo_isAccessTP').sum().alias('policy_options_count'),
        
        # Timing features
        pl.col('legs0_departureAt').first().alias('first_departure'),
        
        # Booking timing
        pl.col('requestDate').first().alias('request_date'),
        
        # Target: does this session have any selection?
        pl.col('selected').max().alias('has_selection'),
        
        # Which position was selected (almost always 1)
        pl.col('selected').arg_max().alias('selected_position')
    ])
    
    # Add derived features
    session_features = session_features.with_columns([
        # Price spread
        (pl.col('max_price') - pl.col('min_price')).alias('price_range')
    ])
    
    session_features = session_features.with_columns([
        # Price spread ratio (using the newly created price_range)
        (pl.col('price_range') / pl.col('min_price')).alias('price_spread_ratio')
    ])
    
    session_features = session_features.with_columns([
        # Lead time - simplified
        pl.lit(7).alias('lead_days'),  # Placeholder for now
        
        # Session quality indicators
        (pl.col('policy_options_count') > 0).alias('has_policy_options'),
        (pl.col('session_size') > 1).alias('has_multiple_options'),
        
        # Conversion indicators (more likely to book)
        (pl.col('policy_compliance_rate') > 0.5).alias('mostly_policy_compliant'),
        (pl.col('price_spread_ratio') < 0.2).alias('narrow_price_range')
    ])
    
    return session_features

def train_binary_model(df: pl.DataFrame) -> dict:
    """Train binary classification model to predict bookings"""
    
    # Create session-level dataset
    session_df = create_session_level_features(df)
    
    logger.info(f"Session-level dataset shape: {session_df.shape}")
    logger.info(f"Booking rate: {session_df['has_selection'].mean():.4f}")
    
    # Features for binary classification
    feature_cols = [
        'session_size', 'min_price', 'max_price', 'avg_price', 'price_std',
        'policy_compliance_rate', 'policy_options_count', 'lead_days',
        'price_range', 'price_spread_ratio', 'has_policy_options', 
        'has_multiple_options', 'mostly_policy_compliant', 'narrow_price_range'
    ]
    
    # Handle missing values
    for col in feature_cols:
        if col in session_df.columns:
            session_df = session_df.with_columns(
                pl.col(col).fill_null(0)
            )
    
    # Split for training
    X = session_df.select(feature_cols).to_pandas()
    y = session_df['has_selection'].to_pandas()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Positive class rate: {y.mean():.4f}")
    
    # Train binary classifier
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 1000
    }
    
    # Simple train/val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    # Evaluate
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_preds)
    logger.info(f"Binary classification AUC: {auc:.4f}")
    
    return {
        'model': model,
        'features': feature_cols,
        'auc': auc,
        'session_df': session_df
    }

def convert_to_ranking_predictions(binary_results: dict, original_df: pl.DataFrame) -> pl.DataFrame:
    """Convert binary predictions back to ranking format"""
    
    session_df = binary_results['session_df']
    model = binary_results['model']
    features = binary_results['features']
    
    # Get session-level probabilities
    X_all = session_df.select(features).to_pandas()
    session_probs = model.predict(X_all, num_iteration=model.best_iteration)
    
    # Add probabilities to session data
    session_df = session_df.with_columns(
        pl.Series('booking_probability', session_probs)
    )
    
    # Join back to original data
    result_df = original_df.join(
        session_df.select(['ranker_id', 'booking_probability']), 
        on='ranker_id'
    )
    
    # Create ranking scores: booking_prob * position_penalty
    # Since selected is almost always position 1, heavily weight first position
    result_df = result_df.with_columns([
        pl.col('booking_probability').rank().over('ranker_id').alias('position_in_session')
    ])
    
    result_df = result_df.with_columns([
        # Score = booking probability / position penalty
        (pl.col('booking_probability') / pl.col('position_in_session')).alias('predicted_score')
    ])
    
    return result_df

def main():
    """Test the binary classification approach"""
    logger.info("Testing binary classification approach...")
    
    # Load data
    config = DataConfig()
    pipeline = DataPipeline(config)
    full_df_lazy = pipeline.loader.load_structured_data("train")
    
    # Sample for testing
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.1, seed=42)
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    full_df = full_df_lazy.collect()
    logger.info(f"Sample data shape: {full_df.shape}")
    
    # Filter to groups >10 items for fair comparison
    group_sizes = full_df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    full_df = full_df.join(group_sizes, on='ranker_id', how='inner')
    large_groups_df = full_df.filter(pl.col('group_size') > 10).drop('group_size')
    logger.info(f"Large groups data shape: {large_groups_df.shape}")
    
    # Train binary model
    binary_results = train_binary_model(full_df)
    
    # Convert to ranking predictions
    ranking_df = convert_to_ranking_predictions(binary_results, large_groups_df)
    
    # Evaluate ranking performance
    eval_df = ranking_df.select(['ranker_id', 'selected', 'predicted_score'])
    hit_rate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
    
    logger.info(f"BINARY APPROACH HitRate@3: {hit_rate:.5f}")
    
    # Show feature importance
    importance = binary_results['model'].feature_importance('gain')
    for i, (feat, imp) in enumerate(zip(binary_results['features'], importance)):
        logger.info(f"  {feat}: {imp}")

if __name__ == "__main__":
    main()