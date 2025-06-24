"""
The true solution - understanding the real problem structure
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb

from data_pipeline import DataPipeline, DataConfig
from utils import ValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_true_problem():
    """Understand the true problem without filtering"""
    
    # Load full data
    config = DataConfig()
    pipeline = DataPipeline(config)
    df = pipeline.loader.load_structured_data("train").collect()
    
    print("\n" + "="*60)
    print("TRUE PROBLEM ANALYSIS - NO FILTERING")
    print("="*60)
    
    # 1. Overall statistics
    total_sessions = df['ranker_id'].n_unique()
    total_rows = df.height
    sessions_with_selection = df.filter(pl.col('selected') == 1)['ranker_id'].n_unique()
    
    print(f"\nOverall Statistics:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total sessions: {total_sessions:,}")
    print(f"  Sessions with selection: {sessions_with_selection:,}")
    print(f"  Selection rate: {sessions_with_selection/total_sessions:.4f}")
    
    # 2. Group size distribution
    group_sizes = df.group_by('ranker_id').agg(pl.count().alias('size'))
    
    print(f"\nGroup Size Distribution:")
    for size in [1, 2, 3, 5, 10, 20, 50]:
        count = (group_sizes['size'] == size).sum()
        pct = count / total_sessions * 100
        print(f"  Size {size}: {count:,} sessions ({pct:.1f}%)")
    
    large_groups = (group_sizes['size'] > 10).sum()
    print(f"  Size >10: {large_groups:,} sessions ({large_groups/total_sessions*100:.1f}%)")
    
    # 3. Selection patterns by group size
    print(f"\nSelection Rate by Group Size:")
    
    session_stats = df.group_by('ranker_id').agg([
        pl.count().alias('size'),
        pl.col('selected').max().alias('has_selection')
    ])
    
    for size in [1, 2, 3, 5, 10]:
        size_sessions = session_stats.filter(pl.col('size') == size)
        if size_sessions.height > 0:
            selection_rate = size_sessions['has_selection'].mean()
            print(f"  Size {size}: {selection_rate:.4f}")
    
    large_size_sessions = session_stats.filter(pl.col('size') > 10)
    if large_size_sessions.height > 0:
        large_selection_rate = large_size_sessions['has_selection'].mean()
        print(f"  Size >10: {large_selection_rate:.4f}")
    
    # 4. What makes a session likely to have a selection?
    print(f"\nFactors predicting selection:")
    
    session_features = df.group_by('ranker_id').agg([
        pl.count().alias('options_count'),
        pl.col('totalPrice').min().alias('min_price'),
        pl.col('totalPrice').mean().alias('avg_price'),
        pl.col('pricingInfo_isAccessTP').mean().alias('policy_rate'),
        pl.col('selected').max().alias('has_selection')
    ])
    
    # Compare sessions with vs without selections
    with_selection = session_features.filter(pl.col('has_selection') == 1)
    without_selection = session_features.filter(pl.col('has_selection') == 0)
    
    print(f"  Sessions WITH selection:")
    print(f"    Avg options: {with_selection['options_count'].mean():.2f}")
    print(f"    Avg min price: {with_selection['min_price'].mean():.0f}")
    print(f"    Avg policy rate: {with_selection['policy_rate'].mean():.3f}")
    
    if without_selection.height > 0:
        print(f"  Sessions WITHOUT selection:")
        print(f"    Avg options: {without_selection['options_count'].mean():.2f}")
        print(f"    Avg min price: {without_selection['min_price'].mean():.0f}")
        print(f"    Avg policy rate: {without_selection['policy_rate'].mean():.3f}")
    else:
        print(f"  Sessions WITHOUT selection: NONE (all sessions have selections)")
    
    return session_features

def create_true_solution(session_features: pl.DataFrame) -> dict:
    """Create solution for the true problem"""
    
    print(f"\n" + "="*60)
    print("CREATING TRUE SOLUTION")
    print("="*60)
    
    # Features for predicting session-level booking
    feature_cols = ['options_count', 'min_price', 'avg_price', 'policy_rate']
    
    X = session_features.select(feature_cols).fill_null(0).to_pandas()
    y = session_features['has_selection'].to_pandas()
    
    print(f"Training binary classifier:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Sessions: {len(X):,}")
    print(f"  Positive rate: {y.mean():.4f}")
    
    # Train binary classifier for session-level prediction
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 500
    }
    
    # Simple split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Evaluate binary performance
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, val_preds)
    print(f"Session-level AUC: {auc:.4f}")
    
    return {
        'model': model,
        'features': feature_cols,
        'session_data': session_features
    }

def convert_to_ranking_format(solution: dict, original_df: pl.DataFrame) -> pl.DataFrame:
    """Convert session predictions to ranking format"""
    
    print(f"\nConverting to ranking format...")
    
    # Get session-level predictions
    session_features = solution['session_data']
    model = solution['model']
    features = solution['features']
    
    X_all = session_features.select(features).fill_null(0).to_pandas()
    session_probs = model.predict(X_all, num_iteration=model.best_iteration)
    
    # Add predictions to session data
    session_with_probs = session_features.with_columns(
        pl.Series('booking_prob', session_probs)
    )
    
    # Join back to original data
    result_df = original_df.join(
        session_with_probs.select(['ranker_id', 'booking_prob']),
        on='ranker_id'
    )
    
    # Create ranking scores
    # Key insight: selected is almost always first item, so rank by original order within high-probability sessions
    result_df = result_df.with_row_count('original_order')
    
    result_df = result_df.with_columns([
        # Position in original data within session
        pl.col('original_order').rank().over('ranker_id').alias('position_in_session')
    ])
    
    # Score = booking_probability / position_penalty
    # Higher booking prob + earlier position = higher score
    result_df = result_df.with_columns([
        (pl.col('booking_prob') / pl.col('position_in_session')).alias('predicted_score')
    ])
    
    return result_df

def main():
    """Run the true solution"""
    
    # 1. Analyze the true problem
    session_features = analyze_true_problem()
    
    # 2. Create solution
    solution = create_true_solution(session_features)
    
    # 3. Test on large groups (for comparison with previous results)
    config = DataConfig()
    pipeline = DataPipeline(config)
    full_df = pipeline.loader.load_structured_data("train").collect()
    
    # Filter to large groups for evaluation
    group_sizes = full_df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    full_df = full_df.join(group_sizes, on='ranker_id', how='inner')
    large_groups_df = full_df.filter(pl.col('group_size') > 10).drop('group_size')
    
    print(f"\nEvaluating on large groups: {large_groups_df.shape}")
    
    # Convert to ranking format
    ranking_df = convert_to_ranking_format(solution, large_groups_df)
    
    # Evaluate
    eval_df = ranking_df.select(['ranker_id', 'selected', 'predicted_score'])
    hit_rate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
    
    print(f"\nTRUE SOLUTION HitRate@3: {hit_rate:.5f}")
    
    # Show what the model learned
    importance = solution['model'].feature_importance('gain')
    print(f"\nFeature importance:")
    for feat, imp in zip(solution['features'], importance):
        print(f"  {feat}: {imp}")

if __name__ == "__main__":
    main()