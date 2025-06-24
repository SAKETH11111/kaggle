"""
HIDDEN SIGNAL DETECTOR - Find the missing signals to reach 0.7+
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path
from itertools import combinations
import lightgbm as lgb
from data_pipeline import DataPipeline, DataConfig
from utils import ValidationUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_leakage_signals(df):
    """Detect potential leakage or hidden perfect signals"""
    
    print("\n" + "="*70)
    print("HIDDEN SIGNAL DETECTION - SEARCHING FOR 0.7+ SIGNALS")
    print("="*70)
    
    # 1. Check for perfect predictors (near 1.0 correlation)
    print("\n1. PERFECT PREDICTOR SEARCH:")
    
    perfect_signals = []
    
    for col in df.columns:
        if col in ['selected', 'ranker_id', 'Id']:
            continue
            
        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean, pl.Int8]:
            try:
                # Calculate correlation with selected
                correlation = df.select([
                    pl.corr('selected', col).alias('corr')
                ])['corr'][0] 
                
                if correlation is not None and abs(correlation) > 0.1:
                    perfect_signals.append((col, correlation))
                    
            except Exception as e:
                continue
    
    # Sort by correlation strength
    perfect_signals.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"   Found {len(perfect_signals)} potentially useful signals:")
    for col, corr in perfect_signals[:20]:
        print(f"   {col}: {corr:.4f}")
    
    # 2. Test combinations of top signals
    print("\n2. SIGNAL COMBINATION TESTING:")
    
    if len(perfect_signals) >= 2:
        top_signals = [col for col, _ in perfect_signals[:10]]
        
        # Test pairwise combinations
        best_combo = None
        best_score = 0
        
        for col1, col2 in combinations(top_signals, 2):
            try:
                # Create combined signal
                df_test = df.with_columns([
                    (pl.col(col1) * pl.col(col2)).alias('combo_signal')
                ])
                
                # Test this signal
                score = test_signal_strength(df_test, 'combo_signal')
                
                if score > best_score:
                    best_score = score
                    best_combo = (col1, col2)
                    
                print(f"      {col1} × {col2}: {score:.4f}")
                
            except:
                continue
        
        print(f"   Best combination: {best_combo} -> {best_score:.4f}")
    
    return perfect_signals

def test_signal_strength(df, signal_col):
    """Test how well a signal predicts selection"""
    
    try:
        # Rank by signal within each session
        df_ranked = df.with_columns([
            pl.col(signal_col).rank(method='ordinal', descending=True).over('ranker_id').alias('signal_rank')
        ])
        
        # Calculate HitRate@3
        results = df_ranked.group_by('ranker_id').agg([
            pl.col('selected').sum().alias('has_selection'),
            (pl.col('selected') * (pl.col('signal_rank') <= 3)).sum().alias('hit_in_top3')
        ])
        
        valid_sessions = results.filter(pl.col('has_selection') > 0)
        if valid_sessions.height == 0:
            return 0.0
        
        hitrate = valid_sessions['hit_in_top3'].sum() / valid_sessions.height
        return hitrate
        
    except:
        return 0.0

def search_temporal_patterns(df):
    """Search for temporal patterns that could boost performance"""
    
    print("\n3. TEMPORAL PATTERN SEARCH:")
    
    # Check if booking time patterns exist
    temporal_features = []
    
    # Day of week patterns
    if 'requestDate' in df.columns:
        try:
            df_temporal = df.with_columns([
                pl.col('requestDate').dt.weekday().alias('request_weekday'),
                pl.col('requestDate').dt.hour().alias('request_hour')
            ])
            
            # Test weekday signal
            weekday_score = test_signal_strength(df_temporal, 'request_weekday')
            print(f"   Weekday pattern: {weekday_score:.4f}")
            
            # Test hour signal  
            hour_score = test_signal_strength(df_temporal, 'request_hour')
            print(f"   Hour pattern: {hour_score:.4f}")
            
            temporal_features.extend([('request_weekday', weekday_score), ('request_hour', hour_score)])
            
        except:
            pass
    
    # Check departure time patterns
    if 'legs0_departureAt' in df.columns:
        try:
            df_temporal = df.with_columns([
                pl.col('legs0_departureAt').dt.weekday().alias('departure_weekday'),
                pl.col('legs0_departureAt').dt.hour().alias('departure_hour')
            ])
            
            weekday_score = test_signal_strength(df_temporal, 'departure_weekday')
            hour_score = test_signal_strength(df_temporal, 'departure_hour')
            
            print(f"   Departure weekday: {weekday_score:.4f}")
            print(f"   Departure hour: {hour_score:.4f}")
            
            temporal_features.extend([('departure_weekday', weekday_score), ('departure_hour', hour_score)])
            
        except:
            pass
    
    return temporal_features

def search_sequence_patterns(df):
    """Search for sequence and ordering patterns"""
    
    print("\n4. SEQUENCE PATTERN SEARCH:")
    
    # Check if there are patterns in how data is ordered
    df_seq = df.with_row_count('global_order')
    
    # Test global ordering signal
    global_score = test_signal_strength(df_seq, 'global_order')
    print(f"   Global order signal: {global_score:.4f}")
    
    # Test within-session ordering
    df_seq = df_seq.with_columns([
        pl.col('global_order').rank().over('ranker_id').alias('session_order')
    ])
    
    session_score = test_signal_strength(df_seq, 'session_order')
    print(f"   Session order signal: {session_score:.4f}")
    
    # Check for repeating patterns
    df_seq = df_seq.with_columns([
        (pl.col('global_order') % 10).alias('order_mod10'),
        (pl.col('global_order') % 7).alias('order_mod7'),
        (pl.col('global_order') % 3).alias('order_mod3')
    ])
    
    mod10_score = test_signal_strength(df_seq, 'order_mod10')
    mod7_score = test_signal_strength(df_seq, 'order_mod7')
    mod3_score = test_signal_strength(df_seq, 'order_mod3')
    
    print(f"   Order mod 10: {mod10_score:.4f}")
    print(f"   Order mod 7: {mod7_score:.4f}")
    print(f"   Order mod 3: {mod3_score:.4f}")
    
    return [
        ('global_order', global_score),
        ('session_order', session_score),
        ('order_mod10', mod10_score),
        ('order_mod7', mod7_score),
        ('order_mod3', mod3_score)
    ]

def create_ultimate_ensemble(df, all_signals):
    """Create ensemble using all discovered signals"""
    
    print("\n5. ULTIMATE ENSEMBLE CREATION:")
    
    # Get top signals
    all_signals.sort(key=lambda x: x[1], reverse=True)
    top_signals = [col for col, score in all_signals[:20] if score > 0.2]  # Only good signals
    
    print(f"   Using {len(top_signals)} top signals for ensemble")
    for col, score in all_signals[:10]:
        print(f"      {col}: {score:.4f}")
    
    if len(top_signals) < 3:
        print("   ❌ Not enough strong signals for ensemble")
        return None
    
    # Prepare ensemble features
    feature_cols = []
    
    # Add basic features
    df_ensemble = df.with_row_count('idx').with_columns([
        pl.col('idx').rank().over('ranker_id').alias('position'),
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank')
    ])
    
    feature_cols.extend(['position', 'price_rank'])
    
    # Add discovered signals
    for signal_col in top_signals:
        if signal_col in df_ensemble.columns:
            feature_cols.append(signal_col)
    
    print(f"   Total features: {len(feature_cols)}")
    
    # Train ensemble model
    sample_size = min(100000, df_ensemble.height)  # Limit size for speed
    df_sample = df_ensemble.sample(n=sample_size, seed=42)
    
    # Prepare training data
    X = df_sample.select(feature_cols).fill_null(0).to_pandas()
    y = df_sample['selected'].to_pandas()
    
    # Get group information
    groups = df_sample.group_by('ranker_id', maintain_order=True).count().get_column('count').to_numpy()
    
    # Train LightGBM ranker
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': 0.1,
        'num_leaves': 512,
        'feature_fraction': 0.9,
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 1000
    }
    
    # Simple split
    split_idx = int(len(groups) * 0.8)
    train_groups = groups[:split_idx]
    val_groups = groups[split_idx:]
    
    train_size = train_groups.sum()
    val_size = val_groups.sum()
    
    X_train, X_val = X[:train_size], X[train_size:train_size+val_size]
    y_train, y_val = y[:train_size], y[train_size:train_size+val_size]
    
    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    # Evaluate ensemble
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Create evaluation DataFrame
    val_df = df_sample[train_size:train_size+val_size].with_columns([
        pl.Series('predicted_score', val_preds)
    ])
    
    eval_df = val_df.select(['ranker_id', 'selected', 'predicted_score'])
    ensemble_hitrate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
    
    print(f"   ENSEMBLE HitRate@3: {ensemble_hitrate:.4f}")
    
    return ensemble_hitrate

def main():
    """Run comprehensive hidden signal detection"""
    
    # Load data
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    # Use substantial sample for robust signal detection
    full_df_lazy = pipeline.loader.load_structured_data("train")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.05, seed=42)  # 5% for thorough analysis
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    df = full_df_lazy.collect()
    logger.info(f"Analysis dataset: {df.shape}")
    
    # Find all potential signals
    all_signals = []
    
    # 1. Basic signal search
    perfect_signals = find_leakage_signals(df)
    all_signals.extend(perfect_signals)
    
    # 2. Temporal patterns
    temporal_signals = search_temporal_patterns(df)
    all_signals.extend(temporal_signals)
    
    # 3. Sequence patterns
    sequence_signals = search_sequence_patterns(df)
    all_signals.extend(sequence_signals)
    
    # 4. Create ultimate ensemble
    final_score = create_ultimate_ensemble(df, all_signals)
    
    print(f"\n" + "="*70)
    print(f"HIDDEN SIGNAL DETECTION COMPLETE")
    print(f"  Final ensemble score: {final_score:.4f}")
    print(f"  Target: 0.7000")
    
    if final_score and final_score >= 0.7:
        print(f"  ✅ BREAKTHROUGH ACHIEVED!")
    else:
        print(f"  ❌ Still need {0.7 - (final_score or 0):.4f} more performance")
        print(f"  Next: Advanced ML techniques, neural networks, external data")
    print(f"="*70)

if __name__ == "__main__":
    main()