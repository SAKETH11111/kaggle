"""
BREAKTHROUGH ANALYSIS - What does >0.7 HitRate@3 actually require?
"""
import polars as pl
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_theoretical_maximum():
    """What's the theoretical maximum possible?"""
    
    # Load data
    df = pl.read_parquet('data/train.parquet')
    logger.info(f"Full dataset: {df.shape}")
    
    # Take larger sample for robust analysis
    unique_sessions = df.select('ranker_id').unique()
    sample_sessions = unique_sessions.sample(fraction=0.1, seed=42)
    df_sample = df.join(sample_sessions, on='ranker_id')
    logger.info(f"Sample: {df_sample.shape}")
    
    print("\n" + "="*70)
    print("BREAKTHROUGH ANALYSIS - PATH TO >0.7 HITRATE@3")
    print("="*70)
    
    # 1. Perfect oracle analysis
    print("\n1. THEORETICAL MAXIMUM ANALYSIS:")
    
    # If we could perfectly predict the selected item position
    selected_positions = df_sample.with_row_count('idx').with_columns([
        pl.col('idx').rank().over('ranker_id').alias('position')
    ]).filter(pl.col('selected') == 1)['position'].to_list()
    
    # What % of selected items are in top-3 positions?
    top3_selected = sum(1 for pos in selected_positions if pos <= 3) / len(selected_positions)
    print(f"   Selected items in top-3 positions: {top3_selected:.4f}")
    print(f"   THEORETICAL MAXIMUM HitRate@3: {top3_selected:.4f}")
    
    if top3_selected < 0.7:
        print(f"   ❌ IMPOSSIBLE to reach 0.7+ with position-only strategy!")
        print(f"   Need to find other signals or perfect ranking")
    
    # 2. Analyze what makes items get selected
    print("\n2. SELECTION PATTERN DEEP DIVE:")
    
    df_analysis = df_sample.with_row_count('idx').with_columns([
        pl.col('idx').rank().over('ranker_id').alias('position'),
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank'),
        pl.col('totalPrice').rank(method='dense').over('ranker_id').alias('price_dense_rank')
    ])
    
    selected_analysis = df_analysis.filter(pl.col('selected') == 1)
    
    print(f"   Position 1 selected: {(selected_analysis['position'] == 1).mean():.4f}")
    print(f"   Cheapest selected: {(selected_analysis['price_rank'] == 1).mean():.4f}")
    print(f"   Top-3 cheapest selected: {(selected_analysis['price_rank'] <= 3).mean():.4f}")
    
    # Combined signals
    pos1_and_cheap = selected_analysis.filter(
        (pl.col('position') == 1) & (pl.col('price_rank') == 1)
    ).height / selected_analysis.height
    print(f"   Position 1 AND cheapest: {pos1_and_cheap:.4f}")
    
    # 3. What if we combined position + price perfectly?
    print("\n3. PERFECT COMBINATION ANALYSIS:")
    
    # Create perfect score: heavily weight position 1 + cheapest
    df_perfect = df_analysis.with_columns([
        # Perfect business logic score
        (
            (pl.col('position') == 1).cast(pl.Int32) * 1000 +  # Position 1 is huge
            (pl.col('price_rank') == 1).cast(pl.Int32) * 500 +  # Cheapest is important
            (pl.col('price_rank') <= 3).cast(pl.Int32) * 100 +  # Top-3 price
            (pl.col('pricingInfo_isAccessTP').cast(pl.Int32)) * 200  # Policy compliance
        ).alias('perfect_score')
    ])
    
    # Rank by perfect score
    df_perfect = df_perfect.with_columns([
        pl.col('perfect_score').rank(method='ordinal', descending=True).over('ranker_id').alias('perfect_rank')
    ])
    
    # Calculate HitRate@3 with perfect score
    perfect_hitrate = calculate_hitrate_simple(df_perfect, 'perfect_rank')
    print(f"   Perfect business logic HitRate@3: {perfect_hitrate:.4f}")
    
    # 4. Find the missing signal
    print("\n4. MISSING SIGNAL DETECTION:")
    
    # What makes the difference between selected and not selected in same position/price?
    # Focus on position 1 items that aren't selected
    pos1_items = df_analysis.filter(pl.col('position') == 1)
    pos1_selected = pos1_items.filter(pl.col('selected') == 1)
    pos1_not_selected = pos1_items.filter(pl.col('selected') == 0)
    
    if pos1_not_selected.height > 0:
        print(f"   Position 1 items - Selected: {pos1_selected.height}, Not selected: {pos1_not_selected.height}")
        
        # Compare features
        for col in ['totalPrice', 'pricingInfo_isAccessTP']:
            if col in df_analysis.columns:
                sel_mean = pos1_selected[col].mean()
                not_sel_mean = pos1_not_selected[col].mean()
                print(f"   {col} - Selected: {sel_mean:.3f}, Not selected: {not_sel_mean:.3f}")
    
    return df_perfect

def calculate_hitrate_simple(df, score_col):
    """Simple HitRate@3 calculation"""
    
    # Group by session and check if selected item is in top-3 predictions
    results = df.group_by('ranker_id').agg([
        pl.col('selected').sum().alias('has_selection'),
        (pl.col('selected') * (pl.col(score_col) <= 3)).sum().alias('hit_in_top3')
    ])
    
    # Only count sessions that have selections
    valid_sessions = results.filter(pl.col('has_selection') > 0)
    if valid_sessions.height == 0:
        return 0.0
    
    hitrate = valid_sessions['hit_in_top3'].sum() / valid_sessions.height
    return hitrate

def test_radical_approaches(df):
    """Test radical approaches to reach 0.7+"""
    
    print("\n5. RADICAL APPROACH TESTING:")
    
    # df already has position and price_rank from previous analysis
    
    # Approach 1: Always predict top-3 positions
    approach1_hitrate = calculate_hitrate_simple(df, 'position')
    print(f"   Approach 1 (top positions): {approach1_hitrate:.4f}")
    
    # Approach 2: Always predict 3 cheapest
    approach2_hitrate = calculate_hitrate_simple(df, 'price_rank')
    print(f"   Approach 2 (cheapest 3): {approach2_hitrate:.4f}")
    
    # Approach 3: Weighted combination
    df = df.with_columns([
        (pl.col('position') * 0.7 + pl.col('price_rank') * 0.3).alias('weighted_rank')
    ])
    approach3_hitrate = calculate_hitrate_simple(df, 'weighted_rank')
    print(f"   Approach 3 (weighted): {approach3_hitrate:.4f}")
    
    # Approach 4: Business rule - first position OR cheapest
    df = df.with_columns([
        pl.when((pl.col('position') == 1) | (pl.col('price_rank') == 1))
        .then(1)
        .when((pl.col('position') <= 3) | (pl.col('price_rank') <= 3))
        .then(2)
        .otherwise(10)
        .alias('business_rule_rank')
    ])
    approach4_hitrate = calculate_hitrate_simple(df, 'business_rule_rank')
    print(f"   Approach 4 (business rule): {approach4_hitrate:.4f}")
    
    return max(approach1_hitrate, approach2_hitrate, approach3_hitrate, approach4_hitrate)

def main():
    """Run breakthrough analysis"""
    
    # 1. Theoretical analysis
    df_perfect = analyze_theoretical_maximum()
    
    # 2. Test radical approaches
    best_simple = test_radical_approaches(df_perfect)
    
    print(f"\n" + "="*70)
    print(f"BREAKTHROUGH SUMMARY:")
    print(f"  Best simple approach: {best_simple:.4f}")
    print(f"  Target: 0.7000")
    print(f"  Gap to close: {0.7 - best_simple:.4f}")
    
    if best_simple >= 0.7:
        print(f"  ✅ BREAKTHROUGH POSSIBLE with simple rules!")
    else:
        print(f"  ❌ Need advanced techniques - simple rules insufficient")
        print(f"  Must explore: ensemble methods, neural ranking, data leakage")
    print(f"="*70)

if __name__ == "__main__":
    main()