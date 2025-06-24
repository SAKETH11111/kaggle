"""
Forensic analysis to find the core signal we're missing
"""
import polars as pl
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset():
    """Deep forensic analysis of the dataset"""
    
    # Load small sample for analysis
    df = pl.read_parquet('data/train.parquet')
    logger.info(f"Full dataset shape: {df.shape}")
    
    # Take 10k samples for analysis
    df_sample = df.sample(n=10000, seed=42)
    
    print("\n" + "="*50)
    print("FORENSIC ANALYSIS - LOOKING FOR THE MISSING SIGNAL")
    print("="*50)
    
    # 1. Analyze the target distribution
    print("\n1. TARGET ANALYSIS:")
    selected_stats = df_sample.group_by('ranker_id').agg([
        pl.col('selected').sum().alias('total_selected'),
        pl.count().alias('group_size')
    ])
    
    print(f"   Groups with selected=1: {(selected_stats['total_selected'] == 1).sum()}")
    print(f"   Groups with selected>1: {(selected_stats['total_selected'] > 1).sum()}")
    print(f"   Groups with selected=0: {(selected_stats['total_selected'] == 0).sum()}")
    print(f"   Average group size: {selected_stats['group_size'].mean():.2f}")
    
    # 2. Look for obvious patterns in selected rows
    print("\n2. SELECTED ROW PATTERNS:")
    
    # Check if position in group matters
    df_with_pos = df_sample.with_row_count('row_idx').with_columns([
        pl.col('row_idx').rank().over('ranker_id').alias('position_in_group')
    ])
    
    selected_positions = df_with_pos.filter(pl.col('selected') == 1)['position_in_group'].to_list()
    print(f"   Selected positions: {selected_positions[:20]}")
    print(f"   First position selected: {(np.array(selected_positions) == 1).mean():.3f}")
    
    # 3. Check if there's a pattern in the data order
    print("\n3. DATA ORDER ANALYSIS:")
    df_with_idx = df_sample.with_row_count('original_idx')
    
    # Group by ranker_id and see if selected is always at a certain position
    position_analysis = df_with_idx.group_by('ranker_id').agg([
        pl.col('original_idx').first().alias('first_idx'),
        pl.col('selected').arg_max().alias('selected_pos'),
        pl.count().alias('size')
    ])
    
    selected_at_first = (position_analysis['selected_pos'] == 0).mean()
    print(f"   Selected item is first in group: {selected_at_first:.3f}")
    
    # 4. Look for any obvious distinguishing features
    print("\n4. FEATURE ANALYSIS:")
    
    selected_rows = df_sample.filter(pl.col('selected') == 1)
    not_selected_rows = df_sample.filter(pl.col('selected') == 0)
    
    # Check key features
    if 'totalPrice' in df_sample.columns:
        print(f"   Selected avg price: {selected_rows['totalPrice'].mean():.2f}")
        print(f"   Not selected avg price: {not_selected_rows['totalPrice'].mean():.2f}")
    
    if 'pricingInfo_isAccessTP' in df_sample.columns:
        sel_policy = selected_rows['pricingInfo_isAccessTP'].mean()
        not_sel_policy = not_selected_rows['pricingInfo_isAccessTP'].mean()
        print(f"   Selected policy compliance: {sel_policy:.3f}")
        print(f"   Not selected policy compliance: {not_sel_policy:.3f}")
    
    # 5. Check for potential leakage
    print("\n5. POTENTIAL LEAKAGE ANALYSIS:")
    
    # Check if any column has perfect correlation with selected
    for col in df_sample.columns:
        if col == 'selected':
            continue
            
        if df_sample[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]:
            try:
                correlation = df_sample.select([
                    pl.corr('selected', col).alias('corr')
                ])['corr'][0]
                
                if correlation is not None and abs(correlation) > 0.8:
                    print(f"   HIGH CORRELATION: {col} = {correlation:.4f}")
            except:
                pass
    
    # 6. Analyze session-level patterns  
    print("\n6. SESSION PATTERNS:")
    
    # Check if certain ranker_ids always have the same pattern
    session_patterns = df_sample.group_by('ranker_id').agg([
        pl.col('selected').sum().alias('selected_count'),
        pl.count().alias('total_count'),
        pl.col('totalPrice').min().alias('min_price'),
        pl.col('totalPrice').max().alias('max_price')
    ])
    
    # Check if selected is always the cheapest
    df_price_rank = df_sample.with_columns([
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank')
    ])
    
    selected_price_ranks = df_price_rank.filter(pl.col('selected') == 1)['price_rank'].to_list()
    cheapest_selected = (np.array(selected_price_ranks) == 1).mean()
    print(f"   Selected is cheapest: {cheapest_selected:.3f}")
    
    # Check if selected is always policy compliant
    if 'pricingInfo_isAccessTP' in df_sample.columns:
        policy_pattern = df_sample.filter(pl.col('selected') == 1)['pricingInfo_isAccessTP'].mean()
        print(f"   Selected is always policy compliant: {policy_pattern:.3f}")
    
    # 7. Simple baseline prediction
    print("\n7. SIMPLE BASELINE TESTS:")
    
    # Test: always pick cheapest policy-compliant option
    if 'pricingInfo_isAccessTP' in df_sample.columns:
        test_predictions = df_sample.with_columns([
            pl.when(pl.col('pricingInfo_isAccessTP') == True)
              .then(pl.col('totalPrice').rank().over('ranker_id'))
              .otherwise(999)
              .alias('policy_price_rank')
        ])
        
        # Predict: cheapest policy-compliant = selected
        simple_pred = test_predictions.with_columns([
            (pl.col('policy_price_rank') == pl.col('policy_price_rank').min().over('ranker_id')).alias('simple_pred')
        ])
        
        accuracy = simple_pred.filter(pl.col('simple_pred') == pl.col('selected')).height / simple_pred.height
        print(f"   'Cheapest policy-compliant' accuracy: {accuracy:.3f}")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    analyze_dataset()