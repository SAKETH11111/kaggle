"""
Aggressive feature engineering approach for breakthrough performance
"""
import polars as pl
import numpy as np
import logging
from utils import TargetEncodingUtils

logger = logging.getLogger(__name__)

class AggressiveFeatureEngineer:
    """Ultra-aggressive feature engineering for maximum signal extraction"""
    
    def __init__(self):
        pass
    
    def create_ultra_target_encoding(self, df: pl.DataFrame, cv_splits: list) -> pl.DataFrame:
        """Target encode EVERYTHING that's categorical or high-cardinality"""
        logger.info("Ultra-aggressive target encoding...")
        
        # Find all potential categorical columns
        target_encode_candidates = []
        
        for col in df.columns:
            if col in ['Id', 'ranker_id', 'selected', 'requestDate']:
                continue
                
            # Target encode if:
            # 1. String/categorical type
            # 2. High cardinality (>10 unique values)  
            # 3. Any column that looks like an identifier
            
            if df[col].dtype in [pl.Utf8, pl.Categorical]:
                target_encode_candidates.append(col)
            elif df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                # Check if it's actually categorical (like codes)
                unique_count = df[col].n_unique()
                total_count = df[col].count()
                if unique_count < total_count * 0.5 and unique_count > 10:  # Seems categorical
                    target_encode_candidates.append(col)
        
        logger.info(f"Target encoding {len(target_encode_candidates)} columns: {target_encode_candidates[:10]}...")
        
        # Apply target encoding
        if target_encode_candidates:
            df = TargetEncodingUtils.kfold_target_encode(
                df=df,
                cat_cols=target_encode_candidates,
                target='selected',
                folds=cv_splits,
                noise=0.005  # Less noise for more signal
            )
        
        return df
    
    def create_collaborative_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Collaborative filtering style features"""
        logger.info("Creating collaborative filtering features...")
        
        # User-based collaborative filtering
        if 'profileId' in df.columns:
            # What do similar users typically choose?
            user_stats = df.group_by('profileId').agg([
                pl.col('totalPrice').mean().alias('user_avg_price'),
                pl.col('is_direct_flight').mean().alias('user_direct_preference'),
                pl.col('policy_compliant_flag').mean().alias('user_policy_compliance'),
                pl.col('legs0_segments0_operatingCarrier_code').mode().first().alias('user_preferred_airline')
            ])
            
            df = df.join(user_stats, on='profileId', how='left')
            
            # How different is this choice from user's typical pattern?
            df = df.with_columns([
                (pl.col('totalPrice') - pl.col('user_avg_price')).abs().alias('price_deviation_from_user'),
                (pl.col('is_direct_flight') - pl.col('user_direct_preference')).abs().alias('direct_deviation_from_user'),
                (pl.col('legs0_segments0_operatingCarrier_code') == pl.col('user_preferred_airline')).cast(pl.Int8).alias('user_airline_match')
            ])
        
        # Item-based collaborative filtering
        route_cols = ['legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_arrivalTo_airport_iata']
        if all(col in df.columns for col in route_cols):
            # How popular is this specific flight option globally?
            flight_popularity = df.group_by(route_cols + ['legs0_segments0_operatingCarrier_code']).agg([
                pl.count().alias('flight_global_frequency'),
                pl.col('selected').mean().alias('flight_global_selection_rate')
            ])
            
            df = df.join(flight_popularity, on=route_cols + ['legs0_segments0_operatingCarrier_code'], how='left')
        
        return df
    
    def create_power_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Most powerful combination features"""
        logger.info("Creating power combination features...")
        
        # The "perfect business traveler choice" feature
        df = df.with_columns([
            # Ultimate score: policy + direct + reasonable price + good time
            (
                pl.col('policy_compliant_flag') * 10 +  # Policy is everything
                pl.col('is_direct_flight') * 5 +       # Direct highly valued  
                (pl.col('rel_price_vs_min') < 0.3).cast(pl.Int8) * 3 +  # Reasonable price
                pl.col('is_business_hours') * 2 +      # Good timing
                (pl.col('duration_vs_min') < 1.2).cast(pl.Int8) * 1     # Reasonable duration
            ).alias('business_traveler_score'),
            
            # The "forced choice" feature (when no good options exist)
            pl.when(pl.col('policy_compliant_flag').sum().over('ranker_id') == 0)
              .then(
                  # If no policy options, choose least bad
                  (pl.col('is_direct_flight') * 3 + 
                   (1 / (1 + pl.col('rel_price_vs_min'))) * 2 +
                   pl.col('is_business_hours'))
              )
              .otherwise(pl.col('business_traveler_score'))
              .alias('forced_choice_score'),
            
            # Rank these scores within session
            pl.col('business_traveler_score').rank(method='ordinal', descending=True).over('ranker_id').alias('business_score_rank'),
        ])
        
        # Is this the obvious best choice?
        df = df.with_columns([
            (pl.col('business_score_rank') == 1).cast(pl.Int8).alias('obvious_best_choice'),
            (pl.col('business_score_rank') <= 3).cast(pl.Int8).alias('top3_business_choice'),
            
            # How much better is this than the average option in session?
            (pl.col('business_traveler_score') - pl.col('business_traveler_score').mean().over('ranker_id')).alias('score_above_average')
        ])
        
        return df
    
    def engineer_aggressive_features(self, df: pl.DataFrame, cv_splits: list) -> pl.DataFrame:
        """Apply all aggressive feature engineering"""
        logger.info("Starting aggressive feature engineering...")
        
        # 1. Ultra-aggressive target encoding
        df = self.create_ultra_target_encoding(df, cv_splits)
        
        # 2. Collaborative filtering features  
        df = self.create_collaborative_features(df)
        
        # 3. Power combination features
        df = self.create_power_features(df)
        
        logger.info(f"Aggressive feature engineering complete. Final shape: {df.shape}")
        return df