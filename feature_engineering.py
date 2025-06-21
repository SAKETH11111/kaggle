"""
FlightRank 2025 - Feature Engineering Module
Advanced feature extraction for flight recommendation ranking
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """A robust feature engineering pipeline for the FlightRank 2025 competition."""

    def __init__(self):
        """Initializes the FeatureEngineer."""
        logger.info("FeatureEngineer initialized.")

    def create_price_intelligence_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Creates advanced price-based features based on session-level statistics."""
        logger.info("Creating price intelligence features...")
        
        if "totalPrice" not in df.columns:
            logger.warning("'totalPrice' not found. Skipping price intelligence features.")
            return df

        session_stats = df.group_by("ranker_id").agg([
            pl.min("totalPrice").alias("min_price"),
            pl.median("totalPrice").alias("median_price"),
            pl.mean("totalPrice").alias("mean_price"),
            pl.std("totalPrice").alias("std_price")
        ])

        df = df.join(session_stats, on="ranker_id", how="left")

        df = df.with_columns([
            (((pl.col("totalPrice") - pl.col("min_price")) / pl.col("min_price")))
             .fill_nan(0).fill_null(0).alias("rel_price_vs_min"),
            
            pl.col("totalPrice").rank(method="dense").over("ranker_id")
             .alias("price_rank_in_session"),

            (pl.col("totalPrice") - pl.col("median_price")).abs()
             .alias("price_vs_median_gap"),

            (pl.col("totalPrice") <= (pl.col("min_price") * 1.10))
             .alias("is_cheapest_or_near"),

            (((pl.col("totalPrice") - pl.col("mean_price")) / pl.col("std_price")))
             .fill_nan(0).fill_null(0).alias("price_z_score")
        ])
        
        return df.drop(["min_price", "median_price", "mean_price", "std_price"])

    def create_policy_compliance_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create features that capture compliance with corporate travel policy."""
        logger.info("Creating corporate policy compliance features...")

        # Use a fallback for policy_compliant if it's missing
        policy_col = "pricingInfo_isAccessTP"
        if policy_col not in df.columns:
            df = df.with_columns(pl.lit(False).alias(policy_col))
        
        # Use a fallback for cabin_class if it's missing
        cabin_col = "legs0_segments0_cabinClass"
        if cabin_col not in df.columns:
            df = df.with_columns(pl.lit("Unknown").alias(cabin_col))

        df = df.with_columns([
            pl.col(policy_col).fill_null(False).cast(pl.Int8).alias("policy_compliant_flag"),
            pl.when(
                (pl.col(cabin_col).fill_null("Unknown").is_in(["Economy", "PremiumEconomy"])) |
                (pl.col(policy_col).fill_null(False) == True)
            ).then(1).otherwise(0).cast(pl.Int8).alias("cabin_allowed")
        ])
        
        return df

    def create_fare_flexibility_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create features that quantify fare flexibility."""
        logger.info("Creating fare flexibility features...")

        # Define source columns and ensure they exist with safe defaults
        cancellation_fee_col = "miniRules0_monetaryAmount"
        exchange_fee_col = "miniRules1_monetaryAmount"
        baggage_col = "legs0_segments0_baggageAllowance_quantity"

        if cancellation_fee_col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(cancellation_fee_col))
        if exchange_fee_col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(exchange_fee_col))
        if baggage_col not in df.columns:
            df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias(baggage_col))

        return df.with_columns([
            pl.when(pl.col(cancellation_fee_col).is_null())
              .then(pl.lit("NonRefundable"))
              .when(pl.col(cancellation_fee_col) > 0)
              .then(pl.lit("Fee"))
              .otherwise(pl.lit("Free"))
              .alias("cancellation_policy_tier"),

            pl.when(pl.col(exchange_fee_col).is_null())
              .then(pl.lit("NoChange"))
              .when(pl.col(exchange_fee_col) > 0)
              .then(pl.lit("FeeChange"))
              .otherwise(pl.lit("FreeChange"))
              .alias("change_policy_category"),

            (pl.col(baggage_col).fill_null(0) > 0).cast(pl.Int8).alias("free_baggage")
        ])

    def create_temporal_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create temporal features for business travel preferences."""
        logger.info("Creating temporal features...")
        
        dep_time_col = "legs0_departureAt"
        if dep_time_col not in df.columns:
            logger.warning(f"'{dep_time_col}' not found. Skipping temporal features.")
            return df

        # Assuming departure time is in local time as per competition docs.
        df = df.with_columns(
            pl.col(dep_time_col).str.to_datetime(cache=False).alias("departure_time")
        )

        df = df.with_columns([
            pl.col("departure_time").dt.hour().alias("dep_hour"),
            pl.col("departure_time").dt.weekday().alias("dep_weekday") # Monday=1, Sunday=7
        ])

        df = df.with_columns([
            ((pl.col("dep_hour") >= 6) & (pl.col("dep_hour") < 22)).cast(pl.Int8).alias("is_business_hours"),
            ((pl.col("dep_hour") >= 22) | (pl.col("dep_hour") < 6)).cast(pl.Int8).alias("is_redeye_flight"),
            (pl.col("dep_weekday").is_in([6, 7])).cast(pl.Int8).alias("is_weekend_flight"),
            (pl.col("dep_hour") // 6).alias("departure_hour_bin"),
            (pl.sin(2 * np.pi * pl.col("dep_hour") / 24)).alias("dep_hour_sin"),
            (pl.cos(2 * np.pi * pl.col("dep_hour") / 24)).alias("dep_hour_cos")
        ])
        
        return df.drop("departure_time")

    def create_booking_context_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create booking urgency and lead time features."""
        logger.info("Creating booking context features...")

        req_date_col = "requestDate"
        req_dep_date_col = "requestDepartureDate"

        if req_date_col not in df.columns or req_dep_date_col not in df.columns:
            logger.warning("Request date columns not found. Skipping booking context features.")
            return df
        
        df = df.with_columns([
            pl.col(req_date_col).str.to_datetime(cache=False).cast(pl.Date).alias("request_date_parsed"),
            pl.col(req_dep_date_col).str.to_datetime(cache=False).cast(pl.Date).alias("departure_date_parsed")
        ])

        df = df.with_columns([
            (pl.col("departure_date_parsed") - pl.col("request_date_parsed")).dt.days().alias("booking_lead_days")
        ])

        df = df.with_columns([
            pl.when(pl.col("booking_lead_days") <= 3).then(pl.lit("last_minute"))
             .when(pl.col("booking_lead_days") <= 14).then(pl.lit("short_term"))
             .otherwise(pl.lit("planned"))
             .alias("booking_urgency_category")
        ])
        
        return df.drop(["request_date_parsed", "departure_date_parsed"])
    
    def create_route_quality_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Creates features based on the route's quality, like connections."""
        logger.info("Creating route quality features...")

        segments_count_col = "segments_count"
        if segments_count_col not in df.columns:
            # Fallback: assume direct flight if segment count is missing
            df = df.with_columns(pl.lit(1, dtype=pl.Int32).alias(segments_count_col))
        
        return df.with_columns([
            (pl.col(segments_count_col).fill_null(1) - 1).alias("connection_count"),
            (pl.col(segments_count_col).fill_null(1) == 1).cast(pl.Int8).alias("is_direct_flight")
        ])

    def create_duration_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Creates duration-based features like ranking and ratios."""
        logger.info("Creating duration features...")

        duration_col = "total_duration_minutes"
        if duration_col not in df.columns:
            # Attempt to create it from legs0_duration if available
            if "legs0_duration" in df.columns:
                df = df.with_columns(
                    (pl.col("legs0_duration").str.extract(r"(\d+):(\d+):(\d+)", 1).cast(pl.Int32) * 60 +
                     pl.col("legs0_duration").str.extract(r"(\d+):(\d+):(\d+)", 2).cast(pl.Int32))
                    .alias(duration_col)
                )
            else:
                logger.warning(f"'{duration_col}' not found. Skipping duration features.")
                return df

        session_min_dur = df.group_by("ranker_id").agg(
            pl.min(duration_col).alias("min_dur")
        )

        df = df.join(session_min_dur, on="ranker_id", how="left")

        df = df.with_columns([
            (pl.col(duration_col) / pl.col("min_dur"))
             .fill_nan(1.0).fill_null(1.0).alias("duration_vs_min"),
            pl.col(duration_col).rank(method="dense").over("ranker_id").alias("duration_rank")
        ])
        
        return df.drop("min_dur")
    
    def engineer_all_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply all feature engineering steps from the Week 1 roadmap in a logical,
        robust, and sequential manner.
        """
        logger.info("Starting Week 1 advanced feature engineering pipeline...")

        # The order is intentional to build features upon each other if needed.
        df = self.create_price_intelligence_features(df)
        df = self.create_policy_compliance_features(df)
        df = self.create_fare_flexibility_features(df)
        df = self.create_temporal_features(df)
        df = self.create_booking_context_features(df)
        df = self.create_route_quality_features(df)
        df = self.create_duration_features(df)

        logger.info("Advanced feature engineering complete!")
        return df
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get categorized feature lists for analysis"""
        # This can be updated to reflect the new feature names
        return {
            "price_features": [
                "rel_price_vs_min", "price_rank_in_session", "price_vs_median_gap",
                "is_cheapest_or_near", "price_z_score"
            ],
            "policy_features": [
                "policy_compliant_flag", "cabin_allowed"
            ],
            "flexibility_features": [
                "cancellation_policy_tier", "change_policy_category", "free_baggage"
            ],
            "temporal_features": [
                "is_business_hours", "is_redeye_flight", "is_weekend_flight",
                "departure_hour_bin", "dep_hour_sin", "dep_hour_cos"
            ],
            "context_features": [
                "booking_lead_days", "booking_urgency_category"
            ],
            "route_features": [
                "connection_count", "is_direct_flight"
            ],
            "duration_features": [
                "duration_vs_min", "duration_rank"
            ],
            "route_features": [
                "is_roundtrip", "origin_airport", "destination_airport",
                "marketing_carrier", "same_marketing_operating", "has_connection",
                "cabin_class_name", "seats_available", "baggage_quantity"
            ],
            "policy_features": [
                "has_corporate_tariff", "policy_compliant", "is_vip", "books_by_self",
                "can_cancel", "can_exchange", "cancellation_fee_rate", "exchange_fee_rate"
            ],
            "group_features": [
                "group_size", "price_position_in_group", "group_price_std",
                "unique_carriers_in_group", "group_policy_compliance_rate"
            ],
            "interaction_features": [
                "price_duration_interaction", "combined_rank_score", "vip_price_interaction",
                "corporate_policy_interaction", "convenient_time_score", "weekend_departure"
            ]
        }

class FeatureSelector:
    """Feature selection and importance analysis"""
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        
    def select_high_variance_features(self, df: pl.LazyFrame, threshold: float = 0.01) -> List[str]:
        """Select features with variance above threshold"""
        numeric_cols = [col for col, dtype in df.schema.items() 
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        # Calculate variance for numeric columns
        variance_df = df.select([
            pl.col(col).var().alias(f"{col}_var") for col in numeric_cols
        ]).collect()
        
        high_var_features = []
        for col in numeric_cols:
            var_val = variance_df.select(pl.col(f"{col}_var")).item()
            if var_val and var_val > threshold:
                high_var_features.append(col)
                
        return high_var_features
    
    def get_feature_correlation_matrix(self, df: pl.LazyFrame, features: List[str]) -> pl.DataFrame:
        """Calculate correlation matrix for feature selection"""
        corr_df = df.select(features).collect()
        
        # Convert to pandas for correlation calculation
        pandas_df = corr_df.to_pandas()
        numeric_df = pandas_df.select_dtypes(include=[np.number])
        
        correlation_matrix = numeric_df.corr()
        
        return pl.from_pandas(correlation_matrix.reset_index())

if __name__ == "__main__":
    # Test feature engineering
    from data_pipeline import DataPipeline, DataConfig
    
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    # Load sample data
    train_lf = pipeline.loader.load_structured_data("train").limit(1000)
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    enriched_df = engineer.engineer_all_features(train_lf)
    
    # Collect sample results
    sample_results = enriched_df.limit(10).collect()
    print(f"Sample engineered features shape: {sample_results.shape}")
    print(f"New columns created: {len(enriched_df.columns) - len(train_lf.columns)}") 