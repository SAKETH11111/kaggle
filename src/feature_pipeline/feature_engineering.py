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
    """Advanced feature engineering for flight ranking"""
    
    def __init__(self):
        self.price_features = []
        self.time_features = []
        self.route_features = []
        self.policy_features = []
        
    def create_price_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create price-related features"""
        logger.info("Creating price features...")
        
        price_features = df.with_columns([
            # Basic price features
            pl.col("totalPrice").alias("total_price"),
            pl.col("taxes").alias("tax_amount"),
            (pl.col("totalPrice") - pl.col("taxes")).alias("base_price"),
            (pl.col("taxes") / pl.col("totalPrice")).alias("tax_rate"),
            
            # Price ranking within group
            pl.col("totalPrice").rank(method="ordinal").over("ranker_id").alias("price_rank"),
            pl.col("totalPrice").rank(method="ordinal", descending=True).over("ranker_id").alias("price_rank_desc"),
            
            # Price statistics within group
            (pl.col("totalPrice") - pl.col("totalPrice").mean().over("ranker_id")).alias("price_diff_from_mean"),
            (pl.col("totalPrice") - pl.col("totalPrice").min().over("ranker_id")).alias("price_diff_from_min"),
            (pl.col("totalPrice") - pl.col("totalPrice").max().over("ranker_id")).alias("price_diff_from_max"),
            
            # Price percentiles within group
            ((pl.col("totalPrice").rank().over("ranker_id") - 1) / 
             (pl.col("totalPrice").count().over("ranker_id") - 1)).alias("price_percentile"),
             
            # Price bins
            pl.col("totalPrice").qcut(5, labels=["very_cheap", "cheap", "medium", "expensive", "very_expensive"]).alias("price_bin")
        ])
        
        return price_features
    
    def create_time_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create time-related features"""
        logger.info("Creating time features...")
        
        # Convert duration strings to minutes
        def duration_to_minutes(duration_col):
            duration_struct = (
                pl.col(duration_col)
                .str.extract_groups(r"(\d+):(\d+):(\d+)")
                .struct.rename_fields(["hours", "minutes", "seconds"])
            )
            return (
                duration_struct.struct.field("hours").cast(pl.Int32) * 60 +
                duration_struct.struct.field("minutes").cast(pl.Int32)
            )
        
        time_features = df.with_columns([
            # Parse departure and arrival times
            pl.col("legs0_departureAt").str.to_datetime().alias("departure_datetime"),
            pl.col("legs0_arrivalAt").str.to_datetime().alias("arrival_datetime"),
            
            # Duration features
            duration_to_minutes("legs0_duration").alias("total_duration_minutes"),
            
            # Time of day features
            pl.col("legs0_departureAt").str.to_datetime().dt.hour().alias("departure_hour"),
            pl.col("legs0_arrivalAt").str.to_datetime().dt.hour().alias("arrival_hour"),
            
            # Day of week features
            pl.col("legs0_departureAt").str.to_datetime().dt.weekday().alias("departure_weekday"),
            
            # Time categories
            pl.when(pl.col("legs0_departureAt").str.to_datetime().dt.hour().is_between(6, 11))
              .then(pl.lit("morning"))
              .when(pl.col("legs0_departureAt").str.to_datetime().dt.hour().is_between(12, 17))
              .then(pl.lit("afternoon"))
              .when(pl.col("legs0_departureAt").str.to_datetime().dt.hour().is_between(18, 21))
              .then(pl.lit("evening"))
              .otherwise(pl.lit("night"))
              .alias("departure_time_category")
        ])
        
        # Add duration ranking and statistics within group
        time_features = time_features.with_columns([
            pl.col("total_duration_minutes").rank().over("ranker_id").alias("duration_rank"),
            (pl.col("total_duration_minutes") - pl.col("total_duration_minutes").mean().over("ranker_id")).alias("duration_diff_from_mean"),
            ((pl.col("total_duration_minutes").rank().over("ranker_id") - 1) / 
             (pl.col("total_duration_minutes").count().over("ranker_id") - 1)).alias("duration_percentile")
        ])
        
        return time_features
    
    def create_route_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create route and airline features"""
        logger.info("Creating route features...")
        
        route_features = df.with_columns([
            # Route features
            pl.col("searchRoute").alias("search_route"),
            pl.col("searchRoute").str.contains("/").alias("is_roundtrip"),
            
            # Airport features
            pl.col("legs0_segments0_departureFrom_airport_iata").alias("origin_airport"),
            pl.col("legs0_segments0_arrivalTo_airport_iata").alias("destination_airport"),
            
            # Airline features
            pl.col("legs0_segments0_marketingCarrier_code").alias("marketing_carrier"),
            pl.col("legs0_segments0_operatingCarrier_code").alias("operating_carrier"),
            (pl.col("legs0_segments0_marketingCarrier_code") == 
             pl.col("legs0_segments0_operatingCarrier_code")).alias("same_marketing_operating"),
             
            # Aircraft features
            pl.col("legs0_segments0_aircraft_code").alias("aircraft_code"),
            
            # Connection features (if segment 1 exists)
            pl.col("legs0_segments1_departureFrom_airport_iata").is_not_null().alias("has_connection"),
            
            # Cabin class features
            pl.col("legs0_segments0_cabinClass").alias("cabin_class"),
            pl.when(pl.col("legs0_segments0_cabinClass") == 1.0)
              .then(pl.lit("economy"))
              .when(pl.col("legs0_segments0_cabinClass") == 2.0)
              .then(pl.lit("business"))
              .when(pl.col("legs0_segments0_cabinClass") == 4.0)
              .then(pl.lit("premium"))
              .otherwise(pl.lit("unknown"))
              .alias("cabin_class_name"),
              
            # Seats available
            pl.col("legs0_segments0_seatsAvailable").alias("seats_available"),
            
            # Baggage allowance
            pl.col("legs0_segments0_baggageAllowance_quantity").alias("baggage_quantity")
        ])
        
        return route_features
    
    def create_policy_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create policy and compliance features"""
        logger.info("Creating policy features...")
        
        policy_features = df.with_columns([
            # Corporate features
            pl.col("corporateTariffCode").alias("corporate_tariff"),
            pl.col("corporateTariffCode").is_not_null().alias("has_corporate_tariff"),
            
            # Travel policy compliance
            pl.col("pricingInfo_isAccessTP").alias("policy_compliant"),
            
            # User features
            pl.col("profileId").alias("profile_id"),
            pl.col("companyID").alias("company_id"),
            pl.col("sex").alias("user_gender"),
            pl.col("nationality").alias("user_nationality"),
            pl.col("frequentFlyer").alias("frequent_flyer_programs"),
            pl.col("isVip").alias("is_vip"),
            pl.col("bySelf").alias("books_by_self"),
            
            # Cancellation and exchange rules
            pl.col("miniRules0_statusInfos").alias("can_cancel"),
            pl.col("miniRules1_statusInfos").alias("can_exchange"),
            pl.col("miniRules0_monetaryAmount").alias("cancellation_fee"),
            pl.col("miniRules1_monetaryAmount").alias("exchange_fee"),
            
            # Rule costs as percentage of ticket price
            (pl.col("miniRules0_monetaryAmount") / pl.col("totalPrice")).alias("cancellation_fee_rate"),
            (pl.col("miniRules1_monetaryAmount") / pl.col("totalPrice")).alias("exchange_fee_rate")
        ])
        
        return policy_features
    
    def create_group_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create group-level features within ranker_id"""
        logger.info("Creating group features...")
        
        group_features = df.with_columns([
            # Group size & position
            pl.count().over("ranker_id").alias("group_size"),
            pl.int_range(1, pl.count() + 1).over("ranker_id").alias("position"),
            
            # Position in group (by price, duration, etc.)
            pl.col("totalPrice").rank().over("ranker_id").alias("price_position_in_group"),
            pl.col("legs0_duration").rank().over("ranker_id").alias("duration_position_in_group"),
            
            # Group statistics
            pl.col("totalPrice").std().over("ranker_id").alias("group_price_std"),
            pl.col("totalPrice").max().over("ranker_id") - pl.col("totalPrice").min().over("ranker_id").alias("group_price_range"),
            
            # Company diversity in group
            pl.col("legs0_segments0_marketingCarrier_code").n_unique().over("ranker_id").alias("unique_carriers_in_group"),
            pl.col("legs0_segments0_cabinClass").n_unique().over("ranker_id").alias("unique_cabin_classes_in_group"),
            
            # Policy compliance rate in group
            pl.col("pricingInfo_isAccessTP").mean().over("ranker_id").alias("group_policy_compliance_rate")
        ])
        
        return group_features
    
    def create_json_interaction_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create interaction features with the aggregated JSON data."""
        logger.info("Creating interaction features with JSON data...")

        json_interaction_features = df.with_columns([
            # Price relative to the average price in the search
            (pl.col("total_price") / pl.col("avg_total_price")).alias("price_vs_avg_search_price"),

            # Whether this flight's price is above or below the average
            (pl.col("total_price") > pl.col("avg_total_price")).cast(pl.Int32).alias("is_pricier_than_average"),

            # Interaction between policy compliance of the flight and the search average
            (pl.col("policy_compliant") * pl.col("avg_policy_compliant")).alias("policy_compliance_interaction"),
        ])

        return json_interaction_features

    def create_carrier_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create features specific to airline carriers."""
        logger.info("Creating carrier-specific features...")

        carrier_features = df.with_columns([
            # Carrier Frequency: How many flights a carrier has in the search
            (pl.col("marketing_carrier").count().over("ranker_id", "marketing_carrier")).alias("carrier_frequency_in_search"),

            # Carrier Market Share: The percentage of flights a carrier has in the search
            (pl.col("marketing_carrier").count().over("ranker_id", "marketing_carrier") / pl.col("ranker_id").count().over("ranker_id")).alias("carrier_market_share")
        ])

        return carrier_features

    def create_route_complexity_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create features related to the complexity of the flight route."""
        logger.info("Creating route complexity features...")

        # Count the number of segments for the first leg
        # This requires checking for the existence of segment columns
        segment_cols = [col for col in df.columns if "legs0_segments" in col and "departureFrom" in col]
        num_segments = len(segment_cols)

        # Calculate layover duration only when both segment timestamp columns are present
        seg1_dep_col = "legs0_segments1_departureAt"
        seg0_arr_col = "legs0_segments0_arrivalAt"

        if seg1_dep_col in df.columns and seg0_arr_col in df.columns:
            layover_duration_expr = (
                pl.col(seg1_dep_col).str.to_datetime() - pl.col(seg0_arr_col).str.to_datetime()
            ).dt.total_seconds() / 60
            layover_duration_expr = layover_duration_expr.cast(pl.Int32).alias("layover_duration_minutes")
        else:
            layover_duration_expr = pl.lit(0).alias("layover_duration_minutes")

        route_complexity_features = df.with_columns([
            pl.lit(num_segments).alias("num_segments"),
            (pl.lit(num_segments) == 1).cast(pl.Int32).alias("is_direct"),
            layover_duration_expr
        ])

        return route_complexity_features

    def create_advanced_time_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create advanced time-based features, including booking window and trip duration."""
        logger.info("Creating advanced time features...")

        # Ensure optional request-related columns exist so that Polars' expression planner
        # does not raise ColumnNotFound errors when they are absent in the input schema.
        optional_cols = ["requestDate", "requestDepartureDate", "requestReturnDate"]
        for col_name in optional_cols:
            if col_name not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col_name))

        # Ensure all request-related columns are Utf8 so .str namespace is available regardless of source dtype.
        df = df.with_columns([
            pl.col("requestDate").cast(pl.Utf8).alias("requestDate"),
            pl.col("requestDepartureDate").cast(pl.Utf8).alias("requestDepartureDate"),
            pl.col("requestReturnDate").cast(pl.Utf8).alias("requestReturnDate"),
        ])

        advanced_time_features = df.with_columns([
            # Booking window: difference between search request date and flight departure date.
            # Prefer a dedicated search timestamp column if present; otherwise, use requestDepartureDate (date selected by traveller).
            (
                pl.col("departure_datetime")
                - pl.coalesce([
                    pl.when(pl.col("requestDate").is_not_null())
                      .then(pl.col("requestDate").str.to_datetime())
                      .otherwise(pl.col("requestDepartureDate").str.to_datetime())
                ])
            ).dt.total_days().cast(pl.Int32).alias("booking_window_days"),

            # Trip Duration: Difference between return and departure date (in days)
            (
                pl.col("requestReturnDate").str.to_datetime()
                - pl.col("requestDepartureDate").str.to_datetime()
            ).dt.total_days().cast(pl.Int32).alias("trip_duration_days"),

            # "Red-Eye" Flight Indicator: Departs late (e.g., after 10 PM) and arrives early (e.g., before 5 AM)
            (
                (pl.col("departure_hour") >= 22) & (pl.col("arrival_hour") <= 5)
            ).cast(pl.Int32).alias("is_red_eye_flight"),
        ])

        # Remove raw request date columns to avoid dtype conflicts downstream
        advanced_time_features = advanced_time_features.drop(optional_cols)
        return advanced_time_features

    def create_interaction_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create interaction features between different aspects"""
        logger.info("Creating interaction features...")
        
        interaction_features = df.with_columns([
            # Price vs Duration trade-off
            (pl.col("totalPrice") * pl.col("total_duration_minutes")).alias("price_duration_interaction"),
            (pl.col("price_rank") + pl.col("duration_rank")).alias("combined_rank_score"),
            
            # VIP and price interaction
            (pl.col("isVip").cast(pl.Int32) * pl.col("totalPrice")).alias("vip_price_interaction"),
            
            # Corporate tariff and policy compliance
            (pl.col("has_corporate_tariff").cast(pl.Int32) * pl.col("pricingInfo_isAccessTP").cast(pl.Int32)).alias("corporate_policy_interaction"),
            
            # Time convenience score (prefer morning/afternoon flights)
            pl.when(pl.col("departure_time_category").is_in(["morning", "afternoon"]))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .alias("convenient_time_score"),
              
            # Weekend travel indicator
            (pl.col("departure_weekday") >= 5).cast(pl.Int32).alias("weekend_departure"),
            
            # Premium service indicators
            ((pl.col("cabin_class") > 1.0) | pl.col("isVip")).cast(pl.Int32).alias("premium_service_indicator")
        ])
        
        return interaction_features
    
    def engineer_all_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering...")
        
        # Apply all feature engineering steps
        df = self.create_price_features(df)
        df = self.create_time_features(df)
        df = self.create_advanced_time_features(df)
        # Route-related base features first (creates marketing_carrier etc.)
        df = self.create_route_features(df)
        df = self.create_policy_features(df)
        df = self.create_carrier_features(df)
        df = self.create_route_complexity_features(df)
        df = self.create_json_interaction_features(df)
        df = self.create_group_features(df)
        df = self.create_interaction_features(df)
        
        # Drop raw request date columns if they remain to avoid dtype issues
        for col_to_drop in ["requestDate", "requestDepartureDate", "requestReturnDate"]:
            if col_to_drop in df.columns:
                df = df.drop(col_to_drop)

        logger.info("Feature engineering complete!")
        return df
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get categorized feature lists for analysis"""
        return {
            "price_features": [
                "total_price", "tax_amount", "base_price", "tax_rate",
                "price_rank", "price_diff_from_mean", "price_percentile", "price_bin"
            ],
            "time_features": [
                "total_duration_minutes", "departure_hour", "arrival_hour",
                "departure_weekday", "departure_time_category", "duration_rank", "duration_percentile"
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