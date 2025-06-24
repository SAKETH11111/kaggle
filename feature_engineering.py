import polars as pl
import numpy as np
import logging
from utils import FeatureUtils

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Implements the advanced feature engineering plan (Week 1) for the FlightRank competition.
    """
    def __init__(self, baseline_features: list = None):
        self.baseline_features = baseline_features if baseline_features else []

    def create_price_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 1: Pricing Intelligence Features"""
        logger.info("Creating Day 1: Pricing Intelligence Features...")
        
        session_stats = df.group_by("ranker_id").agg([
            pl.min("totalPrice").alias("min_price"),
            pl.median("totalPrice").alias("median_price"),
            pl.mean("totalPrice").alias("mean_price"),
            pl.std("totalPrice").alias("std_price")
        ])
        
        df = df.join(session_stats, on="ranker_id")
        
        df = df.with_columns([
            ((pl.col("totalPrice") - pl.col("min_price")) / pl.col("min_price")).alias("rel_price_vs_min"),
            pl.col("totalPrice").rank("dense").over("ranker_id").alias("price_rank_in_session"),
            # Percentile (0-1) within session for smoother price position signal
            ((pl.col("totalPrice").rank().over("ranker_id") - 1) /
             (pl.count().over("ranker_id") - 1)).alias("price_percentile_in_session"),
            (pl.col("totalPrice") - pl.col("median_price")).abs().alias("price_vs_median_gap"),
            (pl.col("totalPrice") <= pl.col("min_price") * 1.10).alias("is_cheapest_or_near"),
            ((pl.col("totalPrice") - pl.col("mean_price")) / pl.col("std_price")).alias("price_z_score")
        ])
        
        return FeatureUtils.guard_against_infinities(df, ["rel_price_vs_min", "price_z_score"])

    def create_policy_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 2: Corporate Policy Compliance Features"""
        logger.info("Creating Day 2: Corporate Policy Compliance Features...")
        
        return df.with_columns([
            pl.col("pricingInfo_isAccessTP").cast(pl.Int8).alias("policy_compliant_flag"),
            pl.when(
                (pl.col("legs0_segments0_cabinClass").is_in([1.0, 4.0])) | (pl.col("pricingInfo_isAccessTP") == True)
            ).then(1).otherwise(0).cast(pl.Int8).alias("cabin_allowed")
        ])

    def create_flexibility_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 3: Fare Flexibility Analysis Features"""
        logger.info("Creating Day 3: Fare Flexibility Analysis Features...")
        
        df = df.with_columns([
            pl.when(pl.col("miniRules0_monetaryAmount") == 0).then(pl.lit("Free"))
              .when(pl.col("miniRules0_monetaryAmount") > 0).then(pl.lit("Fee"))
              .otherwise(pl.lit("NonRefundable")).alias("cancellation_policy_tier"),
            
            pl.when(pl.col("miniRules1_monetaryAmount") == 0).then(pl.lit("FreeChange"))
              .when(pl.col("miniRules1_monetaryAmount") > 0).then(pl.lit("FeeChange"))
              .otherwise(pl.lit("NoChange")).alias("change_policy_category"),
              
            (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) > 0).cast(pl.Int8).alias("free_baggage")
        ])
        
        return FeatureUtils.label_encode_categorical(df, ["cancellation_policy_tier", "change_policy_category"])

    def create_temporal_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 4: Temporal Preferences – Business Hours Modeling"""
        logger.info("Creating Day 4: Temporal Preferences...")
        
        # Ensure datetime type
        df = df.with_columns([
            pl.col("legs0_departureAt").cast(pl.Datetime, strict=False).alias("_tmp_dep_dt")
        ])

        df = df.with_columns([
            pl.col("_tmp_dep_dt").dt.hour().alias("dep_hour"),
            pl.col("_tmp_dep_dt").dt.weekday().alias("dep_weekday")
        ])

        df = df.drop(["_tmp_dep_dt"])
        
        df = df.with_columns([
            ((pl.col("dep_hour") >= 6) & (pl.col("dep_hour") < 22)).cast(pl.Int8).alias("is_business_hours"),
            (((pl.col("dep_hour") >= 22) | (pl.col("dep_hour") < 6))).cast(pl.Int8).alias("is_redeye_flight"),
            (pl.col("dep_weekday").is_in([5, 6])).cast(pl.Int8).alias("is_weekend_flight"),
            (pl.col("dep_hour") // 6).alias("departure_hour_bin"),
            (2 * np.pi * pl.col("dep_hour") / 24).sin().alias("dep_hour_sin"),
            (2 * np.pi * pl.col("dep_hour") / 24).cos().alias("dep_hour_cos")
        ])
        return df
    
    def create_urgency_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 5: Booking Urgency & Lead Time Features"""
        logger.info("Creating Day 5: Booking Urgency Features...")
        
        df = df.with_columns([
            (
                pl.col("legs0_departureAt").cast(pl.Datetime, strict=False)
                - pl.col("requestDate").cast(pl.Datetime, strict=False)
            ).dt.total_days().alias("booking_lead_days")
        ])
        
        df = df.with_columns([
            pl.when(pl.col("booking_lead_days") <= 3).then(pl.lit("last_minute"))
              .when(pl.col("booking_lead_days") <= 14).then(pl.lit("short_term"))
              .otherwise(pl.lit("planned")).alias("booking_urgency_category")
        ])
        
        return FeatureUtils.label_encode_categorical(df, ["booking_urgency_category"])

    def create_route_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 6: Route and Connection Quality Features - Corrected Implementation"""
        logger.info("Creating Day 6: Route Quality Features (Corrected)...")

        # Correctly count non-null segments per row
        segment_dep_cols = [col for col in df.columns if "legs0_segments" in col and "departureFrom_airport_iata" in col]
        
        df = df.with_columns(
            sum(pl.col(c).is_not_null() for c in segment_dep_cols).alias("num_segments")
        )
        
        df = df.with_columns(
            (pl.col("num_segments") - 1).alias("connection_count"),
            (pl.col("num_segments") == 1).cast(pl.Int8).alias("is_direct_flight")
        )

        # Advanced layover calculations
        seg_arrival_cols = sorted([c for c in df.columns if "legs0_segments" in c and "arrivalAt" in c])
        seg_depart_cols = sorted([c for c in df.columns if "legs0_segments" in c and "departureAt" in c])

        layover_exprs = []
        if len(seg_arrival_cols) > 1 and len(seg_depart_cols) > 1:
            for i in range(len(seg_arrival_cols) - 1):
                arr_col = seg_arrival_cols[i]
                dep_col = seg_depart_cols[i+1]
                layover_exprs.append(
                    (
                        pl.col(dep_col).cast(pl.Datetime, strict=False)
                        -
                        pl.col(arr_col).cast(pl.Datetime, strict=False)
                    ).dt.total_minutes()
                )
        
        if layover_exprs:
            df = df.with_columns(
                pl.concat_list(layover_exprs).alias("layovers_list")
            )
            df = df.with_columns([
                pl.col("layovers_list").list.min().alias("shortest_layover_minutes"),
                pl.col("layovers_list").list.max().alias("longest_layover_minutes"),
                pl.col("layovers_list").list.sum().alias("total_layover_minutes"),
                (pl.col("layovers_list").list.max() > 8 * 60).cast(pl.Int8).alias("has_overnight_connection")
            ])
        else:
            df = df.with_columns([
                pl.lit(0).alias("shortest_layover_minutes"),
                pl.lit(0).alias("longest_layover_minutes"),
                pl.lit(0).alias("total_layover_minutes"),
                pl.lit(0).cast(pl.Int8).alias("has_overnight_connection")
            ])
            
        return df

    def create_duration_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Day 7: Flight Duration Features"""
        logger.info("Creating Day 7: Duration Features...")
        
        # Convert duration string to minutes
        duration_struct = (
            pl.col("legs0_duration")
            .str.extract_groups(r"(\d+):(\d+):(\d+)")
            .struct.rename_fields(["hours", "minutes", "seconds"])
        )
        df = df.with_columns(
            (
                duration_struct.struct.field("hours").cast(pl.Int32) * 60 +
                duration_struct.struct.field("minutes").cast(pl.Int32)
            ).alias("total_duration_minutes")
        )
        
        session_dur_stats = df.group_by("ranker_id").agg(pl.min("total_duration_minutes").alias("min_dur"))
        df = df.join(session_dur_stats, on="ranker_id")
        
        df = df.with_columns([
            (pl.col("total_duration_minutes") / pl.col("min_dur")).alias("duration_vs_min"),
            pl.col("total_duration_minutes").rank("dense").over("ranker_id").alias("duration_rank"),
            ((pl.col("total_duration_minutes").rank().over("ranker_id") - 1) /
             (pl.count().over("ranker_id") - 1)).alias("duration_percentile_in_session")
        ])
        
        return FeatureUtils.guard_against_infinities(df, ["duration_vs_min"])

    def create_advanced_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Advanced features for higher performance"""
        logger.info("Creating Advanced Features...")
        
        # Flight popularity features
        df = df.with_columns([
            # Session competition level
            pl.count().over("ranker_id").alias("session_options_count")
        ])
        
        # Add airline features if available
        if "legs0_segments0_operatingCarrier_code" in df.columns:
            df = df.with_columns([
                # Count how many times this exact flight appears across all sessions
                pl.col("legs0_segments0_operatingCarrier_code").count().over(["legs0_segments0_operatingCarrier_code", "legs0_segments0_departureFrom_airport_iata", "legs0_segments0_arrivalTo_airport_iata"]).alias("flight_popularity"),
                
                # Airline market share on this route
                pl.col("legs0_segments0_operatingCarrier_code").count().over(["legs0_segments0_operatingCarrier_code", "legs0_segments0_departureFrom_airport_iata", "legs0_segments0_arrivalTo_airport_iata"]).alias("airline_route_frequency")
            ])
        
        # Interaction features - these are critical for ranking
        df = df.with_columns([
            # Price-Policy interaction
            (pl.col("rel_price_vs_min") * pl.col("policy_compliant_flag")).alias("price_policy_interaction"),
            
            # Duration-Stops interaction  
            (pl.col("duration_vs_min") * pl.col("connection_count")).alias("duration_stops_interaction"),
            
            # Time-Policy interaction
            (pl.col("is_business_hours") * pl.col("policy_compliant_flag")).alias("time_policy_interaction"),
            
            # Price-Duration efficiency score
            ((1 / (1 + pl.col("rel_price_vs_min"))) * (1 / (1 + pl.col("duration_vs_min")))).alias("price_duration_efficiency"),
            
            # Convenience score (direct flights at good times)
            (pl.col("is_direct_flight") * pl.col("is_business_hours") * pl.col("policy_compliant_flag")).alias("convenience_score")
        ])
        
        # User/Company preference indicators
        user_cols = [col for col in df.columns if col in ["profileId", "companyID"]]
        if user_cols:
            user_col = user_cols[0]  # Use first available
            df = df.with_columns([
                # User's typical price range
                pl.col("totalPrice").mean().over(user_col).alias("user_avg_price"),
                pl.col("totalPrice").std().over(user_col).alias("user_price_std")
            ])
            
            df = df.with_columns([
                # Price deviation from user's norm
                ((pl.col("totalPrice") - pl.col("user_avg_price")) / pl.col("user_price_std")).alias("price_vs_user_norm")
            ])
            
            # Add airline affinity if airline column exists
            if "legs0_segments0_operatingCarrier_code" in df.columns:
                df = df.with_columns([
                    # How often this user/company picks this airline
                    pl.col("legs0_segments0_operatingCarrier_code").count().over([user_col, "legs0_segments0_operatingCarrier_code"]).alias("user_airline_affinity")
                ])
        
        # Route characteristics
        df = df.with_columns([
            # Route distance proxy (could be enhanced with actual distance)
            (pl.col("total_duration_minutes") / (1 + pl.col("connection_count"))).alias("route_efficiency"),
            
            # Peak travel indicator (Monday morning, Friday evening)
            ((pl.col("dep_weekday") == 0) & (pl.col("dep_hour") <= 10)).cast(pl.Int8).alias("monday_morning"),
            ((pl.col("dep_weekday") == 4) & (pl.col("dep_hour") >= 17)).cast(pl.Int8).alias("friday_evening"),
        ])
        
        return FeatureUtils.guard_against_infinities(df, ["price_vs_user_norm", "route_efficiency"])

    def create_competitive_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Competitive and ranking-based features"""
        logger.info("Creating Competitive Features...")
        
        # Multi-dimensional ranking features
        df = df.with_columns([
            # Combined efficiency rank (price + duration + stops)
            ((pl.col("rel_price_vs_min") + pl.col("duration_vs_min") + pl.col("connection_count")) / 3).alias("efficiency_composite"),
            
            # Is this the best option in multiple dimensions?
            (pl.col("price_rank_in_session") + pl.col("duration_rank") + pl.col("connection_count")).alias("multi_rank_sum"),
            
            # How many "wins" does this flight have vs others in session
            ((pl.col("price_rank_in_session") == 1).cast(pl.Int8) + 
             (pl.col("duration_rank") == 1).cast(pl.Int8) + 
             (pl.col("is_direct_flight") == 1).cast(pl.Int8) + 
             (pl.col("policy_compliant_flag") == 1).cast(pl.Int8)).alias("advantages_count"),
        ])
        
        # Rank the composite efficiency within session
        df = df.with_columns([
            pl.col("efficiency_composite").rank().over("ranker_id").alias("efficiency_rank"),
            pl.col("multi_rank_sum").rank().over("ranker_id").alias("overall_rank")
        ])
        
        # Session context features
        df = df.with_columns([
            # Percentage of flights in session that are direct
            (pl.col("is_direct_flight").sum().over("ranker_id") / pl.count().over("ranker_id")).alias("session_direct_ratio"),
            
            # Percentage of flights in session that are in policy
            (pl.col("policy_compliant_flag").sum().over("ranker_id") / pl.count().over("ranker_id")).alias("session_policy_ratio"),
            
            # Price range in session (max - min)
            (pl.col("totalPrice").max().over("ranker_id") - pl.col("totalPrice").min().over("ranker_id")).alias("session_price_range"),
            
            # This flight's advantage over median option
            (pl.col("totalPrice").median().over("ranker_id") - pl.col("totalPrice")).alias("price_vs_session_median"),
            (pl.col("total_duration_minutes").median().over("ranker_id") - pl.col("total_duration_minutes")).alias("duration_vs_session_median")
        ])
        
        # Binary competitive features
        df = df.with_columns([
            # Is this significantly better than alternatives?
            (pl.col("rel_price_vs_min") < 0.1).alias("much_cheaper"),  # <10% more than cheapest
            (pl.col("duration_vs_min") < 1.1).alias("similar_duration"),  # <10% longer than fastest
            (pl.col("price_vs_session_median") > 0).alias("cheaper_than_median"),
            (pl.col("duration_vs_session_median") < 0).alias("faster_than_median"),
            
            # Golden combination flags
            ((pl.col("is_direct_flight") == 1) & (pl.col("policy_compliant_flag") == 1) & (pl.col("is_business_hours") == 1)).alias("golden_option"),
            ((pl.col("price_rank_in_session") <= 2) & (pl.col("duration_rank") <= 2) & (pl.col("policy_compliant_flag") == 1)).alias("top_contender")
        ])
        
        return df

    def create_decision_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Business decision-making features"""
        logger.info("Creating Business Decision Features...")
        
        # Traveler preference modeling - these are critical
        df = df.with_columns([
            # Perfect option score (combines all key factors)
            (pl.col("policy_compliant_flag") * 3 +  # Policy is critical
             pl.col("is_direct_flight") * 2 +      # Direct flights highly valued
             pl.col("is_business_hours") * 1 +     # Good timing
             (pl.col("rel_price_vs_min") < 0.2).cast(pl.Int8) * 1 +  # Reasonable price
             (pl.col("duration_vs_min") < 1.2).cast(pl.Int8) * 1     # Reasonable duration
            ).alias("traveler_preference_score"),
            
            # Compromise assessment (when perfect option isn't available)
            pl.when(pl.col("policy_compliant_flag") == 0)
              .then(10)  # Heavy penalty for out of policy
              .otherwise(
                  pl.col("rel_price_vs_min") * 2 +  # Price sensitivity
                  pl.col("duration_vs_min") * 1 +   # Time sensitivity
                  pl.col("connection_count") * 1.5  # Connection penalty
              ).alias("compromise_penalty"),
            
            # Best available option flags
            (pl.col("policy_compliant_flag") & pl.col("is_direct_flight") & pl.col("is_business_hours")).alias("ideal_flight"),
            (pl.col("policy_compliant_flag") & pl.col("is_direct_flight")).alias("acceptable_flight"),
            (pl.col("policy_compliant_flag")).alias("policy_ok_flight"),
        ])
        
        # Session quality assessment
        df = df.with_columns([
            # How many good options are in this session?
            pl.col("ideal_flight").sum().over("ranker_id").alias("session_ideal_count"),
            pl.col("acceptable_flight").sum().over("ranker_id").alias("session_acceptable_count"),
            pl.col("policy_ok_flight").sum().over("ranker_id").alias("session_policy_count"),
            
            # Is this session problematic (few good options)?
            (pl.col("policy_ok_flight").sum().over("ranker_id") == 0).alias("session_no_policy_options"),
            (pl.col("acceptable_flight").sum().over("ranker_id") == 0).alias("session_no_direct_options"),
        ])
        
        # Relative standing within session
        df = df.with_columns([
            # Among policy-compliant options, how does this rank?
            pl.when(pl.col("policy_compliant_flag") == 1)
              .then(pl.col("totalPrice").rank().over(["ranker_id", "policy_compliant_flag"]))
              .otherwise(999).alias("price_rank_among_policy"),
            
            pl.when(pl.col("policy_compliant_flag") == 1)  
              .then(pl.col("total_duration_minutes").rank().over(["ranker_id", "policy_compliant_flag"]))
              .otherwise(999).alias("duration_rank_among_policy"),
            
            # Best-in-class indicators
            (pl.col("traveler_preference_score") == pl.col("traveler_preference_score").max().over("ranker_id")).alias("highest_preference_in_session"),
            (pl.col("compromise_penalty") == pl.col("compromise_penalty").min().over("ranker_id")).alias("lowest_compromise_in_session"),
        ])
        
        # Final decision factors
        df = df.with_columns([
            # If no perfect options exist, this might be the best compromise
            pl.when(pl.col("session_ideal_count") == 0)
              .then(pl.col("lowest_compromise_in_session").cast(pl.Int8))
              .otherwise(pl.col("highest_preference_in_session").cast(pl.Int8))
              .alias("likely_choice"),
            
            # Emergency booking indicators (when normal rules don't apply)
            ((pl.col("booking_lead_days") <= 1) & (pl.col("session_no_policy_options"))).alias("emergency_booking"),
            ((pl.col("session_policy_count") / pl.col("session_options_count")) < 0.3).alias("limited_policy_session")
        ])
        
        return df

    def engineer_features(self, df: pl.LazyFrame, is_train: bool = True) -> pl.LazyFrame:
        """
        Applies the full feature engineering pipeline.
        """
        logger.info("Starting comprehensive feature engineering pipeline...")
        
        # Pre-cast all date/time columns to avoid schema conflicts in lazy mode
        schema = df.schema
        datetime_cols = [c for c in schema if "At" in c or "Date" in c]
        for col in datetime_cols:
            if schema[col] == pl.Utf8:
                 df = df.with_columns(pl.col(col).str.to_datetime().alias(col))

        # Retain baseline features if explicitly requested
        if self.baseline_features:
            existing_cols = df.columns
            baseline_keep = [c for c in self.baseline_features if c in existing_cols]
            remaining = [c for c in existing_cols if c not in baseline_keep]
            df = df.select(baseline_keep + remaining)

        df = self.create_price_features(df)
        df = self.create_policy_features(df)
        df = self.create_flexibility_features(df)
        df = self.create_temporal_features(df)
        df = self.create_urgency_features(df)
        df = self.create_route_features(df)
        df = self.create_duration_features(df)
        df = self.create_advanced_features(df)
        df = self.create_competitive_features(df)
        df = self.create_decision_features(df)
        
        # High-cardinality frequency encoding (airline/route) after all numeric features
        high_card_cols = [c for c in ["legs0_segments0_operatingCarrier_code", "searchRoute", "legs0_segments0_departureFrom_airport_iata", "legs0_segments0_arrivalTo_airport_iata"] if c in df.columns]
        for hc in high_card_cols:
            df = FeatureUtils.encode_categorical(df, hc, method="frequency")

        # Final check for group integrity
        df = df.sort("ranker_id")

        logger.info("Feature engineering complete!")
        return df

    def get_feature_names(self, df: pl.DataFrame) -> list[str]:
        """
        Get the list of engineered feature names, excluding identifiers and the target.
        """
        exclude_cols = ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute',
                        'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
                        'cancellation_policy_tier', 'change_policy_category', 'booking_urgency_category',
                        'layovers_list', 'user_avg_price', 'user_price_std'] # Exclude list-type and intermediate columns
        
        # Also exclude original categorical columns that have been label encoded
        encoded_cols = [col for col in df.columns if col.endswith("_le")]
        original_cats = [col.replace("_le", "") for col in encoded_cols]
        exclude_cols.extend(original_cats)

        # Exclude datetime columns explicitly (dtype equality for Polars dtypes)
        datetime_cols = [col for col, dtype in df.schema.items() if dtype == pl.Datetime]
        exclude_cols.extend(datetime_cols)

        features = [col for col in df.columns if col not in exclude_cols and df[col].dtype != pl.Utf8]
        
        # Ensure baseline features are included if they exist
        if self.baseline_features:
            for f in self.baseline_features:
                if f not in features and f not in exclude_cols:
                    features.append(f)
                    
        # Add freq-encoded cols to feature list if present
        freq_cols = [f"{c}_freq" for c in ["legs0_segments0_operatingCarrier_code", "searchRoute", "legs0_segments0_departureFrom_airport_iata", "legs0_segments0_arrivalTo_airport_iata"] if f"{c}_freq" in df.columns]
        features.extend(freq_cols)
        
        return list(dict.fromkeys(features)) # Remove duplicates while preserving order