import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np

from feature_engineering import FeatureEngineer
from utils import FeatureUtils

@pytest.fixture
def sample_lazy_frame() -> pl.LazyFrame:
    """Provides a sample LazyFrame for testing."""
    df = pl.DataFrame({
        "ranker_id": ["a", "a", "b", "b", "b"],
        "totalPrice": [100.0, 200.0, 150.0, 50.0, 300.0],
        "pricingInfo_isAccessTP": [True, False, True, True, False],
        "legs0_segments0_cabinClass": [1.0, 2.0, 1.0, 4.0, 1.0],
        "miniRules0_monetaryAmount": [0.0, 50.0, None, 0.0, 100.0],
        "miniRules1_monetaryAmount": [None, 0.0, 25.0, 0.0, 50.0],
        "legs0_segments0_baggageAllowance_quantity": [0, 1, 1, None, 0],
        "legs0_departureAt": ["2025-10-01T10:00:00", "2025-10-01T23:00:00", "2025-10-02T08:00:00", "2025-10-02T14:00:00", "2025-10-02T04:00:00"],
        "requestDate": ["2025-09-20T12:00:00"] * 5,
        "legs0_duration": ["01:30:00", "02:00:00", "05:00:00", "01:00:00", "03:30:00"],
        "legs0_segments0_departureFrom_airport_iata": ["JFK", "JFK", "LHR", "LHR", "LHR"],
        "legs0_segments1_departureAt": [None, None, "2025-10-02T15:00:00", None, None],
        "legs0_segments0_arrivalAt": [None, None, "2025-10-02T13:00:00", None, None],
    })
    return df.lazy()

def test_create_price_features(sample_lazy_frame: pl.LazyFrame):
    """Tests the creation of price-related features."""
    engineer = FeatureEngineer()
    result_lf = engineer.create_price_features(sample_lazy_frame)
    result_df = result_lf.collect()

    assert "rel_price_vs_min" in result_df.columns
    assert "price_rank_in_session" in result_df.columns
    assert "price_z_score" in result_df.columns
    # Check for infinities
    assert result_df.select(
        pl.col("rel_price_vs_min").is_infinite().sum() + pl.col("price_z_score").is_infinite().sum()
    ).item() == 0

def test_create_policy_features(sample_lazy_frame: pl.LazyFrame):
    """Tests the creation of policy compliance features."""
    engineer = FeatureEngineer()
    result_lf = engineer.create_policy_features(sample_lazy_frame)
    result_df = result_lf.collect()

    assert "policy_compliant_flag" in result_df.columns
    assert "cabin_allowed" in result_df.columns
    assert result_df["policy_compliant_flag"].dtype == pl.Int8
    assert result_df["cabin_allowed"].dtype == pl.Int8

def test_create_flexibility_features(sample_lazy_frame: pl.LazyFrame):
    """Tests the creation of fare flexibility features."""
    engineer = FeatureEngineer()
    result_lf = engineer.create_flexibility_features(sample_lazy_frame)
    result_df = result_lf.collect()

    assert "cancellation_policy_tier_le" in result_df.columns
    assert "change_policy_category_le" in result_df.columns
    assert "free_baggage" in result_df.columns
    assert result_df["cancellation_policy_tier_le"].dtype == pl.Int16
    assert result_df["free_baggage"].dtype == pl.Int8

def test_create_temporal_features(sample_lazy_frame: pl.LazyFrame):
    """Tests the creation of temporal features."""
    engineer = FeatureEngineer()
    result_lf = engineer.create_temporal_features(sample_lazy_frame)
    result_df = result_lf.collect()

    assert "dep_hour" in result_df.columns
    assert "is_redeye_flight" in result_df.columns
    assert "dep_hour_sin" in result_df.columns
    assert result_df["is_redeye_flight"].dtype == pl.Int8

def test_create_urgency_features(sample_lazy_frame: pl.LazyFrame):
    """Tests the creation of booking urgency features."""
    engineer = FeatureEngineer()
    result_lf = engineer.create_urgency_features(sample_lazy_frame)
    result_df = result_lf.collect()

    assert "booking_lead_days" in result_df.columns
    assert "booking_urgency_category_le" in result_df.columns
    assert result_df["booking_urgency_category_le"].dtype == pl.Int16

def test_create_route_features():
    """Tests the corrected creation of route and connection features with a dedicated dataframe."""
    engineer = FeatureEngineer()
    
    # Create a specific dataframe for this test to ensure correctness
    df = pl.DataFrame({
        "ranker_id": ["a", "a", "a"],
        "totalPrice": [100.0, 200.0, 300.0],
        # Segment 0 (all flights have this)
        "legs0_segments0_departureFrom_airport_iata": ["JFK", "JFK", "JFK"],
        "legs0_segments0_arrivalAt": ["2025-10-01T12:00:00", "2025-10-01T12:00:00", "2025-10-01T12:00:00"],
        "legs0_segments0_departureAt": ["2025-10-01T10:00:00", "2025-10-01T10:00:00", "2025-10-01T10:00:00"],
        # Segment 1
        "legs0_segments1_departureFrom_airport_iata": [None, "SFO", "SFO"],
        "legs0_segments1_arrivalAt": [None, "2025-10-01T16:00:00", "2025-10-01T16:00:00"],
        "legs0_segments1_departureAt": [None, "2025-10-01T14:00:00", "2025-10-01T14:00:00"],
        # Segment 2
        "legs0_segments2_departureFrom_airport_iata": [None, None, "EWR"],
        "legs0_segments2_arrivalAt": [None, None, "2025-10-01T20:00:00"],
        "legs0_segments2_departureAt": [None, None, "2025-10-01T18:00:00"],
    }).lazy()

    result_df = engineer.create_route_features(df).collect()

    # Assert correctness for direct flight (row 0, price 100)
    direct_flight = result_df.row(0, named=True)
    assert direct_flight["num_segments"] == 1
    assert direct_flight["connection_count"] == 0
    assert direct_flight["is_direct_flight"] == 1
    assert direct_flight["total_layover_minutes"] == 0

    # Assert correctness for 1-stop flight (row 1, price 200)
    one_stop_flight = result_df.row(1, named=True)
    assert one_stop_flight["num_segments"] == 2
    assert one_stop_flight["connection_count"] == 1
    assert one_stop_flight["is_direct_flight"] == 0
    assert one_stop_flight["total_layover_minutes"] == 120

    # Assert correctness for 2-stop flight (row 2, price 300)
    two_stop_flight = result_df.row(2, named=True)
    assert two_stop_flight["num_segments"] == 3
    assert two_stop_flight["connection_count"] == 2
    assert two_stop_flight["is_direct_flight"] == 0
    assert two_stop_flight["total_layover_minutes"] == 240
    assert two_stop_flight["shortest_layover_minutes"] == 120
    assert two_stop_flight["longest_layover_minutes"] == 120

def test_create_duration_features(sample_lazy_frame: pl.LazyFrame):
    """Tests the creation of flight duration features."""
    engineer = FeatureEngineer()
    result_lf = engineer.create_duration_features(sample_lazy_frame)
    result_df = result_lf.collect()

    assert "total_duration_minutes" in result_df.columns
    assert "duration_vs_min" in result_df.columns
    assert "duration_rank" in result_df.columns
    assert result_df.select(pl.col("duration_vs_min").is_infinite()).sum().item() == 0

def test_full_pipeline(sample_lazy_frame: pl.LazyFrame):
    """Tests the full feature engineering pipeline."""
    engineer = FeatureEngineer()
    result_lf = engineer.engineer_features(sample_lazy_frame)
    result_df = result_lf.collect()

    # Check a few key features from different steps
    assert "price_z_score" in result_df.columns
    assert "policy_compliant_flag" in result_df.columns
    assert "cancellation_policy_tier_le" in result_df.columns
    assert "is_redeye_flight" in result_df.columns
    assert "booking_lead_days" in result_df.columns
    assert "is_direct_flight" in result_df.columns
    assert "duration_vs_min" in result_df.columns
    
    # Ensure no obvious errors like all-null columns for key features
    for col in ["price_z_score", "booking_lead_days", "duration_vs_min"]:
        assert result_df[col].is_not_null().all()