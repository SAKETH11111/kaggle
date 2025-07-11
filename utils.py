"""
FlightRank 2025 - Utility Functions
Common utilities for data processing and validation
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import StratifiedGroupKFold
import json
import pickle
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_submission_format(df: pl.DataFrame, test_df: pl.DataFrame) -> Dict[str, bool]:
        """Validate submission format requirements"""
        validations = {}
        
        # Check if same number of rows
        validations["correct_row_count"] = df.shape[0] == test_df.shape[0]
        
        # Check if all required columns present
        required_cols = ["Id", "ranker_id", "selected"]
        validations["has_required_columns"] = all(col in df.columns for col in required_cols)
        
        # Check if IDs match
        if "Id" in df.columns and "Id" in test_df.columns:
            validations["ids_match"] = df.select("Id").equals(test_df.select("Id"))
        else:
            validations["ids_match"] = False
        
        # Check rank validity within groups
        try:
            rank_check = (
                df.group_by("ranker_id")
                .agg([
                    pl.col("selected").min().alias("min_rank"),
                    pl.col("selected").max().alias("max_rank"),
                    pl.col("selected").n_unique().alias("unique_ranks"),
                    pl.count().alias("group_size")
                ])
                .with_columns([
                    (pl.col("min_rank") == 1).alias("starts_with_1"),
                    (pl.col("max_rank") == pl.col("group_size")).alias("ends_with_group_size"),
                    (pl.col("unique_ranks") == pl.col("group_size")).alias("all_ranks_unique")
                ])
            )
            
            rank_summary = rank_check.select([
                pl.col("starts_with_1").all(),
                pl.col("ends_with_group_size").all(),
                pl.col("all_ranks_unique").all()
            ]).to_dict(as_series=False)
            
            validations["valid_rank_permutations"] = all(rank_summary.values())
        except:
            validations["valid_rank_permutations"] = False
        
        return validations
    
    @staticmethod
    def calculate_hitrate_at_k(predictions: pl.DataFrame, k: int = 3, min_group_size: int = 11) -> float:
        """
        Calculate HitRate@k metric based on predicted scores.

        Args:
            predictions (pl.DataFrame): DataFrame with ranker_id, predicted_score, and selected columns.
            k (int): The 'k' in HitRate@k.
            min_group_size (int): The minimum number of items in a group to be included in the evaluation.
                                 Per competition rules, we exclude groups with 10 or fewer items,
                                 so the default is 11 (groups > 10).

        Returns:
            float: The HitRate@k score.
        """
        try:
            # Add group size and rank based on predicted score
            ranked_preds = predictions.with_columns([
                pl.col("ranker_id").count().over("ranker_id").alias("group_size"),
                pl.col("predicted_score").rank(method="ordinal", descending=True).over("ranker_id").alias("predicted_rank")
            ])

            # Filter for groups large enough to be evaluated
            valid_groups = ranked_preds.filter(pl.col("group_size") >= min_group_size)

            if valid_groups.height == 0:
                logger.warning("No groups met the minimum size requirement for HitRate calculation.")
                return 0.0

            # Find the rank of the truly selected item in each valid group
            hits = valid_groups.filter(
                (pl.col("selected") == 1) & (pl.col("predicted_rank") <= k)
            )

            # Total number of unique groups that were evaluated
            total_evaluated_groups = valid_groups.select("ranker_id").n_unique()

            if total_evaluated_groups == 0:
                return 0.0

            return hits.height / total_evaluated_groups

        except Exception as e:
            logger.error(f"Error calculating HitRate@{k}: {e}")
            return 0.0

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def memory_usage_mb(df: Union[pl.DataFrame, pl.LazyFrame]) -> float:
        """Calculate memory usage in MB"""
        if isinstance(df, pl.LazyFrame):
            # Estimate based on schema
            return 0.0  # Cannot estimate lazy frame memory
        return df.estimated_size("mb")
    
    @staticmethod
    def optimize_datatypes(df: pl.LazyFrame) -> pl.LazyFrame:
        """Optimize data types to reduce memory usage"""
        optimizations = []
        
        for col, dtype in df.schema.items():
            if dtype == pl.Int64:
                # Try to downcast integers
                optimizations.append(pl.col(col).cast(pl.Int32).alias(col))
            elif dtype == pl.Float64:
                # Try to downcast floats
                optimizations.append(pl.col(col).cast(pl.Float32).alias(col))
            else:
                optimizations.append(pl.col(col))
        
        return df.with_columns(optimizations)
    
    @staticmethod
    def split_by_groups(df: pl.LazyFrame, group_col: str = "ranker_id", 
                       train_ratio: float = 0.8, seed: int = 42) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Split data by groups while maintaining group integrity"""
        # Get unique groups
        unique_groups = df.select(group_col).unique().collect()
        n_groups = unique_groups.shape[0]
        n_train_groups = int(n_groups * train_ratio)
        
        # Randomly sample groups for training
        np.random.seed(seed)
        train_groups = unique_groups.sample(n_train_groups)
        
        # Split data
        train_df = df.join(train_groups, on=group_col, how="inner")
        val_df = df.join(train_groups, on=group_col, how="anti")
        
        return train_df, val_df
    
    @staticmethod
    def create_submission_template(test_df: pl.DataFrame) -> pl.DataFrame:
        """Create submission template with proper format"""
        return test_df.select([
            "Id",
            "ranker_id",
            pl.lit(1).alias("selected")  # Default rank of 1 for all
        ])

class FeatureUtils:
    """Feature engineering utilities"""
    
    @staticmethod
    def encode_categorical(df: pl.LazyFrame, col: str, method: str = "ordinal") -> pl.LazyFrame:
        """Encode categorical variables"""
        if method == "ordinal":
            # Simple ordinal encoding
            unique_values = df.select(col).unique().collect().to_series().to_list()
            mapping = {val: i for i, val in enumerate(unique_values) if val is not None}
            
            encoded_col = pl.col(col).replace(mapping).alias(f"{col}_encoded")
            return df.with_columns(encoded_col)
        
        elif method == "frequency":
            # Frequency encoding
            freq_map = (
                df.group_by(col)
                .len()
                .collect()
                .to_dict(as_series=False)
            )
            freq_dict = dict(zip(freq_map[col], freq_map["len"]))
            
            encoded_col = pl.col(col).replace(freq_dict).alias(f"{col}_freq")
            return df.with_columns(encoded_col)
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    @staticmethod
    def create_time_features(df: pl.LazyFrame, datetime_col: str) -> pl.LazyFrame:
        """Extract time-based features from datetime column"""
        time_features = [
            pl.col(datetime_col).dt.hour().alias(f"{datetime_col}_hour"),
            pl.col(datetime_col).dt.weekday().alias(f"{datetime_col}_weekday"),
            pl.col(datetime_col).dt.month().alias(f"{datetime_col}_month"),
            pl.col(datetime_col).dt.day().alias(f"{datetime_col}_day")
        ]
        
        return df.with_columns(time_features)
    
    @staticmethod
    def create_ranking_features(df: pl.LazyFrame, value_col: str, group_col: str = "ranker_id") -> pl.LazyFrame:
        """Create ranking-based features within groups"""
        ranking_features = [
            pl.col(value_col).rank().over(group_col).alias(f"{value_col}_rank"),
            pl.col(value_col).rank(method="dense").over(group_col).alias(f"{value_col}_dense_rank"),
            ((pl.col(value_col).rank().over(group_col) - 1) / 
             (pl.col(value_col).count().over(group_col) - 1)).alias(f"{value_col}_percentile")
        ]
        
        return df.with_columns(ranking_features)

class IOUtils:
    """Input/Output utilities"""
    
    @staticmethod
    def save_processed_data(df: pl.DataFrame, filename: str, processed_dir: Path = Path("processed")):
        """Save processed data in efficient format"""
        processed_dir.mkdir(exist_ok=True)
        filepath = processed_dir / filename
        
        if filename.endswith('.parquet'):
            df.write_parquet(filepath)
        elif filename.endswith('.csv'):
            df.write_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Saved {df.shape[0]} rows to {filepath}")
    
    @staticmethod
    def load_processed_data(filename: str, processed_dir: Path = Path("processed")) -> pl.DataFrame:
        """Load processed data"""
        filepath = processed_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filename.endswith('.parquet'):
            return pl.read_parquet(filepath)
        elif filename.endswith('.csv'):
            return pl.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    @staticmethod
    def save_model_artifacts(artifacts: Dict, filename: str, processed_dir: Path = Path("processed")):
        """Save model artifacts"""
        processed_dir.mkdir(exist_ok=True)
        filepath = processed_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Saved model artifacts to {filepath}")
    
    @staticmethod
    def load_model_artifacts(filename: str, processed_dir: Path = Path("processed")) -> Dict:
        """Load model artifacts"""
        filepath = processed_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class CrossValidationUtils:
    """Cross-validation utilities for ranking problems"""
    
    @staticmethod
    def group_kfold_split(df: pl.LazyFrame, group_col: str = "ranker_id", 
                         n_splits: int = 5, seed: int = 42) -> List[Tuple[pl.LazyFrame, pl.LazyFrame]]:
        """Create group-based K-fold splits"""
        # Get unique groups
        unique_groups = df.select(group_col).unique().collect()
        n_groups = unique_groups.shape[0]
        
        # Shuffle groups
        np.random.seed(seed)
        shuffled_groups = unique_groups.sample(n_groups, seed=seed)
        
        # Create splits
        splits = []
        fold_size = n_groups // n_splits
        
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else n_groups
            
            val_groups = shuffled_groups.slice(start_idx, end_idx - start_idx)
            train_groups = shuffled_groups.join(val_groups, on=group_col, how="anti")
            
            train_df = df.join(train_groups, on=group_col, how="inner")
            val_df = df.join(val_groups, on=group_col, how="inner")
            
            splits.append((train_df, val_df))
        
        return splits

    @staticmethod
    def temporal_split(df: pl.LazyFrame, date_col: str, train_ratio: float = 0.8) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Create temporal split based on date column"""
        # Sort by date and split
        sorted_df = df.sort(date_col)
        n_rows = df.select(pl.count()).collect().item()
        split_point = int(n_rows * train_ratio)
        
        train_df = sorted_df.limit(split_point)
        val_df = sorted_df.slice(split_point, n_rows - split_point)
        
        return train_df, val_df

class PerformanceUtils:
    """Performance monitoring utilities"""
    
    @staticmethod
    def profile_memory_usage(func):
        """Decorator to profile memory usage"""
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            logger.info(f"Memory usage for {func.__name__}: {mem_diff:.2f} MB")
            
            return result
        return wrapper
    
    @staticmethod
    def time_execution(func):
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Execution time for {func.__name__}: {duration:.2f} seconds")
            
            return result
        return wrapper

if __name__ == "__main__":
    # Test utilities
    logger.info("Testing utility functions...")
    
    # Test validation utils
    sample_data = pl.DataFrame({
        "Id": [1, 2, 3, 4],
        "ranker_id": ["A", "A", "B", "B"],
        "selected": [2, 1, 1, 2]
    })
    
    test_data = pl.DataFrame({
        "Id": [1, 2, 3, 4],
        "ranker_id": ["A", "A", "B", "B"]
    })
    
    validations = ValidationUtils.validate_submission_format(sample_data, test_data)
    logger.info(f"Validation results: {validations}")
    
    # Test HitRate calculation
    hitrate = ValidationUtils.calculate_hitrate_at_k(sample_data, k=3, min_group_size=1)
    logger.info(f"HitRate@3: {hitrate}")
    
    logger.info("Utility tests complete!") 