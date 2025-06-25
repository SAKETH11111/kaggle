"""
Evaluation Metrics for FlightRank 2025 RecSys Cup
Precise HitRate@3 implementation following competition rules
"""

import polars as pl
import numpy as np
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)

class RankingMetrics:
    """
    Evaluation metrics for ranking problems, specifically optimized for the
    FlightRank 2025 RecSys Cup competition rules.
    """
    
    @staticmethod
    def hitrate_at_k(
        predictions: pl.DataFrame, 
        k: int = 3, 
        min_group_size: int = 11,
        group_col: str = "ranker_id",
        score_col: str = "predicted_score",
        target_col: str = "selected"
    ) -> float:
        """
        Calculate HitRate@k metric following exact competition rules.
        
        This is the primary evaluation metric for FlightRank 2025.
        Only groups with MORE than 10 options are evaluated (min_group_size=11).
        
        Args:
            predictions: DataFrame with columns [group_col, score_col, target_col]
            k: The 'k' in HitRate@k (default 3 for competition)
            min_group_size: Minimum group size to include in evaluation (default 11)
            group_col: Column name for grouping (default "ranker_id")  
            score_col: Column name for predicted scores (default "predicted_score")
            target_col: Column name for true target (default "selected")
            
        Returns:
            float: HitRate@k score between 0 and 1
            
        Example:
            >>> predictions = pl.DataFrame({
            ...     "ranker_id": ["A", "A", "A", "B", "B", "B"],
            ...     "predicted_score": [0.9, 0.7, 0.3, 0.8, 0.6, 0.4],
            ...     "selected": [0, 1, 0, 1, 0, 0]
            ... })
            >>> RankingMetrics.hitrate_at_k(predictions, k=3, min_group_size=1)
            1.0
        """
        try:
            # Validate inputs
            required_cols = [group_col, score_col, target_col]
            missing_cols = [col for col in required_cols if col not in predictions.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add group size and predicted rank based on scores (higher score = better rank)
            ranked_predictions = predictions.with_columns([
                pl.col(group_col).count().over(group_col).alias("group_size"),
                pl.col(score_col).rank(method="ordinal", descending=True).over(group_col).alias("predicted_rank")
            ])
            
            # Filter for groups that meet minimum size requirement
            valid_groups = ranked_predictions.filter(pl.col("group_size") >= min_group_size)
            
            if valid_groups.height == 0:
                logger.warning(f"No groups met minimum size requirement ({min_group_size})")
                return 0.0
            
            # Count hits: selected items (target_col=1) ranked in top-k by prediction
            hits = valid_groups.filter(
                (pl.col(target_col) == 1) & (pl.col("predicted_rank") <= k)
            )
            
            # Count total number of evaluated groups
            total_groups = valid_groups.select(group_col).n_unique()
            
            if total_groups == 0:
                return 0.0
                
            hitrate = hits.height / total_groups
            
            logger.debug(f"HitRate@{k}: {hits.height}/{total_groups} = {hitrate:.4f}")
            return hitrate
            
        except Exception as e:
            logger.error(f"Error calculating HitRate@{k}: {e}")
            return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        predictions: pl.DataFrame,
        min_group_size: int = 11,
        group_col: str = "ranker_id", 
        score_col: str = "predicted_score",
        target_col: str = "selected"
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) metric.
        
        Args:
            predictions: DataFrame with predictions and targets
            min_group_size: Minimum group size to include
            group_col: Column name for grouping
            score_col: Column name for predicted scores  
            target_col: Column name for true target
            
        Returns:
            float: MRR score between 0 and 1
        """
        try:
            # Add rankings
            ranked_predictions = predictions.with_columns([
                pl.col(group_col).count().over(group_col).alias("group_size"),
                pl.col(score_col).rank(method="ordinal", descending=True).over(group_col).alias("predicted_rank")
            ])
            
            # Filter valid groups
            valid_groups = ranked_predictions.filter(pl.col("group_size") >= min_group_size)
            
            if valid_groups.height == 0:
                return 0.0
            
            # Calculate reciprocal ranks for selected items
            selected_items = valid_groups.filter(pl.col(target_col) == 1)
            reciprocal_ranks = selected_items.with_columns(
                (1.0 / pl.col("predicted_rank")).alias("reciprocal_rank")
            )
            
            # Calculate MRR
            total_groups = valid_groups.select(group_col).n_unique()
            if total_groups == 0:
                return 0.0
                
            mrr = reciprocal_ranks.select("reciprocal_rank").sum().item() / total_groups
            return mrr
            
        except Exception as e:
            logger.error(f"Error calculating MRR: {e}")
            return 0.0
    
    @staticmethod
    def ndcg_at_k(
        predictions: pl.DataFrame,
        k: int = 3,
        min_group_size: int = 11,
        group_col: str = "ranker_id",
        score_col: str = "predicted_score", 
        target_col: str = "selected"
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            predictions: DataFrame with predictions and targets
            k: The 'k' in NDCG@k
            min_group_size: Minimum group size to include
            group_col: Column name for grouping
            score_col: Column name for predicted scores
            target_col: Column name for true target
            
        Returns:
            float: NDCG@k score between 0 and 1
        """
        try:
            # Add rankings
            ranked_predictions = predictions.with_columns([
                pl.col(group_col).count().over(group_col).alias("group_size"),
                pl.col(score_col).rank(method="ordinal", descending=True).over(group_col).alias("predicted_rank")
            ])
            
            # Filter valid groups and top-k predictions
            valid_groups = ranked_predictions.filter(
                (pl.col("group_size") >= min_group_size) & 
                (pl.col("predicted_rank") <= k)
            )
            
            if valid_groups.height == 0:
                return 0.0
            
            # Calculate DCG: sum of (2^relevance - 1) / log2(rank + 1)
            dcg_df = valid_groups.with_columns([
                ((2 ** pl.col(target_col) - 1) / (pl.col("predicted_rank") + 1).log()).alias("dcg_contrib")
            ])
            
            dcg_by_group = dcg_df.group_by(group_col).agg(
                pl.col("dcg_contrib").sum().alias("dcg")
            )
            
            # Calculate IDCG (ideal DCG) - sort by true relevance
            ideal_ranked = ranked_predictions.filter(pl.col("group_size") >= min_group_size).with_columns(
                pl.col(target_col).rank(method="ordinal", descending=True).over(group_col).alias("ideal_rank")
            ).filter(pl.col("ideal_rank") <= k)
            
            idcg_df = ideal_ranked.with_columns([
                ((2 ** pl.col(target_col) - 1) / (pl.col("ideal_rank") + 1).log()).alias("idcg_contrib")
            ])
            
            idcg_by_group = idcg_df.group_by(group_col).agg(
                pl.col("idcg_contrib").sum().alias("idcg")
            )
            
            # Calculate NDCG = DCG / IDCG
            ndcg_df = dcg_by_group.join(idcg_by_group, on=group_col).with_columns(
                (pl.col("dcg") / pl.col("idcg")).alias("ndcg")
            )
            
            # Return mean NDCG across all groups
            mean_ndcg = ndcg_df.select("ndcg").mean().item()
            return mean_ndcg if mean_ndcg is not None else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating NDCG@{k}: {e}")
            return 0.0
    
    @staticmethod
    def ranking_summary(
        predictions: pl.DataFrame,
        k_values: list = [1, 3, 5, 10],
        min_group_size: int = 11,
        group_col: str = "ranker_id",
        score_col: str = "predicted_score",
        target_col: str = "selected"
    ) -> dict:
        """
        Calculate comprehensive ranking metrics summary.
        
        Args:
            predictions: DataFrame with predictions and targets
            k_values: List of k values to evaluate
            min_group_size: Minimum group size to include
            group_col: Column name for grouping
            score_col: Column name for predicted scores
            target_col: Column name for true target
            
        Returns:
            dict: Summary of all ranking metrics
        """
        summary = {
            "total_groups": predictions.select(group_col).n_unique(),
            "total_items": predictions.height,
            "min_group_size": min_group_size
        }
        
        # Add group size filter stats
        with_group_size = predictions.with_columns(
            pl.col(group_col).count().over(group_col).alias("group_size")
        )
        
        valid_groups_df = with_group_size.filter(pl.col("group_size") >= min_group_size)
        summary["evaluated_groups"] = valid_groups_df.select(group_col).n_unique()
        summary["evaluated_items"] = valid_groups_df.height
        
        # Calculate metrics for each k
        for k in k_values:
            summary[f"hitrate_at_{k}"] = RankingMetrics.hitrate_at_k(
                predictions, k, min_group_size, group_col, score_col, target_col
            )
            summary[f"ndcg_at_{k}"] = RankingMetrics.ndcg_at_k(
                predictions, k, min_group_size, group_col, score_col, target_col
            )
        
        # Add MRR
        summary["mrr"] = RankingMetrics.mean_reciprocal_rank(
            predictions, min_group_size, group_col, score_col, target_col
        )
        
        return summary


# Compatibility aliases for existing code
def calculate_hitrate_at_k(predictions: pl.DataFrame, k: int = 3, min_group_size: int = 11) -> float:
    """Compatibility function for existing code."""
    return RankingMetrics.hitrate_at_k(predictions, k, min_group_size)


if __name__ == "__main__":
    # Test the metrics implementation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_data = pl.DataFrame({
        "ranker_id": ["A"] * 15 + ["B"] * 12 + ["C"] * 8,  # Group C will be excluded (< 11)
        "predicted_score": [
            # Group A: true item ranked 2nd
            [0.9, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05],
            # Group B: true item ranked 1st  
            [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08],
            # Group C: true item ranked 1st (excluded from evaluation)
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        ][0] + [0.9, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05] + 
              [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08] +
              [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "selected": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] +  # Group A: 2nd highest score
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] +              # Group B: highest score  
                   [1, 0, 0, 0, 0, 0, 0, 0]                            # Group C: highest score (excluded)
    })
    
    print("Test data shape:", test_data.shape)
    print("Groups and sizes:")
    print(test_data.group_by("ranker_id").agg(pl.count()))
    
    # Test HitRate@3
    hitrate_3 = RankingMetrics.hitrate_at_k(test_data, k=3)
    print(f"\nHitRate@3: {hitrate_3:.4f}")
    print("Expected: 1.0 (both Group A and B have selected item in top 3)")
    
    # Test comprehensive summary
    summary = RankingMetrics.ranking_summary(test_data)
    print(f"\nMetrics Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")