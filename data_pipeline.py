"""
FlightRank 2025 - Data Pipeline Architecture
Core Philosophy: Build once, iterate fast. Memory-efficient, scalable data processing.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tarfile
import gzip
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp
from functools import partial
import logging
from dataclasses import dataclass
from feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data processing pipeline"""
    data_dir: Path = Path("data")
    processed_dir: Path = Path("processed")
    chunk_size: int = 10000
    n_workers: int = 8
    memory_limit_gb: float = 8.0
    use_lazy_loading: bool = True
    extract_json_features: bool = True

class MemoryEfficientLoader:
    """Memory-efficient data loading with lazy evaluation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.config.processed_dir.mkdir(exist_ok=True)
        
    def load_structured_data(self, dataset: str = "train", columns: Optional[List[str]] = None) -> pl.LazyFrame:
        """Load structured parquet data with lazy evaluation"""
        file_path = self.config.data_dir / f"{dataset}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset {dataset} not found at {file_path}")
        
        logger.info(f"Loading {dataset} data from {file_path}")
        
        if columns:
            lf = pl.scan_parquet(file_path).select(columns)
        else:
            lf = pl.scan_parquet(file_path)
            
        return lf
    
    def get_schema_info(self, dataset: str = "train") -> Dict:
        """Get schema information without loading data"""
        file_path = self.config.data_dir / f"{dataset}.parquet"
        schema = pl.read_parquet_schema(file_path)
        
        return {
            "total_columns": len(schema),
            "schema": dict(schema),
            "column_types": {col: str(dtype) for col, dtype in schema.items()}
        }
    
    def load_chunked(self, dataset: str = "train", chunk_size: Optional[int] = None) -> pl.LazyFrame:
        """Load data in chunks for memory-efficient processing"""
        chunk_size = chunk_size or self.config.chunk_size
        lf = self.load_structured_data(dataset)
        return lf.with_row_count().filter(pl.col("row_nr") < chunk_size)

class DataQualityValidator:
    """Data quality validation and integrity checks"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def validate_training_data(self, train_lf: pl.LazyFrame) -> Dict[str, any]:
        """Validate training data integrity"""
        logger.info("Validating training data integrity...")
        
        # Collect essential validations
        validations = {}
        
        # Check selected column integrity
        selected_check = (
            train_lf
            .group_by("ranker_id")
            .agg(pl.col("selected").sum().alias("selected_sum"))
            .collect()
        )
        
        invalid_groups = selected_check.filter(pl.col("selected_sum") != 1)
        validations["invalid_groups_count"] = invalid_groups.shape[0]
        validations["total_groups"] = selected_check.shape[0]
        
        # Group size analysis
        group_sizes = (
            train_lf
            .group_by("ranker_id")
            .len()
            .collect()
        )
        
        validations["group_size_stats"] = {
            "min": group_sizes.select(pl.col("len").min()).item(),
            "max": group_sizes.select(pl.col("len").max()).item(),
            "median": group_sizes.select(pl.col("len").median()).item(),
            "groups_gt_10": group_sizes.filter(pl.col("len") > 10).shape[0],
            "groups_lte_10": group_sizes.filter(pl.col("len") <= 10).shape[0]
        }
        
        # Basic data stats
        basic_stats = train_lf.select([
            pl.col("Id").n_unique().alias("unique_ids"),
            pl.col("ranker_id").n_unique().alias("unique_ranker_ids"),
            pl.count().alias("total_rows")
        ]).collect()
        
        validations["basic_stats"] = basic_stats.to_dict(as_series=False)
        
        return validations
    
    def detect_missing_patterns(self, lf: pl.LazyFrame, sample_size: int = 10000) -> Dict[str, float]:
        """Analyze missing value patterns efficiently using lazy evaluation"""
        sample_lf = lf.limit(sample_size)
        
        # Compute null counts lazily
        null_counts_lf = sample_lf.select([
            pl.col(c).is_null().sum().alias(c) for c in sample_lf.columns
        ])
        
        # Collect only the small result dataframe
        null_counts = null_counts_lf.collect()
        
        if null_counts.shape[0] == 0:
            return {col: 0.0 for col in sample_lf.columns}

        missing_rates = {col: null_counts[col][0] / sample_size for col in null_counts.columns}
        return missing_rates
    
    def detect_outliers(self, lf: pl.LazyFrame, numeric_columns: List[str]) -> Dict[str, Dict]:
        """Detect outliers in numeric columns using a single lazy query"""
        if not numeric_columns:
            return {}

        quantile_exprs = []
        quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        
        for col in numeric_columns:
            quantile_exprs.extend([pl.col(col).quantile(q).alias(f"{col}_q{int(q*100)}") for q in quantiles])
            quantile_exprs.extend([pl.col(col).min().alias(f"{col}_min"), pl.col(col).max().alias(f"{col}_max")])

        # Collect all stats in one go
        all_stats_df = lf.select(quantile_exprs).collect()

        outlier_stats = {}
        for col in numeric_columns:
            stats = {
                "min": all_stats_df.select(f"{col}_min").item(),
                "max": all_stats_df.select(f"{col}_max").item()
            }
            for q in quantiles:
                stats[f"q{int(q*100)}"] = all_stats_df.select(f"{col}_q{int(q*100)}").item()
            outlier_stats[col] = stats
            
        return outlier_stats

class JSONFeatureExtractor:
    """Extract features from raw JSON files"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.json_archive_path = self.config.data_dir / "jsons_raw.tar.kaggle"
        
    def extract_archive(self) -> Path:
        """Extract JSON archive if not already extracted"""
        extracted_dir = self.config.processed_dir / "jsons_raw"
        
        if extracted_dir.exists():
            logger.info(f"JSON archive already extracted to {extracted_dir}")
            return extracted_dir
            
        logger.info("Extracting JSON archive...")
        extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # The .kaggle file is actually a .tar.gz
        with tarfile.open(self.json_archive_path, 'r') as tar:
            tar.extractall(extracted_dir)
            
        logger.info(f"JSON archive extracted to {extracted_dir}")
        return extracted_dir
    
    def process_json_file(self, json_path: Path) -> Dict:
        """Process a single JSON file and extract key features"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features = {
                'ranker_id': data.get('ranker_id'),
                'request_time': data.get('request_time')
            }
            
            # Extract personal data features
            personal_data = data.get('personalData', {})
            features.update({
                'yearOfBirth': personal_data.get('yearOfBirth'),
                'hasAssistant': personal_data.get('hasAssistant'),
                'isGlobal': personal_data.get('isGlobal')
            })
            
            # Extract route data features
            route_data = data.get('routeData', {})
            features.update({
                'requestDepartureDate': route_data.get('requestDepartureDate'),
                'requestReturnDate': route_data.get('requestReturnDate')
            })
            
            # Extract flight options features
            flight_data = data.get('data', [])
            if not isinstance(flight_data, list):
                flight_data = flight_data.get('$values', [])

            features['num_flight_options'] = len(flight_data)
            
            # Enhanced feature extraction from flight options
            all_flight_features = []
            for option in flight_data:
                if not isinstance(option, dict): continue
                
                option_id = option.get('id')
                
                # Extract from pricings
                pricings = option.get('pricings', [])
                if not isinstance(pricings, list): continue
                
                for pricing in pricings:
                    if not isinstance(pricing, dict): continue
                    
                    pricing_info = pricing.get('pricingInfo', [])
                    if not isinstance(pricing_info, list): continue

                    for info in pricing_info:
                        if not isinstance(info, dict): continue
                        
                        fares_info = info.get('faresInfo', [])
                        if not isinstance(fares_info, list): continue

                        for fare in fares_info:
                            if not isinstance(fare, dict): continue
                            
                            flight_features = {
                                'ranker_id': data.get('ranker_id'),
                                'flight_id': option_id,
                                'pricing_id': pricing.get('id'),
                                'total_price': pricing.get('totalPrice'),
                                'policy_compliant': info.get('isAccessTP'),
                                'cabin_class': fare.get('cabinClass'),
                                'seats_available': fare.get('seatsAvailable'),
                                'baggage_quantity': fare.get('baggageAllowance', {}).get('quantity'),
                                'cancellation_fee': None,
                                'exchange_fee': None,
                            }

                            # Extract fare rules
                            mini_rules = pricing.get('miniRules', [])
                            if isinstance(mini_rules, list):
                                for rule in mini_rules:
                                    if not isinstance(rule, dict): continue
                                    if rule.get('category') == 31: # Cancellation
                                        flight_features['cancellation_fee'] = rule.get('monetaryAmount')
                                    elif rule.get('category') == 33: # Exchange
                                        flight_features['exchange_fee'] = rule.get('monetaryAmount')
                            
                            all_flight_features.append(flight_features)

            # This returns a list of dicts, which will be converted to a DataFrame
            # The calling function needs to handle this list of features
            return all_flight_features
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            return {'ranker_id': json_path.stem, 'error': str(e)}
    
    def extract_features_parallel(self, max_files: Optional[int] = None) -> pl.DataFrame:
        """Extract features from JSON files using parallel processing"""
        json_dir = self.extract_archive()
        json_files = list(json_dir.glob("**/*.json"))
        
        if max_files:
            json_files = json_files[:max_files]
            
        logger.info(f"Processing {len(json_files)} JSON files with {self.config.n_workers} workers")
        
        with mp.Pool(self.config.n_workers) as pool:
            results = pool.map(self.process_json_file, json_files)
        
        # Flatten the list of lists into a single list of features
        all_features = [item for sublist in results for item in sublist if isinstance(sublist, list)]
        
        if not all_features:
            return pl.DataFrame()

        # Convert results to DataFrame
        features_df = pl.DataFrame(all_features)

        # Pre-aggregate features to the ranker_id level
        aggregated_features = features_df.group_by("ranker_id").agg([
            pl.col("total_price").mean().alias("avg_total_price"),
            pl.col("policy_compliant").mean().alias("avg_policy_compliant"),
            pl.col("seats_available").sum().alias("total_seats_available"),
            pl.col("cancellation_fee").mean().alias("avg_cancellation_fee"),
            pl.col("exchange_fee").mean().alias("avg_exchange_fee"),
            pl.count().alias("json_options_count")
        ])
        
        # Save processed features
        output_path = self.config.processed_dir / "json_features_aggregated.parquet"
        aggregated_features.write_parquet(output_path)
        logger.info(f"Aggregated JSON features saved to {output_path}")
        
        return aggregated_features

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.loader = MemoryEfficientLoader(self.config)
        self.validator = DataQualityValidator(self.config)
        self.json_extractor = JSONFeatureExtractor(self.config)
        # Expose a single FeatureEngineer instance so that external callers
        # (e.g. inference pipelines) can reuse the same transformations.
        # This avoids duplicated initialization and keeps feature logic
        # centralized in one place.
        from feature_engineering import FeatureEngineer  # local import to avoid cycles
        self.feature_engineer = FeatureEngineer()
        
    def initialize_pipeline(self) -> Dict:
        """Initialize the data pipeline and run basic validation"""
        logger.info("Initializing FlightRank 2025 Data Pipeline...")
        
        # Get schema information
        train_schema = self.loader.get_schema_info("train")
        test_schema = self.loader.get_schema_info("test")
        
        logger.info(f"Train data: {train_schema['total_columns']} columns")
        logger.info(f"Test data: {test_schema['total_columns']} columns")
        
        # Load minimal data for validation
        train_lf = self.loader.load_structured_data("train", columns=["Id", "ranker_id", "selected"])
        
        # Validate data quality
        validation_results = self.validator.validate_training_data(train_lf)
        
        # Log validation results
        logger.info(f"Data validation complete:")
        logger.info(f"  Total groups: {validation_results['total_groups']}")
        logger.info(f"  Invalid groups: {validation_results['invalid_groups_count']}")
        logger.info(f"  Group size stats: {validation_results['group_size_stats']}")
        
        return {
            "train_schema": train_schema,
            "test_schema": test_schema,
            "validation_results": validation_results
        }
    
    def run_full_analysis(self, extract_json: bool = False, max_json_files: Optional[int] = 1000) -> Dict:
        """Run comprehensive data analysis"""
        logger.info("Running full data analysis...")
        
        results = self.initialize_pipeline()
        
        # Load full training data lazily
        train_lf = self.loader.load_structured_data("train")
        
        # Analyze missing patterns on sample
        missing_patterns = self.validator.detect_missing_patterns(train_lf, sample_size=10000)
        results["missing_patterns"] = missing_patterns
        
        # Extract JSON features if requested
        if extract_json:
            logger.info("Extracting JSON features...")
            json_features = self.json_extractor.extract_features_parallel(max_files=max_json_files)
            results["json_features_shape"] = json_features.shape
            results["json_sample"] = json_features.head(5).to_dict(as_series=False)
        
        return results

    def prepare_training_data(self, profile_ids: Optional[List[int]] = None) -> pl.LazyFrame:
        """
        Prepares a complete, feature-engineered dataset for training.

        This method integrates structured data with JSON-derived features,
        applies all feature engineering transformations, and optionally filters
        the data for a specific set of profile IDs for cross-validation.

        Args:
            profile_ids: An optional list of profile IDs to filter the dataset.
                         If provided, only these profiles will be included.

        Returns:
            A polars.LazyFrame ready for model training, including the label,
            ranker_id for grouping, and profileId for cross-validation.
        """
        logger.info("Preparing training data...")

        # Load the full structured training data
        structured_lf = self.loader.load_structured_data("train")

        # Load the aggregated JSON features
        json_features_path = self.config.processed_dir / "json_features_aggregated.parquet"
        if not json_features_path.exists():
            raise FileNotFoundError(f"JSON features not found at {json_features_path}. Please run JSON extraction first.")
        
        json_features_lf = pl.scan_parquet(json_features_path)

        # Join structured data with JSON features
        combined_lf = structured_lf.join(json_features_lf, on="ranker_id", how="left")

        # Apply the same feature engineering pipeline that will also be used
        # at inference time.
        engineered_lf = self.feature_engineer.engineer_all_features(combined_lf)

        # Filter by profile_ids if provided (for cross-validation)
        if profile_ids:
            logger.info(f"Filtering training data for {len(profile_ids)} profile IDs.")
            engineered_lf = engineered_lf.filter(pl.col("profileId").is_in(profile_ids))

        # Ensure key columns are present
        final_columns = ["selected", "ranker_id", "profileId"] + [
            col for col in engineered_lf.columns if col not in ["selected", "ranker_id", "profileId"]
        ]
        
        logger.info("Training data preparation complete.")
        return engineered_lf.select(final_columns)


if __name__ == "__main__":
    # Initialize pipeline
    config = DataConfig(
        n_workers=8,
        memory_limit_gb=8.0,
        extract_json_features=True
    )
    
    pipeline = DataPipeline(config)
    
    # Run initial analysis
    results = pipeline.run_full_analysis(extract_json=False, max_json_files=100)
    
    # Save results
    import pickle
    with open("processed/pipeline_analysis.pkl", "wb") as f:
        pickle.dump(results, f)
    
    logger.info("Pipeline analysis complete!") 