"""
FlightRank 2025 - Quick Analysis Script
Test the data pipeline and generate initial insights
"""

import logging
import json
from pathlib import Path
import polars as pl
from data_pipeline import DataPipeline, DataConfig
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_analysis():
    """Run the full data analysis pipeline, including JSON extraction and feature engineering."""
    logger.info("="*50)
    logger.info("FLIGHTRANK 2025 - FULL DATA ANALYSIS PIPELINE")
    logger.info("="*50)

    config = DataConfig(n_workers=8, memory_limit_gb=250.0) # Increased memory
    pipeline = DataPipeline(config)
    
    try:
        # 1. Initial Validation
        logger.info("🔍 Step 1: Initial Data Validation")
        results = pipeline.initialize_pipeline()

        # 2. Enhanced JSON Feature Extraction
        logger.info("🔍 Step 2: Enhanced JSON Feature Extraction")
        json_features_df = pipeline.json_extractor.extract_features_parallel(max_files=1000) # Use a larger sample
        logger.info(f"✅ Extracted {json_features_df.shape[0]} records from {json_features_df['ranker_id'].n_unique()} JSON files.")

        # 3. Join Structured and JSON Data
        logger.info("🔍 Step 3: Joining Structured and JSON Data")
        train_lf = pipeline.loader.load_structured_data("train")
        enriched_lf = train_lf.join(json_features_df.lazy(), on="ranker_id", how="left")

        # 4. Full Feature Engineering
        logger.info("🔍 Step 4: Full Feature Engineering on Enriched Data")
        engineer = FeatureEngineer()
        final_lf = engineer.engineer_all_features(enriched_lf)
        
        # 5. Comprehensive Data Quality Analysis
        logger.info("🔍 Step 5: Comprehensive Data Quality Analysis")
        # Missing patterns on the final dataset
        missing_patterns = pipeline.validator.detect_missing_patterns(final_lf, sample_size=10000)
        high_missing = {k: v for k, v in missing_patterns.items() if v > 0.5}
        
        # Outlier detection
        numeric_cols = [col for col, dtype in final_lf.schema.items() if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        outlier_stats = pipeline.validator.detect_outliers(final_lf.limit(50000), numeric_columns=numeric_cols)

        # 6. Generate Final Report
        logger.info("🔍 Step 6: Generating Final Analysis Report")
        final_sample = final_lf.limit(10).collect()
        report = {
            "data_overview": results["validation_results"],
            "json_extraction": {
                "files_processed": json_features_df['ranker_id'].n_unique(),
                "records_extracted": json_features_df.shape[0],
                "features_created": json_features_df.shape[1],
            },
            "feature_engineering": {
                "original_columns": len(train_lf.columns),
                "enriched_columns": len(enriched_lf.columns),
                "final_columns": len(final_lf.columns),
                "new_features_created": len(final_lf.columns) - len(train_lf.columns),
            },
            "data_quality": {
                "high_missing_columns_count": len(high_missing),
                "outlier_analysis_sample": outlier_stats,
            }
        }

        processed_dir = Path("processed")
        processed_dir.mkdir(exist_ok=True)
        with open(processed_dir / "final_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("="*50)
        logger.info("✅ PHASE 1 STRATEGIC FOUNDATION COMPLETE")
        logger.info("="*50)
        logger.info(f"📊 Final dataset has {report['feature_engineering']['final_columns']} columns.")
        logger.info(f"💎 Extracted {report['json_extraction']['records_extracted']} detailed records from JSON files.")
        logger.info("🚀 Ready for Phase 2: Model Development.")
        logger.info("="*50)

        return report

    except Exception as e:
        logger.error(f"Full analysis pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        run_full_analysis()
        logger.info("🎉 Full analysis complete! Check 'processed/final_analysis_report.json' for detailed results.")
    except Exception as e:
        logger.error(f"Script failed dramatically: {e}")
        exit(1)