# FlightRank 2025 - Data Architecture & Processing Pipeline

## 🎯 Strategic Foundation Overview

This repository implements the **Phase 1: Strategic Foundation** for the FlightRank 2025 competition - building robust, memory-efficient data processing infrastructure that can handle 200+ features, multiple model types, and complex validation scenarios.

**Core Philosophy**: Build once, iterate fast.

## 📊 Dataset Overview

### Data Scale & Structure
- **Training Data**: 18.1M rows, 127 columns, 105,539 search sessions
- **Test Data**: 126 columns (missing `selected` target)
- **JSON Archive**: 150k files (~50GB) with rich feature opportunities
- **Evaluation Target**: 89,646 groups with >10 options (84.9% of total)

### Key Statistics
```
📊 Dataset: 18,145,372 rows, 127 columns
👥 Groups: 105,539 search sessions  
🎯 Evaluation groups: 89,646 (84.9%)
⚠️  Data quality: 77 columns with >50% missing
📈 Group sizes: 1-8,236 options (median: 50)
✅ Data validation: PASSED - All groups have exactly 1 selected flight
```

## 🏗️ Architecture Components

### 1. Memory-Efficient Data Pipeline (`data_pipeline.py`)

**Core Classes:**
- `MemoryEfficientLoader`: Lazy loading with Polars for 8GB+ datasets
- `DataQualityValidator`: Automated integrity checks and validation  
- `JSONFeatureExtractor`: Parallel processing of 150k JSON files
- `DataPipeline`: Main orchestrator with 8-core multiprocessing

**Key Features:**
- Lazy evaluation prevents memory crashes
- Streaming processing for large datasets  
- Chunked processing with configurable batch sizes
- Robust error handling for malformed data

### 2. Advanced Feature Engineering (`feature_engineering.py`)

**Feature Categories:**
- **Price Features**: Rankings, percentiles, group statistics
- **Time Features**: Departure patterns, duration analysis
- **Route Features**: Airlines, airports, connections
- **Policy Features**: Corporate compliance, cancellation rules
- **Group Features**: Within-session comparisons and rankings
- **Interaction Features**: Complex business logic combinations

### 3. Utility Framework (`utils.py`)

**Utility Classes:**
- `ValidationUtils`: Submission format validation, HitRate@k calculation
- `DataUtils`: Memory optimization, group-aware data splitting
- `FeatureUtils`: Categorical encoding, ranking features
- `CrossValidationUtils`: Group-based K-fold for ranking problems
- `IOUtils`: Efficient data serialization and model artifacts

### 4. Analysis & Testing (`run_analysis.py`)

Comprehensive data analysis pipeline:
- Schema validation and data quality assessment
- Missing pattern analysis across 127 columns
- Feature engineering testing and validation
- JSON extraction testing with parallel processing
- Automated report generation

## 🚀 Quick Start

### Setup Environment
```bash
pip install polars pyarrow fastparquet
```

### Run Initial Analysis
```bash
python run_analysis.py
```

This will:
1. Validate data integrity (✅ All 105,539 groups passed)
2. Analyze missing patterns (77 columns with >50% missing)
3. Test feature engineering pipeline
4. Extract sample JSON features
5. Generate detailed analysis report

### Key Files Generated
- `processed/initial_analysis_report.json`: Comprehensive data analysis
- `processed/jsons_raw/`: Extracted JSON archive (50GB)
- `analysis.log`: Detailed processing logs

## 📋 Critical Findings

### Data Quality Assessment
- **✅ Perfect Data Integrity**: All 105,539 groups have exactly 1 selected flight
- **⚠️ High Missing Rate**: 77/127 columns (60%) have >50% missing values
- **🎯 Evaluation Impact**: 84.9% of groups qualify for scoring (>10 options)

### Missing Data Patterns
Top missing columns indicate optional flight segments and corporate features:
- `frequentFlyer`: 75% missing
- `corporateTariffCode`: 60% missing  
- `legs0_segments1_*`: 57% missing (connecting flights)
- `legs1_*`: High missing (return flights)

### JSON Feature Opportunity
- **Competitive Advantage**: 150k JSON files contain rich features most teams will skip
- **Technical Challenge**: 50GB archive requires parallel processing
- **Key Features Available**: Fare rules, policy compliance, real-time availability

## 🔧 Advanced Features

### Memory Management
```python
config = DataConfig(
    memory_limit_gb=8.0,
    use_lazy_loading=True,
    chunk_size=10000
)
```

### Parallel JSON Processing
```python
json_features = pipeline.json_extractor.extract_features_parallel(
    max_files=1000,  # Process subset for testing
    n_workers=8      # 8-core parallel processing
)
```

### Group-Aware Validation
```python
train_df, val_df = DataUtils.split_by_groups(
    df, group_col="ranker_id", 
    train_ratio=0.8, 
    seed=42
)
```

## 🎯 Next Steps (Phase 2-3)

### Immediate Priorities
1. **JSON Feature Extraction**: Process full 150k archive for competitive advantage
2. **Baseline Model**: Implement LightGBM ranking with cross-validation
3. **Feature Selection**: Analyze importance on larger samples
4. **Submission Pipeline**: Automated validation and format checking

### Advanced Development
1. **Multi-Model Ensemble**: Combine ranking models with different architectures  
2. **Real-time Processing**: Optimize pipeline for inference speed
3. **Feature Engineering**: Domain-specific business travel patterns
4. **Hyperparameter Optimization**: Bayesian optimization with ranking metrics

## 📖 API Reference

### Core Pipeline Usage
```python
from data_pipeline import DataPipeline, DataConfig
from feature_engineering import FeatureEngineer

# Initialize pipeline
config = DataConfig(n_workers=8, extract_json_features=True)
pipeline = DataPipeline(config)

# Load and process data
train_lf = pipeline.loader.load_structured_data("train")
engineer = FeatureEngineer()
features_df = engineer.engineer_all_features(train_lf)

# Validate and save
results = pipeline.run_full_analysis()
```

### Submission Validation
```python
from utils import ValidationUtils

validations = ValidationUtils.validate_submission_format(
    submission_df, test_df
)
hitrate = ValidationUtils.calculate_hitrate_at_k(predictions, k=3)
```

## 📊 Performance Benchmarks

- **Data Loading**: Lazy evaluation handles 18M+ rows efficiently
- **Feature Engineering**: Processes 1000 samples in <1 second  
- **JSON Extraction**: 8-core parallel processing of archive
- **Memory Usage**: <8GB peak with optimized data types
- **Validation**: Complete integrity check in ~2 seconds

## 🔍 Monitoring & Debugging

All operations include comprehensive logging:
```bash
tail -f analysis.log  # Monitor real-time processing
```

Performance profiling decorators available:
```python
@PerformanceUtils.time_execution
@PerformanceUtils.profile_memory_usage
def your_function():
    pass
```

---

**Competition Goal**: HitRate@3 ≥ 0.7 for bonus prize (double payout)  
**Architecture Status**: ✅ Phase 1 Complete - Ready for model development 