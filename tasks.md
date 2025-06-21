# FlightRank 2025 Project Tasks

## Phase 1: Data Architecture & Processing (Days 1-2) - COMPLETED

- [x] Create reproducible Docker environment.
- [x] Implement memory-efficient data loader for parquet files.
- [x] Implement data integrity and leakage checks.
- [x] Analyze JSON data and generate schema overview.
- [x] Create feature map for high-value feature extraction.
- [x] Design architecture for parallel JSON extractor.
- [x] Set up Makefile and GitHub Actions CI workflow.
- [x] Execute full-scale parallel JSON extraction.
- [x] Post-process raw features into curated, model-ready feature blocks.
- [x] Merge structured data with new JSON features.
- [x] Benchmark performance of the new dataset.
- [x] Implement data leakage guards.
- [x] Create feature store API with caching.
- [x] Generate comprehensive report documenting the work.

- [x] 1. **Environment bootstrap**: Set up the Conda environment and Dockerfile.
- [x] 2. **Structured parquet ingestion skeleton**: Create a memory-efficient data loader for parquet files.
- [x] 3. **Integrity & leakage audit**: Build a script to validate data integrity and prevent leakage.
- [x] 4. **JSON archive reconnaissance**: Analyze a sample of the JSON data to understand its structure.
- [x] 5. **Feature-selection workshop**: Define the high-value features to be extracted from the JSON files.
- [x] 6. **Parallel-extraction architecture design**: Design the architecture for parallel JSON processing.
- [x] 7. **Data-flow automation**: Create a Makefile and CI pipeline for automation.
- [x] 8. **Full-scale JSON extraction run**: Execute the parallel extraction on the entire dataset.
- [x] 9. **Feature block post-processing**: Clean and transform the extracted JSON features.
- [x] 10. **Merge structured + JSON features**: Join the structured data with the new JSON features.
- [x] 11. **Memory & performance benchmark**: Benchmark the performance of the new dataset.
- [x] 12. **Data-leakage guards**: Implement safeguards to prevent data leakage between train and test sets.
- [x] 13. **Feature-store API scaffolding**: Create an API for accessing the feature store.
- [x] 14. **Documentation & hand-off**: Document the work and prepare for the next phase.

## Phase 2: LGBM Ranker Integration (Day 3) - COMPLETED

This phase focuses on building and evaluating a baseline `LGBMRanker` model.

- **Task 1: Extend Data Pipeline for Model Training.**
  - **Goal:** Modify `data_pipeline.py` to produce a training-ready DataFrame. This involves joining all features and ensuring data is correctly formatted for the model, parameterized by fold to prevent leakage.
  - **Status:** ✅ Completed

- **Task 2: Implement `StratifiedGroupKFold` Cross-Validation Logic.**
  - **Goal:** Create a reusable utility for 5-fold cross-validation that groups by `profileId` to prevent data leakage and stratifies by the `selected` target to ensure balanced folds.
  - **Status:** ✅ Completed

- **Task 3: Implement the `RankerTrainer` Class.**
  - **Goal:** Develop a `RankerTrainer` class in a new `ranker_trainer.py` file to encapsulate all logic for training, prediction, and model persistence for the `LGBMRanker`.
  - **Status:** ✅ Completed

- **Task 4: Implement `HitRate@3` Evaluation Metric.**
  - **Goal:** Create a precise function to calculate the `HitRate@3` metric, which is the primary success measure for the ranking model, ensuring it excludes groups with 10 or fewer items as per competition rules.
  - **Status:** ✅ Completed

- **Task 5: End-to-End Pipeline Integration.**
  - **Goal:** Create a `run_training.py` script that connects all components: data preparation, cross-validation, training, evaluation, and logging. This script will execute the full baseline model training and evaluation workflow.
  - **Status:** ✅ Completed

## Phase 3: Advanced Feature Engineering & Tuning (Day 4)

This phase focuses on advanced feature engineering and optimizing the `LGBMRanker` model.

- **Task 1: Feature Engineering Sprint.**
  - **Goal:** Brainstorm and implement new features based on domain knowledge and data analysis to improve model performance.
  - **Status:** ✅ Completed

- **Task 2: Hyperparameter Tuning.**
  - **Goal:** Use a systematic approach (e.g., Optuna) to tune the `LGBMRanker` hyperparameters and optimize for `NDCG@3`.
  - **Status:** ✅ Completed

- **Task 3: Feature Importance and SHAP Analysis.**
  - **Goal:** Generate and analyze feature importance and SHAP values from the trained model to understand key drivers of performance and validate feature engineering decisions.
  - **Status:** ✅ Completed

- **Task 4: Enhance Logging and Reproducibility.**
  - **Goal:** Improve the logging to capture all critical information for reproducibility, including best hyperparameters, fold-wise performance metrics, and random seeds.
  - **Status:** ✅ Completed

- **Task 5: Final Predictions and Ensembling.**
  - **Goal:** Generate final predictions using an ensemble of the 5-fold models and the single model trained on the full dataset, preparing the results for submission.
  - **Status:** ✅ Completed
