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

## Phase 4: Advanced Feature Engineering Roadmap (Week 1)

This phase focuses on implementing a comprehensive set of advanced features to significantly boost model performance.

### Day 1: Pricing Intelligence Features
- [x] **Task:** Implement within-session price competitiveness metrics.
  - **Features:** `rel_price_vs_min`, `price_rank_in_session`, `price_vs_median_gap`, `is_cheapest_or_near`, `price_z_score`.
  - **Goal:** Quantify the "deal appeal" of each flight option to improve ranking based on cost-effectiveness.

### Day 2: Corporate Policy Compliance Features
- [x] **Task:** Implement features capturing adherence to corporate travel policies.
  - **Features:** `policy_compliant_flag`, `cabin_allowed`, `preferred_airline_match`, `expense_approval_required`.
  - **Goal:** Align model predictions with the critical business constraint of policy compliance.

### Day 3: Fare Flexibility Analysis Features
- [x] **Task:** Engineer features to quantify the flexibility of a fare (cancellation, changes).
  - **Features:** `cancellation_policy_tier`, `change_policy_category`, `free_baggage`.
  - **Goal:** Capture the value business travelers place on flexible tickets.

### Day 4: Temporal Preferences – Business Hours Modeling
- [x] **Task:** Implement features to model preferences for business-friendly travel times.
  - **Features:** `departure_hour_bin`, `is_business_hours`, `is_redeye_flight`, `is_weekend_flight`, cyclical hour encodings.
  - **Goal:** Align model with known preferences for flights during reasonable hours and days.

### Day 5: Booking Urgency & Lead Time Features
- [x] **Task:** Implement features based on the booking lead time.
  - **Features:** `booking_lead_days`, `booking_urgency_category`.
  - **Goal:** Provide the model with context about the booking window, which influences decision-making.

### Day 6: Route and Connection Quality Features
- [x] **Task:** Engineer features to evaluate itinerary quality, focusing on connections.
  - **Features:** `is_direct_flight`, `connection_count`, `shortest_layover_minutes`, `longest_layover_minutes`, `has_overnight_connection`.
  - **Goal:** Capture the strong preference for non-stop and efficient routes.

### Day 7: Flight Duration and Final Integration
- [x] **Task:** Implement features for total travel duration and integrate all new features.
  - **Features:** `total_duration_minutes`, `duration_vs_min`, `duration_rank`.
  - **Goal:** Finalize the feature set, perform integration tests, and ensure the entire pipeline is robust and free of data leakage.
