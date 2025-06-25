# Phase 1: Solidify the Foundation & Validate Assumptions

This phase is critical for establishing a robust, reproducible workflow that prevents data leakage and confirms our foundational hypotheses before building complex features.

## 1. Upgrade the Code Repository Structure
- [ ] Create a new `/src` directory for all source code.
- [ ] Create modular subdirectories:
  - `/src/feature_pipeline/`
  - `/src/json_extract/`
  - `/src/eval/`
  - `/src/inference/`
- [ ] Migrate existing scripts to the new structure:
  - `run_training.py` -> `/src/train_baseline.py`
  - `json_feature_extractor.py` -> `/src/json_extract/parallel_extractor.py`
  - Feature engineering scripts -> `/src/feature_pipeline/`
- [ ] Create new placeholder scripts:
  - `/src/eval/metrics.py`
  - `/src/inference/compile_model.py`

## 2. Implement Leakage-Proof Cross-Validation
- [ ] Modify `/src/train_baseline.py` to implement `StratifiedGroupKFold`.
- [ ] Use `profileId` as the grouping column to prevent user history leakage.
- [ ] Ensure all subsequent model evaluation uses this CV scheme.

## 3. Train Baseline Ranker & Quantify Feature Impact
- [ ] Train a baseline `LGBMRanker` model using the new `/src/train_baseline.py` script.
- [ ] Use only the ~40 existing "obvious" features for this baseline.
- [ ] Generate and save a gain-based feature importance plot.
- [ ] Verify that core drivers (`price_rank`, `duration_rank`, `position`, `pricingInfo_isAccessTP`) dominate the importances.
