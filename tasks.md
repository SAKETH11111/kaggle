# FlightRank 2025 Implementation Plan

This file tracks the high-level tasks for building the data architecture and processing systems.

## Day 1: Data Architecture & Processing Foundation

- [x] 1. **Environment bootstrap**: Set up the Conda environment and Dockerfile.
- [x] 2. **Structured parquet ingestion skeleton**: Create a memory-efficient data loader for parquet files.
- [x] 3. **Integrity - [ ] 3. **Integrity & leakage audit** leakage audit**: Build a script to validate data integrity and prevent leakage.
- [x] 4. **JSON archive reconnaissance**: Analyze a sample of the JSON data to understand its structure.
- [x] 5. **Feature-selection workshop**: Define the high-value features to be extracted from the JSON files.
- [x] 6. **Parallel-extraction architecture design**: Design the architecture for parallel JSON processing.
- [x] 7. **Data-flow automation**: Create a Makefile and CI pipeline for automation.

## Day 2: Full-Scale Extraction & Feature Store

- [ ] 1. **Full-scale JSON extraction run**: Execute the parallel extraction on the entire dataset.
- [ ] 2. **Feature block post-processing**: Clean and transform the extracted JSON features.
- [ ] 3. **Merge structured + JSON features**: Join the structured data with the new JSON features.
- [ ] 4. **Memory & performance benchmark**: Benchmark the performance of the new dataset.
- [ ] 5. **Data-leakage guards**: Implement safeguards to prevent data leakage between train and test sets.
- [ ] 6. **Feature-store API scaffolding**: Create an API for accessing the feature store.
- [ ] 7. **Documentation & hand-off**: Document the work and prepare for the next phase.
