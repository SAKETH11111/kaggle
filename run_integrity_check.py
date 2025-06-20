import polars as pl
from data_loading import load_dataset
import json
import os

def run_integrity_checks():
    """
    Performs data integrity and leakage checks on the dataset.
    """
    # Load training data
    train_df = load_dataset("train")

    # Assert that for every group, the sum of the `selected` column is exactly 1.
    selected_sum_check = train_df.group_by("ranker_id").agg(
        pl.sum("selected")
    ).collect()
    assert (selected_sum_check["selected"] == 1).all(), "Integrity check failed: Not every ranker_id has exactly one selected item."

    # Load test data
    test_df = load_dataset("test")

    # Assert that the 'selected' column is not in the test data
    assert "selected" not in test_df.columns, "Data leakage check failed: 'selected' column found in test data."

    # Generate a histogram of the group sizes
    group_sizes = train_df.group_by("ranker_id").count().collect()
    
    histogram = group_sizes.group_by("count").agg(
        pl.count().alias("num_groups")
    ).sort("count")

    histogram_dict = {row["count"]: row["num_groups"] for row in histogram.to_dicts()}

    # Ensure the output directory exists
    os.makedirs("processed", exist_ok=True)
    
    # Save the histogram to a JSON file
    output_path = "processed/00_integrity_report.json"
    with open(output_path, "w") as f:
        json.dump(histogram_dict, f, indent=4)

    print("Integrity and leakage checks passed.")

if __name__ == "__main__":
    run_integrity_checks()