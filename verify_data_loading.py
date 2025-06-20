import polars as pl
from data_loading import load_dataset

# Verify that loading 'train' returns a LazyFrame
train_df = load_dataset("train")
assert isinstance(train_df, pl.LazyFrame), "load_dataset('train') did not return a LazyFrame"

# Verify that loading 'test' returns a LazyFrame
test_df = load_dataset("test")
assert isinstance(test_df, pl.LazyFrame), "load_dataset('test') did not return a LazyFrame"

# Verify column selection
cols_to_select = ["Id", "totalPrice"]
train_subset_df = load_dataset("train", cols=cols_to_select)
assert train_subset_df.columns == cols_to_select, f"Columns did not match. Expected {cols_to_select}, got {train_subset_df.columns}"

print("Verification successful!")