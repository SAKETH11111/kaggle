import pandas as pd
from typing import List, Dict

def assert_no_ranker_id_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Asserts that there are no ranker_id values that appear in both the train and test DataFrames.

    Args:
        train_df (pd.DataFrame): The training DataFrame, must contain a 'ranker_id' column.
        test_df (pd.DataFrame): The testing DataFrame, must contain a 'ranker_id' column.
    """
    train_ranker_ids = set(train_df['ranker_id'].unique())
    test_ranker_ids = set(test_df['ranker_id'].unique())

    intersection = train_ranker_ids.intersection(test_ranker_ids)

    assert len(intersection) == 0, f"Data leakage detected! The following ranker_ids are in both train and test sets: {intersection}"

def fold_aware_aggregation(
    df: pd.DataFrame, 
    group_by_cols: List[str], 
    agg_cols: Dict[str, List[str]], 
    fold_col: str
) -> pd.DataFrame:
    """
    Performs fold-aware aggregation to prevent data leakage from the validation fold.

    The goal of this function is to create aggregated features for a dataset that is used in
    cross-validation. A common mistake is to calculate aggregations (like mean, sum, etc.)
    over the entire training set and then merge them back. This leads to data leakage, as
    the features for a given validation fold will have been influenced by the data within
    that same fold.

    This function avoids that by performing the aggregations for each fold using only the data
    from the *other* folds.

    How it should work:
    1. Iterate through each unique fold in the `fold_col`.
    2. For the current fold, create a temporary training set consisting of all other folds.
    3. Calculate the aggregations on this temporary training set.
    4. Merge these aggregations back to the data corresponding to the current fold.
    5. Concatenate the results for all folds.

    For now, this implementation is a basic placeholder that performs a simple group-by
    and aggregation across the entire DataFrame. This is NOT safe for use in cross-validation
    as it will leak information. It is provided here with this detailed docstring to illustrate
    the intended future functionality.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_by_cols (List[str]): A list of column names to group by.
        agg_cols (Dict[str, List[str]]): A dictionary where keys are column names to aggregate
                                         and values are lists of aggregation functions (e.g., 'mean', 'sum').
        fold_col (str): The name of the column that contains the fold identifiers.

    Returns:
        pd.DataFrame: The original DataFrame with aggregated features merged back.
                      NOTE: The current basic implementation is not fold-aware.
    """
    # Basic (and leaky) group-by aggregation for placeholder purposes.
    # A real implementation would involve iterating over folds.
    
    agg_spec = {col: funcs for col, funcs in agg_cols.items()}
    
    grouped = df.groupby(group_by_cols).agg(agg_spec)
    
    # Flatten multi-level column names
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
    df = df.merge(grouped, on=group_by_cols, how='left')
    
    return df