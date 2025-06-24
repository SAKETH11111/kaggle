import lightgbm as lgb
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RankerTrainer:
    """
    A class to encapsulate the training, prediction, and persistence of an LGBMRanker model.
    """
    def __init__(self, model_params: dict, model_dir: str = "processed/models"):
        """
        Initializes the RankerTrainer.

        Args:
            model_params (dict): Parameters for the LGBMRanker model.
            model_dir (str): Directory to save model artifacts.
        """
        self.model_params = model_params
        self.model = lgb.LGBMRanker(**self.model_params)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X_train, y_train, group_train, X_val, y_val, group_val, categorical_features: list = None, callbacks: list = None):
        """
        Trains the LGBMRanker model.
        """
        logger.info("Training LGBMRanker model...")
        
        X_train_proc = self._preprocess_features(X_train)
        X_val_proc = self._preprocess_features(X_val)
        
        # Identify categorical features for LightGBM
        if categorical_features is None:
            categorical_features = [col for col in X_train_proc.columns if X_train_proc[col].dtype.name == 'category']
        
        self.model.fit(
            X_train_proc, y_train, group=group_train,
            eval_set=[(X_val_proc, y_val)], eval_group=[group_val],
            eval_at=[3], eval_metric="ndcg",
            categorical_feature=categorical_features,
            callbacks=callbacks
        )
        logger.info("Model training complete.")

    def predict(self, X) -> np.ndarray:
        """
        Generates predictions using the trained model.
        """
        X_proc = self._preprocess_features(X)
        return self.model.predict(X_proc)

    @staticmethod
    def _preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe has dtypes compatible with LightGBM."""
        df = df.copy()
        # Remove accidental index column that LightGBM might treat as feature
        if not df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1)):
            df.reset_index(drop=True, inplace=True)

        # LightGBM may still pick up an implicit index column created during pandas conversion
        if "__index_level_0__" in df.columns:
            df = df.drop(columns=["__index_level_0__"])

        for col in df.columns:
            if df[col].dtype.name == 'object':
                df[col] = df[col].astype('category')
        return df

    def save_model(self, filename: str):
        """
        Saves the trained model booster to a file.
        """
        filepath = self.model_dir / filename
        self.model.booster_.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")