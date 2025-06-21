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

    def train(self, X_train, y_train, group_train, X_val, y_val, group_val, callbacks: list = None):
        """
        Trains the LGBMRanker model.
        """
        logger.info("Training LGBMRanker model...")
        # Convert object/datetime/bool columns to dtypes LightGBM accepts
        X_train = self._preprocess_features(X_train)
        X_val = self._preprocess_features(X_val)
        self.model.fit(
            X_train, y_train, group=group_train,
            eval_set=[(X_val, y_val)], eval_group=[group_val],
            eval_at=[3], eval_metric="ndcg",
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
        """Ensure dataframe has dtypes compatible with LightGBM.

        * object  -> category
        * datetime64[ns/us] -> int64
        * bool    -> int8
        """
        df = df.copy()
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_object_dtype(series):
                df[col] = series.astype("category")
            elif pd.api.types.is_bool_dtype(series):
                df[col] = series.astype("int8")
            elif pd.api.types.is_datetime64_any_dtype(series):
                df[col] = series.view("int64")
        return df

    def save_model(self, filename: str):
        """
        Saves the trained model booster to a file.
        """
        filepath = self.model_dir / filename
        self.model.booster_.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")