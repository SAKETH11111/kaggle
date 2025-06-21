import optuna
import polars as pl
import numpy as np
import logging
from pathlib import Path
import json
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback

from data_pipeline import DataPipeline, DataConfig
from ranker_trainer import RankerTrainer
from utils import CrossValidationUtils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

N_TRIALS = 50  # Number of Optuna trials to run
N_SPLITS = 5   # Using full 5-fold CV for robust evaluation in each trial
RANDOM_SEED = 42

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    The objective function for Optuna to optimize. Trains a model with a set of
    hyperparameters and returns the average validation score.
    """
    # --- 1. Hyperparameter Search Space ---
    # We define the search space for Optuna to explore.
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        # Use CUDA device for NVIDIA GPUs.
        'device': 'cuda',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
    }

    # --- 2. Data Preparation ---
    # This part is consistent for each trial.
    config = DataConfig()
    pipeline = DataPipeline(config)
    full_df = pipeline.prepare_training_data().collect()
    
    feature_cols = [
        col for col in full_df.columns 
        if col not in [
            'Id', 'ranker_id', 'selected', 'profileId', 'search_route', 
            'price_bin', 'departure_time_category', 'cabin_class_name', 
            'origin_airport', 'destination_airport', 'marketing_carrier', 
            'operating_carrier', 'aircraft_code'
        ]
    ]

    # --- 3. Cross-Validation ---
    cv_splits = CrossValidationUtils.stratified_group_kfold_split(
        df=full_df,
        group_col='profileId',
        stratify_col='selected',
        n_splits=N_SPLITS,
        seed=RANDOM_SEED
    )

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        train_fold_df = full_df[train_idx].sort('ranker_id')
        val_fold_df = full_df[val_idx].sort('ranker_id')

        X_train = train_fold_df[feature_cols].to_pandas()
        y_train = train_fold_df['selected'].to_pandas()
        group_train = train_fold_df.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()

        X_val = val_fold_df[feature_cols].to_pandas()
        y_val = val_fold_df['selected'].to_pandas()
        group_val = val_fold_df.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()

        # --- 4. Model Training ---
        trainer = RankerTrainer(model_params=params)
        pruning_callback = LightGBMPruningCallback(trial, "ndcg@3", valid_name="valid_0")
        early_stopping_cb = lgb.early_stopping(stopping_rounds=100, verbose=False)
        callbacks = [pruning_callback, early_stopping_cb]
        trainer.train(X_train, y_train, group_train, X_val, y_val, group_val, callbacks=callbacks)
        
        # The evaluation metric (NDCG@3) is automatically calculated by LightGBM's fit method.
        # We retrieve the best score from the validation set.
        best_score = trainer.model.best_score_['valid_0']['ndcg@3']
        fold_scores.append(best_score)

    # --- 5. Return Average Score ---
    # Optuna will aim to maximize this value.
    average_ndcg = np.mean(fold_scores)
    logger.info(f"Trial {trial.number} completed. Average NDCG@3: {average_ndcg:.4f}")
    return average_ndcg

# --- Main Execution Block ---
def main():
    """
    Main function to run the Optuna hyperparameter search.
    """
    logger.info(f"Starting Optuna hyperparameter search with {N_TRIALS} trials...")

    # Create a study object and specify the direction as "maximize"
    study = optuna.create_study(direction='maximize')
    
    # Start the optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Results ---
    logger.info("Hyperparameter search complete.")
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info(f"Best NDCG@3 score: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save the best parameters to a JSON file
    best_params_path = Path("processed/optuna_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best parameters saved to {best_params_path}")

    # Save the full study results
    study_results_path = Path("processed/optuna_study_results.pkl")
    with open(study_results_path, "wb") as f:
        import pickle
        pickle.dump(study, f)
    logger.info(f"Full Optuna study results saved to {study_results_path}")

if __name__ == "__main__":
    main()