"""
NEURAL RANKING APPROACH - Deep learning to reach 0.7+ HitRate@3
"""
import polars as pl
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from pathlib import Path

from data_pipeline import DataPipeline, DataConfig
from utils import ValidationUtils, TargetEncodingUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RankingDataset(Dataset):
    """Dataset for neural ranking"""
    
    def __init__(self, features, targets, groups):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.groups = groups
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DeepRankingModel(nn.Module):
    """Deep neural network for ranking"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final scoring layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

def create_extreme_features(df):
    """Create extreme feature engineering for neural network"""
    
    logger.info("Creating extreme features for neural ranking...")
    
    # Start with basic features
    df = df.with_row_count('idx')
    
    # 1. POSITION FEATURES (many variants)
    df = df.with_columns([
        pl.col('idx').rank().over('ranker_id').alias('position'),
        pl.col('idx').rank(method='dense').over('ranker_id').alias('position_dense'),
        ((pl.col('idx').rank().over('ranker_id') - 1) / (pl.count().over('ranker_id') - 1)).alias('position_pct'),
        (1.0 / pl.col('idx').rank().over('ranker_id')).alias('position_inverse'),
        np.log(pl.col('idx').rank().over('ranker_id')).alias('position_log'),
        (pl.col('idx').rank().over('ranker_id') == 1).cast(pl.Float32).alias('is_first'),
        (pl.col('idx').rank().over('ranker_id') <= 3).cast(pl.Float32).alias('is_top3'),
        (pl.col('idx').rank().over('ranker_id') <= 5).cast(pl.Float32).alias('is_top5')
    ])
    
    # 2. PRICE FEATURES (many variants)
    session_price_stats = df.group_by('ranker_id').agg([
        pl.col('totalPrice').min().alias('session_min_price'),
        pl.col('totalPrice').max().alias('session_max_price'),
        pl.col('totalPrice').mean().alias('session_avg_price'),
        pl.col('totalPrice').median().alias('session_median_price'),
        pl.col('totalPrice').std().alias('session_price_std'),
        pl.col('totalPrice').quantile(0.25).alias('session_p25_price'),
        pl.col('totalPrice').quantile(0.75).alias('session_p75_price')
    ])
    
    df = df.join(session_price_stats, on='ranker_id')
    
    df = df.with_columns([
        # Price rankings
        pl.col('totalPrice').rank().over('ranker_id').alias('price_rank'),
        pl.col('totalPrice').rank(method='dense').over('ranker_id').alias('price_rank_dense'),
        ((pl.col('totalPrice').rank().over('ranker_id') - 1) / (pl.count().over('ranker_id') - 1)).alias('price_rank_pct'),
        
        # Price ratios
        (pl.col('totalPrice') / pl.col('session_min_price')).alias('price_vs_min'),
        (pl.col('totalPrice') / pl.col('session_max_price')).alias('price_vs_max'),
        (pl.col('totalPrice') / pl.col('session_avg_price')).alias('price_vs_avg'),
        (pl.col('totalPrice') / pl.col('session_median_price')).alias('price_vs_median'),
        
        # Price z-scores and normalized
        ((pl.col('totalPrice') - pl.col('session_avg_price')) / pl.col('session_price_std')).alias('price_zscore'),
        ((pl.col('totalPrice') - pl.col('session_min_price')) / (pl.col('session_max_price') - pl.col('session_min_price'))).alias('price_normalized'),
        
        # Price binary indicators
        (pl.col('totalPrice') == pl.col('session_min_price')).cast(pl.Float32).alias('is_cheapest'),
        (pl.col('totalPrice') <= pl.col('session_p25_price')).cast(pl.Float32).alias('is_bottom_quartile'),
        (pl.col('totalPrice') <= pl.col('session_median_price')).cast(pl.Float32).alias('is_below_median'),
        
        # Price deviations
        (pl.col('totalPrice') - pl.col('session_min_price')).alias('price_above_min'),
        (pl.col('session_max_price') - pl.col('totalPrice')).alias('price_below_max')
    ])
    
    # 3. INTERACTION FEATURES
    df = df.with_columns([
        # Position × Price interactions
        (pl.col('position') * pl.col('price_rank')).alias('pos_price_mult'),
        (pl.col('position') + pl.col('price_rank')).alias('pos_price_sum'),
        (pl.col('position') / pl.col('price_rank')).alias('pos_price_ratio'),
        
        # Complex interactions
        (pl.col('is_first') * pl.col('is_cheapest')).alias('first_and_cheapest'),
        (pl.col('is_top3') * pl.col('is_below_median')).alias('top3_and_cheap'),
        (pl.col('position_pct') * pl.col('price_vs_median')).alias('pos_pct_price_ratio'),
        
        # Business logic combinations
        (pl.col('is_first').cast(pl.Float32) * 10 + pl.col('is_cheapest').cast(pl.Float32) * 5).alias('business_score'),
        (pl.col('position_inverse') + (1.0 / pl.col('price_rank'))).alias('inverse_combined')
    ])
    
    # 4. POLICY FEATURES
    if 'pricingInfo_isAccessTP' in df.columns:
        df = df.with_columns([
            pl.col('pricingInfo_isAccessTP').cast(pl.Float32).alias('policy_compliant'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_first')).alias('policy_and_first'),
            (pl.col('pricingInfo_isAccessTP').cast(pl.Float32) * pl.col('is_cheapest')).alias('policy_and_cheap')
        ])
    
    # 5. SESSION CONTEXT FEATURES
    session_context = df.group_by('ranker_id').agg([
        pl.count().alias('session_size'),
        pl.col('pricingInfo_isAccessTP').mean().alias('session_policy_rate'),
        (pl.col('session_max_price') - pl.col('session_min_price')).first().alias('session_price_range')
    ])
    
    df = df.join(session_context, on='ranker_id')
    
    df = df.with_columns([
        # Session-level features
        np.log(pl.col('session_size')).alias('session_size_log'),
        (pl.col('session_price_range') / pl.col('session_min_price')).alias('session_price_spread'),
        (pl.col('position') / pl.col('session_size')).alias('position_in_session_pct')
    ])
    
    return df

def prepare_neural_data(df):
    """Prepare data for neural network training"""
    
    # Get feature columns (exclude metadata)
    exclude_cols = {
        'Id', 'ranker_id', 'selected', 'idx', 
        'session_min_price', 'session_max_price', 'session_avg_price', 
        'session_median_price', 'session_price_std', 'session_p25_price', 
        'session_p75_price', 'requestDate', 'legs0_departureAt'
    }
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]:
            feature_cols.append(col)
    
    logger.info(f"Using {len(feature_cols)} features for neural network")
    
    # Convert to numpy
    X = df.select(feature_cols).fill_null(0).to_numpy()
    y = df['selected'].to_numpy().astype(np.float32)
    
    # Get group information
    groups = df.group_by('ranker_id', maintain_order=True).agg(pl.count().alias('count'))['count'].to_numpy()
    
    return X, y, groups, feature_cols

def train_neural_ranker(X, y, groups):
    """Train deep neural ranking model"""
    
    logger.info(f"Training neural ranker on {X.shape[0]} samples with {X.shape[1]} features")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(len(groups) * 0.8)
    train_groups = groups[:split_idx]
    val_groups = groups[split_idx:]
    
    train_size = train_groups.sum()
    val_size = val_groups.sum()
    
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:train_size+val_size]
    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    
    # Create datasets
    train_dataset = RankingDataset(X_train, y_train, train_groups)
    val_dataset = RankingDataset(X_val, y_val, val_groups)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Create model
    model = DeepRankingModel(X.shape[1])
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):  # Max epochs
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_neural_ranker.pth')
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        if patience_counter >= 10:  # Early stopping
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_neural_ranker.pth'))
    
    return model, scaler

def evaluate_neural_ranker(model, scaler, df, feature_cols):
    """Evaluate neural ranking model"""
    
    model.eval()
    
    # Prepare features
    X = df.select(feature_cols).fill_null(0).to_numpy()
    X_scaled = scaler.transform(X)
    
    # Get predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = torch.sigmoid(model(X_tensor)).numpy()
    
    # Create evaluation DataFrame
    eval_df = df.select(['ranker_id', 'selected']).with_columns([
        pl.Series('predicted_score', predictions)
    ])
    
    # Calculate HitRate@3
    hitrate = ValidationUtils.calculate_hitrate_at_k(eval_df, k=3)
    
    return hitrate

def main():
    """Run neural ranking approach"""
    
    logger.info("NEURAL RANKING APPROACH - TARGETING 0.7+ HITRATE")
    
    # Load data
    config = DataConfig()
    pipeline = DataPipeline(config)
    
    # Use substantial sample for training
    full_df_lazy = pipeline.loader.load_structured_data("train")
    unique_ranker_ids = full_df_lazy.select("ranker_id").unique().collect()
    sampled_ranker_ids = unique_ranker_ids.sample(fraction=0.03, seed=42)  # 3% for neural network
    full_df_lazy = full_df_lazy.join(sampled_ranker_ids.lazy(), on="ranker_id")
    
    df = full_df_lazy.collect()
    logger.info(f"Neural training data: {df.shape}")
    
    # Filter to meaningful groups
    group_sizes = df.group_by('ranker_id').agg(pl.count().alias('group_size'))
    df = df.join(group_sizes, on='ranker_id', how='inner')
    df = df.filter(pl.col('group_size') > 5).drop('group_size')  # At least 6 options
    logger.info(f"Filtered data: {df.shape}")
    
    # Create extreme features
    df = create_extreme_features(df)
    logger.info(f"With extreme features: {df.shape}")
    
    # Prepare data
    X, y, groups, feature_cols = prepare_neural_data(df)
    
    # Train neural ranker
    model, scaler = train_neural_ranker(X, y, groups)
    
    # Evaluate
    hitrate = evaluate_neural_ranker(model, scaler, df, feature_cols)
    
    logger.info(f"NEURAL RANKER HitRate@3: {hitrate:.4f}")
    
    if hitrate >= 0.7:
        logger.info("🎉 BREAKTHROUGH ACHIEVED WITH NEURAL RANKING!")
    else:
        logger.info(f"❌ Still need {0.7 - hitrate:.4f} more performance")
        
        # Try ensemble with LightGBM
        logger.info("Training LightGBM ensemble...")
        
        # Quick LightGBM for comparison
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': 0.05,
            'num_leaves': 1024,
            'feature_fraction': 0.9,
            'random_state': 42,
            'verbose': -1,
            'n_estimators': 2000
        }
        
        split_idx = int(len(groups) * 0.8)
        train_groups = groups[:split_idx]
        val_groups = groups[split_idx:]
        
        train_size = train_groups.sum()
        val_size = val_groups.sum()
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_train = y[:train_size]
        y_val = y[train_size:train_size+val_size]
        
        train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_data)
        
        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        # Evaluate LightGBM
        lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        
        val_df = df[train_size:train_size+val_size].with_columns([
            pl.Series('predicted_score', lgb_preds)
        ])
        
        lgb_eval_df = val_df.select(['ranker_id', 'selected', 'predicted_score'])
        lgb_hitrate = ValidationUtils.calculate_hitrate_at_k(lgb_eval_df, k=3)
        
        logger.info(f"LightGBM HitRate@3: {lgb_hitrate:.4f}")
        
        # Show best result
        best_hitrate = max(hitrate, lgb_hitrate)
        logger.info(f"BEST APPROACH HitRate@3: {best_hitrate:.4f}")
        
        if best_hitrate >= 0.7:
            logger.info("🎉 BREAKTHROUGH ACHIEVED!")
        else:
            logger.info(f"❌ Still {0.7 - best_hitrate:.4f} short of target")

if __name__ == "__main__":
    main()