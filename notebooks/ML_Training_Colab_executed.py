# %% [markdown]
# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [2]</a>'.</span>

# %% [markdown]
# # ML Pattern Classifier Training - Google Colab
#
# **Instructions:**
# 1. Click "Runtime" → "Change runtime type" → Select "GPU" (optional but recommended)
# 2. Click "Runtime" → "Run all" to execute all cells
# 3. Download trained model from Google Drive link at the end
#
# **Memory Optimization:** This notebook is configured for limited RAM (12-25GB on Colab)

# %% [markdown]
# ## Step 1: Install Dependencies

# %%
# Install required packages
# !pip install yfinance lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn tqdm -q
print("✓ Dependencies installed")

# %% [markdown]
# ## Step 2: Mount Google Drive (for saving models)

# %% [markdown]
# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

# %%
from google.colab import drive
import os

drive.mount('/content/drive')

# Create output directory
os.makedirs('/content/drive/MyDrive/ML_Models', exist_ok=True)
print("✓ Google Drive mounted")

# %% [markdown]
# ## Step 3: Configure Training Parameters

# %%
# Configuration - Modify these settings as needed
CONFIG = {
    'symbol': 'SPY',
    'start_date': '2019-01-01',  # Reduced date range for memory
    'end_date': '2024-12-31',
    'model_type': 'lightgbm',     # Options: lightgbm, xgboost, random_forest
    'prediction_horizon': 3,       # Days ahead to predict
    'test_size': 0.2,             # 80/20 train/test split
    'use_walk_forward': False,    # Set True for more robust validation (uses more RAM)
    'memory_safe': True,          # Enable memory optimizations
    'random_state': 42
}

print(f"Configuration:")
print(f"  Symbol: {CONFIG['symbol']}")
print(f"  Date Range: {CONFIG['start_date']} to {CONFIG['end_date']}")
print(f"  Model: {CONFIG['model_type']}")
print(f"  Memory Safe Mode: {CONFIG['memory_safe']}")

# %% [markdown]
# ## Step 4: Download Price Data

# %%
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

print(f"Downloading {CONFIG['symbol']} data...")

# Download data
df = yf.download(CONFIG['symbol'], start=CONFIG['start_date'], end=CONFIG['end_date'], progress=True)

# Handle multi-level columns if present
if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(1, axis=1)

# Clean data
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.dropna()

print(f"\n✓ Data downloaded: {len(df)} rows")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {list(df.columns)}")

# Display memory usage
memory_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage: {memory_mb:.2f} MB")

# %% [markdown]
# ## Step 5: Feature Engineering

# %%
from tqdm import tqdm

def create_features(df, memory_safe=True):
    """Create ML features optimized for memory usage."""
    print("Creating features...")

    features = pd.DataFrame(index=df.index)

    # Price features
    print("  - Price features...")
    features['returns_1d'] = df['Close'].pct_change(1)
    features['returns_5d'] = df['Close'].pct_change(5)
    features['returns_10d'] = df['Close'].pct_change(10)
    features['ma_ratio_20'] = df['Close'] / df['Close'].rolling(20).mean()
    features['ma_ratio_50'] = df['Close'] / df['Close'].rolling(50).mean()
    features['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                                  (df['High'].rolling(20).max() - df['Low'].rolling(20).min() + 1e-9)

    # Momentum features
    print("  - Momentum features...")
    # RSI
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()

    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    features['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-9)

    # Volatility features
    print("  - Volatility features...")
    features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
    features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
    features['atr_14'] = (df['High'] - df['Low']).rolling(14).mean()

    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    features['bb_upper'] = (sma_20 + 2 * std_20) / df['Close']
    features['bb_lower'] = (sma_20 - 2 * std_20) / df['Close']
    features['bb_width'] = (4 * std_20) / df['Close']

    # Volume features
    print("  - Volume features...")
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    features['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Pattern features
    print("  - Pattern features...")
    features['body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
    features['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-9)
    features['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-9)

    # Doji pattern
    features['is_doji'] = (features['body_ratio'] < 0.1).astype(int)

    # Engulfing pattern
    features['bullish_engulfing'] = ((df['Close'] > df['Open']).rolling(2).apply(
        lambda x: x.iloc[0] == False and x.iloc[1] == True and
                  df['Close'].iloc[1] > df['Open'].iloc[0] and
                  df['Close'].iloc[0] < df['Open'].iloc[1]
    )).astype(int)

    # Create label (future returns)
    print("  - Creating labels...")
    horizon = CONFIG['prediction_horizon']
    features['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    features['label'] = (features['future_return'] > 0).astype(int)

    # Drop NA values
    features = features.dropna()

    print(f"✓ Created {len(features.columns)} features")
    return features

# Generate features
features_df = create_features(df, memory_safe=CONFIG['memory_safe'])
print(f"\nFeature matrix shape: {features_df.shape}")

# %% [markdown]
# ## Step 6: Prepare Training Data

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc

# Drop rows with infinite values
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.dropna()

# Separate features and target
feature_columns = [col for col in features_df.columns if col not in ['future_return', 'label']]
X = features_df[feature_columns].values
y = features_df['label'].values

print(f"Feature matrix: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Class balance: {np.mean(y):.2%} positive")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Clear memory
del X, y
gc.collect()

print("✓ Data preparation complete")

# %% [markdown]
# ## Step 7: Train ML Model

# %%
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Training {CONFIG['model_type']} model...")
print(f"Training samples: {len(X_train_scaled)}")
print(f"Features: {X_train_scaled.shape[1]}")

# Memory-safe LightGBM parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,          # Limited for memory
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'force_col_wise': True,    # Memory optimization
    'max_bin': 255,            # Limited bins for memory
    'n_jobs': -1,
    'random_state': CONFIG['random_state']
}

# Calculate scale_pos_weight for imbalanced classes
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
lgb_params['scale_pos_weight'] = scale_pos_weight
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Create datasets
train_data = lgb.Dataset(X_train_scaled, label=y_train)
test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

# Train model
model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print(f"\n✓ Model trained with {model.best_iteration} iterations")
print(f"Best iteration AUC: {model.best_score['valid_0']['auc']:.4f}")

# %% [markdown]
# ## Step 8: Evaluate Model Performance

# %%
# Predictions
y_pred_proba = model.predict(X_test_scaled, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"\nAUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=150)
plt.show()

# Feature importance
print("\n" + "=" * 60)
print("TOP 20 FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importance(importance_type='gain')
})
importance = importance.sort_values('importance', ascending=False)

print(importance.head(20).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 12))
plt.barh(range(20), importance['importance'].head(20).values[::-1])
plt.yticks(range(20), importance['feature'].head(20).values[::-1])
plt.xlabel('Importance (Gain)')
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.savefig('/content/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 9: Save Model and Artifacts

# %%
import pickle
import json
from datetime import datetime

# Create timestamp for filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save LightGBM model
model_path = f'/content/pattern_classifier_{timestamp}.pkl'
model.save_model(model_path)
print(f"✓ Model saved: {model_path}")

# Save scaler
scaler_path = f'/content/scaler_{timestamp}.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_path}")

# Save metadata
metadata = {
    'symbol': CONFIG['symbol'],
    'start_date': CONFIG['start_date'],
    'end_date': CONFIG['end_date'],
    'model_type': CONFIG['model_type'],
    'prediction_horizon': CONFIG['prediction_horizon'],
    'n_features': len(feature_columns),
    'feature_columns': feature_columns,
    'train_samples': len(X_train_scaled),
    'test_samples': len(X_test_scaled),
    'auc_score': float(roc_auc_score(y_test, y_pred_proba)),
    'best_iteration': model.best_iteration,
    'timestamp': timestamp
}

metadata_path = f'/content/metadata_{timestamp}.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

# Copy to Google Drive
drive_model_path = f"/content/drive/MyDrive/ML_Models/pattern_classifier_{timestamp}.pkl"
drive_scaler_path = f"/content/drive/MyDrive/ML_Models/scaler_{timestamp}.pkl"
drive_metadata_path = f"/content/drive/MyDrive/ML_Models/metadata_{timestamp}.json"

import shutil
shutil.copy(model_path, drive_model_path)
shutil.copy(scaler_path, drive_scaler_path)
shutil.copy(metadata_path, drive_metadata_path)

print(f"\n✓ All files backed up to Google Drive!")
print(f"Google Drive folder: /content/drive/MyDrive/ML_Models/")

# %% [markdown]
# ## Step 10: Download Files

# %%
from google.colab import files
import zipfile

# Create zip file
zip_name = f'ml_model_{CONFIG["symbol"]}_{timestamp}.zip'
with zipfile.ZipFile(zip_name, 'w') as zipf:
    zipf.write(model_path)
    zipf.write(scaler_path)
    zipf.write(metadata_path)
    zipf.write('/content/confusion_matrix.png')
    zipf.write('/content/feature_importance.png')

print(f"Created: {zip_name}")
print(f"\nDownloading ZIP file...")
files.download(zip_name)

print(f"\nYou can also find files in your Google Drive:")
print(f"  - Model: ML_Models/pattern_classifier_{timestamp}.pkl")
print(f"  - Scaler: ML_Models/scaler_{timestamp}.pkl")
print(f"  - Metadata: ML_Models/metadata_{timestamp}.json")

# %% [markdown]
# ## Step 11: How to Use Trained Model Locally

# %%
print("""
USAGE INSTRUCTIONS:
===================

1. Download the ZIP file (auto-downloaded above)
   OR copy from Google Drive: ML_Models/ folder

2. Place files in your project's models/ directory:
   models/pattern_classifier_YYYYMMDD_HHMMSS.pkl
   models/scaler_YYYYMMDD_HHMMSS.pkl
   models/metadata_YYYYMMDD_HHMMSS.json

3. Use the model in your local code:

```python
from src.ml.pattern_classifier import PatternClassifier
from src.ml.feature_engineering import FeatureExtractor
import pickle

# Load model
classifier = PatternClassifier()
classifier.load('models/pattern_classifier_20240421_120000.pkl')

# Load scaler
with open('models/scaler_20240421_120000.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Extract features from your data
extractor = FeatureExtractor()
features = extractor.extract_all_features(df)

# Scale features
features_scaled = scaler.transform(features)

# Make predictions
predictions = classifier.predict_proba(features_scaled)
```

4. Run ML-enhanced backtest:
```bash
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --model-path models/pattern_classifier_20240421_120000.pkl
```

""")
