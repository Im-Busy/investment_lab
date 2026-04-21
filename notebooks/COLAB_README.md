# ML Training on Google Colab

## Quick Start

1. **Open the Notebook**
   - Upload `notebooks/ML_Training_Colab.ipynb` to Google Drive
   - OR open directly: File → Upload notebook in Google Colab

2. **Configure Runtime**
   - Click "Runtime" → "Change runtime type"
   - Select "GPU" (recommended: T4)
   - RAM: 12GB (free tier) or 25GB (Colab Pro)

3. **Run All Cells**
   - Click "Runtime" → "Run all"
   - Wait for completion (~5-15 minutes)

4. **Download Model**
   - Model auto-downloads as ZIP at end
   - Also saved to Google Drive: `/ML_Models/`

## Configuration Options

Edit the CONFIG cell to customize:

```python
CONFIG = {
    'symbol': 'SPY',              # Ticker symbol
    'start_date': '2019-01-01',   # Training start (reduce for less RAM)
    'end_date': '2024-12-31',     # Training end
    'model_type': 'lightgbm',     # lightgbm, xgboost, random_forest
    'prediction_horizon': 3,      # Days ahead to predict
    'memory_safe': True,          # Enable memory optimizations
}
```

## Memory Optimization Tips

| Setting | Free Colab (12GB) | Colab Pro (25GB) |
|---------|-------------------|------------------|
| Date Range | 3-5 years | 5-10 years |
| Features | 20-30 | 80+ |
| Walk-Forward | False | True |
| Model Type | LightGBM | Any |

### For Low RAM (<8GB available)
```python
CONFIG = {
    'start_date': '2020-01-01',   # Only 2 years
    'memory_safe': True,
    'use_walk_forward': False,
}
```

## Installed Packages

The notebook auto-installs:
- yfinance (data download)
- lightgbm (default model)
- xgboost (alternative model)
- scikit-learn (preprocessing)
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- tqdm (progress bars)

## Output Files

After training, you'll receive:

| File | Description |
|------|-------------|
| `pattern_classifier_*.pkl` | Trained LightGBM model |
| `scaler_*.pkl` | Feature scaler (StandardScaler) |
| `metadata_*.json` | Model config and feature list |
| `confusion_matrix.png` | Evaluation visualization |
| `feature_importance.png` | Top features plot |

## Using Trained Model Locally

```python
from src.ml.pattern_classifier import PatternClassifier
import pickle

# Load model
classifier = PatternClassifier()
classifier.load('models/pattern_classifier_20240421_120000.pkl')

# Load scaler
with open('models/scaler_20240421_120000.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict
predictions = classifier.predict(scaled_features)
```

## Cost Estimates

| Option | Cost/Month | Training Runs | Best For |
|--------|-----------|---------------|----------|
| Colab Free | $0 | ~10/week | Testing, small models |
| Colab Pro | $10 | Unlimited | Regular training |
| Colab Pro+ | $50 | Unlimited + A100 GPU | Large-scale training |

## Troubleshooting

### "Ran out of memory" error
- Reduce date range: use only 2-3 years
- Set `memory_safe: True`
- Disable walk-forward validation

### "GPU not available" error
- Runtime → Change runtime type → Select GPU
- Free tier GPU may be unavailable during peak hours

### Model download failed
- Check Google Drive: `/ML_Models/` folder
- Files are automatically backed up there

## Next Steps

After training:

1. Download ZIP file or copy from Google Drive
2. Extract to `models/` directory in your project
3. Run ML-enhanced backtest:
   ```bash
   uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --model-path models/pattern_classifier_*.pkl
   ```
