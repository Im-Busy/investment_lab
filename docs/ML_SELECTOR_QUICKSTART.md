# ML Selector Quick Start Guide

## Overview
The ML Selector provides intelligent model selection and evaluation for trading trading strategies through both CLI and web UI.

## Files Created
- `src/ml/model_selector.py` - Core selection logic
- `scripts/run_ml.py` - CLI wrapper
- `scripts/ml_selector_app.py` - Streamlit web UI

## CLI Usage

### Get Recommendation Only
```bash
uv run python scripts/run_ml.py --recommend
```

### Auto-Select and Train Best Model
```bash
uv run python scripts/run_ml.py --auto
```

### Compare All Models
```bash
uv run python scripts/run_ml.py --compare
```

### With Custom Data
```bash
uv run python scripts/run_ml.py --auto --data features.csv
```

### Priority Modes
```bash
uv run python scripts/run_ml.py --auto --priority fast
uv run python scripts/run_ml.py --auto --priority accurate
uv run python scripts/run_ml.py --auto --priority interpretable
```

### Export Results
```bash
uv run python scripts/run_ml.py --auto --export json
uv run python scripts/run_ml.py --compare --export csv
```

### Force Specific Model/Validation
```bash
uv run python scripts/run_ml.py --auto --model lightgbm --validation walk_forward
```

## Web UI Usage

### Start Streamlit App
```bash
uv run streamlit run scripts/ml_selector_app.py
```

### Access
Open browser to: http://localhost:8501

### Features
- **Sidebar Configuration**: Upload data, select task type, priority mode
- **Get Recommendation**: Instant model and validation recommendation
- **Auto-Select Best Model**: Trains all models and picks best
- **Compare All Models**: Side-by-side comparison with charts
- **Export Results**: Download as JSON/CSV
- **Feature Importance**: Interactive charts
- **Overfitting Visualization**: Gap analysis

## Model Types Supported
- **LightGBM** (default): Fast, accurate, gradient boosting
- **XGBoost**: Advanced gradient boosting
- **Random Forest**: Robust ensemble method
- **Gradient Boosting**: Classic boosting
- **Logistic Regression**: Interpretable baseline

## Validation Methods
- **Train/Test Split**: Fast, suitable for non-time-series
- **Walk-Forward**: Time-series appropriate, prevents lookahead bias
- **Purged K-Fold**: Time-series cross-validation

## Example Workflows

### Workflow 1: Quick Classification
```bash
# 1. Get recommendation
uv run python scripts/run_ml.py --recommend

# 2. Train best model
uv run python scripts/run_ml.py --auto

# 3. Export results
uv run python scripts/run_ml.py --auto --export json
```

### Workflow 2: Model Comparison
```bash
# Compare all models with fast priority
uv run python scripts/run_ml.py --compare --priority fast

# Compare with accurate priority and export
uv run python scripts/run_ml.py --compare --priority accurate --export json
```

### Workflow 3: Web UI Exploration
```bash
# Start web UI
uv run streamlit run scripts/ml_selector_app.py

# Then in browser:
# 1. Load sample data or upload CSV/parquet
# 2. Select task type and priority
# 3. Click "Get Recommendation"
# 4. Click "Auto-Select Best Model" to train
# 5. View feature importance charts
# 6. Export results
```

## Integration with Existing Code

### Use ModelSelector in Scripts
```python
from src.ml.model_selector import ModelSelector

selector = ModelSelector()

# Recommendation
rec = selector.recommend(X, y, task="classification", priority="balanced")

# Training
result = selector.train_and_evaluate(X, y, rec.model, rec.validation)

# Comparison
results = selector.compare_models(X, y, task="classification")
```

### Save to Model Registry
```bash
uv run python scripts/run_ml.py --auto --registry --registry-name my_model
```

## Data Format Requirements

### Input Data
- **CSV/Parquet**: Features as columns, target as last column or named 'target'
- **Time Series**: Index should be datetime for walk-forward validation
- **Features**: Numeric values, handle NaNs automatically

### Target Variable
- **Classification**: Binary (0/1) or multiclass (0, 1, 2, ...)
- **Regression**: Continuous numeric values

## Performance Tips

1. **Use `--priority fast`** for quick iterations
2. **Use `--validation train_test`** for non-time-series data
3. **Force specific models** if you know what works best
4. **Use web UI** for interactive exploration and visualization

## Troubleshooting

### Import Errors
```bash
# Ensure dependencies are installed
uv add lightgbm xgboost streamlit plotly seaborn
```

### Memory Issues
```bash
# Use train_test validation instead of walk_forward
uv run python scripts/run_ml.py --auto --validation train_test
```

### Slow Training
```bash
# Use fast priority or force simpler models
uv run python scripts/run_ml.py --auto --priority fast --model logistic_regression
```
