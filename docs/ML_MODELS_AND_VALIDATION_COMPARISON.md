# ML Models & Validation Methods: Complete Comparison Guide

## Overview

This guide explains the different machine learning models and validation techniques available in this trading system, and when to use each combination.

---

## Part 1: Model Types

### 1.1 Gradient Boosting Models (GBMs)

#### **LightGBM**
**How it works:** Builds decision trees sequentially, each tree correcting errors of previous trees. Uses histogram-based algorithms for speed.

**Key Characteristics:**
- Fast training and inference
- Handles large datasets efficiently
- Good with categorical features
- Lower memory usage than XGBoost
- Can handle missing values natively

**When to use:**
- Large datasets (100K+ samples)
- Need fast training cycles
- Features include categorical data
- Trading on multiple symbols simultaneously

**Best for:** Pattern classification with many features, high-frequency data

**Hyperparameters that matter:**
- `learning_rate` (0.01-0.1): Controls contribution of each tree
- `n_estimators` (100-500): Number of trees
- `max_depth` (3-8): Prevents overfitting
- `num_leaves` (31-127): Controls tree complexity

**Pros:**
- Fastest training speed
- Excellent accuracy
- Handles missing data automatically
- Good GPU support

**Cons:**
- Can overfit on small datasets
- Requires careful hyperparameter tuning
- Less interpretable than single decision trees

---

#### **XGBoost**
**How it works:** Similar to LightGBM but uses different tree-splitting algorithms (pre-sorted algorithm vs histogram). More mature, widely battle-tested.

**Key Characteristics:**
- Robust and battle-tested
- Excellent regularization options
- Slightly slower than LightGBM
- Very reliable performance

**When to use:**
- Need proven, stable performance
- Smaller datasets (<100K samples)
- Want strong regularization controls
- Production-critical models

**Best for:** Regime classification, feature importance analysis

**Hyperparameters that matter:**
- `eta` / `learning_rate` (0.01-0.3): Step size shrinkage
- `max_depth` (3-10): Tree depth limit
- `subsample` (0.5-1.0): Row sampling for randomness
- `colsample_bytree` (0.5-1.0): Feature sampling

**Pros:**
- Best-in-class performance on tabular data
- Strong regularization to prevent overfitting
- Excellent documentation and community
- Very stable predictions

**Cons:**
- Slower training than LightGBM
- Higher memory usage
- Can still overfit if not regularized

---

### 1.2 Ensemble Tree Models

#### **Random Forest**
**How it works:** Builds many decision trees independently using random feature subsets, then averages their predictions (bagging).

**Key Characteristics:**
- Trains trees in parallel (embarrassingly parallel)
- Very robust and hard to overfit
- Provides natural uncertainty estimates
- Good out-of-the-box performance

**When to use:**
- Want model that won't overfit
- Need uncertainty estimates
- Interpretable feature importance needed
- Quick prototyping and baseline

**Best for:** Initial model development, feature selection, regime detection

**Hyperparameters that matter:**
- `n_estimators` (100-500): Number of trees
- `max_depth` (3-15): Individual tree depth
- `min_samples_leaf` (5-20): Leaf node size
- `max_features` ("sqrt" or 0.3-0.7): Features per split

**Pros:**
- Very resistant to overfitting
- Good parallelization
- Natural feature importance
- Works well out-of-the-box

**Cons:**
- Less accurate than boosting methods
- Larger model size (many trees)
- Slower inference than single tree
- Can't learn complex patterns as deeply

---

#### **Gradient Boosting (sklearn)**
**How it works:** Sklearn's implementation of gradient boosting. Less optimized than XGBoost/LightGBM but simpler API.

**Key Characteristics:**
- Pure Python, no external dependencies
- Good for small datasets
- Easier to debug
- Slower than specialized GBMs

**When to use:**
- Want to avoid extra dependencies
- Small datasets (<10K samples)
- Debugging model behavior
- Educational purposes

**Best for:** Quick experiments, educational demos, small-scale trading

**Pros:**
- No extra dependencies
- Easy to understand
- Good sklearn integration
- Reliable on small data

**Cons:**
- Much slower than XGBoost/LightGBM
- Less feature-rich
- Worse performance on large data

---

### 1.3 Linear Models

#### **Logistic Regression**
**How it works:** Linear model with sigmoid function. Predicts probability of binary outcome.

**Key Characteristics:**
- Simple, interpretable linear relationships
- Very fast training and inference
- Assumes linear decision boundaries
- Provides calibrated probabilities

**When to use:**
- Features have linear relationships with target
- Need interpretable model
- Very fast predictions needed
- Baseline comparison

**Best for:** Simple regime detection, baseline models, feature filtering

**Hyperparameters that matter:**
- `C` (0.01-10): Inverse regularization strength
- `penalty` ("l1", "l2", "elasticnet"): Regularization type
- `max_iter` (100-1000): Optimization iterations

**Pros:**
- Extremely fast
- Highly interpretable (coefficients)
- Works well with few features
- Good calibrated probabilities

**Cons:**
- Can't capture non-linear patterns
- Poor performance on complex relationships
- Requires feature engineering for non-linearity

---

## Part 2: Validation Methods

### 2.1 Train/Test Split

**How it works:** Split data into two parts: first portion for training, last portion for testing.

**Example:**
```python
split_idx = int(len(data) * 0.7)
train = data[:split_idx]
test = data[split_idx:]
```

**When to use:**
- Quick model evaluation
- Large datasets (>10K samples)
- Baseline performance check
- Simple scenarios

**Pros:**
- Simple and fast
- Single performance number
- Easy to understand

**Cons:**
- Only one split, may not be representative
- No cross-validation statistics
- Can be unlucky split

**Trading Context:**
- Good for: Quick backtest of new features
- Bad for: Final model selection, overfitting detection

**Example for your project:**
```python
from sklearn.model_selection import train_test_split

# Time-aware split (no shuffle!)
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

---

### 2.2 Walk-Forward Validation

**How it works:** Expanding window training. Train on period 1, test on period 2. Then train on periods 1+2, test on period 3. Repeat.

**Visualization:**
```
Train: [====] Test: [||]
Train: [=======] Test: [||]
Train: [==========] Test: [||]
Train: [=============] Test: [||]
```

**Parameters:**
- `train_size`: Initial training window (e.g., 500 bars)
- `step_size`: How much to expand each step (e.g., 100 bars)

**When to use:**
- Time-series data (trading data!)
- Simulating real deployment
- Detecting overfitting over time
- Final model selection

**Pros:**
- Respects temporal structure
- Simulates real trading scenario
- Multiple performance metrics
- Detects performance decay

**Cons:**
- More compute-intensive
- Need sufficient data
- More complex to implement

**Trading Context:**
- **Essential** for any trading model
- Prevents look-ahead bias
- Shows how model performs as market conditions change

**Example for your project:**
```python
def walk_forward(X, y, train_size=500, step_size=100):
    results = []
    for start in range(0, len(X) - train_size, step_size):
        train_end = start + train_size
        test_start = train_end
        test_end = min(test_start + step_size, len(X))

        X_train, y_train = X.iloc[start:train_end], y.iloc[start:train_end]
        X_test, y_test = X.iloc[test_start:test:test_end], y.iloc[test_start:test_end]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results.append(score)

    return results
```

**Key advantage in trading:** If walk-forward scores decline over time, your model is overfitting to past market conditions.

---

### 2.3 Purged K-Fold Cross-Validation

**How it works:** K-Fold CV but with temporal splits and gaps between train and test to prevent look-ahead bias.

**Visualization:**
```
Fold 1: Train: [====] [GAP] Test: [||]
Fold 2: Train: [========] [GAP] Test: [||]
Fold 3: Train: [============] [GAP] Test: [||]
```

**Parameters:**
- `n_splits`: Number of folds (typically 5)
- `test_size`: Number of test samples per fold
- `gap`: Number of samples between train and test (e.g., 20 bars = 1 month)

**Why "Purged"?**
Traditional K-Fold uses random splits, which cause look-ahead bias in time series. Purged K-Fold:
1. Uses temporal (time-based) splits only
2. Adds a gap between train and test
3. Ensures no data leakage

**When to use:**
- Limited training data
- Need robust performance estimate
- Multiple market regimes in data
- Compare model architectures

**Pros:**
- Most robust time-series validation
- Multiple performance measurements
- Detects overfitting better than single split
- Accounts for different market conditions

**Cons:**
- Most compute-intensive
- Complex to implement correctly
- Requires careful parameter selection
- May still have some look-ahead if gap too small

**Trading Context:**
- **Gold standard** for trading model validation
- Essential when data is limited (<5K samples)
- Gap parameter critical (prevent short-term autocorrelation)

**Example for your project:**
```python
from src.ml.purged_cv import PurgedKFold

purged_cv = PurgedKFold(
    n_splits=5,
    test_size=100,  # ~5 months of daily data
    gap=20,  # ~1 month gap to prevent look-ahead
)

scores = []
for train_idx, test_idx in purged_cv.split(X):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

**Gap selection for trading:**
- Daily data: `gap=20` (1 month)
- Hourly data: `gap=120` (5 days)
- Minute data: `gap=60` (1 hour)

---

## Part 3: Model + Validation Combinations

### Combination Matrix

| Model | Train/Test | Walk-Forward | Purged K-Fold | Best Use Case |
|-------|------------|---------------|----------------|----------------|
| **LightGBM** | ✅ Quick | ✅ Recommended | ⚠️ Overkill | Large datasets, production |
| **XGBoost** | ✅ Good | ✅ Recommended | ✅ Excellent | Production, critical models |
| **Random Forest** | ✅ Perfect | ✅ Good | ✅ Good | Baseline, feature selection |
| **Logistic Regression** | ✅ Perfect | ✅ Good | ✅ Overkill | Simple baselines, interpretability |

**Legend:**
- ✅ = Good fit
- ⚠️ = Not worth the compute

---

### Recommended Combinations for Your Project

#### **Scenario 1: Initial Development**
```
Model: Random Forest
Validation: Train/Test Split (80/20)
Why: Fast iteration, won't overfit, easy to debug
```

#### **Scenario 2: Pattern Classification (Large Data)**
```
Model: LightGBM
Validation: Walk-Forward (train=500, step=100)
Why: Best performance, respects temporal structure
```

#### **Scenario 3: Regime Classification (Small Data)**
```
Model: XGBoost
Validation: Purged K-Fold (n_splits=5, test_size=100, gap=20)
Why: Robust with limited data, prevents overfitting
```

#### **Scenario 4: Feature Selection**
```
Model: Random Forest
Validation: Walk-Forward
Why: Natural feature importance, robust evaluation
```

#### **Scenario 5: Production Deployment**
```
Model: XGBoost or LightGBM
Validation: Purged K-Fold + Walk-Forward
Why: Most thorough validation before live trading
```

---

## Part 4: Practical Examples

### Example 1: Training Pattern Classifier

```python
from scripts.train_ml_model import load_spy_data, extract_features
from src.ml.pattern_classifier import PatternClassifier

# Load data
df = load_spy_data("SPY", "2015-01-01", "2024-12-31")
features, _ = extract_features(df)

# Generate labels
labels = (df["Close"].shift(-5) / df["Close"] - 1 > 0).astype(int)

# Prepare data
X, y = prepare_training_data(df, features, labels)

# Train with walk-forward
classifier = PatternClassifier(
    model_type="lightgbm",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
)

wf_results = classifier.walk_forward_validation(
    X=X,
    y=y,
    train_size=500,
    step_size=100,
)

print(f"Average test AUC: {np.mean(wf_results['test_auc']):.4f}")
print(f"AUC std: {np.std(wf_results['test_auc']):.4f}")
```

### Example 2: Training Regime Classifier

```python
from src.ml.regime_model import RegimeClassifier
from src.indicators.regime_detector import RegimeDetector
from src.ml.features import FeatureEngineer

# Generate features and rule-based labels
engine = FeatureEngineer()
detector = RegimeDetector()

features = engine.extract_all_features(df)
rule_based_regimes = detector.detect_regimes(df)

# Train ML classifier
classifier = RegimeClassifier(
    model_type="xgboost",
    n_estimators=100,
    max_depth=5,
)

results = classifier.train(features, rule_based_regimes, test_size=0.2)

print(f"Train accuracy: {results['train_accuracy']:.4f}")
print(f"Test accuracy: {results['test_accuracy']:.4f}")

# Walk-forward to check overfitting
wf_results = classifier.walk_forward_validation(
    X=features,
    y=rule_based_regimes,
    train_size=500,
    step_size=100,
)

print(f"Mean test score: {np.mean(wf_results['test_score']):.4f}")
print(f"Score std: {np.std(wf_results['test_score']):.4f}")
```

### Example 3: Purged K-Fold for Robust Validation

```python
from src.ml.purged_cv import PurgedKFold
from src.ml.pattern_classifier import PatternClassifier

# Setup purged CV
purged_cv = PurgedKFold(
    n_splits=5,
    test_size=100,  # ~5 months
    gap=20,  # 1 month gap
)

# Train and validate across folds
fold_scores = []
feature_importances = []

for fold_idx, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
    print(f"Training fold {fold_idx + 1}/5...")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Train model
    classifier = PatternClassifier(model_type="xgboost")
    classifier.fit(X_train, y_train)

    # Evaluate
    score = classifier.score(X_test, y_test)
    fold_scores.append(score)

    # Store feature importance
    importance = classifier.get_feature_importance(top_n=10)
    feature_importances.append(importance)

print(f"\nMean AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

# Check overfitting
train_mean = np.mean([r['train_auc'] for r in classifier.train_results])
test_mean = np.mean(fold_scores)
overfit_gap = train_mean - test_mean

print(f"Overfit gap: {overfit_gap:.4f}")
if overfit_gap > 0.1:
    print("⚠️  Warning: Model may be overfitting!")
```

---

## Part 5: Common Pitfalls & Solutions

### Pitfall 1: Using Random Split on Time Series
```python
# ❌ WRONG - Causes look-ahead bias
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ CORRECT - Time-aware split
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

### Pitfall 2: No Gap Between Train and Test
```python
# ❌ WRONG - Autocorrelation leakage
test_start = train_end

# ✅ CORRECT - Add temporal gap
test_start = train_end + gap  # gap=20 for daily data
```

### Pitfall 3: Using Wrong Model for Data Size
```python
# ❌ WRONG - XGBoost on tiny dataset
if len(X) < 1000:
    model = XGBoost()  # Will overfit

# ✅ CORRECT - Simpler model
if len(X) < 1000:
    model = LogisticRegression()
else:
    model = LightGBM()
```

### Pitfall 4: Not Checking Overfitting
```python
# ❌ WRONG - Only check test score
test_score = model.score(X_test, y_test)
print(f"Test score: {test_score}")

# ✅ CORRECT - Compare train vs test
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
overfit = train_score - test_score
print(f"Train: {train_score:.4f}, Test: {test_score:.4f}, Gap: {overfit:.4f}")
```

---

## Part 6: Quick Reference Cheat Sheet

### Model Selection Flowchart

```
Is data linear?
├─ Yes → Logistic Regression
└─ No → Is data large (>50K samples)?
    ├─ Yes → Need speed?
    │   ├─ Yes → LightGBM
    │   └─ No → XGBoost
    └─ No → Need robustness?
        ├─ Yes → Random Forest
        └─ No → Gradient Boosting (sklearn)
```

### Validation Selection Flowchart

```
Is data limited (<5K samples)?
├─ Yes → Purged K-Fold (n_splits=5, gap=20)
└─ No → Simulating live trading?
    ├─ Yes → Walk-Forward (train=500, step=100)
    └─ No → Quick check?
        ├─ Yes → Train/Test Split (80/20)
        └─ No → Purged K-Fold + Walk-Forward
```

### Command Examples

```bash
# Quick baseline
uv run scripts/train_ml_model.py --symbol SPY --model-type random_forest

# Production model
uv run scripts/train_ml_model.py --symbol SPY --model-type lightgbm --horizon 5

# Robust validation with small data
uv run scripts/train_ml_model.py --symbol SPY --model-type xgboost --start 2020-01-01
```

---

## Summary

| Decision Factor | Recommendation |
|----------------|----------------|
| **Large dataset, production** | LightGBM + Walk-Forward |
| **Small dataset, robustness** | XGBoost + Purged K-Fold |
| **Feature selection** | Random Forest + Walk-Forward |
| **Quick baseline** | Random Forest + Train/Test |
| **Limited data** | XGBoost + Purged K-Fold |
| **Interpretability needed** | Logistic Regression + Train/Test |

**Remember:** For trading models, **always** use time-aware validation (Walk-Forward or Purged K-Fold) to prevent look-ahead bias. Random splits will give overly optimistic results!
