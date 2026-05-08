# ML Training Stack — Beginner's Guide

> **Why this guide exists:** Machine learning in trading is complex. This document explains every component of our ML training stack in plain language — what it does, where it lives, when to use it, and how the pieces fit together. No PhD required.

---

## The Big Picture

```
OHLCV Data → Features → Train Model → Evaluate → Backtest → Deploy
                                    ↑
                            Tune Hyperparameters
                                    ↑
                          Explain Predictions
```

Every trade begins with raw price data (Open, High, Low, Close, Volume). The ML stack transforms this data through stages, ending with a model that can tell you whether a price pattern is likely to be profitable.

---

## 1. Core Components — What They Do

### RegimeClassifier (`src/ml/regime_model.py`)
**What:** Detects which "market regime" we're in — Trending, Ranging, Volatile, or Transition.

**When to use it:**
- Before entering any trade — different patterns work in different regimes
- To filter strategies: only trade breakouts in Trending regimes, reversals in Ranging regimes

**How it works:**
1. Takes features (RSI, volatility, moving averages, etc.)
2. Trains a CatBoost / LightGBM / Random Forest classifier (default: CatBoost)
3. Outputs a regime label for each bar

**Key methods:**
- `train(X, y)` — train on labeled data
- `predict(X)` — get regime predictions
- `train_with_purged_cv(X, y)` — proper time-series cross-validation (no data leaks)

```bash
# Train a regime classifier
uv run python -c "
from src.ml.regime_model import RegimeClassifier
# ... load features and labels
classifier = RegimeClassifier(model_type='catboost')
result = classifier.train(features, labels)
print(f'Test accuracy: {result[\"test_accuracy\"]:.2%}')
"
```

---

### PatternClassifier (`src/ml/pattern_classifier.py`)
**What:** Predicts whether a detected chart pattern (e.g., Head & Shoulders, Double Bottom) will actually be profitable.

**When to use it:**
- After pattern detection: should I trade this Head & Shoulders?
- As a filter: discard low-probability patterns before entering

**How it works:**
1. Uses CatBoost (a powerful gradient boosting library)
2. Takes features about the pattern (size, volume context, preceding trend)
3. Outputs a probability (0.0–1.0) that the pattern will succeed

**Key methods:**
- `train(X, y)` — train classifier
- `predict(X)` — score patterns with probability
- `walk_forward_validation(X, y)` — test without leaking future data

```bash
# Train a pattern classifier
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost --horizon 5
```

---

### EBMRegimeClassifier (`src/ml/ebm_classifier.py`)
**What:** A glassbox version of RegimeClassifier using Explainable Boosting Machines. Unlike black-box models, EBM tells you exactly WHY it made each prediction.

**When to use it:**
- When you need to explain decisions (audit trail, understanding strategy)
- When you're learning ML — the shape functions show exactly how each feature affects predictions

**Key difference from RegimeClassifier:**
- EBM produces per-feature contribution graphs ("shape functions")
- You can decompose any prediction: prediction = intercept + sum(feature_contributions)
- No surrogate explanations needed — it's naturally glassbox

**Key methods (in addition to standard ones):**
- `explain_local(X)` — decompose a single prediction
- `explain_global()` — get all feature shape functions
- `plot_all_shape_functions()` — visual graph of every feature's effect
- `summary()` — human-readable model overview

```bash
# Train a glassbox EBM
uv run python -c "
from src.ml.ebm_classifier import EBMRegimeClassifier
ebm = EBMRegimeClassifier()
result = ebm.train(features, labels)
print(ebm.summary())
"
```

---

### SignalScorer / SignalRegressor (`src/ml/signal_scorer.py`)
**What:** Predicts the quality of a trading signal — how much return to expect.

**When to use it:**
- After a pattern passes the classifier: is the signal strong or weak?
- To rank signals: trade only the top-scoring ones

**How it works:**
1. Trains a regression model (GB, RF, CatBoost)
2. Predicts forward returns (continuous value)
3. Evaluates using Rank IC (Information Coefficient) — a proper financial metric

**Key methods:**
- `train(X, y)` — train regressor
- `predict(X)` — get expected returns
- `purged_kfold_validation(X, y)` — proper cross-validation

---

### PatternScorer (`src/ml/pattern_scorer.py`)
**What:** A unified scoring pipeline that wraps PatternClassifier. Adds per-pattern explainability.

**When to use it:**
- In production: score every detected pattern with a single call
- For debugging: understand WHY a pattern was rejected

**Key methods:**
- `train(X, y)` — train the underlying classifier
- `score_detections(X)` — batch score a DataFrame of detected patterns
- `score_single(features, pattern_name)` — score one pattern with explanation

---

### Feature Engineering (`src/ml/feature_engineering.py`, `src/ml/features.py`)
**What:** Converts raw OHLCV data into features the ML models can learn from.

**When to use it:**
- Every time you train a model or make a prediction
- Before any ML model receives input

**What it produces:**
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, ADX
- Price transformations: returns, volatility, moving averages
- Market structure: support/resistance levels, swing points

```python
from src.ml.feature_engineering import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract_all_features(ohlcv_dataframe)
# features now has 50-100 columns of engineered features
```

---

### MLPipeline (`src/ml/pipeline.py`)
**What:** Orchestrates the full ML workflow — feature extraction → regime detection → signal scoring.

**When to use it:**
- For full end-to-end training and inference
- When you want a single interface that does everything

```python
from src.ml.pipeline import MLPipeline
pipeline = MLPipeline()
pipeline.train(ohlcv_data)      # trains everything
signals = pipeline.inference(ohlcv_data)  # one call for full analysis
```

---

## 2. Training Scripts — Which One to Use

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/train_ml_model.py` | Train PatternClassifier | Training a new model from scratch |
| `scripts/run_ml.py` | Auto-select best model | Quick experiment — let the system choose |
| `scripts/tune_model.py` | HP tuning (GWO/GA/WOA) | Optimize hyperparameters after baseline training |
| `scripts/backtest_ml_enhanced.py` | ML-filtered backtest | Compare ML-enhanced vs baseline strategy |
| `scripts/phase_b_ml_enhancement.py` | Full Phase B pipeline | Complete regime + signal training |
| `scripts/ml_validation_exec.py` | Standalone validation | Validate model quality independent of backtest |

### Typical Workflow

```bash
# Step 1: Train a baseline model
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost

# Step 2: Tune hyperparameters with Grey Wolf Optimizer
uv run scripts/tune_model.py --symbol SPY --wolves 20 --iterations 50

# Step 3: Backtest with the ML filter
uv run scripts/backtest_ml_enhanced.py --symbol SPY

# Step 4 (optional): Use glassbox EBM to understand what drives predictions
uv run python -c "
from src.ml.ebm_classifier import EBMRegimeClassifier
ebm = EBMRegimeClassifier()
ebm.train(features, labels)
print(ebm.summary())
ebm.plot_all_shape_functions(top_n=6)
"
```

---

## 3. Model Architecture Decision Tree

```
What is your goal?
│
├── Classify market regime (Trending/Ranging/Volatile)?
│   ├── Need explainability? → EBMRegimeClassifier
│   └── Need max accuracy? → RegimeClassifier (model_type='catboost') ← DEFAULT
│
├── Score chart pattern profitability?
│   ├── Need explainability? → PatternScorer
│   └── Need max accuracy? → PatternClassifier (model_type='catboost') ← DEFAULT
│
├── Predict signal quality (expected return)?
│   └── SignalRegressor (model_type='catboost') ← DEFAULT
│
├── Optimize hyperparameters?
│   ├── CatBoost params → GWOTuner (Grey Wolf Optimizer)
│   ├── Number of market regimes → GARegimeOptimizer (Genetic Algorithm)
│   └── Per-pattern thresholds → WOATuner (Whale Optimization)
│
├── Select features automatically?
│   └── AROFeatureSelector (Artificial Rabbit Optimization)
│
├── Understand model decisions?
│   ├── For CatBoost/RF/GB → SHAP (already integrated in RegimeClassifier)
│   └── For native glassbox → EBMRegimeClassifier
│
├── Discover new alpha signals?
│   └── EBM shape function analysis (see Section 3b)
│
└── Run everything automatically?
    └── MLPipeline
```

### 3a. Why CatBoost Everywhere? (Model Unification)

All three classifiers now default to CatBoost. Research from 2024-2025 consistently shows CatBoost outperforming alternatives on financial tabular data:

| Study | Task | CatBoost | LightGBM | XGBoost | Random Forest |
|-------|------|----------|----------|---------|---------------|
| Xiao et al. 2025 | Fraud detection (1.85M records) | **F1: 0.916** | F1: 0.881 | F1: 0.881 | — |
| ArXiv benchmark 2024 | 12 datasets (tuned) | **AUC 0.985** | AUC: 0.982 | AUC: 0.980 | — |
| LinkedIn 2025 | Energy forecasting | **MAE: 1381** | MAE: 1395 | MAE: 1419 | MAE: 1413 |
| CatBoost+LightGBM+LSTM paper | Investment prediction | R²: 0.788 | R²: 0.765 | — | — |
| **3-model stacking** | Investment prediction | **R²: 0.815** | — | — | — |

CatBoost wins because: handles categorical data natively, resists overfitting (ordered boosting), and needs the least tuning to reach peak performance. LightGBM is the runner-up. Random Forest is kept as a sanity-check baseline.

### 3b. EBM as Alpha Research Tool (NEW)

EBM isn't just for explaining predictions — it's a legitimate **alpha discovery engine**. The per-feature shape functions reveal non-linear relationships between indicators and outcomes that traditional methods miss.

**The workflow:**

```
1. Train EBM on profitable vs unprofitable patterns
2. Extract shape functions: "When RSI is 30-40 and volume_ratio > 1.5x, contribution is +8.2%"
3. Convert contribution curves into trading rules
4. Backtest the rules independently
5. If they work → new alpha signal, validated and explainable
```

**Example discovery from shape function:**

```
Shape function for "volume_spike" feature:

  Contribution
      ^
   +5%|              ╱‾‾‾‾‾‾‾‾‾‾
      |            ╱
    0%|──────────╱────────────────
      |        ╱
   -3%|  ╱‾‾‾‾
      └──────────────────────────▶ Volume ratio vs 20-day avg
        0.5x   1.0x   1.5x   2.0x   3.0x

Interpretation:
  Volume < 0.7x normal → -3% (dying interest, avoid)
  Volume at 1.0x → neutral (no edge)
  Volume at 1.5x-2.5x → +5% (confirmation spike, trade)
  Volume > 3.0x → still +5% (cap: diminishing returns after 2.5x)
```

This is a **data-driven, auditable trading rule** — not an opaque model output. You can verify it against market logic ("low volume breakouts fail, high volume confirmations work"), quantify the optimal threshold (1.5x), and build it into your strategy.

**Why this beats traditional alpha research:**
- Traditional: test one indicator at a time, miss interactions
- EBM: automatically discovers non-linear patterns AND pairwise interactions across all features simultaneously

---

## 3c. The "Dream Team" Ensemble (Future)

Research consistently shows that combining CatBoost + LightGBM via stacking yields 2-5% better accuracy than either alone. Key paper findings:

- **CatBoost + LightGBM + LSTM ensemble**: R² of 0.815 vs CatBoost alone at 0.788 (3.4% improvement)
- **Stacking consensus**: "Hybrid ensemble approaches achieved the highest direction classification accuracy on SP500" (Springer Journal of Big Data)
- **Three-model recommendation**: CatBoost (primary) + LightGBM (fast) + EBM (glassbox verification)

This is tracked as task RS2 in the master plan for future implementation.

---

## 4. Key Concepts (Quick Definitions)

| Term | Meaning |
|------|---------|
| **Overfitting** | The model memorized training data instead of learning patterns. Test score ≪ train score. |
| **PurgedKFold** | Cross-validation that prevents data leakage in time series by removing samples near fold boundaries. |
| **Walk-forward validation** | Train on past, predict on future — the only honest way to test time series models. |
| **Information Coefficient (IC)** | Spearman correlation between predicted and actual returns. 0.0 = random, >0.03 = useful. |
| **Silhouette score** | Measures how well clusters are separated. 0 to 1, higher is better. Used to find optimal n_regimes. |
| **AUC** | Area Under ROC Curve. 0.5 = random coin flip, 1.0 = perfect classification. |
| **Rank IC** | Same as IC but uses ranks instead of raw values — more robust to outliers. |
| **Glassbox model** | A model where you can see exactly how it makes decisions (EBM, linear regression). Opposite of black-box (neural networks, gradient boosting). |
| **Shape function** | An EBM concept — a graph showing how each feature's value maps to its effect on the prediction. Answer to "what happens if RSI goes from 30 to 70?" |

---

## 5. Metaheuristic Optimizers — Nature-Inspired Tuning

All four optimizers live in `src/ml/tuning/` and share a common base class.

| Optimizer | Inspiration | What It Tunes |
|-----------|-------------|---------------|
| **GWOTuner** | Grey Wolf hierarchy | CatBoost hyperparameters (learning rate, depth, regularization...) |
| **GARegimeOptimizer** | Genetic evolution | Optimal number of market regimes (n_regimes) |
| **WOATuner** | Whale bubble-net hunting | Per-pattern confidence thresholds |
| **AROFeatureSelector** | Rabbit foraging behavior | Which features to keep (subset selection) |

They all work the same way:
1. Create a population of candidate solutions
2. Evaluate each one with a fitness function (AUC, silhouette score, Sharpe ratio)
3. Iteratively improve the population using nature-inspired rules
4. Return the best solution found

```bash
# All four share the same CLI interface:
uv run scripts/tune_model.py --symbol SPY --algo gwo   # Grey Wolf for CatBoost
uv run scripts/tune_model.py --symbol SPY --algo ga --target regime_discovery  # GA for regimes
uv run scripts/tune_model.py --symbol SPY --algo woa --target pattern_threshold  # WOA for thresholds
```

---

## 6. Data Flow — How Data Moves Through the System

```
[Yahoo Finance]                    [Pattern Detectors]
      │                                   │
      v                                   v
  OHLCV DataFrame              Detected Patterns (timestamp + type)
      │                                   │
      v                                   v
Feature Engineering              Pattern Feature Extraction
  │  (returns, RSI, ATR...)       │  (pattern size, volume at neckline...)
  │                               │
  v                               v
┌──────────┐                ┌──────────────┐
│ Features │                │  Pattern      │
│ (50-100  │                │  Features     │
│ columns) │                │  (15-30 cols) │
└────┬─────┘                └──────┬───────┘
     │                             │
     ├─────────────────────────────┤
     │                             │
     v                             v
 ┌─────────┐               ┌───────────────┐
 │ Regime  │               │  Pattern      │
 │Classifier│              │  Classifier   │
 │ ───────  │              │  ───────────  │
 │ Output:  │              │  Output:      │
 │ Trending │              │  P(profit)    │
 │ Ranging  │              │  0.0 - 1.0    │
 │ Volatile │              └───────┬───────┘
 │ Transit. │                      │
 └────┬─────┘                      │
      │                            │
      v                            v
 ┌──────────────────────────────────┐
 │       Signal Aggregation         │
 │  Regime filter + Pattern score   │
 │  → Final trade decision          │
 └──────────────┬───────────────────┘
                │
                v
          ┌──────────┐
          │ Backtest │
          │ Engine   │
          │ ──────── │
          │ Sharpe   │
          │ Drawdown │
          │ Win Rate │
          └──────────┘
```

---

## 7. File Location Reference

```
src/ml/
├── regime_model.py          # RegimeClassifier — black-box regime detector
├── ebm_classifier.py        # EBMRegimeClassifier — glassbox regime detector
├── pattern_classifier.py    # PatternClassifier — pattern profitability predictor
├── pattern_scorer.py        # PatternScorer — unified scoring + explainability
├── signal_scorer.py         # SignalRegressor — expected return predictor
├── pipeline.py              # MLPipeline — orchestrates everything
├── model_selector.py        # ModelSelector — auto-picks best model
├── feature_engineering.py   # FeatureExtractor — OHLCV → features
├── features.py              # FeatureEngineer — original feature engine
├── feature_selector.py      # FeatureSelector — pick best features
├── purged_cv.py             # PurgedKFold — proper time-series CV
├── metrics.py               # IC computation utilities
├── backtest_bridge.py       # BacktestBridge — ML → backtest connection
├── experiment_logger.py     # ExperimentLogger — track experiments
├── registry.py              # ModelRegistry — version models
├── ensemble_regime.py       # EnsembleRegimeDetector — combine detectors
│
├── tuning/                  # Hyperparameter tuning (metaheuristics)
│   ├── base.py              # Shared base classes
│   ├── gwo_tuner.py         # Grey Wolf Optimizer
│   ├── ga_tuner.py          # Genetic Algorithm
│   ├── woa_tuner.py         # Whale Optimization Algorithm
│   └── aro_selector.py      # Artificial Rabbit Optimization
│
├── models/                  # Model wrappers
│   ├── catboost_wrapper.py  # CatBoost
│   ├── chronos.py           # Amazon Chronos-2
│   ├── fincast.py           # FinCast
│   └── xlstm.py             # Extended LSTM

scripts/
├── train_ml_model.py        # Train PatternClassifier
├── run_ml.py                # Auto-select and train
├── tune_model.py            # HP tuning (GWO/GA/WOA)
├── backtest_ml_enhanced.py  # ML-filtered backtest
├── phase_b_ml_enhancement.py # Full Phase B pipeline
├── ml_validation_exec.py    # Standalone validation
├── ml_validation_ext.py     # Extended validation
├── ml_validation_enhanced.py # Cross-asset validation

docs/
└── ML_TRAINING_GUIDE.md     # This file
```

---

## 8. Common Pitfalls

| Pitfall | Why It's Bad | How We Prevent It |
|---------|-------------|-------------------|
| **Look-ahead bias** | Using future data in training | PurgedKFold, walk-forward validation |
| **Overfitting** | 100% train accuracy, 50% test accuracy | Early stopping, regularization, PurgedKFold |
| **Data leakage** | Train/test samples overlap in time | Purge window between train and test sets |
| **Survivorship bias** | Training only on stocks that still exist | Use adjusted close prices, include delisted |
| **False confidence from AUC** | AUC > 0.90 but Rank IC near 0 | Always compute both metrics |
| **Regime label leakage** | Labels use future price info | `purge_window=5` default on all train() methods |

---

## 9. Quick Start — Your First ML Model

```bash
# 1. Train a CatBoost pattern classifier
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost --horizon 5

# 2. Get a glassbox explanation of regime behavior
uv run python -c "
import yfinance as yf
from src.ml.feature_engineering import FeatureExtractor
from src.ml.ebm_classifier import EBMRegimeClassifier

# Load data
df = yf.download('SPY', start='2022-01-01', end='2024-12-31', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Create features
extractor = FeatureExtractor()
features = extractor.extract_all_features(df).dropna()

# Create labels (Trending if 20d return > 0, else Ranging)
returns = df['Close'].shift(-20) / df['Close'] - 1
labels = (returns > 0.02).astype(int).dropna()
aligned = features.index.intersection(labels.index)
X, y = features.loc[aligned], labels.loc[aligned]

# Train glassbox EBM
ebm = EBMRegimeClassifier(max_rounds=500)
result = ebm.train(X, y)
print(f'Accuracy: {result[\"test_accuracy\"]:.2%}')
print(ebm.summary())
"

# 3. Tune hyperparameters
uv run scripts/tune_model.py --symbol SPY --wolves 15 --iterations 30
```

---

## 10. Next Steps

Once comfortable with the basics:
1. **Tune hyperparameters** with GWO/GA/WOA for better performance
2. **Add InterpretML/EBM** when you need to understand WHY the model predicts what it does
3. **Use AutoGluon** for a performance ceiling (how good could we be?)
4. **Compute DSR/PSR/FDR** for statistical soundness of backtest results
5. **Stack multiple models** (CatBoost + LightGBM + RF) via ensemble methods
