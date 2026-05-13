# Investment Trading System - Command Cheatsheet

> **Auto-Update Instruction:** This document is the single source of truth for all project commands. When implementing new features that include CLI scripts, flags, or workflows, the implementing AI MUST update this file immediately. Add new commands to the appropriate section following the existing format. Commit with message: "docs: update command cheatsheet for [feature-name]".

---

## Quick Reference - Environment & Package Management

### Python Environment (uv)
```bash
# Run Python command in project environment
uv run <command>

# Install package
uv add <package>

# Remove package
uv remove <package>

# Sync environment with pyproject.toml
uv sync

# Run script
uv run scripts/<script>.py

# Run pytest
uv run pytest tests/ -v

# Run linting with ruff
uv run ruff check src/
```

---

## ML Training & Model Selection

### Train ML Models
```bash
# Default training (CatBoost)
uv run scripts/train_ml_model.py --symbol SPY --start 2015-01-01 --end 2024-12-31 --model-type catboost

# Train with specific model type
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost
uv run scripts/train_ml_model.py --symbol SPY --model-type chronos
uv run scripts/train_ml_model.py --symbol SPY --model-type fincast
uv run scripts/train_ml_model.py --symbol SPY --model-type xlstm

# Different prediction horizon
uv run scripts/train_ml_model.py --symbol SPY --horizon 5 --model-type catboost

# Add suffix to model
uv run scripts/train_ml_model.py --symbol SPY --suffix v1 --model-type catboost

# Skip walk-forward validation
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost --no-walk-forward

# Compare all model types
uv run scripts/train_ml_model.py --symbol SPY --compare-models

# Continuous training loop (indefinite iterations)
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost --iterations -1 --sleep 60
```

### ML Selector (Auto-select best model)
```bash
# Get recommendation only
uv run scripts/run_ml.py --recommend

# Auto-select and train
uv run scripts/run_ml.py --auto

# Compare all models
uv run scripts/run_ml.py --compare

# With priority mode
uv run scripts/run_ml.py --auto --priority fast
uv run scripts/run_ml.py --auto --priority accurate

# Export results
uv run scripts/run_ml.py --auto --export json

# Web UI - Streamlit
uv run streamlit run scripts/ml_selector_app.py --server.port=8501

# Web UI - Gradio (recommended)
uv run python scripts/ml_selector_gradio.py
```

### ML Model V3 — Honest Foundation Pipeline (Recommended)

Full 9-stage pipeline: features → triple-barrier labels → IC filter → Stability Selection (>=0.6) → GWO HP tuning → Nested PurgedKFold CV → final model → walk-forward → SHAP + regime analysis.

```bash
# Single ticker, full pipeline with walk-forward
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --walk-forward

# Basket training (5 tickers for better generalization)
uv run scripts/train_ml_pipeline_v3.py --basket JOE,SPY,QQQ,TLT,GLD

# Fast mode (skip stability selection + GWO for quick iterations)
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --fast

# Custom stability threshold (lower = more features, higher = stricter)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --stability-threshold 0.7

# Custom horizon and dates
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --horizon 10 --start 2018-01-01 --end 2024-12-31

# Without cross-asset features (baseline comparison)
uv run scripts/train_ml_pipeline_v3.py --symbol JOE --skip-cross-asset

# Per-sector model (B10): train with only intra-sector features, no cross-asset leakage
# Filters tickers to sector only, names model with sector prefix
uv run scripts/train_ml_pipeline_v3.py --sector tech --fast

# Train all 7 sectors in one run
uv run scripts/train_ml_pipeline_v3.py --sector all --fast
```

### Autonomous Training Loop (Orchestration Layer)

Wraps V3 pipeline + backtest + tuning into an iterative refinement loop with guardrails:
independence clustering, consecutive confirmation, cross-group generalization testing, timeout, and BESTS.md integration.

```bash
# Full autonomous loop on 7 tech tickers with 3 consecutive confirmations required
uv run scripts/autonomous_train_loop.py --tickers "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA" --max-iterations 20 --confirmations 3 --timeout-hours 8

# Run only Phase 4 (refinement loop) with existing model, trailing stop
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ,XLK,XLF" --phase 4 --fast --model models/pattern_classifier_v3_SPY_20260511.pkl --trail-stop

# Full loop with custom horizon, fast mode (skip tuning)
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ,IWM,TLT,GLD" --horizon 10 --trail-stop --entry-threshold 0.45 --fast

# Phase 1 only: ticker independence clustering
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ,XLK,XLF,XLE,XLV,XLI,IWM,TLT,GLD" --phase 1

# Phase 5 only: parameter space sweep with existing model
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ,TLT" --phase 5 --fast --model models/pattern_classifier_v3_SPY.pkl

# Skip Phase 4 (no refinement, just clustering + tuning + cross-group test)
uv run scripts/autonomous_train_loop.py --tickers "AAPL,MSFT,GOOGL,AMZN" --skip-phase-4 --max-corr 0.60
```

Key flags:
| Flag | Default | Purpose |
|------|---------|---------|
| `--tickers` | *required* | Comma-separated ticker symbols |
| `--max-iterations` | 20 | Maximum refinement loop iterations |
| `--confirmations` | 3 | Consecutive improvements needed to lock best |
| `--timeout-hours` | 0 | Max runtime (0 = no limit) |
| `--max-corr` | 0.70 | Max absolute correlation within a group |
| `--phase` | all | Run specific phase(s): 1-5 or all |
| `--fast` | False | Skip ARO+GWO tuning |
| `--model` | "" | Existing model path (use instead of training) |
| `--trail-stop` | False | Enable ATR trailing stop for backtests |
| `--conviction` | False | Scale position size by conviction |
| `--entry-threshold` | 0.50 | ML probability threshold |
| `--label-type` | triple_barrier | Label type: `triple_barrier` (forward horizon) or `next_bar` (zero look-ahead) |
| `--no-trail-stop` | False | Disable trailing stop (fixed TP/SL) |
| `--skip-phase-4` | False | Skip the autonomous refinement loop |
| `--resume` | False | Resume Phase 4 from last checkpoint (crash/timeout recovery) |
| `--optuna-trials` | 30 | Number of Optuna trials for Phase 5 Bayesian sweep (0 = grid fallback) |
| `--pareto` | False | Use multi-objective Pareto optimization (Sharpe + MaxDD + WinRate) |

### Loop Hardening Features (Phase 10b — 2026-05-13)

```bash
# Checkpoint + resume: crash-proof long-running loops
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 4 --fast --max-iterations 20 --trail-stop
# Ctrl+C mid-run, then resume from checkpoint:
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 4 --resume --fast --max-iterations 20 --trail-stop

# Pareto multi-objective optimization (Sharpe + MaxDD + WinRate frontier)
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 5 --fast --model models/pattern_classifier_v3_SPY.pkl --pareto --optuna-trials 30

# Next-bar-direction labels (zero look-ahead baseline vs triple-barrier)
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 4 --fast --max-iterations 1 --trail-stop --label-type next_bar

# ETF-component auto-exclusion (automatic — no CLI flag needed)
# Detects SPY+MSFT, QQQ+AAPL pairs in Phase 1 and removes leaky cross-asset features
uv run scripts/autonomous_train_loop.py --tickers "SPY,MSFT,QQQ,AAPL" --phase 1
```

### ML Model V2 (Overfitting-Fixed with Cross-Asset Features) — Legacy
```bash
# Train with cross-asset features (default)
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv --horizon 5 --suffix with_ca_features

# Train without cross-asset (baseline comparison)
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv --horizon 5 --suffix baseline_no_ca --no-cross-asset

# Skip IC filtering (not recommended)
uv run scripts/train_ml_model_v2.py --symbol data/raw/SPY_daily.csv --horizon 5 --no-ic-filter

# Use thresholded binary labels instead of triple-barrier
uv run scripts/train_ml_model_v2.py --symbol data/raw/SPY_daily.csv --horizon 5 --no-triple-barrier --threshold 0.02

# Train all 6 instruments (experiment batch)
for sym in CRVL KODK HIFS JOE SPY QQQ; do
    uv run scripts/train_ml_model_v2.py --symbol data/raw/${sym}_daily.csv --horizon 5 --suffix v3_ca
done
```

### Download Cross-Asset Market Data
```bash
# Download missing market index data (IWM, XLF, XLE, XLK, XLV, EEM)
uv run python -c "
import yfinance as yf
for sym in ['IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'EEM']:
    df = yf.download(sym, start='2015-01-01', end='2025-12-31', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(f'data/raw/{sym}_daily.csv')
    print(f'{sym}: {len(df)} bars')
"
```
```bash
# GWO tune PatternClassifier on SPY
uv run scripts/tune_model.py --symbol SPY --target pattern_classifier --wolves 20 --iterations 50

# GWO tune SignalRegressor
uv run scripts/tune_model.py --symbol SPY --target signal_regressor --wolves 30

# GWO tune RegimeClassifier
uv run scripts/tune_model.py --symbol SPY --target regime --iterations 30

# Custom date range
uv run scripts/tune_model.py --symbol SPY --start 2018-01-01 --end 2024-12-31 --iterations 100
```

### ML Regime Discovery (GA — Genetic Algorithm)
```bash
# GA optimize n_regimes via silhouette score on SPY
uv run scripts/tune_model.py --symbol SPY --target regime_discovery --algo ga --population 20 --iterations 30

# GA with larger population
uv run scripts/tune_model.py --symbol SPY --target regime_discovery --algo ga --population 30 --iterations 50

# GA with custom date range
uv run scripts/tune_model.py --symbol SPY --start 2018-01-01 --end 2024-12-31 --target regime_discovery --algo ga
```

### ML Pattern Threshold Tuning (WOA — Whale Optimization)
```bash
# WOA tune per-pattern confidence thresholds on SPY
uv run scripts/tune_model.py --symbol SPY --target pattern_threshold --algo woa --wolves 30 --iterations 50

# WOA with custom whales count
uv run scripts/tune_model.py --symbol SPY --target pattern_threshold --algo woa --wolves 40 --iterations 80

# WOA with custom date range
uv run scripts/tune_model.py --symbol SPY --start 2018-01-01 --end 2024-12-31 --target pattern_threshold --algo woa
```

### ML Pattern Confidence Scoring
```bash
# Train and score pattern detections (via PatternScorer)
uv run python -c "
from src.ml.pattern_scorer import PatternScorer
import pandas as pd
# ... train and score patterns
"
```

### ML Glassbox Explainability (InterpretML EBM)
```bash
# Train a glassbox EBM regime classifier
uv run python -c "
from src.ml.ebm_classifier import EBMRegimeClassifier
import yfinance as yf
from src.ml.feature_engineering import FeatureExtractor

df = yf.download('SPY', start='2022-01-01', end='2024-12-31', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
extractor = FeatureExtractor()
features = extractor.extract_all_features(df).dropna()
returns = df['Close'].shift(-20) / df['Close'] - 1
labels = (returns > 0.02).astype(int).dropna()
aligned = features.index.intersection(labels.index)
X, y = features.loc[aligned], labels.loc[aligned]

ebm = EBMRegimeClassifier(max_rounds=500)
result = ebm.train(X, y)
print(f'Accuracy: {result[\"test_accuracy\"]:.2%}')
print(ebm.summary())
"

# View per-feature shape functions
uv run python -c "
from src.ml.ebm_classifier import EBMRegimeClassifier
# ... train ebm ...
ebm.plot_all_shape_functions(top_n=6)
"

# Explain a single prediction
uv run python -c "
from src.ml.ebm_classifier import EBMRegimeClassifier
# ... train ebm ...
explanation = ebm.explain_local(X.iloc[0])
for feat, contrib in sorted(explanation['features'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f'{feat}: {contrib:+.4f}')
"
```

### ML AutoML Baseline (AutoGluon)
```bash
# Run AutoML benchmark on SPY (5 min time limit)
uv run scripts/run_automl.py --symbol SPY --time-limit 300

# Compare against hand-tuned CatBoost AUC
uv run scripts/run_automl.py --symbol SPY --time-limit 300 --compare-baseline 0.65

# Best quality preset (more accurate, slower)
uv run scripts/run_automl.py --symbol SPY --presets best_quality --time-limit 600

# Optimize for deployment (small/fast models)
uv run scripts/run_automl.py --symbol SPY --presets optimize_for_deployment

# Python API
uv run python -c "
from src.ml.automl import AutoMLBaseline
automl = AutoMLBaseline(label='target', time_limit=120, presets='medium_quality')
result = automl.fit(X, y)
print(result.summary())
comparison = automl.compare_to_baseline(0.65, 'Hand-Tuned CatBoost')
print(f'AutoML uplift: {comparison[\"uplift\"]:+.4f}')
"
```

### ML Validation
```bash
# ML validation enhanced
uv run scripts/ml_validation_enhanced.py
uv run scripts/ml_validation_exec.py
uv run scripts/ml_validation_ext.py

# Grid search features
uv run scripts/grid_search_features.py
```

---

## Backtesting

### ML-Enhanced Backtesting
```bash
# Default ML-enhanced backtest (EMA strategy recommended — most signals)
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy ema

# With date range
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy ema --start 2020-01-01 --end 2024-12-31

# VWAP strategy (may fail with <100 signals on short periods)
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --start 2015-01-01 --end 2024-12-31

# SMA crossover strategy
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy sma --start 2020-01-01 --end 2024-12-31

# Phase B7 ML backtest
uv run scripts/phase_b7_ml_backtest.py --start 2015-01-01 --end 2024-12-31
uv run scripts/phase_b7_custom_backtest.py --start 2015-01-01 --end 2024-12-31
```

### Strategy-Specific Backtests
```bash
# VWAP bounce strategy
uv run scripts/backtest_vwap_bounce.py

# EMA ribbon strategy
uv run scripts/backtest_ema_ribbon.py

# SMA crossover strategy
uv run scripts/backtest_sma_crossover.py

# Keltner channel strategy
uv run scripts/backtest_keltner_channel.py
```

### Multi-Strategy & Portfolio Backtests
```bash
# Backtest all strategies on SPY
uv run scripts/backtest_all_strategies_spy.py

# Backtest all strategies on multiple symbols
uv run scripts/backtest_all_strategies.py

# Portfolio backtest
uv run scripts/portfolio_backtest.py

# Multi-asset backtest
uv run scripts/test_multi_asset.py

# Pair trading backtests
uv run scripts/run_pair_trading_backtests.py
```

### ML Strategy Backtesting (CatBoost Pattern Classifier V3)

```bash
# Single ticker backtest
uv run scripts/run_ml_backtest.py SPY

# 9-ticker comparison (SO,SPY,D,KO,XLK,PEG,QQQ,AVB,XLV)
uv run scripts/run_ml_backtest.py "SO,SPY,D,KO,XLK,PEG,QQQ,AVB,XLV" --compare

# Full 33-ticker comparison
uv run scripts/run_ml_backtest.py "SO,SPY,D,KO,XLK,PEG,QQQ,AVB,XLV,WMT,UNP,JNJ,PG,AAPL,MSFT,GLD,TLT,AMGN,ICE,PFE,GE,CL,MMM,OXY,FDX,SCHW,ORCL,UL,ALL,CTAS,CAT,KODK" --compare

# With specific model
uv run scripts/run_ml_backtest.py SPY --model models/pattern_classifier_v3_SPY_20260511_224704.pkl

# ── C7 Refinement Options ──

# Trailing stop (BEST: +31% Sharpe improvement)
uv run scripts/run_ml_backtest.py "SO,SPY,D,KO,XLK,PEG,QQQ,AVB,XLV" --compare --trail-stop

# Volatility gate (skip entries when vol_regime > 1.5)
uv run scripts/run_ml_backtest.py SPY --vol-gate 1.5

# Consecutive confirmation (require 2 bars above entry threshold)
uv run scripts/run_ml_backtest.py SPY --confirm 2

# Conviction-based position scaling
uv run scripts/run_ml_backtest.py SPY --conviction

# Combined: trail stop + conviction scaling
uv run scripts/run_ml_backtest.py "SO,SPY,D,KO,XLK,PEG,QQQ,AVB,XLV" --compare --trail-stop --conviction

# Custom entry threshold (default 0.50). Lower = more trades, higher = more selective.
# Optimal for SPY 2016-2024: 0.45
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --entry-threshold 0.45

# ── Parameter Sweeps ──

# Sweep entry thresholds across trail/conviction combos
uv run scripts/sweep_entry_thresholds.py

# Results: BESTS.md (leaderboard)
```

### Benchmarking
```bash
# Benchmark vectorized engine
uv run scripts/benchmark_vectorized_engine.py

# Benchmark phase 2
uv run scripts/benchmark_phase2.py

# Benchmark vectorbt
uv run scripts/benchmark_vectorbt.py

# Benchmark ranking
uv run scripts/benchmark_ranking.py

# Validate vectorized engine
uv run scripts/validate_vectorized_engine.py

# Validate all strategies and signals
uv run scripts/validate_all_strategies_signals.py
```

---

## Pattern Detection

### Pattern Detectors
```bash
# Run all pattern detectors
uv run -c "from src.patterns import scan_patterns; scan_patterns('SPY', '2020-01-01', '2024-12-31')"

# Specific pattern detection (via Python API)
# Scripts for individual patterns being developed
```

### Pattern Testing
```bash
# Test pattern selector
uv run scripts/test_pattern_selector.py

# Optimize passing patterns
uv run scripts/optimize_passing_patterns.py

# Test 23 untested patterns
uv run scripts/backtest_23_untested_patterns.py
```

---

## Feature Extraction

### Qlib Alpha158 Factor Extraction & Evaluation
```bash
# Extract Alpha158 factors from Qlib for single-ticker
uv run scripts/extract_alpha158.py
# Output: data/qlib_alpha158_raw.parquet (2515 rows x 158 features)

# Compare Alpha158 factors vs V3 features (nested PurgedKFold CV)
uv run scripts/compare_alpha158.py
# Output: experiments/alpha158_comparison.json

# VERDICT (May 2026): Alpha158 cross-sectional factors add zero value
# for single-ticker prediction (rank IC 0.1501 vs V3 baseline 0.1556).
# Abandon Qlib — focus on V3 pipeline improvements.
```

### V3 Feature Engineering
```bash
# Extract all V3 features
uv run -c "from src.features import extract_all; extract_all('SPY', output='data/features')"

# Technical indicators
uv run -c "from src.features.technical_indicators import extract; extract('SPY')"

# Feature extraction scripts under development
```

---

## Risk Analysis & Optimization

### Risk Metrics
```bash
# Portfolio risk analysis
# Scripts under development in src/risk/

# Friction scoring test
uv run scripts/test_friction_scoring.py

# Diversity score test
uv run scripts/test_diversity_score.py

# Event weighting test
uv run scripts/test_event_weighting.py
```

### Optimization
```bash
# Optimize MACD parameters
uv run scripts/optimize_macd.py

# Optimize RSI parameters
uv run scripts/optimize_rsi.py

# 5+ parameter optimization
uv run scripts/test_5_plus_optimize.py
```

### Optuna Hyperparameter Optimization (NEW — Phase 10a)
```bash
# Tune CatBoost PatternClassifier with Optuna (TPE sampler, 50 trials, PurgedKFold CV)
uv run scripts/tune_model.py --symbol SPY --target pattern_classifier --algo optuna --trials 50

# Tune LightGBM RegimeClassifier (LGBM-first principle)
uv run scripts/tune_model.py --symbol SPY --target regime --algo optuna --model lgbm --trials 30

# Tune SignalRegressor with pruning (stop unpromising trials early)
uv run scripts/tune_model.py --symbol SPY --target signal_regressor --algo optuna --trials 100 --prune

# Tune strategy parameters (RSI, MACD, etc.) — replaces manual grid search scripts
uv run scripts/tune_model.py --symbol SPY --target strategy_params --algo optuna --strategy rsi --trials 50

# Resume interrupted Optuna study
uv run scripts/tune_model.py --symbol SPY --target pattern_classifier --algo optuna --study-name SPY_pc_20260512 --trials 50

# Compare Optuna vs GWO results
uv run scripts/tune_model.py --symbol SPY --target pattern_classifier --algo compare --trials 30
```

### PyPortfolioOpt Integration (NEW — Phase 10a)
```bash
# Hierarchical Risk Parity (HRP) on 33-ticker basket
# Use when: want risk-parity allocation that respects correlation structure
uv run scripts/portfolio_backtest.py --basket 33t --method hrp

# Mean-variance efficient frontier with max Sharpe
uv run scripts/portfolio_backtest.py --basket 33t --method ef --objective max_sharpe

# CVaR (Conditional Value-at-Risk) optimization for tail-risk management
uv run scripts/portfolio_backtest.py --basket 33t --method cvar --beta 0.95

# Black-Litterman with PyPortfolioOpt posterior + views from ML predictions
uv run scripts/portfolio_backtest.py --basket 33t --method bl --views ml_predictions

# Compare allocation methods: equal-weight vs HRP vs EF vs CVaR
uv run scripts/portfolio_backtest.py --basket 33t --method compare
```

---

## Testing & Debugging

### Unit Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Test specific module
uv run pytest tests/test_backtest.py -v
uv run pytest tests/test_patterns.py -v
uv run pytest tests/test_ml.py -v

# Test with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run tests with detailed output
uv run pytest tests/ -v --tb=long
```

### Integration Tests & Debugging
```bash
# Test ensemble strategies
uv run scripts/test_ensemble_strategies.py

# Test phase 6 tier 1 features
uv run scripts/test_phase6_tier1.py
uv run scripts/test_phase6_tier1_r2r4r5.py

# Reverse signals testing
uv run scripts/test_reverse_signals.py
uv run scripts/test_reverse_signals_fast.py

# Setup test
uv run scripts/test_setup.py

# Debug scripts
uv run scripts/debug_imports.py
uv run scripts/debug_trades.py
uv run scripts/debug_trades_simple.py
uv run scripts/debug_get_trades.py
```

---

## Phase Implementation Scripts

### Phase B7 (ML Enhancement)
```bash
# ML enhancement
uv run scripts/phase_b_ml_enhancement.py

# ML backtest
uv run scripts/phase_b7_ml_backtest.py --start 2015-01-01 --end 2024-12-31

# Custom backtest
uv run scripts/phase_b7_custom_backtest.py --start 2015-01-01 --end 2024-12-31

# Phase 6 tier 1 integration
uv run scripts/phase6_tier1_integration.py
```

---

## Pioneer Research (Phase 6b)

### Pattern Detector Ablation Study
```bash
# Run on SPY (2019-2024)
uv run scripts/ablate_patterns.py --symbol SPY --start 2019-01-01 --end 2024-12-31

# Run on Bitcoin
uv run scripts/ablate_patterns.py --symbol BTC-USD --start 2020-01-01 --end 2024-12-31

# Run on both SPY and BTC
uv run scripts/ablate_patterns.py --all --start 2019-01-01 --end 2024-12-31

# Custom output directory
uv run scripts/ablate_patterns.py --symbol SPY --output-dir reports/pioneer
```

### Meta-Labeling (T9)
```bash
# Run meta-labeler tests
uv run pytest tests/test_meta_labeler.py -v

# Train and evaluate meta-labeler
uv run python -c "
from src.ml.meta_labeler import MetaLabeler
import pandas as pd; import numpy as np
# See MetaLabeler.fit() docstring for usage
"
```

### Gap-Fill Prediction (FS19)
```bash
# Run gap-fill tests
uv run pytest tests/test_gap_fill_predictor.py -v

# Train gap-fill predictor
uv run python -c "
from src.ml.gap_fill_predictor import GapFillPredictor
# See GapFillPredictor.fit() docstring for usage
"
```

---

## Notebooks

### Execute Notebooks
```bash
# Execute and convert notebook
uv run jupyter nbconvert --to notebook --execute notebooks/01_multi_pattern_backtest.ipynb --output 01_multi_pattern_backtest_executed.ipynb

# Run specific notebook
uv run scripts/run_nb_02.py
```

### Key Notebooks
- `01_multi_pattern_backtest.ipynb` - Multi-pattern backtest
- `02_smc_backtest.ipynb` - SMC backtest
- `03_strategy_comparison.ipynb` - Strategy comparison
- `04_pattern_visualization.ipynb` - Pattern visualization
- `05_spy_longterm_backtest.ipynb` - Long-term SPY backtest
- `06_pattern_contribution.ipynb` - Pattern contribution analysis
- `07_pattern_selection_framework.ipynb` - Pattern selection framework
- `08_benchmark_ranking.ipynb` - Benchmark ranking
- `09_multi_timeframe_backtest.ipynb` - Multi-timeframe backtest
- `12_regime_aware_backtest.ipynb` - Regime-aware backtest
- `13_ml_validation.ipynb` - ML validation
- `14_regime_comparison.ipynb` - Regime detector comparison
- `15_phase2_validation.ipynb` - Phase 2 validation
- `16_cross_asset_experiment.ipynb` - H14 cross-asset feature experiment (Start/Stop dashboard)
- `ML_Training_Colab.ipynb` - Google Colab training notebook

---

## Model & Pattern Registry

### Model Selector
```bash
# ML selector app (Streamlit)
uv run streamlit run scripts/ml_selector_app.py

# ML selector app (Gradio)
uv run python scripts/ml_selector_gradio.py

# Export to Hugging Face Spaces
# See .useful_commands/ml_selector_commands.txt for deployment instructions
```

---

## Data Ingestion
```bash
# Data fetching (via Python API)
uv run -c "from src.data_ingestion.fetch_data import fetch_yahoo; fetch_yahoo('SPY', '2020-01-01', '2024-12-31')"
```

---

## Visualization & Analysis

### Contribution Analysis
```bash
# Contribution analysis
uv run scripts/contribution_analysis.py
uv run scripts/contribution_analysis_spy.py
```

### Validation
```bash
# Test SPY backtest
uv run scripts/test_spy_backtest.py

# Simple test
uv run scripts/simple_test.py
```

---

## Paper & Research Tools

### Paper Summarization (paper2md)
```bash
cd useful_resources/useful_repos/paper2md

# Summarize papers
uv run python summarize_md_papers.py --papers-dir ..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md

# Clear cache and re-summarize
uv run python summarize_md_papers.py --papers-dir ..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md --clear-cache

# API Configuration
# Edit: paper2md\.env
# OPENAI_BASE_URL: https://openrouter.ai/api/v1
# OPENAI_MODEL: deepseek/deepseek-v4-flash
```

### AI Research Systems (Reference Only — Linux/Docker Required)
```bash
# RD-Agent — Quant factor/model evolution with Qlib
cd useful_resources/useful_repos/RD-Agent
rdagent fin_factor   # Iterative factor evolution
rdagent fin_model    # Iterative model evolution
rdagent fin_quant    # Factor + model joint evolution
rdagent health_check # Validate setup

# DeepScientist — Local-first research OS
cd useful_resources/useful_repos/DeepScientist
npm install -g @researai/deepscientist
ds --here            # Start research workspace

# Idea2Paper — KG-based paper generation
cd useful_resources/useful_repos/Idea2Paper
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "your idea"

# Architecture analysis reference
# See: useful_resources/papers_md/REPOS_ARCHITECTURE_ANALYSIS.md
```

---

## Code Quality & Pre-commit

### Pre-commit Hooks
```bash
# Run all hooks on all files (ruff linter + formatter, mypy type check, whitespace/merge conflict checks)
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files
uv run pre-commit run mypy --all-files

# Install hooks into .git/hooks/ (auto-runs on git commit)
uv run pre-commit install

# Update hook versions
uv run pre-commit autoupdate
```

**Hooks configured** (`.pre-commit-config.yaml`):
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-json`
- `check-added-large-files`, `check-merge-conflict`, `detect-private-key`, `debug-statements`
- **ruff** (linter + formatter with `--fix`)
- **mypy** (type checker with pandas/numpy stubs)

### Ruff Linting
```bash
# Check code
uv run ruff check src/

# Auto-fix issues
uv run ruff check src/ --fix

# Check tests
uv run ruff check tests/
```

### MyPy Type Checking
```bash
# Check types
uv run mypy src/

# Check with strict mode
uv run mypy --strict src/
```

### Bandit Security Linting (NEW — Phase 10b)
```bash
# Run security scan on source code
# Use when: want to catch pickle deserialization, hardcoded keys, subprocess injection
uv run bandit -r src/ -c pyproject.toml

# Scan with severity filter (skip low-severity findings)
uv run bandit -r src/ -ll -c pyproject.toml

# Scan tests directory
uv run bandit -r tests/ -c pyproject.toml

# Output results as JSON for CI integration
uv run bandit -r src/ -f json -o bandit_report.json
```

**Bandit checks enabled** (`[tool.bandit]` in `pyproject.toml`):
- B301-B303 (pickle, marshal, shelve deserialization)
- B104-B107 (hardcoded bind, password, secret key)
- B602-B611 (subprocess, exec, eval, sql injection)
- B701-B703 (jinja2 autoescape, requests without timeout)
- Exclusions: `tests/`, `notebooks/`, `useful_resources/`

---

## Pipeline & Orchestration (Prefect)

### Prefect Trading Pipeline
```bash
# Run the full 6-task pipeline once (fetch -> indicators -> patterns -> signals -> backtest -> report)
uv run python pipeline/prefect_flow.py

# Deploy with cron scheduling (weekdays at 10 PM)
uv run python pipeline/prefect_flow.py serve

# Start Prefect server (UI at http://localhost:4200)
uv run prefect server start
```

**Pipeline tasks:** `fetch_data` -> `compute_indicators` -> `detect_patterns` -> `generate_signals` -> `run_backtest` -> `generate_report`
**Features:** retry logic, 24h input caching, cron scheduling

---

## Experiment Tracking (MLflow)

### MLflow Server
```bash
# Start MLflow tracking server (UI at http://localhost:5000)
uv run mlflow server --host 0.0.0.0 --port 5000

# Start with specific backend store
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

### MlflowExperimentLogger (Python API)
```python
from src.ml.mlflow_logger import MlflowExperimentLogger

logger = MlflowExperimentLogger("rf_regime_classifier", tracking_uri="http://localhost:5000")
logger.log_metadata(model_type="random_forest", task="regime_classification")
logger.log_config(params)
logger.log_fold_metrics(i, {"auc": 0.72, "accuracy": 0.68})
logger.finish()
```

**Module:** `src/ml/mlflow_logger.py` — dual logging to JSONL files AND MLflow server. MLflow failures are caught and logged but do not interrupt experiments.

---

## DuckDB Data Helpers

### DuckDB SQL Over Parquet
```python
from src.data_ingestion.duckdb_helpers import (
    query_feature_store_df,
    cross_ticker_rank,
    feature_summary_stats,
    join_features_labels,
)

# Zero-copy SQL query against Parquet feature store
df = query_feature_store_df(
    "SELECT ticker, date, ma_cross_score FROM feature_store WHERE ticker='SPY'"
)

# Cross-ticker ranking (ROW_NUMBER() OVER PARTITION BY date)
ranks = cross_ticker_rank(metric="momentum_score", top_n=10)

# Feature summary stats (UNPIVOT)
stats = feature_summary_stats(tickers=["SPY", "QQQ"])

# Join features + labels for ML training
X, y = join_features_labels(ticker="SPY")
```

**Module:** `src/data_ingestion/duckdb_helpers.py` — DuckDB context manager with Parquet zero-copy, SQL feature engineering, cross-ticker ranking, feature summary via UNPIVOT.

---

## Jupytext Paired Notebooks

### Jupytext Commands
```bash
# Pair a single notebook (auto-sync .ipynb <-> .py:percent on save)
uv run jupytext --set-formats ipynb,py:percent notebooks/01_multi_pattern_backtest.ipynb

# Pair all notebooks in directory
uv run jupytext --set-formats ipynb,py:percent notebooks/*.ipynb

# Sync all paired notebooks
uv run jupytext --sync notebooks/*.ipynb

# Convert notebook to .py:percent format (one-shot)
uv run jupytext --to py:percent notebooks/01_multi_pattern_backtest.ipynb
```

**Config:** `.jupytext.toml` — auto-syncs `.ipynb` to `.py:percent` format for readable git diffs, strips notebook/cell metadata.
**Status:** All 34 notebooks are jupytext-paired with `.py:percent` sidecar files.

---

## AI Structured Output (Instructor)

### Instructor Module Usage
```python
from src.ai import get_structured_llm, TradingSignalAnalysis, PatternReviewOutput, BacktestSummary

# Initialize OpenAI-compatible client (uses OPENAI_API_KEY / OPENAI_BASE_URL env vars)
llm = get_structured_llm()

# Structured signal analysis (Pydantic-validated output)
result: TradingSignalAnalysis = llm.ask(
    "Analyze SPY breakout signal: RSI=72, volume=+40%, price above 50MA",
    TradingSignalAnalysis,
)

# Pattern review with quality scoring
review: PatternReviewOutput = llm.ask(
    "Review this double bottom pattern...", PatternReviewOutput
)

# Backtest summary generation
summary: BacktestSummary = llm.ask(
    "Summarize these backtest stats...", BacktestSummary
)
```

**Response models:** `TradingSignalAnalysis` (direction, confidence, rationale, risk_flags), `PatternReviewOutput` (pattern_name, is_valid, quality_score, key_levels), `BacktestSummary` (return, sharpe, max_drawdown, win_rate, profit_factor, verdict)
**Module:** `src/ai/__init__.py`

---

## Pydantic Configuration System

### Trading Config (with env-var overrides)
```python
from src.config import TradingConfig, RiskConfig, SignalConfig, BacktestConfig

# Load config with env-var overrides (e.g., BT_RISK__RISK_PER_TRADE=0.03)
config = TradingConfig()

# Access nested configs
print(config.risk.risk_per_trade)  # 0.02 default, overridden by env var
print(config.signal.min_confluence_score)  # 2.0 default
print(config.backtest.initial_capital)  # 100000 default

# Save/load config files
config.save("configs/my_config.yaml")
config = TradingConfig.load("configs/my_config.yaml")
```

**Config sections:** `TradingConfig`, `RiskConfig`, `SignalConfig`, `BacktestConfig`, `RegimeConfig`, `PatternConfig`
**Env var pattern:** `BT_{SECTION}__{KEY}=value` (e.g., `BT_RISK__RISK_PER_TRADE=0.03`)

---

## New Scripts

### Paper Trading (Live Simulation)
```bash
# Single poll (generate signals, simulate fills)
uv run scripts/paper_trade.py --symbol SPY --once

# Continuous mode with 60s interval for 24 hours
uv run scripts/paper_trade.py --symbols SPY QQQ IWM --interval 60 --duration 24

# ML-enhanced paper trading
uv run scripts/paper_trade.py --symbol SPY --once --ml-enabled

# Generate paper trading report
uv run scripts/paper_trade.py --report
```

### ML Enhanced Backtest (Regime Classifier + Signal Scorer)
```bash
# Full ML-enhanced backtest with regime classification
uv run scripts/ml_enhanced_backtest.py --symbol SPY --start 2015-01-01 --end 2024-12-31

# Train only (no backtest)
uv run scripts/ml_enhanced_backtest.py --symbol SPY --ml-only
```

### Feature Selection Pipeline
```bash
# Run full B1 feature selection (126 features -> ~40 high-IC)
uv run scripts/feature_selection_pipeline.py

# Loads multi-asset data, computes IC summary, SFI selector
```

### Multi-Asset Signal Collector
```bash
# Collect 300+ trade samples across universe for ML training
uv run scripts/multi_asset_signal_collector.py --start 2015-01-01 --end 2024-12-31

# With custom output path
uv run scripts/multi_asset_signal_collector.py --start 2015-01-01 --end 2024-12-31 --output output/signals.csv
```

### Validation Scripts
```bash
# Validate engine + 23 untested patterns on SPY + BTC
uv run scripts/validate_engine_and_patterns.py

# Quick test of portfolio module imports + components
uv run scripts/test_portfolio_module.py

# Validate all strategies and signals
uv run scripts/validate_all_strategies_signals.py

# Validate vectorized engine
uv run scripts/validate_vectorized_engine.py
```

### Notebook Maintenance Scripts
```bash
# Fix notebook project roots (Path('..') -> Path('.'))
uv run scripts/fix_project_roots.py

# Colab-to-local notebook transformation
uv run scripts/transform_colab_to_local.py

# Fix specific notebooks (02, 04, 06, 07, 08, 13)
uv run scripts/fix_notebook_<NN>.py
```

---

## Backtest Validation Workflow

### Validate Data Leak Fixes
```bash
# Run ML-enhanced backtest comparison (check for fake accuracy)
uv run scripts/backtest_ml_enhanced.py --symbol SPY --start 2020-01-01 --end 2024-12-31

# Run ML training with walk-forward validation
uv run scripts/train_ml_model.py --symbol SPY --start 2020-01-01 --end 2024-12-31

# Run all strategies backtest on SPY
uv run scripts/backtest_all_strategies_spy.py

# Run full test suite (target: 371 pass, 4 skip)
uv run pytest tests/ --tb=short -p no:warnings -q
```

### Expected Validation Results (post data-leak fix)
| Check | Expected | Actual |
|-------|----------|--------|
| ML accuracy | Not ~100% (no data leak) | Test AUC 0.40-0.58 |
| Train vs Test gap | Train > Test (expected) | Train AUC 1.0, Test 0.58 |
| Walk-forward AUC | Near 0.5 (weak signal) | Avg 0.518 |
| Pattern signals | Generated on correct bars | No look-ahead bias |
| Purge window | Default 5 bars | Active in 3 train() methods |

---

## Model Diagnostics & OOS Analysis

### Model Calibration Audit (Reliability Diagram)
```bash
# Check if P=0.45 actually means 45% win rate (triple-barrier labels)
# Output: reports/calibration/reliability_diagram_triple_barrier.png
uv run scripts/model_calibration.py
```

### Regime Shift Investigation (KS Tests)
```bash
# Compare feature distributions IS (2015-2024) vs OOS (2025-2026)
# Identifies which features broke in OOS period
# Output: reports/calibration/regime_shift_features.png
uv run scripts/investigate_regime_shift.py
```

### Walk-Forward Optimization Comparison
```bash
# Compare WFO (retraining quarterly) vs single-split on OOS 2025-2026
# Uses normalized ATR features (ATR/Close) to fix scale-dependence
# Output: reports/wfo/wfo_comparison.json
uv run scripts/backtest_wfo.py
```

### ML Backtest (Trailing Stop)
```bash
# Run ML strategy backtest with CatBoost V3 model
uv run scripts/run_ml_backtest.py SPY --entry-threshold 0.45 --trail-stop

# OOS test on 2025-2026 data
uv run scripts/run_ml_backtest.py SPY --entry-threshold 0.45 --trail-stop --start 2025-01-01

# Sweep entry thresholds (0.35, 0.40, 0.45, 0.50) with all C7 options
uv run scripts/sweep_entry_thresholds.py SPY
```

---

## CLI Tools (Installed via Scoop)

### File Operations
```bash
# Find files
fd pattern

# Search content
rg "pattern"

# Search in archives/PDFs
rga "pattern"

# View file with syntax
bat file.py

# View plain file (for piping)
bat -p --paging-never file.py
```

### Config Processing
```bash
# Parse JSON
jq '.path.to.value' file.json

# Parse YAML/XML
yq '.path.to.value' file.yaml

# Convert JSON to YAML
cat file.json | yq -o yaml '.'

# Convert YAML to JSON
cat file.yaml | yq -o json '.'
```

### Document Conversion
```bash
# Markdown to HTML
pandoc input.md -o output.html

# Markdown to PDF
pandoc input.md -o output.pdf

# Any format conversion
pandoc input.docx -o output.md
```

### Git & Diffs
```bash
# Pretty diff with delta
git diff | delta
git config --global core.pager delta
```

### Disk Usage
```bash
# Directory sizes
dust

# Show all files
dust -n 999
```

### Repository Packing (for AI analysis)
```bash
# Pack current repo
npx repomix

# Pack remote repo
npx repomix --remote owner/repo

# Pack with specific format
npx repomix --style markdown
```

---

## Streamlit & Gradio Apps

### Streamlit
```bash
# ML Selector App
uv run streamlit run scripts/ml_selector_app.py --server.port=8501

# Paper extraction app (from marker)
uv run streamlit run marker/marker/scripts/streamlit_app.py
```

### Gradio
```bash
# ML Selector Gradio App
uv run python scripts/ml_selector_gradio.py
```

---

## Multi-Agent & Seach Tools (MCP)

### Web Search & Research
- **Tavily**: Web search for real-time information
- **Exa**: Semantic search for finding relevant documents
- **Microsoft Learn**: Search Microsoft/Azure documentation
- **Google Maps Platform**: Location-based services

---

## Pre-Commit Hooks

### Setup & Run
```bash
# Install pre-commit hooks (one-time)
uv run pre-commit install

# Run all hooks on staged files (auto-triggered on git commit)
uv run pre-commit run --all-files

# Run all hooks (unstaged changes)
uv run pre-commit run

# Run specific hook
uv run pre-commit run ruff

# Run specific hook on a file
uv run pre-commit run ruff --files src/ml/pattern_classifier.py
```

### Configured Hooks
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-json` — file hygiene
- `check-added-large-files`, `check-merge-conflict`, `detect-private-key`, `debug-statements` — safety
- `ruff` (lint + format) — code quality (auto-fix enabled)
- `mypy` — type checking (excludes scripts/, notebooks/, .kilo/)

### Skip Hooks (emergency only)
```bash
SKIP=ruff,mypy git commit -m "message"
```

---

## Pipeline Orchestration (Prefect)

### Prefect Flow
```bash
# Run trading pipeline flow
uv run python pipeline/prefect_flow.py

# Start Prefect server (UI at http://127.0.0.1:4200)
uv run prefect server start

# Serve flow for scheduled execution
uv run python pipeline/prefect_flow.py serve
```

### Flow Tasks
- `fetch-data` — downloads OHLCV with 2 retries (30s delay), 24h cache
- `detect-patterns` — runs all pattern detectors
- `generate-signals` — aggregates pattern signals
- `compute-risk` — risk metrics, position sizing
- `backtest` — runs backtest on generated signals
- `generate-report` — produces performance report

---

## ML Experiment Tracking (MLflow)

### MLflow Logger
```bash
# Start MLflow tracking server (UI at http://localhost:5000)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Usage in Python:
# from src.ml.mlflow_logger import MlflowExperimentLogger
# logger = MlflowExperimentLogger("experiment_name", tracking_uri="http://localhost:5000")
# logger.log_metadata(...), logger.log_config(...), logger.log_fold_metrics(...), logger.finish()
```

### MLflow Commands
```bash
# Launch MLflow UI (if server already running)
uv run mlflow ui

# List experiments
uv run mlflow experiments list

# Run experiment via CLI
uv run python -c "
from src.ml.mlflow_logger import MlflowExperimentLogger
logger = MlflowExperimentLogger('my_experiment')
logger.log_metadata(model_type='catboost')
logger.finish()
"
```

---

## DuckDB Helpers

### SQL Feature Querying
```bash
# Query parquet feature store with SQL (zero-copy)
uv run python -c "
from src.data_ingestion.duckdb_helpers import query_feature_store
df = query_feature_store(
    \"SELECT ticker, date, ma_cross_score FROM features WHERE ticker='SPY'\"
)
"

# Register parquet and run arbitrary SQL
uv run python -c "
from src.data_ingestion.duckdb_helpers import duckdb_parquet
with duckdb_parquet(['experiments/features/feature_store.parquet']) as con:
    print(con.sql('SELECT * FROM features LIMIT 5').fetchdf())
"
```

---

## Jupytext Paired Notebooks

### Configuration
```bash
# Configured in .jupytext.toml:
#   formats = "ipynb,py:percent" — auto-syncs .ipynb <-> .py on save
#   34 notebooks are paired

# Manually pair a notebook
uv run jupytext --set-formats ipynb,py:percent path/to/notebook.ipynb

# Sync all paired notebooks
uv run jupytext --sync notebooks/*.ipynb

# Convert notebook to script
uv run jupytext --to py:percent path/to/notebook.ipynb

# Convert script back to notebook
uv run jupytext --to ipynb path/to/notebook.py
```

---

## Instructor AI Module

### Structured LLM Outputs
```bash
# Pydantic-validated LLM responses (in src/ai/)
# Requires: uv add instructor --group dev

# Usage in Python:
# from src.ai import StructuredLLM, TradingSignalAnalysis
# from openai import OpenAI
# client = OpenAI()
# llm = StructuredLLM(client)
# result = llm.ask("Analyze this double bottom pattern...", TradingSignalAnalysis)
```

### Available Structured Models
- `TradingSignalAnalysis` — direction, confidence, rationale, risk_flags, suggested_hold
- `PatternReviewOutput` — pattern validation, quality score, key levels
- `BacktestSummary` — return, sharpe, drawdown, win_rate, profit_factor, verdict

---

## Configuration (Pydantic BaseSettings)

### Settings Management
```bash
# Environment variable format: BT_<SECTION>_<FIELD>
# Example: BT_BACKTEST_COMMISSION_PCT=0.002

# Validate config
uv run python -c "from src.config import TradingConfig; c = TradingConfig(); print(c.model_dump_json(indent=2))"

# Load from custom YAML
uv run python -c "
from src.config import TradingConfig
cfg = TradingConfig.from_yaml('config/custom.yaml')
"
```

---

## Backtest Validation (Data Leak Checks)

```bash
# Validate data leak fixes produce sane metrics (no ~100% fake accuracy)
uv run scripts/backtest_ml_enhanced.py --symbol SPY --start 2020-01-01 --end 2024-12-31 --strategy ema

# Train model and check walk-forward AUC (should be near 0.5, NOT 0.99)
uv run scripts/train_ml_model.py --symbol SPY --start 2020-01-01 --end 2024-12-31 --model-type catboost

# Expected results after data leak fixes:
#   - Walk-forward test AUC: ~0.55 (barely above random — no forward-return leakage)
#   - Test accuracy: ~57% (not ~95-100%)
#   - Pattern classifier signal quality is measured honestly

# Run tests excluding slow hypothesis tests
uv run pytest tests/ --tb=short -p no:warnings -q --ignore=tests/test_hypothesis_properties.py
```

---

## pylint & Code Quality

### Linting (ruff)
```bash
# Check code
uv run ruff check src/

# Check tests
uv run ruff check tests/

# Fix auto-fixable issues
uv run ruff check src/ --fix

# Run ruff via pre-commit (auto-fix)
uv run pre-commit run ruff --all-files
```

### Type Checking (mypy)
```bash
# Check types
uv run mypy src/

# Check with strict mode
uv run mypy --strict src/

# Run mypy via pre-commit
uv run pre-commit run mypy --all-files
```

---

## Git Workflow
```bash
# Status, diff, log (used by AI agents)
git status
git diff
git log

# Create commit
git add .
git commit -m "feat: description"

# Push changes
git push
```

### Sync to Public Repo
```bash
# Dry run (preview what will be synced)
uv run scripts/sync_to_public.py --dry-run

# Push to public repo (requires PAT env var)
set PAT=ghp_...   # Windows
uv run scripts/sync_to_public.py

# With custom author info
set GIT_AUTHOR_NAME=Im-Busy
set GIT_AUTHOR_EMAIL=your@email.com
uv run scripts/sync_to_public.py
```

---

## Quick Command Reference by Task

### Train a model
```bash
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost
```

### Run backtest
```bash
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy ema
```

### Test patterns
```bash
uv run scripts/test_pattern_selector.py
```

### Optimize parameters
```bash
uv run scripts/optimize_macd.py
```

### Validate implementation
```bash
uv run scripts/validate_all_strategies_signals.py
```

### Run ML selection
```bash
uv run scripts/run_ml.py --auto
```

### Analyze contributions
```bash
uv run scripts/contribution_analysis.py
```

---

## File Structure Quick Reference

```
investment_trying/
├── src/                      # Source code
│   ├── backtest/             # Backtesting engine
│   ├── patterns/             # Pattern detectors (34+)
│   ├── strategies/           # Strategy wrappers
│   ├── ml/                   # ML models
│   ├── features/             # Feature extraction
│   ├── signals/              # Signal aggregation
│   ├── risk/                 # Risk management
│   ├── indicators/           # Technical indicators
│   └── data_ingestion/       # Data fetching
├── scripts/                  # CLI scripts (this document)
├── tests/                    # Unit/integration tests
├── notebooks/                # Jupyter notebooks
├── data/                     # Data storage
├── reports/                  # Backtest reports
├── .useful_commands/         # Command documentation
└── docs/                     # Documentation
```

---

## Maintenance

### Updating This Document

**ATTENTION ALL AI ASSISTANTS:** When implementing new functionality:

1. Add CLI commands to appropriate section above
2. Include brief description of what the command does
3. Group related commands together
4. Update file structure if new directories are created
5. Commit with: "docs: update command cheatsheet for [feature]"

### Related Documentation

- `.useful_commands/` - Detailed command files by category
- `src/*/AI_COMMANDS.txt` - Module-specific commands
- `AGENTS.md` - Project-wide AI instructions
- `.kilo/global-rules.md` - Global coding standards
- `.kilo/project-rules.md` - Project-specific rules

---

*Last Updated: 2026-05-13*
*Total Scripts: 65+*
*Categories: 22+*
