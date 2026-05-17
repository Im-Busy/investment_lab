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

Full 9-stage pipeline: features → triple-barrier labels → IC filter → Stability Selection (>=0.6) → GWO HP tuning → CV → final model → walk-forward → SHAP + regime analysis.

Supports two CV methods:
- `--cv-method purged` (default): 5-fold PurgedKFold
- `--cv-method cpcv`: Combinatorial Purged CV — C(6,2)=15 backtest paths. Each path
  tests the model against a different regime sequence. Papers show CPCV has lower PBO
  (Probability of Backtest Overfitting) than both PurgedKFold and Walk-Forward.
  When using CPCV, also trains a bagged ensemble (one model per path) for
  ensemble prediction in MLStrategy.

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

# CPCV cross-validation (B11): 15 backtest paths, lower PBO than PurgedKFold
# Use when verifying model generalization across different regime sequences
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method cpcv

# CPCV with per-sector model (bagged ensemble)
uv run scripts/train_ml_pipeline_v3.py --sector tech --fast --cv-method cpcv
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

### Simple Baseline Backtest (Technical vs ML)

```bash
# Compare RSI, MA Crossover, Buy & Hold vs ML strategy on SPY 2020-2026
uv run scripts/backtest_simple_baselines.py
```
Output table: Return%, Sharpe, MaxDD%, WinRate%, ProfitFactor, #Trades.

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

# B12 Dynamic Ensemble: regime-adaptive prediction via EGD-weighted ensemble (5 CatBoost variants)
# Sharpe +32% vs single model (0.69 → 0.91). Requires trained ensemble from pipeline.
uv run scripts/train_dynamic_ensemble.py --symbol SPY --fast
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.45 --use-dynamic-ensemble

# Or train via full pipeline (with CPCV, IC filter, stability selection):
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method cpcv
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.45 --use-dynamic-ensemble --dynamic-ensemble-path models/dynamic_ensemble_v3_SPY_latest

# B13 Meta-Labeling: secondary CatBoost filter that rejects low-quality signals
# Filters out ~28% of signals, Sharpe +36% (+0.69 → +0.94), Win% +7pp, PF +41%
uv run scripts/train_meta_labeler.py SPY --model models/pattern_classifier_v3_SPY_20260511_224704.pkl --entry-threshold 0.45
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.45 --use-meta-label --meta-label-path models/meta_labeler_v2_SPY_20260514_121006.pkl

# B14 Production Hardening: strict walk-forward invariants, PBO/DSR gates, hold-out validation
# --strict-wf: enforce chronological feature selection, no future data leakage, PurgedKFold only
# --pbo-gate: require PBO < 0.3 and DSR > 1.0 after CV evaluation
# --hold-out: run final untouched hold-out backtest (2025-01-01 → 2026-05-13)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method cpcv --strict-wf --pbo-gate --hold-out

# B14.4 Model Health Dashboard: monitor production models for drift, stability, degradation
# Now includes C1 overfitting detection checks (training history divergence + synthetic OOS comparison)
uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260514_195235.pkl
uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260514_195235.pkl --json
uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260514_195235.pkl --monitor-start 2025-01-01 --output reports/health/health_report.json

# ── Phase 12c: Post-Retrain Architecture Improvements (2026-05-16) ──

# P2-3: Diagnose Dynamic Ensemble Collapse — investigate why EGD ensemble underperforms
# Tests EGD weight trajectory, equal-weight vs stacking vs voting, per-path signal density
uv run scripts/diagnose_ensemble_collapse.py --synthetic

# P3-1: Daily Paper Trading Harness — generates signals without deploying capital
uv run scripts/paper_trade_daily.py --symbol SPY
uv run scripts/paper_trade_daily.py --basket SPY,QQQ,XLK,IWM
uv run scripts/paper_trade_daily.py --symbol SPY --days 90 --ir-weights

# Show recent paper trading activity
uv run scripts/paper_trade_daily.py --status

# P3-2: Kelly Position Sizing — compute optimal capital allocation from trade history
uv run python -c "from src.risk.kelly_allocator import estimate_minimum_capital, compute_kelly_from_history, format_kelly_report; result = compute_kelly_from_history([0.05, -0.03, 0.02, 0.04, -0.01]); print(format_kelly_report(result))"

# P2-4: Walk-Forward Cadence Experiment — compare retraining frequencies
# Tests never/3mo/4mo/6mo/12mo/24mo expanding-window retraining on same data
# Use when: deciding how often to retrain ML models in production
uv run scripts/backtest_wf_cadence.py

# P3-3: Survival Analysis for Time-to-Exit — predict when trades hit TP/SL
# Trains Cox, RSF, GBSA models with censored triple-barrier labels
# Use when: you want S-curve exit probability curves instead of binary signals
uv run scripts/train_survival_exit.py

# P3-4: Regression Labels — CatBoostRegressor for 5-day forward return
# Predicts continuous 5-day return instead of binary TP/SL labels
# Use when: you want a smooth signal (not binary) for position sizing
# Gate: directional accuracy > 0.55 AND IC > 0.03
uv run scripts/train_regression_labels.py

# P3-5: HMM Regime Detection — hmmlearn GaussianHMM vs Simple 200MA rule
# Compares 4-state HMM latent regimes to rule-based Bull/Bear detection
# Trains per-HMM-regime CatBoost models, compares AUC vs single model
# Gate: HMM Ensemble OOS AUC > Single Model OOS AUC
uv run scripts/train_hmm_regime.py

# C1 Overfitting Detection: smoke test both detectors
# Use when: verifying overfitting detectors are working after code changes
uv run python -c "from src.ml.overfitting_detectors import TrainingHistoryOverfitDetector, SyntheticOOSComparator; import numpy as np; np.random.seed(42); epochs = 200; train_losses = 1.0 / (1 + 0.1 * np.arange(epochs)) + 0.01 * np.random.randn(epochs); val_losses = np.concatenate([1.2 / (1 + 0.1 * np.arange(100)) + 0.01 * np.random.randn(100), 0.3 + 0.002 * np.arange(100) + 0.02 * np.random.randn(100)]); detector = TrainingHistoryOverfitDetector(); is_overfit, score, optimal = detector.is_overfit(train_losses.tolist(), val_losses.tolist()); print('TrainingHistoryOverfitDetector:'); print('  is_overfit=%s, divergence_score=%.3f, optimal_epoch=%s' % (is_overfit, score, optimal)); comparator = SyntheticOOSComparator(n_synthetic_runs=100, seed=42); result = comparator.compute_generalization_score(real_oos_sharpe=-1.24, real_is_sharpe=0.85, n_oos_days=252); print('SyntheticOOSComparator:'); print('  PBO=%.3f, generalization_score=%.3f, IS/OOS ratio=%.1f' % (result['pbo'], result['generalization_score'], result['is_oos_ratio']))"

# ── C2 Sentiment Signal Integration ──

# Run ML backtest with synthetic sentiment (AR(1) process)
# Use when: testing sentiment pipeline infrastructure (real data deferred to post-C6)
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --entry-threshold 0.35 --use-sentiment --sentiment-weight 0.15

# Compare sentiment vs baseline
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --entry-threshold 0.35
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --entry-threshold 0.35 --use-sentiment

# With higher sentiment influence
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --entry-threshold 0.35 --use-sentiment --sentiment-weight 0.25

# ── C3 Event Calendar & Event Filtering ──

# Build hardcoded event calendar (FOMC, CPI, NFP, OPEX, Triple Witching for 2020-2026)
uv run scripts/build_event_calendar.py

# Build calendar with custom year range
uv run scripts/build_event_calendar.py --years 2015 2030

# Verify event count
uv run python -c "import duckdb; con=duckdb.connect('data/event_calendar.duckdb'); print(con.execute('SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY event_type').fetchall())"

# Run ML backtest with event-day entry suppression
# Suppresses entries on HIGH-impact event days (FOMC, CPI, NFP, Triple Witching)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-event-filter

# Compare event filter vs baseline
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-event-filter

# ── C4 RL Trade Execution ──

# Train the DQN execution agent on IS data (2016-2024)
uv run scripts/train_rl_executor.py --symbol SPY --start 2016-05-12 --end 2024-12-31 --episodes 500

# Train with more episodes and custom learning rate
uv run scripts/train_rl_executor.py --symbol SPY --start 2016-05-12 --end 2024-12-31 --episodes 2000 --lr 0.0005

# Train on a different symbol
uv run scripts/train_rl_executor.py --symbol QQQ --start 2016-01-01 --end 2024-12-31 --episodes 1000

# Run ML backtest with RL-based entry/exit decisions
# Use when: want RL agent to decide entry/exit instead of fixed thresholds
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --use-rl-execution

# Run RL backtest with custom model path
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --use-rl-execution --rl-model-path models/rl_executor_dqn_custom.pt

# Combine RL execution with event filter
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --use-rl-execution --use-event-filter

# ── C5 Kelly Criterion Position Sizing ──

# Run ML backtest with Kelly criterion position sizing (half-Kelly default)
# Use when: want dynamic position sizing based on model probability edge
# Higher probability → larger position, lower probability → smaller/fractional position
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-kelly

# Quarter-Kelly for conservative sizing (25% of full Kelly)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-kelly --kelly-fraction 0.25

# Full Kelly (aggressive, consider quarter/half for safety)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-kelly --kelly-fraction 1.0

# ── C5 Behavioral Crash Filter ──

# Run ML backtest with crash regime entry suppression
# Use when: want to avoid entering during panic selling / herding / vol jumps
# Crash filter uses behavioral finance indicators from Fang et al. (2022)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-crash-filter

# Stricter crash filtering (higher threshold = more bars suppressed)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-crash-filter --crash-threshold 0.35

# Combined: Kelly sizing + crash filter + event filter
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-kelly --use-crash-filter --use-event-filter

# ── C5 Circuit Overfitting Detection ──

# Run circuit-based overfit detection on model predictions
uv run python -c "
from src.ml.circuit_overfit import CircuitOverfitDetector
import numpy as np
# Generate sample predictions
probs = np.random.beta(5, 5, 1000)
detector = CircuitOverfitDetector()
result = detector.full_check(probs)
print(f'Overfit: {result[\"is_overfit\"]}, Score: {result[\"overall_overfit_score\"]:.4f}')
print(result['recommendation'])
"

# ── C6 Adversarial Overfitting Detection ──

# Run adversarial overfit detection on model predictions
# Use when: want to detect model overfit via feature perturbation sensitivity
uv run python -c "
from src.ml.adversarial_overfit import AdversarialOverfitDetector
import numpy as np
import pandas as pd
rng = np.random.RandomState(42)
n = 500
features = pd.DataFrame({f'f{i}': rng.normal(0, 1, n) for i in range(10)})
predictions = pd.Series(rng.uniform(0.3, 0.7, n))
detector = AdversarialOverfitDetector(noise_scale=0.05, seed=42)
result = detector.analyze(features=features, predictions=predictions)
print(f'Overfit: {result.is_overfit}, Score: {result.overfit_score:.4f}, Risk: {result.risk_level}')
print(result.interpretation)
"

# Run model health with adversarial overfit check (C6 integrated)
uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260514_195235.pkl --monitor-start 2025-01-01

# ── C6 Defensive Backtest + Time-Reversal ──

# Run defensive backtest — re-runs strategy on time-reversed data
# Use when: validating strategy is not overfit to specific market patterns
# If strategy is profitable on reversed data → overfit likely (Svozil 2026)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --defensive

# Combined: all C5+C6 flags
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --trail-stop --entry-threshold 0.35 --use-kelly --use-crash-filter --defensive

# ── E2 Chronos-2 Foundation Model Signal Blend ──

# Run ML backtest with Chronos-2 zero-shot forecast blended into signal
# Use when: experimenting with foundation model signals as augmentation
# Chronos-2 runs transformer inference per bar — slow on CPU (5-10 sec/bar)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --use-chronos

# With custom chronos blend weight (0-1, default 0.30)
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --use-chronos --chronos-weight 0.15

# Note: Chronos-2 needs context_length bars before it generates signals.
# On short windows, add --end to ensure enough data.
uv run scripts/run_ml_backtest.py SPY --start 2018-01-01 --end 2021-01-01 --use-chronos --chronos-weight 0.30

# ── E5 Cross-Asset Feature Inference ──

# Run backtest with cross-asset features (beta, rel_ret, corr, sector)
# Use when: model was trained with cross-asset features (check feature_names)
# Without this flag, cross-asset features are silently zero-filled
uv run scripts/run_ml_backtest.py SPY --model models/pattern_classifier_v3_SPY_20260514_113331.pkl --cross-asset-inference

# Compare cross-asset vs zero-filled baseline
uv run scripts/run_ml_backtest.py SPY --model models/pattern_classifier_v3_SPY_20260514_113331.pkl --start 2020-01-01 --end 2021-01-01 --entry-threshold 0.40
uv run scripts/run_ml_backtest.py SPY --model models/pattern_classifier_v3_SPY_20260514_113331.pkl --start 2020-01-01 --end 2021-01-01 --entry-threshold 0.40 --cross-asset-inference

# ── C6 Production Event Calendar ──

# Query event performance statistics from production calendar
uv run python -c "
from src.data_ingestion.event_calendar_production import ProductionEventCalendar
db = ProductionEventCalendar('data/event_calendar.duckdb')
db.rebuild_performance_summary()
stats = db.get_event_performance_stats()
print(stats.to_string())
db.close()
"

# Compare event-day vs normal-day trading performance
uv run python -c "
from src.data_ingestion.event_calendar_production import ProductionEventCalendar
db = ProductionEventCalendar('data/event_calendar.duckdb')
trade_log = pd.DataFrame({'date': ['2020-03-18'], 'pnl': [100.0], 'win': [True]})
result = db.compare_event_vs_normal_performance(trade_log)
print(result)
db.close()
"

# ── Parameter Sweeps ──

# Sweep entry thresholds across trail/conviction combos
uv run scripts/sweep_entry_thresholds.py

# Results: BESTS.md (leaderboard)
```

# ── E1 Chronos-2 Zero-Shot Forecasting ──

# Run Chronos-2 smoke test on SPY (verifies E1 installation)
uv run scripts/smoke_chronos_spy.py

# Quick forecast: 5-day Chronos-2 prediction on SPY
uv run python -c "
from src.ml.models.chronos import ChronosForecaster
import pandas as pd
df = pd.read_csv('data/raw/SPY_daily.csv', parse_dates=True, index_col=0).dropna()
fc = ChronosForecaster(model_size='chronos-2', device='cpu', prediction_length=5)
result = fc.predict(df['Close'])
print(f'5d expected return: {(result[\"mean\"].iloc[-1] / df[\"Close\"].iloc[-1] - 1) * 100:.2f}%')
"

# Use bolt-tiny (9M) for faster inference on constrained hardware
uv run python -c "
from src.ml.models.chronos import ChronosForecaster
import pandas as pd
df = pd.read_csv('data/raw/SPY_daily.csv', parse_dates=True, index_col=0).dropna()
fc = ChronosForecaster(model_size='bolt-tiny', device='cpu', prediction_length=21)
result = fc.predict(df['Close'])
print(f'10%={result[\"q10\"].iloc[-1]:.2f}  50%={result[\"q50\"].iloc[-1]:.2f}  90%={result[\"q90\"].iloc[-1]:.2f}')
"

# Available models: chronos-2 (120M), chronos-2-small (28M), bolt-tiny (9M) through bolt-base (205M)
# Use --use-chronos flag (coming in E2) to integrate with MLStrategy backtests
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

## Factor DSL & Agentic Discovery (arXiv:2604.26747v1)

### Factor DSL (Constrained Expression Language)
```bash
# Parse and validate a DSL expression
uv run python -c "from src.patterns.dsl import parse_expr, validate_recipe; expr = parse_expr('rank(-log(mcap) + 0.5*ma(hl_range,10))'); print(validate_recipe(expr).report())"

# Evaluate a factor recipe against OHLCV data panel
uv run python -c "from src.patterns.dsl import evaluate_string; scores = evaluate_string('rank(-log(mcap) - ma(realized_vol,20))', df)"

# Evaluate multiple factor recipes as a panel
uv run python -c "from src.patterns.dsl import evaluate_factor_panel; panel = evaluate_factor_panel({'f1': 'rank(-log(mcap))', 'f2': 'rank(ma(ret_log,5))'}, df)"

# Compute daily IC for a factor
uv run python -c "from src.patterns.dsl import compute_ic; ic = compute_ic(scores, forward_returns)"
```

### Factor Discovery Trace
```bash
# Create and use a factor discovery trace
uv run python -c "
from src.patterns.dsl.trace import FactorTrace, CandidateEntry, CandidateType, FailureCategory
trace = FactorTrace('my_session')
candidate = CandidateEntry(id='h1_test', hypothesis='Small-cap outperforms',
    rationale='Size premium in crypto', candidate_type=CandidateType.EXPLORATORY,
    recipe='rank(-log(mcap))', gate_result='PASS')
trace.append_round(1, [candidate], round_summary='Round 1 complete')
print(trace.summary_table())
"
```

### IC-Based Signal Gates
```bash
# Evaluate a single factor with IC gate
uv run python -c "
from src.signals.ic_gate import ICGate
gate = ICGate(min_mean_ic=0.02, min_ic_tstat=2.0)
result = gate.evaluate_panel('my_factor', scores_df, forward_returns_df)
print(result.report())
"
```

### Ridge Signal Combiner
```bash
# Combine multiple DSL factors with Ridge regression
uv run python -c "
from src.signals.ridge_combiner import from_dsl_recipes
recipes = {
    'smallcap': 'rank(-log(mcap))',
    'lowvol': 'rank(-ma(realized_vol,20))',
    'highrange': 'rank(ma(hl_range,10))',
}
result = from_dsl_recipes(recipes, df, fwd_rets, train_start='2020-01-01', train_end='2022-12-31')
print(result.report())
"
```

### Range Persistence Patterns
```bash
# Test range persistence detectors
uv run python -c "
from src.patterns.range_persistence import PersistentRange, ContractingRange, ExpandingRange
import yfinance as yf
df = yf.download('BTC-USD', start='2023-01-01')
pr = PersistentRange()
results = pr.detect(df)
print(f'Detected: {results[\"detected\"].sum()} signals')
"
```

---

---

## Pattern Detection

### Pattern Detectors
```bash# Run all pattern detectors
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

### Researcher Agent (/research)
```bash
# Search for papers, reference implementations, and benchmarks
# Auto-activates when designing new algorithms or encountering unfamiliar methods
/research <query>
/research regime-switching HMM
/research PurgedKFold implementation
/research state of the art volatility forecasting 2024

# The agent searches: Google Scholar → ArXiv → GitHub → web
# Then cross-references findings with project modules via knowledge graph
```

### Paper Summarization (paper2md)
```bash
cd useful_resources/useful_repos/research-tools/paper2md

# Summarize papers
uv run python summarize_md_papers.py --papers-dir ..\..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md

# Clear cache and re-summarize
uv run python summarize_md_papers.py --papers-dir ..\..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md --clear-cache

# API Configuration
# Edit: research-tools\paper2md\.env
# OPENAI_BASE_URL: https://openrouter.ai/api/v1
# OPENAI_MODEL: deepseek/deepseek-v4-flash
```

### AI Research Systems (Reference Only — Linux/Docker Required)
```bash
# RD-Agent — Quant factor/model evolution with Qlib
cd useful_resources/useful_repos/research-tools/RD-Agent
rdagent fin_factor   # Iterative factor evolution
rdagent fin_model    # Iterative model evolution
rdagent fin_quant    # Factor + model joint evolution
rdagent health_check # Validate setup

# DeepScientist — Local-first research OS
cd useful_resources/useful_repos/research-tools/DeepScientist
npm install -g @researai/deepscientist
ds --here            # Start research workspace

# Idea2Paper — KG-based paper generation
cd useful_resources/useful_repos/research-tools/Idea2Paper
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

### Paper Trading — Phase 07 (Production Ready)

```bash
# Production paper trading with go/no-go evaluation (backfill)
uv run scripts/paper_trade_production.py --ticker SPY --start 2025-01-01 --end 2025-12-31

# Multi-ticker basket paper trading
uv run scripts/paper_trade_production.py --basket --start 2025-01-01

# Custom entry threshold and reliability
uv run scripts/paper_trade_production.py --ticker SPY --entry-threshold 0.55 --min-reliability 0.70

# Daily paper trading harness (production config)
uv run scripts/paper_trade_daily.py --symbol SPY

# Daily paper trading with backfill
uv run scripts/paper_trade_daily.py --symbol SPY --days 30

# Basket per-day paper trading
uv run scripts/paper_trade_daily.py --basket SPY,QQQ,GLD

# Honest walk-forward paper trading (no look-ahead)
uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01

# Go/No-Go readiness report
uv run scripts/paper_trade_production.py --ticker SPY --start 2025-01-01
# -> reports/live_readiness.md

# View paper trading log
uv run scripts/paper_trade_daily.py --status
```

### Paper Trading (Legacy — v3)
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

### Statistical Significance Tests (NEW 2026-05-17)
```bash
# Run all significance tests on backtest returns (Bootstrap CI + PSR + DSR + Permutation)
# Prints comprehensive report: observed Sharpe, 95% CI, PSR, DSR, permutation p-value
uv run python -c "
from src.analysis.deflated_sharpe import print_significance_report
import numpy as np
returns = np.array([...])  # daily strategy returns
print(print_significance_report(returns, n_trades=50, n_trials=100))
"

# Individual tests
from src.analysis.deflated_sharpe import bootstrap_sharpe_ci, permutation_test, format_significance_summary

# Bootstrap 95% CI on Sharpe
ci = bootstrap_sharpe_ci(returns, n_bootstrap=5000, use_block_bootstrap=True)

# Permutation test: is strategy Sharpe > random shuffling?
p_result = permutation_test(returns, n_permutations=5000)

# One-shot: all tests in a single call
summary = format_significance_summary(returns, n_trades=50, n_trials=100)
```

### Factor IC/IR Analysis Pipeline (NEW 2026-05-17)
```bash
# Analyze feature predictive power for a single symbol
# Computes IC, rank IC, IC decay, rolling stability, IR per feature
uv run scripts/run_factor_ic_analysis.py SPY

# With custom date range and horizons
uv run scripts/run_factor_ic_analysis.py SPY --start 2023-01-01 --end 2024-12-31 --horizons 5,10,20

# Report only top 20 features
uv run scripts/run_factor_ic_analysis.py SPY --top-n 20

# Save results as JSON for downstream analysis
uv run scripts/run_factor_ic_analysis.py SPY --json-output ic_results.json

# Analyze pre-computed feature file
uv run scripts/run_factor_ic_analysis.py --features-file experiments/features/SPY_features.parquet
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

### PDF to Markdown (markitdown)
```bash
# Extract text from a single PDF
uv run markitdown input.pdf -o output.md

# Extract text from stdin
cat input.pdf | uv run markitdown

# Batch re-extract all project PDFs and merge with existing .md content
# Tier 1 (image-only): keeps old .md as base (markitdown can't read graphs)
# Tier 2 (old has more text): uses old as base, appends raw extraction
# Tier 3 (raw has more text): uses raw as base, adds old supplements
uv run scripts/re_extract_pdfs.py

# Process only papers/ -> papers_md/
uv run scripts/re_extract_pdfs.py --dir papers

# Process only PDFs_Found_Online/
uv run scripts/re_extract_pdfs.py --dir pdfs_found

# Dry run (check mappings, skip writes)
uv run scripts/re_extract_pdfs.py --dry-run
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

### Repo Sync — Private → Public Curation

# Full sync: merge main → public branch → verify → push to public repo
# Uses /repo-sync slash command (invokes repo-syncer agent)
/repo-sync

# Safety check only (no push) — verify no private files on public branch
/repo-sync check

# Show current branch state and remote configuration
/repo-sync status

# Manual dual-remote workflow (if not using /repo-sync):
# 1. Pull latest private changes
git pull origin main
# 2. Merge into public branch
git checkout public && git merge main
# 3. Safety check — ensure no private files leaked
git diff --name-only public@{1}..public | findstr /R "^data\|^models\|^outputs\|^experiments\|^reports\|^notebooks\|^logs"
# 4. If clean, push to public repo
git push public public
# 5. Return to main
git checkout main
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

### Search for papers/implementations
```bash
/research <query>
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

*Last Updated: 2026-05-15*
*Total Scripts: 68+*
*Categories: 25+*

---

## Direction A+B — Regime-Adaptive ML + Rules-First

### RegimeRouter — Per-Regime ML (Sub-track A)

```bash
# OOS backtest with RegimeRouter (2025-2026, the failure period)
uv run scripts/run_ml_backtest.py SPY --start 2025-01-01 --end 2026-05-14 \
  --use-regime-router --entry-threshold 0.40

# Train new per-regime models (after feature changes)
uv run scripts/train_per_regime_models.py SPY --start 2015-01-01 --end 2024-12-31 \
  --exclude vol_regime,volatility_regime,ema_21_55_spread --fast

# Benchmark regime detectors (HMM, GMM, PCA+KMeans)
uv run scripts/benchmark_regimes.py SPY --start 2015-01-01 --end 2024-12-31 \
  --detectors hmm,gmm,pca_kmeans --n-states 4
```

### Rules-First — Pattern-Based Strategy (Sub-track B)

```bash
# IS backtest (2016-2024) — single config
uv run scripts/backtest_rules_first.py SPY --start 2016-01-01 --end 2024-12-31 \
  --min-reliability 0.70 --entry-threshold 0.55

# IS sweep entry thresholds at mr=0.70
uv run scripts/backtest_rules_first.py SPY --start 2016-01-01 --end 2024-12-31 \
  --min-reliability 0.70 --sweep-entry 0.55,0.60,0.65,0.70,0.75

# IS sweep min reliability
uv run scripts/backtest_rules_first.py SPY --start 2016-01-01 --end 2024-12-31 \
  --sweep-reliability 0.4,0.5,0.6,0.7,0.75

# OOS backtest (2025-2026) — the critical test
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-14 \
  --min-reliability 0.70 --sweep-entry 0.55,0.60,0.65,0.70,0.75

# Optimize pattern weights from ablation data
uv run scripts/optimize_pattern_weights.py --ablation-file reports/ablation/solo_results.json
```

### Combined ML+Rules — A+B Convergence (AB1-AB2)

```bash
# Combined strategy with regime-adaptive weighting
uv run scripts/backtest_combined.py SPY --start 2016-01-01 --end 2024-12-31 \
  --weight-mode regime-adaptive

# OOS backtest — sweep all blending modes
uv run scripts/backtest_combined.py SPY --start 2025-01-01 --end 2026-05-14 \
  --weight-mode sweep

# Static weight sweep (find optimal blend)
uv run scripts/backtest_combined.py SPY --start 2025-01-01 --end 2026-05-14 \
  --weight-mode static --sweep-weight 0.3,0.5,0.7,0.9

# Signal-conflict mode (trust rules on ML disagreement)
uv run scripts/backtest_combined.py SPY --start 2025-01-01 --end 2026-05-14 \
  --weight-mode signal-conflict --entry-threshold 0.55

# Single model (no RegimeRouter)
uv run scripts/backtest_combined.py SPY --start 2025-01-01 --end 2026-05-14 \
  --no-regime-router --ml-model-path models/pattern_classifier_v3_SPY.pkl
```

**Weight modes:**
- `static`: Fixed blend `w_rules * rules + (1-w_rules) * ml`
- `regime-adaptive`: Bull → weight ML more, Bear → weight Rules more
- `reciprocal-sharpe`: Rolling 60d Sharpe per source, invert to weight
- `signal-conflict`: When ML and Rules disagree, trust Rules (w_rules→0.8)

**Key baselines (SPY 2025-2026 OOS):**
- ML single model: Sharpe -1.25, Return -6.78%
- ML RegimeRouter: Sharpe +0.35, Return +2.13%
- Rules-First mr=0.70 et=0.55: Sharpe **+0.76**, Return +9.20%
- Combined signal-conflict: Sharpe +0.41, Return +4.6%
- Combined static w=0.9: Sharpe +0.76, Return +9.2% (= rules-first)

**Production decision:** Rules-First is the primary system. ML adds no OOS benefit.
Combined strategy available as reference/validation; static w=0.9 essentially
replicates rules-first. Adding ML weight consistently degrades OOS performance.

### Cross-Instrument Batch Testing (AB3)

```bash
# Run Rules-First on all 16 instruments (IS 2016-2024)
uv run scripts/backtest_rules_batch.py --start 2016-01-01 --end 2024-12-31 \
  --json-output reports/batch/rules_first_IS_2016_2024.json

# Run Rules-First on all 16 instruments (OOS 2025-2026)
uv run scripts/backtest_rules_batch.py --start 2025-01-01 --end 2026-05-14 \
  --json-output reports/batch/rules_first_OOS_2025_2026.json

# Sweep entry thresholds on specific symbols
uv run scripts/backtest_rules_batch.py --start 2016-01-01 --end 2024-12-31 \
  --symbols XLV,XLE,JNJ,KO --sweep-entry 0.55,0.60,0.65,0.70,0.75

# Custom basket
uv run scripts/backtest_rules_batch.py --symbols SPY,QQQ,GLD,BTC_USD --start 2025-01-01

# Summary only (no per-symbol output)
uv run scripts/backtest_rules_batch.py --summary-only --start 2025-01-01
```

**Key insights (IS 2016-2024):**
- 8/16 positive Sharpe (50%). Best: BTC_USD (0.77), XLK (0.77), QQQ (0.77).
- Tech/indices/crypto dominate. Defensive/energy/single-stocks underperform.
- Rules-First works best on momentum/trending instruments.
- See `BESTS.md` cross-instrument section for full results table.

### Comprehensive 125-Instrument Backtest (2026-05-17)

```bash
# Run all 12 batches (125 tickers) with production config
uv run scripts/backtest_all_comprehensive.py --batch all --output-dir reports/comprehensive_batch

# Run a specific batch (1-12)
uv run scripts/backtest_all_comprehensive.py --batch-index 1 --output-dir reports/comprehensive_batch

# List available batches
uv run scripts/backtest_all_comprehensive.py --list-batches

# Run custom ticker list
uv run scripts/backtest_all_comprehensive.py --tickers SPY,QQQ,GLD,BTC_USD --output-dir reports/comprehensive_batch

# Compile master summary from all batch JSONs
uv run scripts/compile_master_summary.py

# Generate per-instrument bests markdown
uv run scripts/_gen_per_instrument_bests.py
```

**Key results (production config, no per-instrument tuning):**
- 125 instruments tested across 13 categories. 57% OOS positive Sharpe.
- Top OOS: SPY +1.675, EEM +1.642, MPC +1.503, INTC +1.426, EOG +1.247.
- IS->OOS correlation = -0.198 — IS performance does NOT predict OOS.
- Energy dominates. Bonds, HK, MicroCaps fail.
- Full report: `reports/comprehensive_batch/MASTER_SUMMARY.md`
- Per-instrument: `docs/PER_INSTRUMENT_BESTS.md`

### Chinese Market Data (AKShare)

> Free, no API token required. Data saved to `data/raw/` with `CN_` prefix.
> Source: `src/data_ingestion/fetch_china.py`

```bash
# Fetch all Chinese instruments (indexes + stocks + futures)
uv run src/data_ingestion/fetch_china.py

# Indexes only (CSI300, CSI500, SSE50, SHCOMP, SZCOMP, CHINEXT, STAR50)
uv run src/data_ingestion/fetch_china.py --indexes

# Stocks only (Moutai, Wuliangye, PingAn, CMB, Midea, CATL, BYD, etc.)
uv run src/data_ingestion/fetch_china.py --stocks

# Futures only (IF, IC, IH, IM, RB, I, CU, AU, AG, SC)
uv run src/data_ingestion/fetch_china.py --futures

# Single stock by code
uv run src/data_ingestion/fetch_china.py --symbol 600519
```

**Available instruments:** 7 indexes, 12 stocks, 10 futures.
Column names mapped to English (Date, Open, High, Low, Close, Volume).
Company names preserved in Chinese (Moutai = 贵州茅台, BYD = 比亚迪).

### Instrument-Specific Alpha Strategies

> 10 standalone strategy files in `src/strategies/` — each a backtesting.py Strategy subclass.
> Per-instrument trailing stop widths (4-10 ATR) tuned for volatility characteristics.

```bash
# SPY Gap Fill (overnight gaps mean-revert)
uv run python -c "
from backtesting import Backtest
from src.strategies.gap_fill_strategy import GapFillStrategy
import pandas as pd
df = pd.read_csv('data/raw/SPY_daily.csv', parse_dates=[0], index_col=0)
bt = Backtest(df, GapFillStrategy, cash=10_000, commission=0.001)
stats = bt.run(gap_threshold=0.005, trail_atr=4.0)
print(f'Sharpe={stats[\"Sharpe Ratio\"]:.3f} Return={stats[\"Return [%]\"]:.1f}%')
"

# Gold Trend (MA crossover + ADX, 6 ATR trail)
uv run python -c "
from src.strategies.gold_trend_strategy import GoldTrendStrategy
# ... same pattern
stats = bt.run(ma_fast=10, ma_slow=50, adx_min=20, trail_atr=6.0)
"

# BTC/ETH Crypto Momentum (10 ATR trail, use --cash 100000 for BTC)
uv run python -c "
from src.strategies.crypto_momentum_strategy import CryptoMomentumStrategy
# ...
stats = bt.run(ma_period=20, vol_mult=1.5, return_min=0.02, trail_atr=10.0)
"
```

**Strategy files:**
| File | Class | Best Instrument | IS Sharpe | OOS Sharpe |
|------|-------|----------------|-----------|------------|
| `gap_fill_strategy.py` | `GapFillStrategy` | SPY | 0.80 | 1.23 |
| `bb_squeeze_strategy.py` | `BBSqueezeStrategy` | QQQ | 0.45 | 0.00 |
| `rsi_oversold_strategy.py` | `RSIOversoldStrategy` | IWM | 0.10 | 0.32 |
| `dip_buy_strategy.py` | `DipBuyStrategy` | SO | 0.56 | 1.14 |
| `lowvol_momentum_strategy.py` | `LowVolMomentumStrategy` | JNJ | -0.06 | -0.28 |
| `sector_oversold_strategy.py` | `SectorOversoldStrategy` | XLV | 0.76 | -0.47 |
| `gold_trend_strategy.py` | `GoldTrendStrategy` | GLD | 0.48 | 1.01 |
| `silver_breakout_strategy.py` | `SilverBreakoutStrategy` | SLV | -0.11 | 0.82 |
| `bond_fade_strategy.py` | `BondFadeStrategy` | TLT | -0.56 | 0.44 |
| `crypto_momentum_strategy.py` | `CryptoMomentumStrategy` | BTC/ETH | 0.67 | -0.67 |

**Key finding:** Instrument-specific alphas outperform universal pattern detection on
5/8 instruments OOS. SPY Gap Fill (+1.23) and Gold Trend (+1.01) are the strongest.
Crypto momentum failed OOS due to 2025 drawdown — needs bear-market adaptation.

---

## Phase 16: NLP & Mean Reversion (NEW 2026-05-15)

### Phase N: Loughran-McDonald Dictionary Sentiment (NLP Foundation)

# Download LM dictionary (one-time setup)
curl -L -o "data/sentiment/LM_Master_Dictionary_1993-2025.csv" "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"

# Run ML backtest with LM sentiment (synthetic headlines from price action)
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --use-lm-sentiment --lm-sentiment-weight 0.15

# Compare baseline vs LM sentiment
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --use-lm-sentiment

# Smoke test LM dictionary (standalone)
uv run python -c "from src.signals.sentiment.dictionary import LMDictionary, LMSentimentScorer; d = LMDictionary(); s = LMSentimentScorer(d); print(s.score_text('The company reported strong profit growth and achieved excellent margins'))"

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/sentiment/dictionary.py` | LMDictionary, LMSentimentScorer, LMTradingSignalModifier |
| `src/signals/sentiment/__init__.py` | Subpackage exports |
| `data/sentiment/LM_Master_Dictionary_1993-2025.csv` | 86K+ word LM financial dictionary |
| `src/strategies/ml_strategy.py` | Integrated via `use_lm_sentiment` + `lm_sentiment_weight` params |
| `scripts/run_ml_backtest.py` | Added `--use-lm-sentiment` + `--lm-sentiment-weight` flags |

**LM Dictionary stats:** 347 positive, 2,345 negative, 297 uncertainty, 903 litigious words.
Uses synthetic headlines from price returns for backtesting (no external text source needed).

### Phase P: Mean Reversion System + Regime Routing

# Mean-reversion only (ADX < 20 gated)
uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01

# Mean-reversion without regime gate (test all markets)
uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01 --no-regime-gate

# Regime-routed (trending + ranging)
uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01 --router

# Compare all three: trend-only vs reversion-only vs routed
uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01 --compare

# Custom stop parameters
uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01 --tp-atr 2.5 --sl-atr 3.5 --max-hold 15

**Strategy files:**
| File | Class | Purpose |
|------|-------|---------|
| `src/strategies/mean_reversion_strategy.py` | `MeanReversionStrategy` | 6-indicator MR (RSI+WR+CCI+MFI+StochRSI+Bollinger), ADX-gated, fixed TP/wider SL/time exit |
| `src/strategies/regime_router_strategy.py` | `RegimeRouterStrategy` | Routes trending (ADX>25) vs ranging (ADX<20) with MR/Trend stops |

**SPY 2016-2026 Comparison:**
| Strategy | Return | Sharpe | Trades | Win% | PF |
|----------|--------|--------|--------|------|----|
| MeanReversion (ADX-gated) | -100% | 0.00 | 1 | 0% | 0.00 |
| RulesFirst (trend, defaults) | -6.4% | -0.05 | 237 | 40.5% | 0.99 |
| **RegimeRouter** | **209.4%** | **0.68** | **58** | **58.6%** | **2.23** |

RegimeRouter outperforms both individual strategies - confirms routing value.

### Phase O: Sentiment Lead/Lag Gate Analysis

# Analyze whether sentiment leads or reflects price
# Gate: build social media infra ONLY if sentiment predicts price (tau > 0, p < 0.05)
uv run scripts/analyze_sentiment_lead_lag.py SPY

# With custom date range and longer lag window
uv run scripts/analyze_sentiment_lead_lag.py SPY --start 2016-01-01 --end 2024-12-31 --max-lag 30

# Generate correlation plot
uv run scripts/analyze_sentiment_lead_lag.py SPY --plot

# Export results as JSON
uv run scripts/analyze_sentiment_lead_lag.py SPY --json

**Key files:**
| File | Purpose |
|------|---------|
| `scripts/analyze_sentiment_lead_lag.py` | Cross-correlation rho(tau) = Corr(S_t, R_{t+tau}) for tau in [-20, +20] |

**Gate result (SPY 2016-2024):** BLOCKED - LM sentiment from price-derived synthetic headlines is reflective (peak at tau=0). Test 3 validates methodology correctly identifies forward-looking sentiment (PASS at tau=+3d). Do not build social media infrastructure. Revisit if real news text source becomes available.

### Phase Q: Multi-Factor Fundamental Factors

# Extract fundamental factors (P/E, P/B, ROE, etc.) via yfinance
uv run python -c "from src.ml.fundamental_features import FundamentalFeatureExtractor; e = FundamentalFeatureExtractor(); print(e.extract(['SPY','AAPL','MSFT'])); print(e.composite_score(e.extract(['SPY','AAPL','MSFT'])))"

# Run multi-factor backtest (long top N by composite score, monthly rebalance)
uv run scripts/backtest_multi_factor.py SPY,QQQ,XLK,JPM,XOM,CVX,JNJ,PG --start 2020-01-01

# Compute factor Information Coefficients
uv run scripts/backtest_multi_factor.py SPY,QQQ,XLK,JPM,XOM,CVX,JNJ,PG --start 2020-01-01 --factor-ic

# Sector-based multi-factor backtest
uv run scripts/backtest_multi_factor.py --sector tech --start 2020-01-01

# Custom top-n selection
uv run scripts/backtest_multi_factor.py SPY,JPM,XOM,CVX,PG,JNJ,WMT,HD --start 2020-01-01 --top-n 5

# Integrate fundamentals into ML backtest
uv run scripts/run_ml_backtest.py SPY --start 2020-01-01 --use-fundamentals

**Key files:**
| File | Purpose |
|------|---------|
| `src/ml/fundamental_features.py` | FundamentalFeatureExtractor, QuarterlyFundamentalProvider - 14 factors (Value/Quality/Size/Growth/Income/Risk/Sentiment) |
| `scripts/backtest_multi_factor.py` | Multi-factor scoring backtest with monthly rebalance, sector presets |
| `src/strategies/ml_strategy.py` | Integrated via `use_fundamentals` toggle |
| `scripts/run_ml_backtest.py` | Added `--use-fundamentals` flag |

**Factor categories:** P/E, P/B, P/S, EV/EBITDA, ROE, ROA, profit margin, debt/equity, market cap, revenue growth, earnings growth, dividend yield, beta, short % float.

### Phase R: Advanced NLP — FinBERT + SEC Filing Analysis

# FinBERT sentiment analysis (ProsusAI/finbert - first run downloads ~1.5GB model)
uv run python -c "from src.signals.sentiment.finbert import FinBERTSentiment; fb = FinBERTSentiment(); print(fb.predict(['Revenue grew 20% YoY', 'Company faces headwinds']))"

# Compare FinBERT vs LM Dictionary on same texts
uv run python -c "from src.signals.sentiment.finbert import FinBERTVsLMComparator; c = FinBERTVsLMComparator(); df = c.compare(['Strong profit growth', 'Revenue decline concerns', 'Market flat']); print(df[['finbert_sentiment','lm_sentiment','sign_agreement']])"

# SEC EDGAR filing scraper (look up CIK + list 10-K filings)
uv run python -c "from src.data_ingestion.sec_filing_scraper import SECFilingScraper; s = SECFilingScraper(); print(s.get_filings('AAPL', filing_type='10-K', limit=3))"

# Filing analyzer (text similarity, tone change, word drift)
uv run python -c "from src.signals.sentiment.filing_analyzer import FilingAnalyzer; a = FilingAnalyzer(); texts = ['Revenue grew 20%', 'Revenue growth slowed', 'Revenue declined sharply']; print(a.filing_features(texts))"

# Multi-source sentiment fusion
uv run python -c "from src.signals.sentiment.multi_source_fusion import MultiSourceFusion; mf = MultiSourceFusion(); r = mf.fuse(lm_score=0.15, finbert_score=0.30, filing_tone=-0.05); print(mf.summarize(r))"

# Train NLP+Financial fusion CatBoost model (Phase R4)
uv run scripts/train_nlp_fusion.py SPY --start 2016-01-01 --end 2024-12-31 --use-finbert --use-filing --json

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/sentiment/finbert.py` | FinBERTSentiment, FinBERTVsLMComparator - contextual financial sentiment |
| `src/data_ingestion/sec_filing_scraper.py` | SECFilingScraper - downloads 10-K/10-Q, extracts MD&A section |
| `src/signals/sentiment/filing_analyzer.py` | FilingAnalyzer - YoY similarity, tone change, emerging keywords |
| `src/signals/sentiment/multi_source_fusion.py` | MultiSourceFusion - combines LM+FinBERT+filing+sources, trade signals |
| `scripts/train_nlp_fusion.py` | NLP+Financial CatBoost fusion training, AUC comparison |

**Gate result (SPY 2016-2024):** FAIL — NLP features don't improve AUC (baseline 0.613 vs fusion 0.595, delta -0.019). NLP features are redundant with price features. Still include for regime-dependent value per plan.

### Phase S: Pairs Trading — Statistical Arbitrage with Cointegration (2026-05-16)

# Backtest a single pair
uv run scripts/backtest_pairs.py KO-PEP --start 2016-01-01 --end 2024-12-31

# Compare all 9 pre-built sector pairs side-by-side
uv run scripts/backtest_pairs.py --compare --start 2016-01-01 --end 2024-12-31

# Auto-discover pairs by correlation and cointegration
uv run scripts/backtest_pairs.py --auto-pairs --min-correlation 0.7 --top-n 5

# Sweep entry z-score thresholds
uv run scripts/backtest_pairs.py CVX-XOM --sweep-entry 1.5,2.0,2.5,3.0

# With ATR trailing stop exit
uv run scripts/backtest_pairs.py CVX-XOM --atr-exit 2.0

# Machine-readable JSON output
uv run scripts/backtest_pairs.py CVX-XOM --json

**Key files:**
| File | Purpose |
|------|---------|
| `src/strategies/pairs_trading_strategy.py` | Cointegration + rolling hedge ratio + z-score mean reversion |
| `scripts/backtest_pairs.py` | CLI with --compare / --sweep-entry / --auto-pairs / --atr-exit |

**Best pairs (2016-2024):**
| Pair | Sector | Return% | Sharpe | CoPval | AvgCorr |
|------|--------|---------|--------|--------|---------|
| CVX-XOM | Energy supermajors | +107.3 | 0.40 | 0.117 | 0.729 |
| DUK-SO | Electric utilities | +12.8 | 0.08 | 0.020 | 0.814 |
| JNJ-MRK | Pharma giants | +2.2 | 0.19 | 0.213 | 0.248 |

### Phase V: Strategy Architecture v2 — Short-Side, Kelly Sizing, Portfolio, Crypto (2026-05-16)

# Rules-first strategy with short-side enabled
uv run scripts/backtest_rules_first.py SPY --start 2022-01-01 --end 2022-12-31
# Note: use_short=True is the default; bearish patterns trigger self.sell()

# Strategy-aware Kelly position sizing (from Python)
uv run python -c "from src.risk.strategy_aware_sizing import StrategyAwareSizer; s = StrategyAwareSizer('trend', 100000); print(s.compute(0.55, avg_win=200, avg_loss=-150))"

# Correlation-aware portfolio allocation
uv run python -c "from src.portfolio.multi_asset_allocator import MultiAssetAllocator; import pandas as pd; import numpy as np; np.random.seed(0); returns = pd.DataFrame({'SPY': np.random.randn(100)*0.01, 'QQQ': np.random.randn(100)*0.015, 'XLK': np.random.randn(100)*0.014, 'TLT': np.random.randn(100)*0.008}, index=pd.date_range('2024-01-01', periods=100)); a = MultiAssetAllocator(); w = a.allocate({'SPY':0.6,'QQQ':0.5,'XLK':0.4,'TLT':-0.2}, returns); print(w)"

# Fetch crypto OHLCV via CCXT (free, no API key)
uv run python -c "from src.data_ingestion.crypto_provider import CCXTCryptoProvider; p = CCXTCryptoProvider(); df = p.fetch_ohlcv('BTC/USDT', '1d', limit=10); print(df)"

**Key files:**
| File | Purpose |
|------|---------|
| `src/strategies/rules_first_strategy.py` | Updated with `use_short` param for bearish pattern shorts |
| `src/risk/strategy_aware_sizing.py` | Kelly-derived sizing per strategy type (trend/MR/pairs/ML) |
| `src/portfolio/multi_asset_allocator.py` | Correlation-clustered multi-asset allocation (max 30% per cluster) |
| `src/data_ingestion/crypto_provider.py` | CCXTCryptoProvider wrapping ccxt (free Binance OHLCV, 8 symbols) |

### FMZ Strategy Conversions — PineScript/JS → Python (2026-05-16)

6 PineScript strategies and 1 JavaScript HFT signal module converted from the FMZ strategies
repository (5,807 files). All integrate with the BasePattern framework and RulesFirstStrategy.

**Pattern Detectors (7 new):**

| Detector | File | Strategy | Type |
|----------|------|----------|------|
| `AlphaBeast` | `src/patterns/fmz/alpha_beast.py` | Supertrend + RSI + Volume triple confirmation | CONTINUATION |
| `MultiFactorTrend` | `src/patterns/fmz/multi_factor_trend.py` | SAR + EMA + RSI + ADX quad confirmation | CONTINUATION |
| `MomentumZigZag` | `src/patterns/fmz/momentum_zigzag.py` | QQE/MACD/MA ZigZag with force detection | REVERSAL |
| `EMAMACDHF` | `src/patterns/fmz/ema_macd_hf.py` | EMA(9/21) + MACD(6,13,4) high-freq crossover | CONTINUATION |
| `AdaptiveBollinger` | `src/patterns/fmz/adaptive_bollinger.py` | BB breakout reversion with 4-layer exit | REVERSAL |
| `AIVolatilityBreakout` | `src/patterns/fmz/ai_volatility_breakout.py` | Gap fill + VWAP + compression breakout | BREAKOUT |

**Signal Module:**

| Module | File | Description |
|--------|------|-------------|
| `OrderFlowAccumulator` | `src/signals/order_flow.py` | Golden ratio (0.382) weighted order flow alpha factor `[-1, 1]` |

**PineScript Helper Functions:**
```bash
# All PineScript→NumPy equivalents in src/indicators/pinescript_helpers.py
uv run python -c "from src.indicators.pinescript_helpers import supertrend, sar, dmi, macd, qqe, vwap_simple; print('Available')"
```

**Test all conversions:**
```bash
uv run python -c "
from src.patterns.fmz import AlphaBeast, MultiFactorTrend, MomentumZigZag, EMAMACDHF, AdaptiveBollinger, AIVolatilityBreakout
from src.signals.order_flow import OrderFlowAccumulator, TradeTick
print('All 7 conversions imported successfully')
"
```

**Quick backtest with FMZ patterns included:**
```bash
# FMZ patterns auto-included in RulesFirstStrategy (reliability weights assigned)
uv run scripts/backtest_rules_first.py SPY --start 2023-01-01 --end 2025-12-31
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/indicators/pinescript_helpers.py` | 17 PineScript functions ported to NumPy (crossover, supertrend, sar, dmi, macd, qqe, vwap_simple, etc.) |
| `src/patterns/fmz/` | 6 FMZ strategy → Python pattern detectors |
| `src/signals/order_flow.py` | HFT order flow alpha factor (JS → Python) |
| `src/strategies/rules_first_strategy.py` | PATTERN_RELIABILITY updated with FMZ pattern weights |
| `src/strategies/backtest_py/multi_pattern_strategy_optimized.py` | _init_patterns() includes all 6 FMZ detectors |

---

## Phase 17: Resource-Driven Enhancements (R1-R4)

*Source: 华泰多因子系列1 + Beyond Fama-French + FMZ Strategies Repository*

### R1: IR-Weighted Pattern Synthesis

Replace hardcoded `PATTERN_RELIABILITY` with rolling Information Ratio weights. Weights adapt to regime shifts automatically.

```bash
# Backtest with IR weights enabled (replaces static reliability)
uv run scripts/backtest_rules_first.py SPY --start 2020-01-01 --end 2024-12-31 --ir-weights

# Custom IR window
uv run scripts/backtest_rules_first.py SPY --start 2020-01-01 --end 2024-12-31 --ir-weights --ir-weighting-window 504

# Sweep entry thresholds with IR weights
uv run scripts/backtest_rules_first.py SPY --sweep-entry 0.3,0.4,0.5,0.6,0.7 --ir-weights
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/ir_weighting.py` | IRWeighting class + ir_from_pattern_signals() convenience function |
| `src/strategies/rules_first_strategy.py` | `use_ir_weights` + `ir_weighting_window` params, `_init_ir_weights()` |

### R2: Factor Purification Module

Regress out sector and market cap from pattern signals before evaluating quality. Reveals whether a pattern has independent alpha or is a sector proxy.

```bash
# Purify a pattern signal (Python API)
uv run python -c "
from src.signals.factor_purification import FactorPurifier
import numpy as np
fp = FactorPurifier()
signal = np.random.randn(500)
sectors = np.array(['tech','fin','tech','health','fin']*100)
log_mcap = np.log(np.random.uniform(1e9, 1e12, 500))
purified, report = fp.purify(signal, sectors, log_mcap)
print(f'Purity: {report.purity_ratio:.3f}, Contaminated: {report.is_contaminated}')
"
```

**Key file:** `src/signals/factor_purification.py` — FactorPurifier class with OLS decomposition, purity_ratio, sector/mcap R2 breakdown.

### R3: 4-Step Pattern Evaluation Gate

Formal entry requirement for any new pattern entering the detector catalog. Four sequential tests: t-stat regression, return/risk classification, IC analysis, quantile backtest.

```bash
# Evaluate a pattern using OHLCV data
uv run scripts/evaluate_pattern.py --name "Gartley Pattern" \
    --cls src.patterns.harmonic.gartley.GartleyPattern \
    --ticker data/raw/SPY_daily.csv

# Evaluate with sector data
uv run scripts/evaluate_pattern.py --name "Alpha Beast" \
    --signals-file outputs/alpha_beast_signals.csv \
    --returns-file outputs/forward_returns.csv \
    --sectors-file outputs/sectors.csv

# Save JSON report
uv run scripts/evaluate_pattern.py --name "Gartley Pattern" \
    --cls src.patterns.harmonic.gartley.GartleyPattern \
    --ticker data/raw/SPY_daily.csv \
    --json-output reports/evaluations/gartley.json

# Custom thresholds
uv run scripts/evaluate_pattern.py --name "My Pattern" \
    --signals-file outputs/signals.csv --returns-file outputs/returns.csv \
    --tstat-threshold 2.5 --significance 0.01 --n-quantiles 10
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/evaluation_gate.py` | PatternEvaluationGate class (t-stat, classification, IC, quantile backtest) |
| `scripts/evaluate_pattern.py` | CLI for 4-step gate with --ticker or --signals-file input |

### R4: HP Filter Return Forecasting

Hodrick-Prescott filter for extracting smooth trend from factor returns. Huatai-validated: HP filter > EWMA > ARIMA > historical mean.

```bash
# Compute HP-filtered trend and expected return (Python API)
uv run python -c "
from src.ml.expected_returns import hp_filter, hp_expected_return, HPFilter
import numpy as np
ret = np.cumsum(np.random.randn(500) * 0.01)
trend = hp_filter(ret, lam=100000)
print(f'HP expected return: {hp_expected_return(ret):.6f}')
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/ml/expected_returns.py` | hp_filter(), hp_forecast(), hp_expected_return(), HPFilter class |
| `src/ml/__init__.py` | Exports HPFilter + lambda constants (DAILY/WEEKLY/MONTHLY/QUARTERLY) |

**Lambda reference:**
| Frequency | Lambda | Constant |
|-----------|--------|----------|
| Daily | 100,000 | HP_LAMBDA_DAILY |
| Weekly | 1,600 | HP_LAMBDA_WEEKLY |
| Monthly | 14,400 | HP_LAMBDA_MONTHLY |
| Quarterly | 1,600 | HP_LAMBDA_QUARTERLY |

---

## Phase 17 P1: R5-R8 — Factor Features, Scoring, Preprocessing, Collinearity

*Source: 华泰多因子 + Beyond Fama-French*

### R6: Liquidity Factor (CEI)

Compute Acharya-Pedersen Capital Efficiency Index: `CEI = 12-month change in market equity − 12-month cumulative return`. Also includes Amihud illiquidity and Roll spread.

```bash
# Compute CEI for SPY (uses yfinance for shares outstanding)
uv run python -c "
from src.ml.factor_features import compute_cei_dataframe, add_liquidity_features
import pandas as pd
df = pd.read_csv('data/raw/SPY_daily.csv', index_col=0, parse_dates=True)
cei = compute_cei_dataframe(df, ticker='SPY')
print(f'CEI range: {cei.min():.6f} to {cei.max():.6f}')
"

# Add all liquidity features (CEI + Amihud + Roll spread)
uv run python -c "
from src.ml.factor_features import add_liquidity_features
import pandas as pd
df = pd.read_csv('data/raw/SPY_daily.csv', index_col=0, parse_dates=True)
features = add_liquidity_features(df, ticker='SPY')
print(features[['cei', 'amihud_illiq', 'roll_spread']].describe())
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/ml/factor_features.py` | compute_cei(), compute_amihud_illiquidity(), compute_roll_spread(), add_liquidity_features() |
| `src/ml/__init__.py` | Exports all liquidity functions + CEI_WINDOW constant |

### R7: Multi-Dimensional Signal Scoring

Replace single `confidence` with 5-axis scoring: IC (predictive accuracy), IR (stability), turnover, diversity, overfitting risk. Composite = geometric mean of all 5 axes.

```bash
# Score signals on all 5 axes
uv run python -c "
from src.signals.scoring import MultiAxisScorer, quick_5axis_report
import numpy as np
np.random.seed(42)
n = 500
forward_ret = np.random.randn(n) * 0.01
signal_good = np.where(forward_ret > 0, 1, -1).astype(float)
signal_noise = np.random.choice([-1, 0, 1], n).astype(float)
scorer = MultiAxisScorer(ir_window=60, is_oos_split=400)
print(quick_5axis_report(signal_good, forward_ret, name='predictive'))
print(quick_5axis_report(signal_noise, forward_ret, name='noise'))

# Rank multiple signals by composite score
ranked = scorer.rank_by_composite(
    {'good': signal_good, 'noise': signal_noise}, forward_ret
)
for i, (name, score) in enumerate(ranked):
    print(f'{i+1}. {name} composite={score.composite:.4f}')
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/scoring.py` | MultiAxisScorer, MultiAxisScore dataclass, quick_5axis_report() |
| `src/signals/__init__.py` | Exports MultiAxisScorer, MultiAxisScore, quick_5axis_report |

### R8: MAD + Rank Data Standardization Pipeline

Two-step sklearn pipeline: (1) MAD-based outlier clipping, (2) non-parametric rank standardization. Handles crypto fat tails without normality assumptions.

```bash
# Test MAD clipping with an extreme outlier
uv run python -c "
from src.ml.preprocessing import MADOutlierClipper, RankStandardizer, mad_rank_pipeline
import numpy as np
np.random.seed(42)
x = np.random.standard_t(3, size=1000)
x[0] = 100  # extreme outlier
clipper = MADOutlierClipper(n_mad=5.0)
clipped = clipper.fit_transform(x.reshape(-1,1))
print(f'Max before: {x.max():.1f}, after: {clipped.max():.1f}')

# Full pipeline: MAD clip -> rank standardize to normal
ranked = mad_rank_pipeline(x, output_distribution='normal')
print(f'Ranked mean: {ranked.mean():.3f}, std: {ranked.std():.3f}')
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/ml/preprocessing.py` | MADOutlierClipper, RankStandardizer (sklearn transformers), mad_rank_pipeline() |
| `src/ml/__init__.py` | Exports all preprocessing classes + convenience functions |

### R5: Collinearity Analysis

Compute VIF (Variance Inflation Factor) matrix across pattern signal series. Flag redundant patterns and recommend synthesis (same category) or discard (different category, lower IC).

```bash
# Analyze collinearity with categories and IC scores
uv run python -c "
from src.signals.collinearity import analyze_collinearity
import pandas as pd, numpy as np
np.random.seed(42)
n = 500
base = np.random.randn(n)
p1 = base + np.random.randn(n) * 0.1
p2 = base + np.random.randn(n) * 0.1
p3 = np.random.randn(n)
df = pd.DataFrame({'p1': p1, 'p2': p2, 'p3': p3})
cats = {'p1': 'trend', 'p2': 'trend', 'p3': 'reversal'}
ic = {'p1': 0.15, 'p2': 0.12, 'p3': 0.08}
result = analyze_collinearity(df, cats, ic)
print('VIF:')
print(result.vif_df.round(1))
print(f'Redundant: {result.n_redundant}, Synthesize: {result.n_synthesize}, Discard: {result.n_discard}')
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/collinearity.py` | analyze_collinearity(), CollinearityReport, RedundancyPair, get_discard/synthesize_recommendations() |
| `src/signals/__init__.py` | Exports all collinearity classes and functions |

**Phase 17 P1 Key Files Summary:**
| File | R# | Purpose |
|------|-----|---------|
| `src/ml/factor_features.py` | R6 | CEI + Amihud + Roll spread liquidity factors |
| `src/signals/scoring.py` | R7 | 5-axis multi-dimensional signal scoring |
| `src/ml/preprocessing.py` | R8 | MAD outlier clipping + rank standardization |
| `src/signals/collinearity.py` | R5 | VIF collinearity analysis + redundancy detection |

### R9: Return Factor vs Risk Factor Classification

Classify each pattern signal as "return factor" (directional, entry signals) or "risk factor" (variance explanatory, position sizing modifiers). Uses OLS regression with t-test via statsmodels.

```bash
# Classify a single pattern signal
uv run python -c "
from src.signals.classification import FactorClassifier
import numpy as np
np.random.seed(42)
signal = np.random.randn(500)  # pattern signal
returns = 0.1 * signal + np.random.randn(500) * 0.1  # strong signal
fc = FactorClassifier()
result = fc.classify('PatternName', signal, returns)
print(f'Class: {result.factor_class.value}, t={result.tstat:.2f}, p={result.pvalue:.4f}')
"

# Bulk classify multiple patterns
uv run python -c "
from src.signals.classification import classify_patterns, get_entry_patterns, get_risk_patterns
import pandas as pd, numpy as np
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'solid': 0.1 + np.random.randn(n),
    'noisy': np.random.randn(n),
    'random': np.random.randn(n) * 0.01,
})
returns = 0.15 * df['solid'] + np.random.randn(n) * 0.1
bulk = classify_patterns(df, returns)
print(f'Entry patterns: {get_entry_patterns(bulk)}')
print(f'Risk patterns:  {get_risk_patterns(bulk)}')
print(f'Return: {bulk.return_count}, Risk: {bulk.risk_count}, Ambiguous: {bulk.ambiguous_count}')
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/classification.py` | FactorClassifier, FactorClassification, FactorClass, BulkClassification, classify_patterns(), get_entry_patterns(), get_risk_patterns() |
| `src/signals/__init__.py` | Exports all classification classes and functions |

### R10: Performance Attribution Decomposition

Decompose strategy returns via OLS factor regression: r_P = alpha + beta_market + beta_sector + beta_style + epsilon. Identifies genuine alpha vs factor beta.

```bash
# Basic: strategy vs market
uv run scripts/attribution.py --strategy outputs/strategy_returns.csv \
    --market data/raw/SPY_daily.csv --name "RulesFirst"

# With sector and style factors
uv run scripts/attribution.py --strategy outputs/strategy_returns.csv \
    --market data/raw/SPY_daily.csv --sector outputs/sector_returns.csv \
    --style outputs/style_returns.csv --name "RulesFirst"

# Save JSON report
uv run scripts/attribution.py --strategy outputs/strategy_returns.csv \
    --market data/raw/SPY_daily.csv --json-output outputs/attribution.json
```

**Key files:**
| File | Purpose |
|------|---------|
| `scripts/attribution.py` | CLI: decompose_strategy(), AttributionReport, FactorExposure |

### R11: Default Risk Factor (Merton DtD)

Merton Distance-to-Default: DtD = (ln(V/D) + (r - sigma^2/2)T) / (sigma * sqrt(T)). Captures credit quality signal that price patterns miss. Uses FMP API for market cap and total debt (free tier).

```bash
# Compute DtD for a single firm
uv run python -c "
from src.ml.factor_features import compute_dtd
dtd = compute_dtd(market_cap=500e9, total_debt=50e9, equity_volatility=0.20)
print(f'Healthy firm DtD: {dtd:.2f}')
dtd = compute_dtd(market_cap=10e9, total_debt=9e9, equity_volatility=0.50)
print(f'Risky firm DtD:   {dtd:.2f}')
"

# Add DtD to OHLCV DataFrame
uv run python -c "
from src.ml.factor_features import add_dtd_features
import pandas as pd, numpy as np
dates = pd.date_range('2024-01-01', periods=300, freq='B')
df = pd.DataFrame({
    'Close': 100 + np.cumsum(np.random.randn(300) * 2),
    'Volume': np.random.randint(1000000, 5000000, 300)
}, index=dates)
result = add_dtd_features(df)
print(result.columns.tolist())
print(f'DtD NaN count (no FMP data): {result.dtd.isna().sum()}')
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/ml/factor_features.py` | compute_dtd(), compute_dtd_dataframe(), add_dtd_features() (appended to R6 liquidity features) |

### R12: QRAFTI Standardized Evaluation Protocol

14-test diagnostic suite (Novy-Marx/Velikov 2023) covering factor construction, signal quality, and implementation feasibility. Standardized audit before production deployment. >= 10/14 tests must pass.

```bash
# Evaluate a single strategy
uv run scripts/evaluate_qrafti.py --signals outputs/pattern_signals.csv \
    --returns outputs/forward_returns.csv --name "MyPattern"

# With sector data for sector neutrality test
uv run scripts/evaluate_qrafti.py --signals outputs/pattern_signals.csv \
    --returns outputs/forward_returns.csv --sectors outputs/sectors.csv \
    --name "MyPattern" --report-path outputs/qrafti_report.json

# Batch evaluate multiple strategies
uv run scripts/evaluate_qrafti.py --batch outputs/strategies/ \
    --returns data/returns.csv --report-dir outputs/qrafti/
```

**QRAFTI Test Categories:**
| # | Test | Category |
|---|------|----------|
| 1 | Data Completeness | Construction |
| 2 | Sufficient Bars | Construction |
| 3 | Signal Variation | Construction |
| 4 | Sector Neutrality | Construction |
| 5 | IC Significance | Quality |
| 6 | IC Stability (IR) | Quality |
| 7 | Quintile Spread | Quality |
| 8 | |t|>2 Ratio | Quality |
| 9 | IC Decay | Quality |
| 10 | Turnover Rate | Feasibility |
| 11 | Signal Stability (ACF1) | Feasibility |
| 12 | IS/OOS IC Ratio | Feasibility |
| 13 | Look-Ahead Bias | Feasibility |
| 14 | Max Neg IC Run | Quality |

**Key files:**
| File | Purpose |
|------|---------|
| `scripts/evaluate_qrafti.py` | evaluate_qrafti(), QRAFTIReport, TestResult, 14-test protocol, batch mode |

**Phase 17 P2 Key Files Summary:**
| File | R# | Purpose |
|------|-----|---------|
| `src/signals/classification.py` | R9 | Return/risk factor classification + bulk classify |
| `scripts/attribution.py` | R10 | Performance attribution decomposition CLI |
| `src/ml/factor_features.py` | R11 | Merton Distance-to-Default default risk factor |
| `scripts/evaluate_qrafti.py` | R12 | 14-test QRAFTI evaluation protocol + batch mode |

---

## Phase 17 P3: R13-R14 — Full Optimization Pipeline + Factor Engine Wrapper

*Source: 华泰多因子系列1 + Beyond Fama-French*

### R13: Full IR→HP→Risk→QP Optimization Pipeline

End-to-end 华泰 4-phase optimization: IR-weighted synthesis + HP-filtered return forecast + covariance shrinkage + quadratic programming via scipy. Optimizes portfolio weights under risk cap and weight constraints.

```bash
# Optimize portfolio weights for multiple assets (with synthetic signals)
uv run scripts/run_huatai_pipeline.py SPY QQQ IWM GLD --start 2020-01-01 --end 2024-12-31

# With stricter risk cap and max weight
uv run scripts/run_huatai_pipeline.py SPY QQQ XLK GLD TLT --risk-cap 0.10 --max-weight 0.25

# Use custom CSV files for signals, returns, and prices
uv run scripts/run_huatai_pipeline.py --signals-file signals.csv --returns-file returns.csv --prices-file prices.csv

# Save results to JSON
uv run scripts/run_huatai_pipeline.py SPY QQQ IWM --start 2020-01-01 --json-output outputs/huatai_weights.json

# Demo mode with synthetic data (no symbols needed)
uv run scripts/run_huatai_pipeline.py --seed 123
```

**Pipeline phases:**
| Phase | Step | Description |
|-------|------|-------------|
| 1 | IR-Weighted Synthesis | Aggregate pattern signals using rolling IR weights |
| 2 | HP-Filtered Return Forecast | Extract smooth trend via Hodrick-Prescott filter |
| 3 | Factor Covariance Risk Model | Ledoit-Wolf shrinkage covariance estimation |
| 4 | Quadratic Programming | max wᵀμ − (γ/2) wᵀΣw via SLSQP with risk cap |

**Key files:**
| File | Purpose |
|------|---------|
| `src/optimization/huatai_pipeline.py` | HuataiPipeline class + HuataiResult dataclass + run_huatai_pipeline() convenience |
| `src/optimization/__init__.py` | Module exports |
| `scripts/run_huatai_pipeline.py` | CLI with argparse, data fetching, synthetic signals, JSON output |

### R14: Factor Engine Library Wrapper

Thin try-install wrapper around the Factor Engine open-source Python library. If available, provides 11 validated factor computations (momentum, size, value, quality, low_volatility, profitability, investment, leverage, dividend_yield, earnings_yield, accruals). Graceful fallback if not installed — no lock-in.

```bash
# Check if Factor Engine is available
uv run python -c "from src.ml.factor_engine import get_factor_engine; e=get_factor_engine(); print(f'Available: {e.available}, Factors: {e.status.supported_factors}')"

# Install Factor Engine (if desired)
uv add factor-engine

# Compute a factor if engine is available
uv run python -c "
from src.ml.factor_engine import FactorEngineWrapper
import pandas as pd
import numpy as np
engine = FactorEngineWrapper()
if engine.available:
    prices = pd.DataFrame({'A': 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))})
    returns = prices.pct_change()
    momentum = engine.compute_factor('momentum', prices, returns)
    print(momentum.tail())
else:
    print('Factor Engine not installed — use fallback factor implementations.')
"

# Add all available Factor Engine features to a dataframe
uv run python -c "
from src.ml.factor_engine import FactorEngineWrapper
import pandas as pd
engine = FactorEngineWrapper()
if engine.available:
    df = pd.DataFrame(index=pd.date_range('2024-01-01', periods=100, freq='B'))
    # ... add factor engine features to df via engine.add_factor_features(df, prices, returns)
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/ml/factor_engine.py` | FactorEngineWrapper, FactorEngineStatus, get_factor_engine(), KNOWN_FACTORS (11 factors) |

**Phase 17 P3 Key Files Summary:**
| File | R# | Purpose |
|------|-----|---------|
| `src/optimization/huatai_pipeline.py` | R13 | 4-phase optimization pipeline (IR→HP→Risk→QP) |
| `scripts/run_huatai_pipeline.py` | R13 | CLI for Huatai pipeline |
| `src/ml/factor_engine.py` | R14 | Factor Engine library wrapper (11 factors, try-install) |

---

## Phase 19: Graphify Knowledge Graph (NEW — 2026-05-16)

### R19a: Graphify — Project Codebase Knowledge Graph

```bash
# Build/update knowledge graph from Python source (AST only, no API cost)
uv run graphify update src/

# Query the knowledge graph with natural language
uv run graphify query "how does rules_first_strategy compute ATR trailing stops"

# Find shortest dependency path between two concepts
uv run graphify path "pattern detection" "backtest engine"

# Get plain-language explanation of a node and its neighbors
uv run graphify explain "<node_label>"

# Benchmark token reduction vs naive full-corpus approach
uv run graphify benchmark src/graphify-out/graph.json

# Re-extract with LLM semantic enrichment (requires API key)
uv run graphify extract src/ --backend gemini --model gemini-2.5-flash

# Run clustering only on existing graph (no re-extraction)
uv run graphify cluster-only src/ --no-viz

# Emit Mermaid-based architecture/call-flow HTML
uv run graphify export callflow-html

# Install graphify hooks for auto-update on commit
uv run graphify hook install
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/graphify-out/graph.json` | Queryable knowledge graph (7068 nodes, 11170 edges, 498 communities) |
| `src/graphify-out/GRAPH_REPORT.md` | Human-readable report with community clusters |

### R19c: Matt Pocock Engineering Skills

```bash
# Use in .kilo/skills/engineering/mattpocock/:
# /diagnose           — 6-phase systematic debugging
# /tdd                — Red-green-refactor with AI
# /grill-with-docs    — Domain glossary-driven development
# /handoff            — Session compaction for next agent
# /improve-codebase-architecture — Architecture debt reduction
# /to-prd             — Feature request → structured PRD
# /to-issues          — PRD → GitHub issues
# /caveman            — Token-efficient prompting
```

### R19b: lean-ctx MCP Server — Context Compression

```bash
# Installed globally via npm: npm install -g lean-ctx-bin
# Setup: lean-ctx setup (configures shell hooks + MCP for all detected AI tools)
# Verify: lean-ctx doctor
# Usage is automatic — compresses shell output + caches file reads (60-95% token savings)
```

## Phase 18: Useful Repos Integration (NEW — 2026-05-16)

### R18a: Eiten Portfolio Optimization (Eigen/MVP/MSR/GA + RMT)

```bash
# Optimize portfolio with MSR strategy (default)
uv run python scripts/optimize_portfolio.py --symbols SPY QQQ TLT GLD XLK --strategy msr

# Compare all 4 optimization strategies
uv run python scripts/optimize_portfolio.py --compare --symbols SPY QQQ TLT GLD XLK IWM

# Disable RMT covariance denoising
uv run python scripts/optimize_portfolio.py --compare --symbols SPY QQQ TLT GLD XLK --no-rmt

# GA with custom parameters + reproducibility
uv run python scripts/optimize_portfolio.py --symbols SPY QQQ TLT GLD XLK --strategy ga --ga-generations 100 --ga-seed 42

# Use specific date range
uv run python scripts/optimize_portfolio.py --compare --symbols SPY QQQ TLT GLD XLK --start 2023-01-01 --end 2024-12-31

# Save results to JSON
uv run python scripts/optimize_portfolio.py --compare --symbols SPY QQQ TLT GLD XLK --json-output reports/portfolio_opt.json

# Optimize from signal scores (JSON file with {symbol: score})
uv run python scripts/optimize_portfolio.py --from-signals reports/batch/signals.json --strategy mvp

# Load returns from CSV instead of yfinance
uv run python scripts/optimize_portfolio.py --returns-csv data/returns.csv --strategy msr
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/portfolio/eiten_builder.py` | Unified portfolio optimizer (Eigen/MVP/MSR/GA + RMT) |
| `src/portfolio/eiten_adapters/` | 5 strategy modules (eigen, mvp, msr, ga, rmt_filtering) |
| `scripts/optimize_portfolio.py` | CLI with --compare, --strategy, --from-signals |

### R18b: Scrapling Financial Data Pipeline

```bash
# Scrape insider trades for a ticker
uv run python scripts/scrape_financial_data.py AAPL --insider --limit 10

# Scrape financial news headlines
uv run python scripts/scrape_financial_data.py AAPL --news --days 7

# List SEC filings (10-K, 10-Q)
uv run python scripts/scrape_financial_data.py AAPL --sec-filings --limit 5

# Get earnings calendar
uv run python scripts/scrape_financial_data.py AAPL --earnings --days-ahead 30

# Run all scrapers for a ticker
uv run python scripts/scrape_financial_data.py AAPL --all

# Batch scrape multiple tickers
uv run python scripts/scrape_financial_data.py --batch AAPL MSFT GOOGL --insider --news

# Save results to JSON
uv run python scripts/scrape_financial_data.py AAPL --all --json-output reports/scraped_AAPL.json
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/data_ingestion/financial_scraper.py` | Scrapling-based scraper (insider/news/SEC/earnings) |
| `scripts/scrape_financial_data.py` | CLI with --insider/--news/--sec-filings/--earnings |

### R18c: Agent-Skills Engineering Workflows

```bash
# 6 engineering workflow skills copied to .kilo/skills/engineering/:
#   spec-driven-development/    — Write spec before code
#   test-driven-development/    — Red-green-refactor cycles
#   code-review-and-quality/    — Systematic code review
#   debugging-and-error-recovery/ — Structured debugging
#   performance-optimization/   — Profiling and optimization
#   shipping-and-launch/        — Production deployment

# Anti-rationalization table added to .kilo/global-rules.md
# (10 trading-specific guardrails preventing common AI excuses)
```

---

## Phase 6d: PDF Insight Integration (NEW — 2026-05-16)

### C9: Honest Walk-Forward Paper Trading

```bash
# Single ticker OOS walk-forward with expanding-window pattern signal precomputation
uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01

# Multi-ticker basket (14 tickers)
uv run scripts/paper_trade_wf_honest.py --basket --start 2025-01-01

# Custom entry threshold and trail stop
uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01 --entry-threshold 0.55 --trail-stop-atr 3.0

# JSON output for portfolio aggregation
uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01 --json reports/wf/SPY.json
```

### C10: Portfolio-Level Backtest

```bash
# Equal-weight portfolio of SPY+QQQ+GLD+XLK with monthly rebalancing
uv run scripts/backtest_portfolio.py --tickers SPY QQQ GLD XLK --start 2016-05-12

# Custom weights and rebalancing period
uv run scripts/backtest_portfolio.py --tickers SPY QQQ TLT GLD --weights 0.3 0.3 0.2 0.2 --rebalance 63

# Basket mode (pre-configured 14-instrument basket)
uv run scripts/backtest_portfolio.py --basket --start 2016-05-12
```

### C11: Volume/OI Pattern Validation

Done via code defaults — `volume_filter=True` in HeadAndShoulders, InverseHeadAndShoulders, DoubleTop, DoubleBottom.
Volume dissipation on right shoulder now required for Head & Shoulders signal generation.

### C12: Multi-TP Exit Logic

Active in `RulesFirstStrategy` via params:
- `use_multi_tp=True` (default: False)
- `tp1_atr=1.5` — first target at 1.5x ATR, closes 50% position
- `tp2_atr=3.0` — second target at 3.0x ATR, closes remaining
- `move_sl_to_be=True` — move stop loss to breakeven after TP1 hit

```bash
# Rules-first backtest with multi-TP enabled
uv run scripts/backtest_rules_first.py SPY --use-multi-tp

# With short-side also enabled
uv run scripts/backtest_rules_first.py SPY --use-multi-tp --use-short

# OOS 2025 with both (recommended production config)
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2025-12-31 \
  --use-multi-tp --use-short --min-reliability 0.70 --entry-threshold 0.55
```

### C13: Gap Pattern Hierarchy

Implemented in `src/patterns/breakout/gap.py` — 4 gap types (Breakaway, Continuation, Exhaustion, Common).
Breakaway gaps traded directionally, Exhaustion gaps faded. Size filter >2.5x ATR skips noise.

### C14: Extended Harmonic Patterns

5 harmonic detectors: Butterfly, Bat, Crab, Cypher, Shark in `src/patterns/harmonic/extended.py`.
All registered in RulesFirstStrategy with reliability weights and paper_trade_wf_honest.py.

### C15: Pipe Pattern Detector

Two-bar mechanical reversal in `src/patterns/complex/pipe.py`. Zero parameters, non-Fibonacci.
Bullish: two red candles, second engulfs first. Bearish: two green candles, second engulfs first.

### C16: Dead Cat Bounce >=15% Threshold

Enforced in `src/patterns/classic/dead_cat_bounce.py`:
- Event-day decline >=15% required (default `event_decline_pct=0.15`)
- Bounce retracement 50-62% required
- Target = 100% of event day range

### C17: Empirical Pattern Reliability Calibration

```bash
# Full calibration on SPY training period (2016-2024)
uv run scripts/calibrate_pattern_reliability.py --ticker SPY --start 2016-01-01 --end 2024-12-31

# Quick test (top 10 patterns only)
uv run scripts/calibrate_pattern_reliability.py --ticker SPY --quick --output reports/calibration/quick.json

# Apply calibrated weights to source files
uv run scripts/calibrate_pattern_reliability.py --ticker SPY --start 2016-01-01 --end 2024-12-31 --apply
```

**Key files:**
| File | Purpose |
|------|---------|
| `scripts/paper_trade_wf_honest.py` | C9: Honest walk-forward paper trading |
| `scripts/backtest_portfolio.py` | C10: Portfolio-level backtest |
| `src/patterns/harmonic/extended.py` | C14: 5 harmonic pattern detectors |
| `src/patterns/complex/pipe.py` | C15: Pipe pattern detector |
| `scripts/calibrate_pattern_reliability.py` | C17: Empirical reliability calibration |

---

## Phase 20: System Hardening & Signal Quality (IMPLEMENTED — 2026-05-17)

> Source: Empirical session testing across all modules. 7 items (H1-H7). Full plan: `progress_docs/plans/20-system-hardening.md`.

### H1: Multi-TP as System Default (P0) ✅

Multi-TP partial take-profit at 1.5x ATR + SL-to-breakeven. Now enabled by default in all strategies. Mechanically boosts Sharpe +124% with zero signal changes.

```bash
# Multi-TP is now the default (no flag needed)
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2025-12-31

# Disable if needed
uv run scripts/backtest_rules_first.py SPY --no-use-multi-tp
```

### H4: Adaptive IR-Weighting Scalar Mode (P1) ✅

IR-weighting now supports two modes: `scalar` (default) applies 0.3x-0.7x IR-derived modifiers to base reliability weights, preserving trade count while filtering noise. `gate` mode (legacy) zeros out negative-IR patterns.

```bash
# Scalar IR mode (default when --ir-weights is set)
uv run scripts/backtest_rules_first.py SPY --ir-weights --ir-weighting-mode scalar

# Legacy gate mode (zeroes negative IR patterns)
uv run scripts/backtest_rules_first.py SPY --ir-weights --ir-weighting-mode gate
```

### H3: Sector-Specific Multi-Factor Scoring (P1) ✅

Per-sector fundamental factor models based on empirical sector IC divergence. Feeds into RulesFirstStrategy as `--use-multi-factor` confidence modifier.

```bash
# RulesFirstStrategy with multi-factor modifier
uv run scripts/backtest_rules_first.py SPY --use-multi-factor --multi-factor-file reports/factors/tech_score.csv

# Sector factor analysis via FundamentalScorer
uv run python -c "from src.signals.fundamental_scorer import FundamentalScorer; \
  s = FundamentalScorer(); print(s.compute_sector_score('Technology', pe_ratio=22, ps_ratio=5.1, ev_ebitda=15))"

# Multi-factor backtest with yfinance auto-fetching (tech sector)
uv run scripts/backtest_multi_factor.py --sector tech --top-n 5 --start 2020-01-01 --factor-ic
```

### H2/H7: Pattern Quality Registry + Evaluation Sweep (P0/P3) ✅

4-step formal validation (t-stat, return/risk, IC, quantile) on all 47 registered pattern detectors. FAIL patterns get 0.5x weight and require 2+ confluence. Registry consumed by RulesFirstStrategy for dynamic signal quality gating.

```bash
# Batch sweep all 47 pattern detectors through 4-step gate
uv run scripts/sweep_pattern_gates.py --symbol SPY \
  --output reports/pattern_gate/all_patterns.json

# Load quality registry from sweep results
uv run python -c "from src.signals.pattern_quality_registry import PatternQualityRegistry; \
  r = PatternQualityRegistry.load('reports/pattern_gate/all_patterns.json'); print(r.summary_table())"

# Single pattern evaluation
uv run scripts/evaluate_pattern.py --name "Head and Shoulders" \
  --ticker data/raw/SPY_daily.csv \
  --cls src.patterns.complex.head_shoulders.HeadAndShoulders
```

### H5: GA Portfolio Post-Backtest Step (P2) ✅

Auto-run GA portfolio optimization on batch backtest results via `--optimize-portfolio` flag.

```bash
# Batch backtest with automatic GA portfolio optimization
uv run scripts/backtest_rules_batch.py --start 2025-01-01 --optimize-portfolio

# Standalone portfolio optimization
uv run scripts/optimize_portfolio.py --from-signals reports/batch/SPY_results.json --strategy ga
```

### H6: BTC Configuration Audit (P2) ✅

Systematic sweep across entry thresholds and reliability values to diagnose BTC IS divergence and find optimal crypto config.

```bash
# Full BTC config sweep
uv run scripts/audit_btc_config.py \
  --entry-sweep 0.45,0.55,0.65,0.75 \
  --reliability-sweep 0.4,0.5,0.6,0.7 \
  --start 2016-01-01 --end 2024-12-31

# With custom output path
uv run scripts/audit_btc_config.py --json-output reports/btc_audit/custom.json
```

**Key files:**
| File | Purpose |
|------|---------|
| `progress_docs/plans/20-system-hardening.md` | Full Phase 20 plan with task table, execution order, expected impact |
| `src/signals/pattern_quality_registry.py` | H2: Pattern→gate result mapping, FAIL→0.5x weight, confluence requirements |
| `src/signals/fundamental_scorer.py` | H3: Sector-specific factor scoring, 11 sector models |
| `src/signals/ir_weighting.py` | H4: Scalar mode (0.3x-0.7x) + legacy gate mode |
| `scripts/sweep_pattern_gates.py` | H7: Batch evaluate all 47 pattern detectors |
| `scripts/audit_btc_config.py` | H6: Systematic BTC config sweep |
| `scripts/backtest_rules_first.py` | Updated: `--use-multi-tp` default=True, `--ir-weighting-mode`, `--use-multi-factor` |
| `scripts/backtest_rules_batch.py` | Updated: `--optimize-portfolio` post-backtest GA step |
| `src/strategies/rules_first_strategy.py` | Updated: multi_tp=True default, scalar IR, multi-factor modifier |
| `src/strategies/combined_strategy.py` | Updated: multi-TP exit logic added |
| `src/strategies/ml_strategy.py` | Updated: multi_tp=True default |

---

## Phase 21: Quant-Resources Signal Enhancers ✅ IMPLEMENTED

### Q1: VIX Regime Gate (P0) ✅

VIX-based regime gating for trade signals. Classifies COMPLACENT/NORMAL/ELEVATED/STRESS regimes and scales signal scores during stress periods.

```bash
# Backtest with VIX regime gate (default: 0.30x during stress, 0.75x during elevated)
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --use-vix-gate

# Custom stress/elevated multipliers
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --use-vix-gate \
  --vix-stress-mult 0.20 --vix-elevated-mult 0.60

# Python API
uv run python -c "from src.signals.vix_regime_gate import VixRegimeGate; gate = VixRegimeGate(); gate.fit(); print(gate.regime(), gate.multiplier())"
```

### Q2: Yield Curve Gate (P0) ✅

Yield curve inversion (2s10s spread) as binary macro regime flag. Inversion → defensive 0.50x signal multiplier. Fetches Treasury yields from FMP free tier.

```bash
# Backtest with yield curve gate
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --use-yield-curve-gate

# Custom inversion multipliers
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --use-yield-curve-gate \
  --yield-inversion-mult 0.30 --yield-near-inversion-mult 0.60

# Python API
uv run python -c "from src.signals.yield_curve_gate import YieldCurveGate; gate = YieldCurveGate(); gate.fit(); print(gate.current_spread(), gate.is_inverted())"
```

### Q3: GARCH/EGARCH Volatility Forecasting (P1) ✅

Forward-looking volatility forecasts using GARCH/EGARCH/GJR-GARCH from the `arch` library. Complements CatBoost VolatilityForecaster.

```bash
# Compare GARCH variants on SPY
uv run scripts/garch_forecast.py SPY --start 2020-01-01 --end 2024-12-31

# Specific model with longer horizon
uv run scripts/garch_forecast.py SPY --model egarch --horizon 10

# Compare against CatBoost baseline
uv run scripts/garch_forecast.py SPY --compare-catboost

# Python API
uv run python -c "from src.ml.garch_forecaster import GARCHForecaster; import numpy as np; fc = GARCHForecaster(); fc.fit(np.random.normal(0, 0.01, 500))
"
```

### Q4: Put/Call Ratio + GEX Sentiment (P1) ✅

Options-derived sentiment signals. Put/Call ratio from FMP (with VIX-based proxy fallback). GEX proxy from VIX level (dealer gamma positioning).

```bash
# Python API
uv run python -c "
from src.signals.options_sentiment import OptionsSentimentProvider
import pandas as pd
provider = OptionsSentimentProvider()
provider.fit(start='2024-01-01')
sentiment = provider.get_sentiment(pd.date_range('2024-01-02', '2024-12-31', freq='B'), 'SPY')
print('Mean sentiment:', sentiment.mean())
"
```

### Q5: Model Validation — PSI/KS/Gini (P2) ✅

Industry-standard model monitoring framework. Population Stability Index for feature drift, KS statistic for discrimination decay, Gini coefficient for overall quality.

```bash
# Python API
uv run python -c "
from src.ml.model_validation import ModelValidator
import numpy as np
v = ModelValidator()
report = v.validate(
    np.random.default_rng(42).beta(3, 2, 500),
    np.random.default_rng(42).beta(2, 3, 200),
    (np.random.default_rng(42).random(500) > 0.5).astype(int),
    (np.random.default_rng(42).random(200) > 0.5).astype(int)
)
print(report.summary())
print('Needs retraining:', report.needs_retraining)
"
```

### Q6: Copula Tail-Risk Models (P2) ✅

Gaussian and t-copulas for joint extreme risk modeling in multi-asset portfolios. Fixes independence assumption in current VaR/CVaR.

```bash
uv run python -c "
from src.risk.copula_risk import GaussianCopulaRisk, TCopulaRisk
import numpy as np
returns = np.random.default_rng(42).normal(0.001, 0.02, (500, 3))
gc = GaussianCopulaRisk(); gc.fit(returns); print(gc.evaluate().summary())
tc = TCopulaRisk(df=4); tc.fit(returns); print(tc.evaluate().summary())
"
```

### Q7: Market Impact — Almgren-Chriss (P3) ✅

Quantifies permanent + temporary price impact to bridge backtest→live.

```bash
uv run python -c "
from src.risk.market_impact import MarketImpact
mi = MarketImpact()
print(mi.evaluate(notional=100_000, daily_volume=50_000_000, annual_vol=0.20).summary())
"
```

### Q8: Order Book Dynamics Features (P3) ✅

Bid-ask imbalance proxy, VPIN (informed trading probability), volume imbalance, streak detection, absorption ratio.

```bash
uv run python -c "
from src.signals.order_book_features import OrderBookFeatures, OrderBookSignal
import pandas as pd; import numpy as np
rng = np.random.default_rng(42)
df = pd.DataFrame({
    'Open': 500 + np.cumsum(rng.normal(0, 5, 252)),
    'High': 510 + np.cumsum(rng.normal(0, 5, 252)),
    'Low': 490 + np.cumsum(rng.normal(0, 5, 252)),
    'Close': 500 + np.cumsum(rng.normal(0.5, 5, 252)),
    'Volume': 1e7 + np.abs(rng.normal(0, 2e6, 252)),
}, index=pd.date_range('2024-01-01', periods=252))
df['High'] = df[['Open','Close']].max(axis=1) + np.abs(rng.normal(2, 1, 252))
df['Low'] = df[['Open','Close']].min(axis=1) - np.abs(rng.normal(2, 1, 252))
print(OrderBookFeatures().compute(df).columns.tolist())
print(OrderBookSignal().compute_signal(df).describe())
"
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/signals/vix_regime_gate.py` | Q1: VIX regime gating (Complacent/Normal/Elevated/Stress) |
| `src/signals/yield_curve_gate.py` | Q2: Yield curve inversion + credit spread macro gate |
| `src/ml/garch_forecaster.py` | Q3: GARCH/EGARCH/GJR-GARCH volatility forecasting |
| `src/signals/options_sentiment.py` | Q4: Put/Call ratio + GEX sentiment provider |
| `src/ml/model_validation.py` | Q5: PSI/KS/Gini model validation framework |
| `src/risk/copula_risk.py` | Q6: Gaussian + t-copula tail-risk models |
| `src/risk/market_impact.py` | Q7: Almgren-Chriss market impact model |
| `src/signals/order_book_features.py` | Q8: Order book dynamics + microstructure signals |
| `scripts/garch_forecast.py` | Q3 CLI: GARCH model comparison + CatBoost baseline |
| `scripts/backtest_rules_first.py` | Updated: `--use-vix-gate` + `--use-yield-curve-gate` flags |
| `src/strategies/rules_first_strategy.py` | Updated: VIX + yield curve gate integration |

### Block D: Structural Break + ARIMA+GARCH + State Space (Implemented 2026-05-18)

#### D3: Structural Break / Unit-Root Tests

ADF, KPSS, Chow, and Bai-Perron breakpoint tests for regime shift detection and stationarity assessment. Essential for mean-reversion eligibility and regime-change alerts.

```bash
# Full analysis on price series
uv run python -c "
from src.ml.structural_break import StructuralBreakDetector
import numpy as np; np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))
r = StructuralBreakDetector(max_breaks=3).run(prices)
print(f'Stationary: {r.is_stationary}, Breaks: {len(r.breakpoints)}')
"

# On a pandas price series (with log transform)
uv run python -c "
from src.ml.structural_break import StructuralBreakDetector
import pandas as pd; import numpy as np
p = pd.Series(100*np.exp(np.cumsum(np.random.randn(300)*0.01)), name='TEST',
    index=pd.date_range('2020-01-01', periods=300, freq='B'))
r = StructuralBreakDetector().run_on_prices(p)
print(f'Stationary: {r.is_stationary}, Breaks: {len(r.breakpoints)}')
# On returns: expected stationary
r2 = StructuralBreakDetector().run_on_returns(p)
print(f'Returns stationary: {r2.is_stationary}')
"

# Standalone ADF/KPSS
uv run python -c "
from src.ml.structural_break import adf_test, kpss_test
import numpy as np
rw = np.cumsum(np.random.randn(200))
print(adf_test(rw))  # non-stationary
print(kpss_test(np.random.randn(200)))  # stationary
"
```

#### D10: Rolling ARIMA+GARCH Hybrid Forecaster

Combined mean + volatility forecasting with rolling window refitting. ARIMA for conditional mean, GARCH for conditional variance. Cached between retrains for performance.

```bash
# Fit and evaluate on synthetic returns
uv run python -c "
from src.ml.arima_garch import ARIMAGARCHForecaster
import numpy as np; import pandas as pd; import warnings
warnings.filterwarnings('ignore'); np.random.seed(42)
rets = pd.Series(np.random.randn(600)*0.01+0.0002, name='ret')
f = ARIMAGARCHForecaster(arima_window=252, garch_window=126, retrain_every=252)
f.fit(rets)
fc = f.forecast(horizon=1)
print(f'Forecasts: {fc.mu.notna().sum()}, Mean sigma: {fc.sigma.mean():.4f}')
r = f.evaluate(horizon=5)
print(f'RMSE mean: {r.rmse_mean:.4f}, DirAcc: {r.direction_accuracy:.3f}, Coverage: {r.coverage_95:.3f}')
"
```

#### D11: State Space Models (Kalman Filter)

Local linear trend extraction, online Kalman signal generation, full structural decomposition (trend+cycle+seasonal). Produces noise-free trend and slope-based trading signals.

```bash
# Local linear trend on synthetic prices
uv run python -c "
from src.ml.state_space import LocalLinearTrend, KalmanSignal
import numpy as np; np.random.seed(42)
prices = 100*np.exp(np.cumsum(np.random.randn(300)*0.01))
llt = LocalLinearTrend(); llt.fit(prices)
sig = llt.slope_signal()
print(f'Bullish: {(sig>0).sum()}, Bearish: {(sig<0).sum()}')
print(f'Slope strength range: [{llt.slope_strength().min():.3f}, {llt.slope_strength().max():.3f}]')

# Online Kalman signal (streaming)
ks = KalmanSignal()
for p in prices:
    level, vel = ks.update(p)
print(f'Final: level={ks.level:.2f}, vel={ks.velocity:.4f}, signal={ks.signal:.3f}')
"

# Full structural decomposition
uv run python -c "
from src.ml.state_space import StateSpaceDecomposer
import numpy as np; np.random.seed(42)
prices = 100*np.exp(np.cumsum(np.random.randn(200)*0.01))
ssd = StateSpaceDecomposer(cycle=True); ssd.fit(prices)
d = ssd.decomposition
print(f'Trend NaN: {np.isnan(d.trend).sum()}, SNR: {ssd.signal_to_noise:.0f}')
"
```

### Block B: Fixed Income Models (Implemented 2026-05-18)

#### B1: Nelson-Siegel Yield Curve

Decomposes yield curve into level (β0), slope (β1), and curvature (β2) factors. Explains 95%+ of Treasury yield variation. Fitted via scipy nonlinear least squares.

```bash
uv run python -c "
from src.ml.fixed_income_models import NelsonSiegel
import numpy as np
mats = np.array([1/12,3/12,6/12,1,2,3,5,7,10,20,30])
yields = np.array([4.5,4.6,4.7,4.9,5.1,5.0,4.8,4.7,4.5,4.3,4.2])
ns = NelsonSiegel(); r = ns.fit(mats, yields)
print(f'Level: {r.level:.2f}%, Slope: {r.slope:.2f}%, Curvature: {r.curvature:.2f}, R2: {r.r_squared:.4f}')
print(f'2Y forecast: {ns.forecast_yield(2):.2f}%')
"
```

#### B4: Credit Spread Gate

IG/HY credit spread as risk appetite signal. Percentile-based regime classification (TIGHT/NORMAL/WIDE/CRISIS) with signal multipliers.

```bash
uv run python -c "
from src.ml.fixed_income_models import CreditSpreadGate
cg = CreditSpreadGate(lookback=252)
print(cg.update(ig_yield=5.2, hy_yield=8.5))  # NORMAL, 330bp spread
print(cg.update(ig_yield=6.0, hy_yield=14.0))  # WIDE/CRISIS
"
```

#### B5: Vasicek & CIR Rate Models

Mean-reverting short-rate models. Vasicek (OU process) with closed-form bond prices. CIR (square-root) prevents negative rates. OLS calibration with half-life estimation.

```bash
uv run python -c "
from src.ml.fixed_income_models import VasicekModel, CIRModel
import numpy as np; np.random.seed(42)
rates = 5.0 + np.cumsum(np.random.randn(500)*0.02)
vm = VasicekModel(); vr = vm.fit(rates)
print(f'Vasicek: k={vr.kappa:.3f}, theta={vr.theta:.2f}%, half-life={vr.half_life:.0f}d')
print(f'21d forecast: {vm.forecast(21):.2f}%')
cm = CIRModel(); cr = cm.fit(rates)
print(f'CIR: k={cr.kappa:.3f}, theta={cr.theta:.2f}%, Feller ok: {cm.feller_condition}')
"
```

### Block A: Options Pricing Models (Implemented 2026-05-18)

Complete quantitative options toolkit: Black-Scholes (A1), Binomial Tree (A2), Monte Carlo (A3), Heston stochastic vol (A4), SABR model (A5), Volatility Surface (A6), Greeks (A7).

```bash
# Black-Scholes pricing + implied vol + Greeks
uv run python -c "
from src.ml.options_pricing import BlackScholes, compute_greeks
bs = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
print(f'Call: {bs.call_price:.2f}, Put: {bs.put_price:.2f}')
print(f'Delta: {bs.delta_call:.3f}, Gamma: {bs.gamma:.6f}')
print(f'Vega: {bs.vega:.4f}, Theta: {bs.theta_call:.6f}')
print(f'Implied vol (3.5 call): {BlackScholes.implied_vol(3.5, 100, 105, 0.25, 0.05):.4f}')

# All Greeks at once
g = compute_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
print(f'ATM: call={g.call_price:.2f}, delta={g.delta_call:.3f}, gamma={g.gamma:.6f}')
"

# Binomial tree (American early exercise)
uv run python -c "
from src.ml.options_pricing import BinomialTree
bt = BinomialTree(S=100, K=105, T=0.25, r=0.05, sigma=0.20, steps=200)
print(f'Euro call: {bt.price(True,False):.4f}, Amer call: {bt.price(True,True):.4f}')
print(f'Euro put:  {bt.price(False,False):.4f}, Amer put:  {bt.price(False,True):.4f}')
"

# Heston stochastic volatility
uv run python -c "
from src.ml.options_pricing import HestonModel, SABRModel
hm = HestonModel(S=100, K=105, T=0.25, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
print(f'Heston call: {hm.price():.4f}')

# SABR implied vol
sabr = SABRModel(F=100, K=105, T=0.25, alpha=0.20, beta=1.0, rho=-0.3, nu=0.3)
print(f'SABR IV: {sabr.implied_vol():.4f}')
"

# Volatility surface (cubic spline)
uv run python -c "
from src.ml.options_pricing import VolSurface
import numpy as np
strikes = np.array([0.85,0.90,0.95,1.0,1.05,1.10,1.15])
mats = np.array([0.25,0.5,1.0])
ivs = np.array([[0.22,0.21,0.20,0.19,0.19,0.18,0.18],
                [0.21,0.20,0.19,0.19,0.18,0.18,0.18],
                [0.20,0.19,0.19,0.18,0.18,0.18,0.18]])
vs = VolSurface(strikes, mats, ivs, F=1.0)
s = vs.slice(0.5)
print(f'VolSurf 0.5Y: ATM={s.atm_vol:.4f}, Skew={s.skew:.4f}, Smile={s.smile:.4f}')
print(f'Term structure ATM: {vs.term_structure().values}')
"
```

**Key files (new this session):**
| File | Purpose |
|------|---------|
| `src/ml/structural_break.py` | D3: ADF/KPSS/Chow/Bai-Perron breakpoint detection |
| `src/ml/arima_garch.py` | D10: Rolling ARIMA+GARCH hybrid forecaster |
| `src/ml/state_space.py` | D11: Kalman filter, local linear trend, structural decomposition |
| `src/ml/fixed_income_models.py` | B1 NelSieg + B4 CreditSpread + B5 Vasicek/CIR |
| `src/ml/options_pricing.py` | A1-A7: BS, Binomial, MC, Heston, SABR, VolSurf, Greeks |
| `src/ml/__init__.py` | Updated: 25 new exports (D3+D10+D11+B1/B4/B5+A1-A7) |
| `scripts/paper_trade_daily.py` | Updated: sys.path fix for production paper trading |
