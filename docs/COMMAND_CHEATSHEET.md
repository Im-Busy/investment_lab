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

### B11: CDS Pricing & Credit Risk ✅

Credit Default Swap pricing with hazard rate bootstrapping and credit risk regime
classification. Market-implied default probabilities as macro stress signals.

```bash
# Bootstrap hazard rates from CDS spreads
uv run python -c "
from src.ml.cds_pricing import CDSPricer
cds = CDSPricer(recovery_rate=0.40)
curve = cds.fit(spreads={1: 0.005, 3: 0.008, 5: 0.012})
print(f'5Y default prob: {curve.five_year_default_prob:.2%}')
print(f'1Y→3Y forward: {curve.forward_default_prob(1,3):.2%}')
"

# Credit risk features for ML pipelines
uv run python -c "
from src.ml.cds_pricing import CDSPricer
cds = CDSPricer()
features = cds.credit_risk_features(spreads={1: 0.005, 3: 0.008, 5: 0.012})
for k, v in features.items():
    print(f'{k}: {v}')
"

# Credit risk regime classification (level + momentum)
uv run python -c "
from src.ml.cds_pricing import CDSPricer
cds = CDSPricer()
regime = cds.credit_risk_regime(spread_5y=0.012, spread_change_30d=0.001, spread_change_90d=0.003)
for k, v in regime.items():
    print(f'{k}: {v}')
"
```

### A12: Options Payoff & Volatility Visualization ✅

Payoff diagram generation, 3D volatility surfaces, theta decay curves, skew charts,
term structure plots, and Greeks heatmaps.

```bash
# Payoff diagram for any of 9 built-in strategies
uv run python -c "
from src.ml.options_visualization import OptionsVisualizer
viz = OptionsVisualizer()
result = viz.payoff_diagram('iron_condor', S_range=(80,120), K=100, premium=[1,2,2,1])
print(f'Max profit: {result[\"max_profit\"]}, Max loss: {result[\"max_loss\"]}')
print(f'BEs: {result[\"breakevens\"]}')
"

# Straddle payoff with breakeven analysis
uv run python -c "
from src.ml.options_visualization import OptionsVisualizer
viz = OptionsVisualizer()
result = viz.payoff_diagram('straddle', K=100, premium=[4,5])
print(f'PnL at S0: {result[\"pnl_at_S0\"]}, BEs: {result[\"breakevens\"]}')
"

# Theta decay curve
uv run python -c "
from src.ml.options_visualization import OptionsVisualizer
viz = OptionsVisualizer()
curve = viz.theta_curve(S=100, K=100, r=0.05, sigma=0.20)
print(f'Theta 30d: {curve[\"theta_30d\"]}, 7d: {curve[\"theta_7d\"]}, 1d: {curve[\"theta_1d\"]}')
print(f'Acceleration at ~{curve[\"theta_acceleration_days\"]:.0f} days to expiry')
"

# Vol skew chart
uv run python -c "
import numpy as np
from src.ml.options_visualization import OptionsVisualizer
viz = OptionsVisualizer()
strikes = np.array([80,85,90,95,100,105,110,115,120])
ivs = np.array([0.28,0.26,0.24,0.22,0.20,0.19,0.18,0.17,0.16])
skew = viz.skew_chart(strikes, ivs)
print(f'ATM vol: {skew[\"atm_vol\"]}, Skew: {skew[\"skew\"]}, Slope: {skew[\"skew_slope\"]}')
"

# Greeks heatmap
uv run python -c "
from src.ml.options_visualization import OptionsVisualizer
viz = OptionsVisualizer()
hm = viz.greeks_heatmap(S_range=(80,120), T=0.25, r=0.05, sigma=0.20, greek='gamma')
print(f'Gamma range: [{hm[\"values\"][0][0]:.4f}, {hm[\"max_abs\"]:.4f}]')
"
```

### A13: Volatility Trading Strategies ✅

Straddle/strangle PnL analysis, variance risk premium signals, vega-neutral
portfolio construction, and rolling VRP time series.

```bash
# Analyze straddle PnL vs realized vol
uv run python -c "
from src.risk.vol_trading import VolTradeAnalyzer
va = VolTradeAnalyzer()
result = va.analyze_straddle(S=100, K=100, T=0.25, r=0.05,
    sigma_implied=0.20, sigma_realized=0.25)
print(f'Expected PnL: {result.expected_pnl}, Prob profit: {result.probability_profit:.2%}')
print(f'BEs: {result.breakeven_range}, RR: {result.risk_reward}')
"

# Variance risk premium signal
uv run python -c "
from src.risk.vol_trading import VolTradeAnalyzer
va = VolTradeAnalyzer()
sig = va.analyze_variance_premium(implied_vol=0.25, realized_vol=0.18)
print(f'Signal: {sig.signal}, Confidence: {sig.confidence}, Sharpe: {sig.expected_sharpe}')
"

# Optimize strangle width
uv run python -c "
from src.risk.vol_trading import VolTradeAnalyzer
va = VolTradeAnalyzer()
df = va.optimize_strangle_width(S=100, T=0.25, r=0.05, sigma=0.20, sigma_realized=0.25)
print(df.to_string())
"

# Multi-factor volatility trading signal
uv run python -c "
from src.risk.vol_trading import VolTradeAnalyzer
va = VolTradeAnalyzer()
sig = va.volatility_signal(implied_vol=0.25, realized_vol=0.18,
    historical_vol=0.20, vol_percentile=75)
print(f'Composite: {sig[\"composite_signal\"]}, Action: {sig[\"action\"]}')
"

# Vega-neutral portfolio construction
uv run python -c "
from src.risk.vol_trading import VolTradeAnalyzer
va = VolTradeAnalyzer()
pf = va.vega_neutral_portfolio([
    {'S': 100, 'K': 105, 'T': 0.25, 'sigma': 0.20, 'quantity': 10, 'option_type': 'call'},
    {'S': 100, 'K': 95, 'T': 0.25, 'sigma': 0.20, 'quantity': 8, 'option_type': 'put'},
])
print(f'Portfolio vega: {pf[\"portfolio_vega\"]}, Delta hedge: {pf[\"delta_hedge_shares\"]} shares')
"
```

### B7: Interest Rate Derivatives Pricing ✅

Swap pricing (IRS/fixed-float NPV, par rate, DV01), swaption via Black normal
model, cap/floor as caplet portfolios. Rate signals for ML pipelines.

```bash
# Build discount curve + price swap
uv run python -c "
from src.ml.rate_derivatives import DiscountCurve, InterestRateSwap
curve = DiscountCurve.from_flat(0.05)
irs = InterestRateSwap(curve)
r = irs.price(notional=1_000_000, fixed_rate=0.05, maturity=5.0)
print(f'NPV: {r.npv:,.0f}, Swap rate: {r.swap_rate*100:.3f}%, DV01: {r.dv01:,.2f}')
"

# Swaption pricing via Black normal model
uv run python -c "
from src.ml.rate_derivatives import DiscountCurve, SwaptionPricer
sp = SwaptionPricer(DiscountCurve.from_flat(0.05))
r = sp.price(expiry=1.0, swap_maturity=5.0, strike=0.05, normal_vol=0.008)
print(f'Payer NPV: {r.npv:,.2f}, Fwd: {r.forward_swap_rate*100:.3f}%')
"

# Rate derivative signals for ML
uv run python -c "
from src.ml.rate_derivatives import DiscountCurve, RateDerivativeSignal
sig = RateDerivativeSignal(DiscountCurve.from_flat(0.05))
for k, v in list(sig.macro_rate_signals().items())[:8]: print(f'{k}: {v}')
"
```

### D5: PPO/SAC Trade Execution (SB3) ✅

PPO and SAC agents via stable-baselines3 for trade execution.
Replaces hand-rolled DQN with production-grade algorithms.

```bash
# Compare all algorithms (PPO, SAC, CQL)
uv run scripts/train_rl_advanced.py --symbol SPY --compare-all --timesteps 50000

# Train PPO only
uv run scripts/train_rl_advanced.py --symbol SPY --algo PPO --timesteps 50000

# Train SAC only
uv run scripts/train_rl_advanced.py --symbol SPY --algo SAC --timesteps 100000

# Compare with JSON output
uv run scripts/train_rl_advanced.py --symbol SPY --compare-all --json-output
```

### D12: Offline CQL (Conservative Q-Learning) ✅

Conservative Q-Learning learns optimal trade execution from historical data
without live environment interaction. Critical for finance where live
experimentation is expensive.

```bash
# Build offline dataset + train CQL
uv run scripts/train_rl_advanced.py --symbol SPY --algo CQL --iterations 10000

# CQL inference example
uv run python -c "
from src.rl.offline_rl import CQLAgent, OfflineDataset; import numpy as np
cql = CQLAgent(cql_alpha=5.0)
ds = OfflineDataset.from_env_trajectories({}, n_episodes=100)
cql.train(ds, n_iterations=1000, batch_size=128)
print(f'Action: {cql.predict(np.zeros(8, dtype=np.float32))}')
print(f'Q-values: {cql.predict_q(np.zeros(8, dtype=np.float32))}')
"
```

**Key files (this session):**
| File | Purpose |
|------|---------|
| `src/ml/rate_derivatives.py` | **B7 NEW**: IRS, swaption, cap/floor pricing + rate derivative signals |
| `src/rl/sb3_executors.py` | **D5 NEW**: PPO/SAC trade executors via stable-baselines3 |
| `src/rl/offline_rl.py` | **D12 NEW**: CQL offline RL for historical trade data |
| `scripts/train_rl_advanced.py` | **D5/D12 CLI**: Train & compare PPO/SAC/CQL trade executors |
| `src/rl/__init__.py` | Updated: +11 exports (PPO/SAC executors, CQL agent, dataset) |
