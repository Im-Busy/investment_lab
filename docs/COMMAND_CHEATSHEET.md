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

# Phase 25 Anti-Overfitting gates (LockBox, BlindAnalysis, LabelShuffling, NestedCV, DTW)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --use-lock-box
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --blind-analysis
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --label-shuffling
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --use-phase25-cv
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --dtw-overfit-detect

# Full anti-overfitting suite (all gates)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --use-lock-box --blind-analysis --label-shuffling --use-phase25-cv --dtw-overfit-detect
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

---

## SMC Intraday Strategy (2026-05-21 — gates OFF by default, conviction grading ON)

> **Module:** `src/strategies/smc_strategy.py` (unified, ~1703 loc)
> **CLI:** `scripts/backtest_smc.py`
> **New defaults (2026-05-21):** VIX gate=OFF, Yield curve gate=OFF, Killzone gate=OFF, HTF gate=OFF, Multi-TP=ON, SMC sessions=ON, min_confluence=0 (no gating).
> Use `--use-vix-gate`, `--use-yield-curve-gate`, `--use-killzone-gate` to opt-in.
> Gates were reverted to OFF after postmortem showed they destroyed returns across all instruments.
> **Conviction grading (2026-05-21):** Sweeps are now graded continuously (depth × reversal strength, 0.1-2.0x) — scores are no longer binary ±0.905.
> Entry threshold sweeps produce meaningful trade count variation. Phase 6 blocks (breaker/mitigation/rejection) are opt-in.

### Run SMC Backtest (current defaults)
```bash
# Default config (gates OFF, sessions ON): VIX gate=OFF, yield curve gate=OFF, killzone gate=OFF, multi-TP=ON
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h

# Run on all instruments with sweep (verify conviction grading)
uv run scripts/backtest_smc.py --all --interval 1h --sweep-entry "0.40,0.45,0.50,0.55,0.60"

# Opt-in: enable Phase 6 blocks (breaker/mitigation/rejection) — helps equities
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h \
  --use-breaker-blocks --use-mitigation-blocks --use-rejection-blocks \
  --entry-threshold 0.45 --use-multi-tp
```

### Sweep Entry Thresholds
```bash
# Find optimal entry threshold
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-11-01 --end 2025-04-30 --no-short --sweep-entry 0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

### Test on Other Instruments
```bash
# Gold futures (GC=F hourly)
uv run scripts/backtest_smc.py --symbol GC=F --interval 1h --start 2024-11-01 --end 2025-04-30 --no-short --entry-threshold 0.5 --use-multi-tp --trail-stop-atr 2.0
```

### Available Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | BTC-USD | Ticker symbol |
| `--interval` | 1h | Data interval (1h, 5m) |
| `--entry-threshold` | 0.55 | Minimum score to enter |
| `--exit-threshold` | 0.30 | Score drop to exit |
| `--trail-stop-atr` | 3.0 | ATR multiplier for trailing stop |
| `--session-bars` | 24 | Bars per session range |
| `--use-multi-tp` | **True** | Enable partial TP at 1.5x ATR (default ON) |
| `--no-multi-tp` | — | Disable multi-TP exit |
| `--use-vix-gate` | **False** | Opt-in: enable VIX regime score modulation (default OFF) |
| `--no-vix-gate` | — | Disable VIX regime gate (redundant) |
| `--use-yield-curve-gate` | **False** | Opt-in: enable yield curve macro gate (default OFF) |
| `--no-yield-curve-gate` | — | Disable yield curve macro gate (redundant) |
| `--no-short` | False | Disable short entries |
| `--sweep-buffer-mult` | 0.5 | ATR buffer for sweep detection |
| `--use-breaker-blocks` | False | Enable breaker block detection (Phase 6) |
| `--use-mitigation-blocks` | False | Enable mitigation block detection (Phase 6) |
| `--use-rejection-blocks` | False | Enable rejection block detection (Phase 6) |
| `--use-ir-weights` | False | Enable component IR weighting (Phase 5) |
| `--use-dow-gate` | False | Enable day-of-week gate (Phase 12) |
| `--use-90min-cycle` | False | Enable 90-minute cycle (Phase 12) |
| `--use-frankfurt-gate` | False | Enable Frankfurt fake move gate (Phase 12) |
    | `--use-smc-sessions` | **True** | Enable killzone sessions (Phase 9 — default ON) |
    | `--use-smc-phl` | False | Enable previous high/low (Phase 9) |
    | `--use-smc-retrace` | False | Enable retracement tracking (Phase 9) |
    | `--use-htf-gate` | False | Opt-in: enable hard HTF trend gate (default OFF) |
    | `--use-killzone-gate` | **False** | Opt-in: restrict trades to killzones only (default OFF) |
    | `--no-killzone-gate` | — | Disable killzone gate (redundant) |
    | `--min-confluence` | 0 | Minimum components agreeing for entry (0=no gating) |
    | `--vol-gate` | False | Enable volatility gate |
    | `--session-gate` | False | Enable session time gate |
    | `--crash-gate` | False | Enable crash factor gate |
    | `--volume-pressure` | False | Enable tick direction volume pressure (Phase 5) |
    | `--order-blocks` | False | Enable order block detection (Phase 5) |
    | `--sweep-entry` | — | Comma-separated thresholds to sweep (e.g. "0.40,0.45,0.50") |
    | `--json` | — | Save results as JSON file |
| `--crash-gate` | False | Enable crash factor gate |
| `--volume-pressure` | False | Enable volume pressure features |
| `--order-blocks` | False | Enable order block detection |
| `--crypto-mode` | False | Auto-adjust gates for crypto |
| `--msl-msh-lookback` | 3 | Duddella MSL/MSH lookback bars |
| `--crp-lookback` | 20 | Crash factor rolling window |
| `--ob-lookback` | 5 | Order block lookback bars |
| `--htf-bias-weight` | 0.5 | HTF bias penalty weight (0-1) |
| `--json` | None | Save results as JSON |

---

## ICT Strategy Plugins (2026-05-20 — new-tech defaults)

> **Modules:** `src/strategies/silver_bullet.py`, `turtle_soup.py`, `cameron_model.py`
> **Indicators:** `src/indicators/cisd.py`, `power_of_3.py`, `crt.py`, `ote.py`
> **New-tech defaults:** VIX gate=ON, Yield curve gate=ON, Multi-TP=OFF (hurts ICT win rates — strategies need big wins).
> **Best result:** Silver Bullet on DOGE-USD: Sharpe 0.81 (london_open, trail=2.0, sweep_lb=20).

### Backtest Silver Bullet (Kill Zone FVG Entry)
```bash
# London Open kill zone (default)
uv run python -c "from backtesting import Backtest; from src.strategies.silver_bullet import SilverBulletStrategy; from src.data_ingestion.fetch_data import fetch_data; df = fetch_data('SPY', '1h', '2024-01-01', '2025-01-01'); bt = Backtest(df, SilverBulletStrategy, cash=10000, commission=0.001); stats = bt.run(kill_zone='london_open', trail_stop_atr=2.0); print(stats)"

# New York AM kill zone
uv run python -c "from backtesting import Backtest; from src.strategies.silver_bullet import SilverBulletStrategy; from src.data_ingestion.fetch_data import fetch_data; df = fetch_data('SPY', '1h', '2024-01-01', '2025-01-01'); bt = Backtest(df, SilverBulletStrategy, cash=10000, commission=0.001); stats = bt.run(kill_zone='new_york_am', trail_stop_atr=2.0); print(stats)"
```

### Backtest Turtle Soup (False Breakout Trap)
```bash
uv run python -c "from backtesting import Backtest; from src.strategies.turtle_soup import TurtleSoupStrategy; from src.data_ingestion.fetch_data import fetch_data; df = fetch_data('BTC-USD', '1h', '2024-01-01', '2025-01-01'); bt = Backtest(df, TurtleSoupStrategy, cash=10000, commission=0.001); stats = bt.run(trail_stop_atr=2.0); print(stats)"
```

### Backtest Cameron's Model (Draw on Liquidity)
```bash
uv run python -c "from backtesting import Backtest; from src.strategies.cameron_model import CameronModelStrategy; from src.data_ingestion.fetch_data import fetch_data; df = fetch_data('SPY', '1h', '2024-01-01', '2025-01-01'); bt = Backtest(df, CameronModelStrategy, cash=10000, commission=0.001); stats = bt.run(swing_lookback=50, trail_stop_atr=2.0); print(stats)"
```

### Test All ICT Plugins via Registry
```bash
uv run python -c "from src.strategies.strategy_registry import SilverBulletPlugin, TurtleSoupPlugin, CameronModelPlugin, SMCPlugin, StrategyRegistry; from src.data_ingestion.fetch_data import fetch_data; r = StrategyRegistry(); r.register(SMCPlugin()); r.register(SilverBulletPlugin()); r.register(TurtleSoupPlugin()); r.register(CameronModelPlugin()); df = fetch_data('SPY', '1h', '2024-06-01', '2024-07-01'); signals = r.evaluate(df); print(f'{len(signals)} signals from {len(r.get_plugin_names())} plugins')"
```

### CISD (Change in State of Delivery) Detection
```bash
uv run python -c "from src.indicators.cisd import detect_cisd_vectorized; import pandas as pd; import numpy as np; df = pd.DataFrame({'Open': np.random.randn(100).cumsum()+100, 'High': np.random.randn(100).cumsum()+101, 'Low': np.random.randn(100).cumsum()+99, 'Close': np.random.randn(100).cumsum()+100}); result = detect_cisd_vectorized(df); print(f'Bullish: {result.bullish_cisd.sum()}, Bearish: {result.bearish_cisd.sum()}')"
```

### PO3 (Power of 3) Phase Detection
```bash
uv run python -c "from src.indicators.power_of_3 import detect_po3_daily; import pandas as pd; import numpy as np; df = pd.DataFrame({'Open': np.random.randn(200)*0.5+100, 'High': np.random.randn(200)*0.5+101, 'Low': np.random.randn(200)*0.5+99, 'Close': np.random.randn(200)*0.5+100}); po3 = detect_po3_daily(df); print(po3['phase'].value_counts().to_string())"
```

---

## SMC Phase 6-12: Advanced SMC Detectors & Integration (2026-05-20)

> **Plan:** `progress_docs/plans/smc-modernization-phase2-discoveries.md`
> **Prerequisite:** Phases 1-5 complete per `smc-modernization-implementation.md`

### Breaker/Mitigation/Rejection Block Detection
```bash
# Test breaker block detection
uv run python -c "
from smartmoneyconcepts import smc as smc_lib
from src.patterns.smc.breaker import detect_breaker_blocks
import pandas as pd; import numpy as np
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
shl = smc_lib.swing_highs_lows(df, swing_length=20)
atr = np.full(len(df), df['Close'].std())
vol_sma = df['Volume'].rolling(20).mean().bfill().to_numpy()
bull, bear, breakers = detect_breaker_blocks(df, shl, atr, vol_sma)
print(f'Breaker blocks: {len(breakers)}, Bullish signals: {bull.sum()}, Bearish: {bear.sum()}')
"

# Test all three new SMC detectors
uv run python -c "
from src.patterns.smc.breaker import detect_breaker_blocks
from src.patterns.smc.mitigation import detect_mitigation_blocks
from src.patterns.smc.rejection import detect_rejection_blocks
print('All SMC pattern modules import OK')
"
```

### New Chart Patterns (7 new detectors)
```bash
# Test Island Reversal detection
uv run python -c "
from src.patterns.event.island_reversal import detect_island_reversal
import pandas as pd
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
signals = detect_island_reversal(df)
print(f'Island reversals: bullish={int((signals==1).sum())}, bearish={int((signals==-1).sum())}')
"

# Test NR4 + Inside Bar
uv run python -c "
from src.patterns.volatility.nr4_inside_bar import detect_nr4, detect_inside_bar
import pandas as pd
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
nr4, buy, sell = detect_nr4(df)
inside = detect_inside_bar(df)
print(f'NR4: {nr4.sum()} bars, Breakouts: buy={buy.sum()}, sell={sell.sum()}')
print(f'Inside bars: {int((inside!=0).sum())}')
"

# Test all new pattern detectors
uv run python -c "
from src.patterns.event import island_reversal
from src.patterns.exotic import dragon
from src.patterns.volatility import nr4_inside_bar
from src.patterns.complex import quasimodo
from src.patterns.classic import adam_eve, three_valleys
from src.patterns.candlestick import shooting_star
from src.patterns.basic import key_reversal
print('All 8 new pattern modules import OK')
"
```

### SMT Divergence (Multi-Asset SMC Divergence)
```bash
# Detect SMT divergence between SPY and QQQ
uv run python -c "
from src.signals.smc_divergence import detect_smt_divergence, SMC_CORRELATED_PAIRS
from src.data_ingestion.fetch_data import fetch_data
spy = fetch_data('SPY', '1h', '2024-01-01', '2024-06-30')
qqq = fetch_data('QQQ', '1h', '2024-01-01', '2024-06-30')
signals, divs = detect_smt_divergence(spy, qqq)
print(f'SMT divergences: {len(divs)}, Signals: bullish={(signals==1).sum()}, bearish={(signals==-1).sum()}')
print(f'Correlated pairs: {SMC_CORRELATED_PAIRS}')
"
```

### PD Array Matrix (Premium/Discount Hierarchy)
```bash
# Build PD Array Matrix for Bitcoin
uv run python -c "
from src.patterns.smc.pd_array_matrix import build_pd_array_matrix, PDZone
from smartmoneyconcepts import smc as smc_lib
import pandas as pd
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
shl = smc_lib.swing_highs_lows(df, swing_length=20)
fvg = smc_lib.fvg(df, join_consecutive=True)
ob = smc_lib.ob(df, shl)
matrix = build_pd_array_matrix(df, fvg, ob, [], [], [], '1H')
print(f'Equilibrium: {matrix.equilibrium:.2f}')
print(f'Arrays: {len(matrix.arrays)}')
print(f'In premium: {len(matrix.get_active_arrays(PDZone.PREMIUM))}')
print(f'In discount: {len(matrix.get_active_arrays(PDZone.DISCOUNT))}')
"
```

### SMC-Aware Risk Management
```bash
# Compute SMC structural stop
uv run python -c "
from src.risk.smc_aware import compute_smc_structural_stop, compute_smc_partial_exit, compute_ob_based_position_size
import numpy as np
stop = compute_smc_structural_stop(100.0, 1, np.array([98, 95]), np.array([1, 1]), np.array([97]), np.array([1]), 2.0)
size = compute_ob_based_position_size(1.0, 75.0, True, 3)
print(f'SMC structural stop: {stop:.2f}')
print(f'OB-based position risk: {size:.1f}%')
"
```

### Volume Confirmation & Reliability Registry
```bash
uv run python -c "
from src.signals.pattern_reliability_registry import PATTERN_RELIABILITY, MIN_PATTERN_RELIABILITY
from src.signals.volume_confirmation_rules import get_volume_rule
print(f'Pattern H&S reliability: {PATTERN_RELIABILITY[\"head_and_shoulders\"]}')
print(f'Minimum reliability: {MIN_PATTERN_RELIABILITY}')
print(f'Gap volume rule: {get_volume_rule(\"gap\")}')
print(f'Top-5 patterns by reliability:')
sorted_patterns = sorted(PATTERN_RELIABILITY.items(), key=lambda x: x[1], reverse=True)[:5]
for name, rel in sorted_patterns:
    print(f'  {name}: {rel}')
"
```

### smartmoneyconcepts Library — Full Integration Commands
```bash
# List all available sessions (9 predefined killzones)
uv run python -c "
from smartmoneyconcepts import smc
import pandas as pd
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
for s in ['Sydney','Tokyo','London','New York','Asian kill zone','London open kill zone','New York kill zone','london close kill zone']:
    sess = smc.sessions(df, s)
    active = sess['Active'].sum()
    print(f'{s}: {active} active bars')
"

# Previous daily/weekly high/low
uv run python -c "
from smartmoneyconcepts import smc
import pandas as pd
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
phl_d = smc.previous_high_low(df, '1D')
phl_w = smc.previous_high_low(df, '1W')
print(f'Daily breaks: high={int(phl_d[\"BrokenHigh\"].sum())}, low={int(phl_d[\"BrokenLow\"].sum())}')
print(f'Weekly breaks: high={int(phl_w[\"BrokenHigh\"].sum())}, low={int(phl_w[\"BrokenLow\"].sum())}')
"

# Fibonacci retracement tracking
uv run python -c "
from smartmoneyconcepts import smc
import pandas as pd
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
shl = smc.swing_highs_lows(df, swing_length=20)
ret = smc.retracements(df, shl)
print(f'Direction changes: {int((ret[\"Direction\"]!=0).sum())}')
print(f'Deepest retrace mean: {ret[\"DeepestRetracement%\"].mean():.1f}%')
"
```

### Phase 6-12 Backtest (full enhanced SMC)
```bash
# Full enhanced SMC backtest (all Phase 6-12 features)
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --no-short \
  --entry-threshold 0.40 --trail-stop-atr 2.0 --sweep-buffer-mult 1.20 --use-multi-tp \
  --use-breaker-blocks --use-mitigation-blocks --use-rejection-blocks \
  --use-smc-sessions --use-smc-phl --use-smc-retrace \
  --json reports/smc/full_enchilada.json

# Full enchilada with time gates (equities only)
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --no-short \
  --entry-threshold 0.40 --trail-stop-atr 2.0 --sweep-buffer-mult 1.20 --use-multi-tp \
  --use-breaker-blocks --use-mitigation-blocks --use-rejection-blocks \
  --use-dow-gate --use-90min-cycle --use-frankfurt-gate \
  --use-smc-sessions --use-smc-phl --use-smc-retrace --use-ir-weights \
  --json reports/smc/full_enchilada.json

# Compare pre/post enhancement
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --no-short --entry-threshold 0.40 --trail-stop-atr 2.0 --sweep-buffer-mult 1.00 --use-multi-tp
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --no-short --entry-threshold 0.40 --trail-stop-atr 2.0 --sweep-buffer-mult 1.20 --use-multi-tp --use-breaker-blocks --use-mitigation-blocks

# Phase G3: Judas Swing gate (mandatory ICT entry filter)
uv run scripts/backtest_smc.py --symbol SPY --interval 1h --no-short --entry-threshold 0.45 \
  --use-judas-swing --judas-swing-atr-mult 0.3

# Phase G4: Orphaned indicator wiring (PO3 + OTE + CISD + CRT + SMT)
uv run scripts/backtest_smc.py --symbol SPY --interval 1h --no-short --entry-threshold 0.45 \
  --use-po3-gate --use-ote-confluence --use-cisd --use-crt

# Phase G5: Structural entries with S&D zones + Unicorn
uv run scripts/backtest_smc.py --symbol SPY --interval 1h --no-short --entry-threshold 0.45 \
  --use-sd-zones --use-unicorn --use-breaker-blocks

# Phase G6: POI grading (4-criteria quality filter)
uv run scripts/backtest_smc.py --symbol SPY --interval 1h --no-short --entry-threshold 0.45 \
  --use-poi-grading --use-breaker-blocks --use-order-blocks

# Phase G7: Risk management (daily loss limit + structural TP + 1% rule)
uv run scripts/backtest_smc.py --symbol SPY --interval 1h --no-short --entry-threshold 0.45 \
  --use-daily-loss-limit --daily-loss-limit 0.03 --use-structural-tp --max-risk-pct 0.01

# Phase G9: SMC Trade Plan checklist (10-point pre-trade validator)
uv run scripts/backtest_smc.py --symbol SPY --interval 1h --no-short --entry-threshold 0.45 \
  --use-trade-plan smc --trade-plan-strictness 0.7

# Phase G10: Market-specific kill zones (auto-detect gold/crypto/stocks/forex)
uv run scripts/backtest_smc.py --symbol GC=F --interval 1h --no-short --entry-threshold 0.45 \
  --instrument-class gold --use-smc-sessions
```
```

### Run All ICT Module Tests
```bash
uv run pytest tests/test_ict_new_modules.py -v
```

---

## Rules-First Parameter Tuning (2026-05-20)

> **Speed-optimized IS/OOS parameter sweeper with multiprocessing.**
> 60 param combos × 16 instruments (960 IS backtests) + OOS validation + universal best finder.
> **Full report:** `reports/parameter_tuning/RULES_TUNING.md`

### Per-Instrument Tuning (Batch)
```bash
# Fast grid: 60 combos per instrument, IS-only, multiprocessing
uv run scripts/tune_rules_params.py --fast --is-only --workers 5

# Single instrument, fast grid
uv run scripts/tune_rules_params.py --symbols SPY --fast --is-only

# Ultra-mini grid (12 combos) for quick screening
uv run scripts/tune_rules_params.py --symbols SPY --mini --is-only

# Full grid (243 combos) - slow, use for final tuning
uv run scripts/tune_rules_params.py --symbols SPY --workers 1
```

### OOS Validation + Universal Best
```bash
# After batch IS tuning, run OOS + universal best finder
uv run scripts/tune_rules_oos.py
```

### Universal Best Config (IS=2016-2024, OOS=2025-2026)
```bash
# Run Rules-First with universal params
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-16 \
    --entry-threshold 0.60 --min-reliability 0.70 --trail-stop-atr 2.0 --confluence-bonus 0.10
```

### Per-Instrument Best Configs (Top 3 OOS)
| Symbol | Entry Thr | Min Rel | Trail ATR | Confl Bonus | OOS Sharpe | OOS Ret% |
|--------|-----------|---------|-----------|-------------|------------|----------|
| SPY | 0.50 | 0.70 | 4.0 | 0.05 | **1.48** | +8.6% |
| JNJ | 0.55 | 0.70 | 4.0 | 0.10 | **1.24** | +8.3% |
| XLK | 0.55 | 0.70 | 4.0 | 0.10 | **1.19** | +17.6% |

---

## Arsenal ALL-ON Sweep — Per-Ticker Optimization (2026-05-27)

> **Script:** `scripts/sweep_arsenal_all_on.py`
> **Parameters swept:** max_loss_pct × min_confluence × entry_threshold
> **Fixed:** ALL signal enhancers ON + max_concurrent_orders=3

```bash
# Full sweep (17 tickers, 27 combos each, ~15 min)
uv run scripts/sweep_arsenal_all_on.py

# Single ticker
uv run scripts/sweep_arsenal_all_on.py --single-ticker SPY

# Custom period
uv run scripts/sweep_arsenal_all_on.py --start 2025-01-01 --end 2026-06-01
```

### New Parameters (wired 2026-05-27)

```bash
# ALL ON with wide stop-loss + multi-position concurrency
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 \
    --entry-threshold 0.65 --max-loss-pct 0.10 --max-concurrent-orders 3 --min-confluence 0 \
    --use-voting-signal --use-rules-catalog --use-divergence --use-wm-bollinger \
    --use-garch-atr --use-signal-strength-sizing --use-kelly-sizing
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-loss-pct` | 0.10 | Hard stop-loss as fraction of entry (10% = higher than typical 2-5%) |
| `--max-concurrent-orders` | 3 | Max simultaneous open positions (was 1) |
| `--min-confluence` | 0 | Min patterns that must agree for entry (0=no gating) |

### Results (OOS 2025→now, 17 tickers)

| Metric | Value |
|--------|-------|
| Mean OOS Sharpe | **+0.642** |
| Positive % | **88% (15/17)** |
| Top performer | HAL +1.781 |
| Best max_loss | 0.06-0.10 (88% of tickers) |
| Best min_confl | 0 (100% — gating is harmful) |

---

## Advanced Signal Flags — Complete Reference (Phases 21/23/24/27)

> **ALL flags default OFF.** Production config uses only Multi-TP + Quality Registry (mean OOS Sharpe 1.135).
> Opt-in individually or stack for experiments. Adding flags without OOS validation = uncertain effect.
> Enabling all 13+ flags simultaneously degraded performance to Sharpe 0.195 in testing (ALL OPTIONS ON test).

### Master Flag Table

| # | Flag | Source | What It Does | Tunable Params |
|---|------|--------|-------------|----------------|
| 1 | `--use-garch-atr` | RF3.1 | Replace static ATR trail with EGARCH dynamic vol (wider stops in low vol, tighter in high vol) | `--garch-model` (egarch/garch/gjr-garch) |
| 2 | `--use-options-sentiment` | RF3.2 | Scale entry signals by put/call ratio + gamma exposure proxy | `--options-sentiment-weight` (default 0.10) |
| 3 | `--use-kelly-sizing` | RF3.3 | Kelly-derived fraction-of-equity sizing by signal confidence | `--kelly-fraction` (default 0.5 = half-Kelly) |
| 4 | `--use-order-book` | RF3.4 | Bid-ask imbalance + microstructure signal enhancement | (none — fixed 0.05 weight) |
| 5 | `--use-vix-regime-sizing` | P1.5 | Cap position size by VIX regime (50% in ELEVATED, 25% in STRESS) | `--vix-size-high-vol-cap` (0.50), `--vix-size-crisis-cap` (0.25) |
| 6 | `--use-signal-strength-sizing` | P24-17 | Dynamic lot sizing: |score|≥0.45→3 lots, <0.15→2 lots, <0.05→1 lot | (none) |
| 7 | `--use-voting-signal` | B6 | 6-indicator majority voting (RSI/ROC/SMA/EMA/WMA/MACD) | `--voting-signal-weight` (default 0.15) |
| 8 | `--use-rules-catalog` | B1 | 35-rule catalog: 22 crossover + 6 Bollinger + 7 divergence rules | `--rules-catalog-weight` (default 0.10) |
| 9 | `--use-divergence` | B10 | RSI + MFI divergence detection (bullish/bearish divergences) | `--divergence-weight` (default 0.20) |
| 10 | `--use-wm-bollinger` | B2 | W-bottom + M-top Bollinger Band reversal patterns | `--wm-bollinger-weight` (default 0.15) |
| 11 | `--use-vix-gate` | Q1 | VIX regime gate: scale signals down in ELEVATED (×0.75) and STRESS (×0.30) regimes | `--vix-stress-mult` (0.30), `--vix-elevated-mult` (0.75) |
| 12 | `--use-yield-curve-gate` | Q2 | Yield curve inversion gate: scale signals down when inverted (×0.50) or near (×0.75) | `--yield-inversion-mult` (0.50), `--yield-near-inversion-mult` (0.75) |
| 13 | `--use-tadgan-gate` | 27C | Block entries during TadGAN-detected anomaly bars (crisis periods) | `--tadgan-model` (path), `--tadgan-threshold-pct` (95.0) |
| 14 | `--use-multi-factor` | Q | Fundamental score modifier (combine with price signals) | `--multi-factor-file`, `--multi-factor-weight` (0.15) |
| 15 | `--ir-weights` | R1 | Rolling IR-weighted pattern synthesis (scale pattern weights by recent IC) | `--ir-weighting-window` (252), `--ir-weighting-mode` (scalar/gate) |

### Basic Usage — Single Flag

```bash
# Backtest: any single flag
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-06-01 \
    --entry-threshold 0.55 --min-reliability 0.70 --use-garch-atr

# Tune sub-params on a flag
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-06-01 \
    --use-kelly-sizing --kelly-fraction 0.25
```

### Stacking — Multiple Flags

```bash
# Conservative stack: GARCH trail + VIX sizing (vol-aware risk management, no signal changes)
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-06-01 \
    --entry-threshold 0.55 --min-reliability 0.70 \
    --use-garch-atr --use-vix-regime-sizing

# Signal enhancer stack: voting + catalog + divergence + W/M Bollinger (adds confluence)
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-06-01 \
    --entry-threshold 0.55 --min-reliability 0.70 \
    --use-voting-signal --use-rules-catalog --use-divergence --use-wm-bollinger

# Full risk stack: GARCH + Kelly + VIX sizing + VIX gate + yield gate
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-06-01 \
    --entry-threshold 0.55 --min-reliability 0.70 \
    --use-garch-atr --use-kelly-sizing --use-vix-regime-sizing \
    --use-vix-gate --use-yield-curve-gate

# ALL ON (signal enhancers + risk + sizing + gates) — WARNING: 5.8x worse than production
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-06-01 \
    --entry-threshold 0.55 --min-reliability 0.70 \
    --use-garch-atr --use-options-sentiment --use-kelly-sizing --use-order-book \
    --use-vix-regime-sizing --use-signal-strength-sizing --use-voting-signal \
    --use-rules-catalog --use-divergence --use-wm-bollinger \
    --use-vix-gate --use-yield-curve-gate
```

### Performance Impact Reference (from ALL OPTIONS ON test, 2026-05-27)

| Config | Mean OOS Sharpe | OOS Positive | Notes |
|--------|----------------|-------------|-------|
| **Production (bare)** | **1.135** | 17/17 (100%) | Multi-TP + Quality Registry only |
| Full stack ON | 0.195 | 11/18 (61%) | Signal saturation degrades performance |
| B-tier boosted (phoenix plays) | Varies | — | INTC +0.54Δ, MRK +0.53Δ, NEM +0.39Δ |
| S-tier degraded (core ETFs) | Varies | — | GLD -1.08Δ, CN_CATL -1.15Δ |

**Recommendation:** Start with bare production. Add ONE flag at a time, validate OOS positive. Stack only compatible categories (risk stack vs signal stack). B-tier phoenix plays (INTC/MRK/NEM) benefit most from signal enhancers — apply selectively.

### Using Advanced Signals in Paper Trading

`scripts/paper_trade_daily.py` does NOT currently expose these flags. To paper trade with advanced signals:

```bash
# Option A: Use backtest_rules_first.py directly (backtest mode, no signal logging)
uv run scripts/backtest_rules_first.py SPY --start 2026-05-01 --end 2026-05-27 \
    --entry-threshold 0.55 --min-reliability 0.70 --use-garch-atr --json-output reports/tmp.json

# Option B: Run paper_trade_daily.py (bare signal monitoring) and cross-reference
# with backtest results to estimate stacked-signal effect
```

---

## Phase 25 — Post-Backtest Statistical Validation (2026-05-25)

> **CRITICAL:** Run these validation gates before any deployment decision.
> DSR, Purged WFA, Regime Audit, and Monte Carlo robustness all in one script.

### Comprehensive Validation Suite
```bash
# Full validation of a backtest result (JSON)
uv run scripts/validate_strategy.py --json reports/batch/rules_first_OOS_2025_2026.json --symbol SPY --full

# Full validation from CSV returns column
uv run scripts/validate_strategy.py --csv data/strategy_returns.csv --column daily_returns --full

# Quick validation (reduced simulations for speed)
uv run scripts/validate_strategy.py --json results.json --symbol SPY --quick

# Save report to file
uv run scripts/validate_strategy.py --json results.json --symbol SPY --full -o reports/validation/SPY_OOS.md
```

### Individual Validation Components
```bash
# DSR/PBO only (fast, < 1 second)
uv run scripts/validate_strategy.py --json results.json --symbol SPY --dsr-only

# Purged Walk-Forward only
uv run scripts/validate_strategy.py --json results.json --symbol SPY --wfa-only --is-days 1008 --oos-days 252 --purge-days 21

# Regime audit only (with VIX + SPY for regime classification)
uv run scripts/validate_strategy.py --json results.json --symbol SPY --regime-only --vix data/vix.csv --spy data/spy.csv

# Monte Carlo robustness only
uv run scripts/validate_strategy.py --json results.json --symbol SPY --mc-only --mc-simulations 10000
```

### Python API
```python
from src.analysis.deflated_sharpe import compute_dsr_from_returns, format_significance_summary

# DSR: Probability true Sharpe > expected max from multiple testing
dsr = compute_dsr_from_returns(returns, n_trials=200)
print(f"DSR: {dsr.psr:.3f} (1.0 = definitely not overfit)")

# Full significance summary (DSR + bootstrap CI + permutation test)
summary = format_significance_summary(returns, n_trials=200, n_trades=50)
print(f"Significant: {summary.significant}")

# Purged Walk-Forward Analysis
from src.analysis.purged_walk_forward import PurgedWalkForwardValidator
wfa = PurgedWalkForwardValidator(is_days=1008, oos_days=252, purge_days=21, step_days=126)
report = wfa.validate(returns)
print(f"WFE: {report.mean_wfe:.3f}, Chained OOS Sharpe: {report.chained_oos_sharpe:.3f}")

# Regime Audit
from src.analysis.regime_audit import audit_regimes, format_regime_report
ra = audit_regimes(returns, vix=vix_array, spy_returns=spy_returns)
print(format_regime_report(ra))

# Monte Carlo Robustness
from src.analysis.monte_carlo_robustness import run_full_robustness_check
mc = run_full_robustness_check(returns, params={"entry_threshold": 0.55, "min_reliability": 0.70})
print(f"Score: {mc.combined_score:.0f}/100, Pass: {mc.overall_pass}")
```

### Validation Flags Reference
| Flag | Default | Description |
|------|---------|-------------|
| `--n-trials` | 200 | Strategy variants for DSR multiple-testing correction |
| `--is-days` | 1008 | WFA training window in bars (default: 4 years) |
| `--oos-days` | 252 | WFA test window in bars (default: 1 year) |
| `--purge-days` | 21 | Purge gap between IS and OOS (default: 1 month) |
| `--step-days` | 126 | WFA rolling step size (default: 6 months) |
| `--mc-simulations` | 5000 | Monte Carlo reshuffling iterations |
| `--mc-perturbations` | 100 | Parameter perturbation draws per param |
| `--vix` | None | Path to VIX CSV for regime classification |
| `--spy` | None | Path to SPY CSV for bear/bull drawdown calculation |
| `--use-svm-regime` | False | Use SVM classifier for regime labeling |
| `--quick` | False | Reduced simulations (1000 MC, 30 perturbations) |

### Acceptance Criteria
| Gate | Threshold | What |
|------|-----------|------|
| DSR > 0.95 | P(SR > E[max SR]) | Statistically significant after multiple testing |
| Bootstrap CI > 0 | 95% CI lower bound | Sharpe unlikely due to chance |
| Permutation p < 0.05 | Empirical p-value | Strategy beats random return shuffling |
| WFE > 0.50 | OOS/IS return ratio | Strategy stable across walk-forward windows |
| Majority-Pass > 50% | Windows with OOS Sharpe > 0 | Sufficient robustness |
| No Catastrophic Veto | No window < -30% return | Strategy has no hidden tail risk |
| Per-regime Sharpe > 0 | All market regimes | Edge exists across conditions (or sizing issue) |
| MC Score > 60/100 | Combined robustness | Passes return reshuffling + param perturbation |

---

## Phase 24 P0 — Anti-Overfitting Validation Gates (2026-05-21)

> **New pipeline flags:** Label-shuffling baseline, three-value labeling (UP/DOWN/UNKNOWN), Huber loss, lock box, blind analysis.
> All default OFF — opt-in via `--label-shuffling`, `--label-type three_value`.

### Pipeline with Anti-Overfitting Gates
```bash
# Three-value labeling: UP 35% / DOWN 35% / UNKNOWN 30% (P24-7)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --label-type three_value

# Label-shuffling baseline test: verify model beats random labels (P24-3)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --label-shuffling

# Both anti-overfitting gates together
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --label-type three_value --label-shuffling
```

### Validation API
```python
# MRE-gap: dimensionless overfitting score (P24-4)
from src.ml.model_validation import compute_mre_gap
mre = compute_mre_gap(is_error=0.35, oos_error=0.45)  # → 0.222

# Causal masking verification for transformer models (P24-6)
from src.ml.model_validation import verify_causal_masking
result = verify_causal_masking("fincast")

# Lock Box: permitted ONCE after all decisions final (P24-1)
from scripts.train_ml_pipeline_v3 import lock_box_open
lock_box_open("data/lockbox/test.parquet")

# Blind analysis: shuffle labels during HP tuning (P24-2)
from scripts.train_ml_pipeline_v3 import blind_analysis_labels
y_blinded, mapping = blind_analysis_labels(y, random_state=42)
```

## Phase 24 P1 — Signal Quality & New Indicators (2026-05-21)

```bash
# HBar indicator: (Close−Open)/(High−Low) normalized ratio (P24-12)
uv run python -c "from src.indicators.hbar import compute_hbar; import numpy as np; print(compute_hbar(*np.random.randn(4,100)))"

# iV volume indicator: short-period vol / long-period vol (P24-13)
uv run python -c "from src.indicators.ivol import compute_ivol; import pandas as pd; import numpy as np; df = pd.DataFrame({'Close': np.cumsum(np.random.randn(100))+100}); print(compute_ivol(df)[-5:])"

# 4-indicator trend confirmation: RSI>50 AND CCI≥+100 AND MACD_Line>Signal AND ATR_rising (P24-15)
uv run python -c "from src.signals.combined_trend import four_indicator_trend; import pandas as pd; import numpy as np; data = pd.DataFrame({'High': np.cumsum(np.random.randn(200)*0.1)+100-np.arange(200)*0.05, 'Low': np.cumsum(np.random.randn(200)*0.1)+90-np.arange(200)*0.05, 'Close': np.cumsum(np.random.randn(200)*0.1)+95, 'Open': np.cumsum(np.random.randn(200)*0.1)+94}, index=pd.date_range('2020-01-01', periods=200)); print(four_indicator_trend(data, 199))"

# Signal alignment: technical + fundamental MUST match (P24-16)
uv run python -c "from src.signals.signal_alignment import check_signal_alignment; print(check_signal_alignment(0.15, 0.20))"

# Volatility no-trade switch + FOMC/NFP calendar (P24-14)
uv run python -c "from src.risk.vol_no_trade import VolNoTradeSwitch; import pandas as pd; import numpy as np; df = pd.DataFrame({'Close': np.cumsum(np.random.randn(100))+100}, index=pd.date_range('2026-05-01', periods=100)); g = VolNoTradeSwitch(); g.precompute(df); print(g.check(50))"

# Feature importance regime monitoring (P24-8)
uv run python -c "from src.ml.model_validation import FeatureImportanceMonitor; monitor = FeatureImportanceMonitor(); monitor.set_baseline({'atr': 0.2, 'rsi': 0.15}, 0.55); monitor.record({'atr': 0.1, 'rsi': 0.3}, 0.35); print(monitor.check_degradation())"

# Training history overfit detection (P24-9)
uv run python -c "from src.ml.overfitting_detector import TrainingHistoryOverfitDetector; import numpy as np; d = TrainingHistoryOverfitDetector(); val = np.linspace(0.3,0.8,50); print(d.detect(val).is_overfit)"

# Profit Mirage counterfactual evaluation (P24-10)
uv run python -c "from src.ml.profit_mirage import run_profit_mirage; import numpy as np; feats = np.random.randn(200,5); report = run_profit_mirage(feats, lambda x: np.random.random(len(x)), feature_names=['a','b','c','d','e']); print(report.summary())"

# Bootstrap 95% confidence intervals on performance (P24-11)
uv run python -c "from src.ml.model_validation import bootstrap_performance_ci; import numpy as np; rets = np.random.randn(252)*0.01+0.001; ci = bootstrap_performance_ci(rets); [print(str(v)) for v in ci.values()]"

# Signal-strength position sizing (P24-17) — wired into RulesFirstStrategy
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --trail-stop --et 0.50 --mr 0.70 --use-signal-strength-sizing --multi-tp
```

## PineScript→Python Indicator Conversions (2026-05-24)

New indicators converted from TradingView PineScript v5/v6 to Python/NumPy.

### Individual Indicator Tests

```bash
# FVG Profile + Rolling POC [BigBeluga] — FVG histogram + POC
uv run python -c "from src.indicators.fvg_profile import compute_fvg_profile; import numpy as np; h=np.cumsum(np.random.randn(500)*0.02)+100; l=h-np.abs(np.random.randn(500)*0.3); v=np.ones(500); r=compute_fvg_profile(h,l,v,period=200); print(f'Gaps:{r.total_gap_count} POC:{r.poc_price:.2f}')"

# TASC AutoTune Filter [Ehlers] — adaptive cycle-based band-pass
uv run python -c "from src.indicators.autotune_filter import auto_tune_filter; import numpy as np; c=np.cumsum(np.random.randn(500)*0.01)+100; hp,mc,dc,bp=auto_tune_filter(c,window=20); print(f'Cycle:{dc[-1]:.1f} BP:{bp[-1]:.4f}')"

# KNN Pivot Points [SS] — KNN slope classification for pivot prediction
uv run python -c "from src.indicators.knn_pivots import compute_knn_pivots; import numpy as np; h=np.random.randn(200).cumsum()+100; l=h-0.3; c=h-0.15; r=compute_knn_pivots(h,l,c); print([x.classification for x in r[-5:]])"

# Swing Structure Forecast [BOSWaves] — statistical swing projection
uv run python -c "from src.indicators.swing_forecast import forecast_next_swing; import numpy as np; h=np.cumsum(np.random.randn(300)*0.02)+100; l=h-np.abs(np.random.randn(300)*0.3); c=(h+l)/2; f=forecast_next_swing(h,l,c); print(f'Target:{f.target_price:.2f} Pct:{f.projected_pct:.2f}%')"

# Whale Liquidity & Absorption [AlgoAlpha] — volume/delta/absorption profile
uv run python -c "from src.indicators.whale_profile import compute_whale_profile; import numpy as np; n=200; c=np.cumsum(np.random.randn(n)*0.02)+100; h=c+np.abs(np.random.randn(n)*0.3); l=c-np.abs(np.random.randn(n)*0.3); o=np.roll(c,1); o[0]=c[0]; v=np.ones(n)*1000; r=compute_whale_profile(h,l,c,v,o); print(f'POC bin:{r.poc_index} MaxVol:{r.max_bin_volume:.0f}')"

# Market Microstructure Analytics — 7 academic spread/liquidity models
uv run python -c "from src.indicators.microstructure import compute_microstructure; import numpy as np; n=500; c=np.cumsum(np.random.randn(n)*0.01)+100; h=c+0.2; l=c-0.2; o=np.roll(c,1); o[0]=c[0]; v=np.ones(n)*1000; ms=compute_microstructure(h,l,c,v,o); print(f'LSI:{ms.lsi[-1]:.2f} Regime:{ms.regime_strs[-1]}')"

# Neural Weight Oscillator [Zeiierman] — BWM + adaptive learning oscillator
uv run python -c "from src.indicators.nwo import compute_nwo; import numpy as np; n=300; c=np.cumsum(np.random.randn(n)*0.02)+100; h=c+0.3; l=c-0.3; osc,sig,hist,_=compute_nwo(h,l,c); print(f'OSC:{osc[-1]:.2f} SIG:{sig[-1]:.2f}')"

# Market Structure Volume Profiles [Kioseff] — BoS/CHoCH-anchored volume profiles
uv run python -c "from src.indicators.structure_volume import compute_structure_cvd; import numpy as np; n=300; c=np.cumsum(np.random.randn(n)*0.02)+100; h=c+0.3; l=c-0.3; v=np.ones(n)*1000; cvd,atr,profs=compute_structure_cvd(h,l,c,v); print(f'Profiles:{len(profs)} CVD:{cvd[-1]:.2f}')"
```

### Composite & Strategy

```bash
# Asian Sweep Composite — Asian range + sweep + MSS + IFVG + HTF bias
uv run python -c "from src.indicators.asian_sweep_composite import detect_asian_sweep_setup; import pandas as pd; import numpy as np; dates=pd.date_range('2024-01-01',periods=500,freq='5min'); df=pd.DataFrame({'High':np.cumsum(np.random.randn(500)*0.02)+100,'Low':np.cumsum(np.random.randn(500)*0.02)+99.5,'Close':np.cumsum(np.random.randn(500)*0.02)+99.8,'Volume':np.ones(500)*1000},index=dates); df['High']=df[['High','Low','Close']].max(axis=1); df['Low']=df[['High','Low','Close']].min(axis=1); sigs=detect_asian_sweep_setup(df); print(f'{len(sigs)} signals detected')"

# Asian Sweep Strategy — full backtest-capable strategy
uv run python -c "from src.strategies.asian_sweep_strategy import run_asian_sweep_strategy; import pandas as pd; import numpy as np; dates=pd.date_range('2024-01-01',periods=500,freq='5min'); df=pd.DataFrame({'High':np.cumsum(np.random.randn(500)*0.02)+100,'Low':np.cumsum(np.random.randn(500)*0.02)+99.5,'Close':np.cumsum(np.random.randn(500)*0.02)+99.8,'Volume':np.ones(500)*1000},index=dates); df['High']=df[['High','Low','Close']].max(axis=1); df['Low']=df[['High','Low','Close']].min(axis=1); result=run_asian_sweep_strategy(df); longs=result['entry_long'].sum(); shorts=result['entry_short'].sum(); print(f'{longs} long entries, {shorts} short entries')"
```

## Phase 24 P2 — Strategy Components (2026-05-21)

```bash
# Event-type specific trading: classify event + get optimal config (P24-20)
uv run python -c "from src.signals.event_type_trading import classify_event, get_event_config; ec = classify_event('AAPL Q4 Earnings Beat'); print(get_event_config(ec))"

# Event-weighted sentiment decay: 10-day half-life (P24-21)
uv run python -c "from src.signals.event_weighted_sentiment import compute_time_weighted_sentiment, WeightedSentimentEvent; import pandas as pd; e = [WeightedSentimentEvent(pd.Timestamp.now()-pd.Timedelta(days=d), 0.5, 1.0, 0.5, d) for d in range(20)]; print(f'Aggregate: {compute_time_weighted_sentiment(e):.4f}')"

# ETF portfolio rotation: train 45d → top-10 → trade 45d → repeat (P24-22)
uv run python -c "from src.strategies.etf_rotation import run_etf_rotation, ETF_UNIVERSE; import yfinance as yf; data = {t: yf.download(t,'2020-01-01','2026-05-20',auto_adjust=False) for t in ETF_UNIVERSE[:5]}; result = run_etf_rotation({k:v for k,v in data.items() if not v.empty}); print(f'Rotation return: {result.total_return:.1%}, {result.num_windows} windows')"

# Instance normalization for financial data (P24-23)
uv run python -c "from src.ml.preprocessing import instance_normalize; import numpy as np; x = np.random.randn(100,5)*np.array([0.01,0.05,0.02,0.1,0.03]); print(f'Norm shape: {instance_normalize(x).shape}')"

# Triangular hedge correlation-based pair selection (P24-25)
uv run python -c "from src.strategies.triangular_hedge import select_hedge_pairs; import pandas as pd; import numpy as np; s1 = pd.Series(np.cumsum(np.random.randn(200)*0.01)+100); s2 = pd.Series(np.cumsum(np.random.randn(200)*(-0.01))+105); pairs = select_hedge_pairs({'A': s1, 'B': s2}); print([(p.asset_a, p.asset_b, round(p.correlation,3)) for p in pairs])"

# Hedge-only triangular variant (no averaging/martingale) (P24-28)
uv run python -c "from src.strategies.triangular_hedge import run_hedge_only_backtest, HedgeOnlyConfig; import pandas as pd; import numpy as np; pa = pd.Series(np.cumsum(np.random.randn(300)*0.01)+100); pb = pd.Series(np.cumsum(np.random.randn(300)*(-0.008))+100); trades = run_hedge_only_backtest(pa, pb, HedgeOnlyConfig()); print(f'{len(trades)} trades')"
```

## Phase 24 P3 — Deferred Heavy Lifts (2026-05-27)

### P24-29: Two-Phase GA Rule Combination

```python
from src.optimization.two_phase_ga import TwoPhaseGA, RuleDef, TwoPhaseGAResult
from src.optimization.nsga2_optimizer import ParamDef

# Define rules with parameter bounds and evaluation functions
rules = [
    RuleDef(
        name="double_bottom",
        params=[ParamDef("lookback", 5, 50, is_integer=True),
                ParamDef("threshold", 0.5, 0.95)],
        eval_fn=lambda p: evaluate_double_bottom(**p),
    ),
    RuleDef(
        name="head_shoulders",
        params=[ParamDef("min_period", 10, 60, is_integer=True),
                ParamDef("neckline_tolerance", 0.01, 0.05)],
        eval_fn=lambda p: evaluate_head_shoulders(**p),
    ),
]

two_phase = TwoPhaseGA(rules)
result = two_phase.optimize()
print(result.summary())
```

### P24-32: Divergence-in-Bits Strategy Comparison

```bash
# Compare two strategies using divergence-in-bits (unit-independent, more robust than Sharpe)
uv run python -c "
from src.analysis.divergence_bits import compute_divergence_bits, compare_vs_benchmark, compare_multiple
import numpy as np
r_a = np.random.randn(200)*0.01 + 0.001
r_b = np.random.randn(200)*0.015 + 0.0005
r_bench = np.random.randn(200)*0.01
print(compare_vs_benchmark(r_a, r_bench, 'Strategy', 'SPY'))
print(compare_vs_benchmark(r_b, r_bench, 'Strategy', 'SPY'))
"

# Multi-strategy pairwise comparison
uv run python -c "
from src.analysis.divergence_bits import compare_multiple
import numpy as np
strategies = {
    'Rules-First': np.random.randn(200)*0.01 + 0.001,
    'SMC/ICT': np.random.randn(200)*0.012 + 0.0003,
    'Combined': np.random.randn(200)*0.011 + 0.0007,
}
for r in compare_multiple(strategies):
    print(f'{r.strategy_a_name} vs {r.strategy_b_name}: Δg={r.delta_g_bits:.3f} bits → {r.winner} wins')
"
```

### P24-33: Binomial VAR for Event-Driven Risk

```bash
# Compute binomial VaR for event-driven strategy (N trades × success probability)
uv run python -c "
from src.risk.binomial_var import compute_binomial_var, size_position_binomial
result = compute_binomial_var(n_trades=50, success_prob=0.60, avg_win_pct=0.02, avg_loss_pct=0.015)
print(f'95% VaR: {result.var_pct*100:.2f}%, CVaR: {result.cvar_pct*100:.2f}%')
print(f'Breakeven: {result.loss_breakeven_k} losses out of {result.n_trades} trades')

# Size position based on VaR constraint
sizing = size_position_binomial(capital=100000, n_trades=10, success_prob=0.55, max_var_pct=0.05)
print(f'Max size per trade: \${sizing[\"max_size_per_trade\"]:.0f}, Risk util: {sizing[\"risk_utilization\"]:.1%}')
"
```

### P24-35: W-Type Bottom & M-Type Top Bollinger Patterns

```bash
# Detect W-Bottom and M-Top patterns (already in src/patterns/bollinger/wm_patterns.py)
uv run python -c "
from src.patterns.bollinger import detect_w_bottom, detect_m_top, detect_wm_bollinger
import yfinance as yf
df = yf.download('SPY', '2025-01-01', '2026-05-01', auto_adjust=False)
result = detect_wm_bollinger(df)
w_count = result['w_bottom'].sum()
m_count = (result['m_top'] != 0).sum()
print(f'W-Bottom signals: {w_count}, M-Top signals: {m_count}')
"
```

### Phase 24 P3 File Changes

| File | Change |
|------|--------|
| `src/optimization/two_phase_ga.py` | NEW — TwoPhaseGA, RuleDef, TwoPhaseGAResult, Phase1Result (Phase1 per-rule + Phase2 weighted voting) |
| `src/analysis/divergence_bits.py` | NEW — compute_divergence_bits, compare_vs_benchmark, compare_multiple (Δg = D_KL divergence) |
| `src/risk/binomial_var.py` | NEW — compute_binomial_var, size_position_binomial, BinomialVaRResult (forward-looking risk) |
| `src/patterns/bollinger/wm_patterns.py` | EXISTING — detect_w_bottom, detect_m_top, detect_wm_bollinger (Phase 25) |
| `src/optimization/__init__.py` | MODIFIED — +TwoPhaseGA, TwoPhaseGAResult, Phase1Result, RuleDef |
| `src/analysis/__init__.py` | MODIFIED — +DivergenceBitsResult, compute_divergence_bits, compare_vs_benchmark, compare_multiple |
| `src/risk/__init__.py` | MODIFIED — +compute_binomial_var, size_position_binomial, BinomialVaRResult |

## Phase 23 RF1 — Cross-Asset Tuning (2026-05-21)

```bash
# RF1.1: Per-instrument optimal config auto-discovery (fast grid)
uv run scripts/auto_tune_per_instrument.py --symbols SPY,QQQ,XLK,GLD,TLT --fast --json-output outputs/auto_tune_fast.json

# RF1.1: Full grid sweep (720 combos per instrument)
uv run scripts/auto_tune_per_instrument.py --symbols SPY --full --json-output outputs/auto_tune_full.json

# RF1.3: Dynamic et/mr per regime backtest
uv run scripts/backtest_dynamic_regime.py SPY --start 2016-01-01 --end 2026-05-20 --bull-et 0.50 --bull-mr 0.70 --bear-et 0.35 --bear-mr 0.50
```

## Phase 23 RF2 — Short-Side Production (2026-05-21)

```bash
# RF2.2: Short-side sweep across instruments
uv run scripts/sweep_short_side.py --tickers SPY,QQQ,XLK,XLF,GLD,TLT,EEM,EFA --start 2025-01-01 --json-output outputs/short_sweep.json

# RF2.2: Sweep with custom params
uv run scripts/sweep_short_side.py --tickers SPY,QQQ,IWM --et 0.50 --mr 0.70 --start 2016-01-01 --end 2026-05-20
```

## Phase 23 RF4 — Documentation & Production (2026-05-21)

```bash
# RF4.1: Auto-update BESTS.md from JSON backtest results
uv run scripts/update_bests.py --json outputs/backtest_result.json --strategy rules-first --condition production

# RF4.2: Generate per-instrument config cards
uv run scripts/generate_config_cards.py --json outputs/auto_tune_fast.json --output reports/per_instrument_configs.md

# RF4.3: Post-trade analysis checklist
uv run scripts/post_trade_check.py --results-dir reports/backtests/ --output reports/post_trade_checklist.md
```

## New Techstack Backtests — VIX + Yield Curve Gates ON (2026-05-21)

> **Config:** mr=0.70, et=0.55, trail=3.0, multi-TP=ON, quality-registry=ON, VIX-gate=ON, yield-gate=ON
> **IS=2016-2024, OOS=2025-2026. Run in small batches (5-6 instruments).**

```bash
# RulesFirst Multipattern — Comprehensive batch (IS+OOS in one run)
# Use for: full new-techstack evaluation per instrument
# Production config (2026-05-21): VIX+yield gates both OFF

# Batch 1: Indices + Sector ETFs (5 instruments)
uv run scripts/backtest_all_comprehensive.py --tickers SPY,QQQ,XLK,XLF --output-dir reports/comprehensive_batch --period both

# Batch 2: Sectors + Commodities + Stocks (5 instruments)
uv run scripts/backtest_all_comprehensive.py --tickers XLE,XLV,GLD,KO,XOM --output-dir reports/comprehensive_batch --period both

# Batch 3: Single Stocks + Crypto (4 instruments — IWM, JPM, TLT, EURUSD_X dropped)
uv run scripts/backtest_all_comprehensive.py --tickers JNJ,SO,BTC_USD --output-dir reports/comprehensive_batch --period both

# Per-instrument best configs (from BESTS.md tuning sweep)
# Use for: production OOS validation with tuned params per instrument
uv run scripts/backtest_rules_batch.py --use-best --start 2025-01-01 --end 2026-05-21 --json-output outputs/batch_best_oos.json

uv run scripts/backtest_all_comprehensive.py --batch all --use-best --output-dir reports/comprehensive_batch

# All 12 batches (120+ instruments) — use with care
uv run scripts/backtest_all_comprehensive.py --batch all --output-dir reports/comprehensive_batch

# SMC/ICT — NQ=F + GC=F only (forex/crypto dropped from hourly basket)
uv run scripts/backtest_smc.py --all --interval 1h --start 2025-01-01 --json outputs/smc_all_fixed_20260521.json

# SMC/ICT — Enable VIX+yield gates (opt-in)
uv run scripts/backtest_smc.py --all --interval 1h --start 2025-01-01 --use-vix-gate --use-yield-curve-gate --json outputs/smc_all_gates_ON.json

# SMC/ICT — Single instrument sweep
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --start 2025-01-01 --sweep-entry 0.35,0.40,0.45,0.50,0.55
```

### New Techstack Gate Impact (2026-05-21)

| Strategy | Gates OFF | Gates ON | Δ | Verdict |
|----------|-----------|----------|----|---------|
| RulesFirst SPY 2025 OOS | Sharpe +0.92, +0.42% ret | Sharpe +0.705, +0.04% ret | +0.215 Sharpe, +0.38% ret | Gates OFF significantly better |
| SMC NQ=F OOS | Sharpe -1.17 (fixed) | Sharpe -1.13 | N/A | SMC sweep weak on NQ=F 1h — Phase 6 blocks help (Sharpe +0.32 opt-in) |

**Recommendation:** VIX+yield gates default to OFF for production. Opt-in via `--use-vix-gate` / `--use-yield-curve-gate` when volatility regime demands it.
**SMC/ICT (2026-05-21 update):** Binary scoring FIXED — sweep conviction grading provides continuous scores (depth × reversal strength). Entry threshold sweeps now produce meaningful trade variation. Phase 6 blocks (breaker/mitigation/rejection) improve NQ=F Sharpe from 0.12 → 0.32 when opted in. Strategy still needs dedicated overhaul for weak forex/crypto instruments.

### Gate Fix Postmortem (2026-05-21)

**Round 1 — Gate defaults reverted to OFF (12 files):**
| File | Change |
|------|--------|
| `src/strategies/smc_strategy.py` | `use_vix_gate`/`use_yield_curve_gate`/`use_killzone_gate` → False |
| `src/strategies/silver_bullet.py` | `use_vix_gate`/`use_yield_curve_gate` → False; hard block `<0.4` → `<0.15` |
| `src/strategies/turtle_soup.py` | Same as Silver Bullet |
| `src/strategies/cameron_model.py` | Same as Silver Bullet |
| `src/strategies/combined_strategy.py` | `use_vix_gate`/`use_yield_curve_gate` → False |
| `scripts/backtest_all_comprehensive.py` | PRODUCTION_CONFIG gates → False |
| `scripts/backtest_smc.py` | Argparse defaults → False; killzone → False |
| `scripts/backtest_combined.py` | Argparse defaults → False |
| `scripts/auto_tune_per_instrument.py` | Gate kwarg → False |
| `scripts/backtest_smc.py` | Fixed `compute_trade_metrics` bug (Size=0 filter removed) |
| `scripts/backtest_rules_first.py` | Fixed `--end` default (was 2024-12-31, now None); JSON serialization fix; missing params added |
| `scripts/backtest_smc.py` | JSON serialization fix (_trades excluded) |

**Round 2 — Conviction grading fix (2026-05-21):**
| File | Change |
|------|--------|
| `src/strategies/smc_strategy.py` | `_compute_sweep_signals()`: sweep_depth × reversal_strength → continuous conviction (0.1-2.0x) |
| `src/strategies/smc_strategy.py` | `_compute_smc_score()`: base_score uses conviction instead of binary ±1 |
| `src/strategies/smc_strategy.py` | Docstring updated to reflect new defaults |
| `scripts/backtest_smc.py` | CLI defaults synced: `use_smc_sessions=True`; order_blocks/volume_pressure/Phase6 blocks → False |
| `.kilo/agent/smc-trader.md` | Updated with conviction grading docs, new baselines, pitfall fixes |

**Round 3 — Dead parameter audit & fix (2026-05-21):**
| File | Change |
|------|--------|
| `src/strategies/smc_strategy.py` | `min_confluence` wired as hard entry gate (default 0 = no gating); Phase 6 blocks gated on toggles; `_calculate_size()` wired into `next()`; deleted `use_smc_phl` + `_precompute_smc_previous_levels()` (dead arrays); 7 dead params deleted |
| `src/strategies/rules_first_strategy.py` | Deleted 5 dead params (tp2_atr, rsi_oversold/overbought, fixed_tp_pct/sl_pct) |
| `src/strategies/combined_strategy.py` | Deleted dead `tp2_atr` param |
| `src/strategies/silver_bullet.py` | Deleted dead `fvg_min_gap` param |
| `src/strategies/turtle_soup.py` | Deleted dead `fvg_min_gap` param |
| `scripts/backtest_smc.py` | Deleted 5 dead CLI flags + their kwargs; wired orphaned `--no-*` flags to kwargs; `min_confluence` default → 0 |
| `scripts/backtest_combined.py` | Added missing CLI flags: `--no-multi-tp`, `--tp1-atr`, `--tp1-size`, `--no-volume-confirm` |
| `BESTS.md` | Added Phase 6 ON vs OFF NQ=F results; defaults updated |
| `AGENTS.md` | End-to-End Wiring Protocol (added 2026-05-21) |

**Round 4 — Paper comparison report gaps closed (2026-05-21):**
| File | Change |
|------|--------|
| `src/ml/volatility_forecaster.py` | `loss_function="RMSE"` → `"Huber:delta=1.0"` (3 regressors) |
| `src/ml/stop_loss_optimizer.py` | `loss_function="RMSE"` → `"Huber:delta=1.0"` |
| `src/ml/models/catboost_wrapper.py` | `loss_function="RMSE"` → `"Huber:delta=1.0"` |
| `src/indicators/swing_point_detector.py` | NEW — configurable %-threshold swing point detector |
| `src/patterns/candlestick/ict_single_patterns.py` | NEW — 12 ICT single-candlestick patterns (WM/CWM/OWM/etc.) |
| `src/signals/indicator_voting.py` | NEW — 6-indicator majority voting (RSI/ROC/SMA/EMA/WMA/MACD) |
| `src/signals/cross_currency_signals.py` | NEW — signal propagation across correlated currency pairs |
| `src/signals/divergence_detector.py` | NEW — regular+hidden divergence (RSI/MFI/MACD) |
| `src/patterns/bollinger/wm_patterns.py` | NEW — W-bottom/M-top 4-step Bollinger confirmation |
| `src/signals/rules_catalog.py` | NEW — 35-rule catalog (22 crossover + 6 BB + 7 divergence) |
| `src/strategies/rules_first_strategy.py` | +`use_voting_signal`/`voting_signal_weight` + `_init_voting_signal()` |
| `src/strategies/rules_first_strategy.py` | +`use_rules_catalog`/`rules_catalog_weight` + `_init_rules_catalog()` |
| `src/strategies/rules_first_strategy.py` | +`use_divergence`/`divergence_weight` + `_init_divergence()` |
| `src/strategies/rules_first_strategy.py` | +`use_wm_bollinger`/`wm_bollinger_weight` + `_init_wm_bollinger()` |
| `src/strategies/smc_strategy.py` | +`use_swing_points`/`swing_point_weight` + `_init_swing_points()` + scoring bonus |
| `src/strategies/smc_strategy.py` | +`use_ict_patterns`/`ict_pattern_weight` + `_init_ict_patterns()` + scoring bonus |
| `scripts/backtest_rules_first.py` | +6 new CLI flags: `--use-voting-signal`, `--use-rules-catalog`, `--use-divergence`, `--use-wm-bollinger` |
| `scripts/backtest_smc.py` | +4 new CLI flags: `--use-swing-points`, `--use-ict-patterns` |
| `src/indicators/__init__.py` | +swing_point_detector exports |
| `src/patterns/candlestick/__init__.py` | +detect_all_twelve export |
| `src/patterns/bollinger/__init__.py` | NEW — W/M pattern exports |
| `src/signals/__init__.py` | +4 new signal module exports |
| `BESTS.md` | Updated — 7/8 modules wired end-to-end, validation confirmed signal propagation |

---

## Phase 25: Per-Instrument Configuration & Performance Tracking

### Production Basket Backtest

```bash
# Run production basket (22 instruments, Tier S+A+B) OOS with per-instrument best params
uv run scripts/backtest_rules_batch.py --use-best --start 2025-01-01 --json-output outputs/production_oos.json

# Run all 108 instruments across 11 batches (IS + OOS)
uv run scripts/backtest_all_comprehensive.py --batch-index 1 --use-best --period both --output-dir outputs/comprehensive

# Run a specific batch
uv run scripts/backtest_all_comprehensive.py --batch-index 5 --use-best --period oos

# Run custom ticker list
uv run scripts/backtest_all_comprehensive.py --tickers SPY,QQQ,XLK,GLD --use-best --period oos
```

### Performance Log Query

```bash
# Show aggregate statistics across all logged backtests
uv run scripts/log_query.py --summary

# Show latest OOS results for a specific instrument
uv run scripts/log_query.py --instrument SPY --period OOS

# Show performance evolution (sharpe over time) for an instrument
uv run scripts/log_query.py --instrument XLK --history --metric sharpe

# Show all PASS verdicts
uv run scripts/log_query.py --verdict PASS

# Show all FAIL verdicts for rules_first strategy
uv run scripts/log_query.py --verdict FAIL --strategy rules_first

# Show the last 50 records
uv run scripts/log_query.py --all --limit 50
```

### Per-Instrument Config & Strategy

```bash
# Get recommended backtest command for an instrument
uv run python -c "from src.per_instrument.strategy_selector import get_backtest_command; print(get_backtest_command('SPY'))"

# Get production config for an instrument
uv run python -c "from src.per_instrument.instrument_config import get_config_for; import json; print(json.dumps(get_config_for('XLK', use_best=True), indent=2))"

# Check if an instrument is in its trading window
uv run python -c "from src.per_instrument.strategy_selector import should_trade_now; print(should_trade_now('SPY'))"

# List all instruments by tier
uv run python -c "from src.per_instrument.instrument_config import INSTRUMENT_CONFIG; tiers = {}; [tiers.setdefault(c['tier'], []).append(s) for s, c in INSTRUMENT_CONFIG.items()]; [print(f'Tier {k}: {sorted(v)}') for k in sorted(tiers)]"

# Get timezone for an instrument
uv run python -c "from src.per_instrument.timezone_registry import get_session_for; print(get_session_for('CN_CATL'))"
```

### New Infrastructure Files

| File | Purpose |
|------|---------|
| `src/per_instrument/instrument_config.py` | Single source of truth for per-instrument tuned params (22 instruments, 3 tiers) |
| `src/per_instrument/timezone_registry.py` | 8 trading sessions mapped to ~130 instruments (US Equity, Futures, Crypto, China, HK, Forex, Commodity, Bond) |
| `src/per_instrument/performance_tracker.py` | Append-only JSONL ledger at `logs/per_instrument_performance.jsonl` |
| `src/per_instrument/strategy_selector.py` | Instrument → strategy class dispatch + CLI command generation |
| `scripts/log_query.py` | Query performance history with filters |
| `config_files/production_basket.yaml` | Human-readable basket config with allocations (22 instruments, sum=1.0) |

### Phase 25 File Changes

| File | Change |
|------|--------|
| `src/per_instrument/__init__.py` | NEW — module exports |
| `src/per_instrument/instrument_config.py` | NEW — `INSTRUMENT_CONFIG` (22 instruments), `PRODUCTION_BASKET`, `get_config_for()`, `get_allocation()` |
| `src/per_instrument/timezone_registry.py` | NEW — `TIMEZONE_SESSIONS` (8 sessions), `get_session_for()`, `is_in_trading_window()` |
| `src/per_instrument/performance_tracker.py` | NEW — `log_performance()`, `query_log()`, `get_latest_for()`, `compare_across_runs()`, `get_summary_stats()` |
| `src/per_instrument/strategy_selector.py` | NEW — `get_strategy_class()`, `get_backtest_command()`, `should_trade_now()` |
| `scripts/log_query.py` | NEW — CLI: `--summary`, `--instrument`, `--period`, `--verdict`, `--history`, `--all` |
| `config_files/production_basket.yaml` | NEW — 22-instrument production basket with tier labels, allocations, and excluded-instrument rationale |
| `scripts/backtest_rules_batch.py` | MODIFIED — `PER_INSTRUMENT_BEST` rebuilt from `instrument_config.py`; `log_performance()` wired after each `run_single()` |
| `scripts/backtest_all_comprehensive.py` | MODIFIED — `PER_INSTRUMENT_BEST` rebuilt from `instrument_config.py`; `log_performance()` wired for both IS and OOS |
| `BESTS.md` | MODIFIED — Phase 25 section with tier list, top-15 performers, production basket, and infrastructure table |
| `docs/COMMAND_CHEATSHEET.md` | MODIFIED — Phase 25 section with all new commands |
| `progress_docs/plans/full.md` | MODIFIED — Phase 25 added to master plan |
| `progress_docs/current.md` | MODIFIED — Session log entry for 2026-05-21 |

## Phase 27B — Wavelet Feature Preprocessor (2026-05-26)

Multi-level Daubechies-4 wavelet decomposition as deterministic feature pipeline. Extracts 34-114 features per instrument series (per-level stats, cross-level correlation, volatility decomposition). Precomputed wavelet features can be joined with existing CatBoost/LSTM feature matrices.

**Prerequisite:** `pywt` (PyWavelets) — already installed in project environment.

### Training with Wavelet Features

```bash
# Train CatBoost with wavelet features enabled
# Adds 114 wavelet features (Close+High+Low × 34 price-wavelet + 4 volatility-wavelet)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --wavelet-features --fast

# Basket training with wavelet features
uv run scripts/train_ml_pipeline_v3.py --basket SPY,QQQ,GLD,XLK --wavelet-features --fast

# With CPCV cross-validation
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --wavelet-features --cv-method cpcv --fast
```

### Benchmark Wavelet vs Baseline

```bash
# Single instrument benchmark (4 models: IS+OOS × baseline+wavelet)
uv run scripts/benchmark_wavelet_features.py --symbol SPY --fast

# Multi-instrument benchmark
uv run scripts/benchmark_wavelet_features.py --basket SPY,QQQ,GLD,XLK --fast

# Custom IS/OOS split
uv run scripts/benchmark_wavelet_features.py --symbol SPY --is-start 2016-01-01 --is-end 2021-12-31 --oos-start 2022-01-01 --oos-end 2026-05-01
```

### SHAP Feature Importance Analysis

```bash
# Analyze which wavelet features rank highest
uv run scripts/analyze_wavelet_importance.py --symbol SPY

# Multi-instrument SHAP analysis (gate: ≥3/5 must have wavelet in top-20)
uv run scripts/analyze_wavelet_importance.py --basket SPY,QQQ,GLD,XLK,SLV
```

### Wavelet Feature API

```python
from src.features.wavelet_features import (
    WaveletFeatureExtractor,
    compute_wavelet_features,
    compute_wavelet_volatility_features,
    compute_wavelet_price_volume_features,
)

# Extract wavelet features from OHLCV DataFrame
wfx = compute_wavelet_price_volume_features(df)  # 114 features

# Single-series wavelet decomposition (34 features)
wf = compute_wavelet_features(df['Close'], window=128, levels=5)

# Volatility decomposition (4 features: structural, micro, macro, regime shift ratio)
vf = compute_wavelet_volatility_features(df['Close'])

# Sklearn-compatible transformer
extractor = WaveletFeatureExtractor(output='reduced')  # 20 most important
features = extractor.transform(df)
```

### Phase 27B File Changes

| File | Change |
|------|--------|
| `src/features/wavelet_features.py` | NEW — `WaveletFeatureExtractor`, `compute_wavelet_features`, `compute_wavelet_volatility_features`, `compute_wavelet_price_volume_features` |
| `src/features/__init__.py` | MODIFIED — +4 exports |
| `scripts/train_ml_pipeline_v3.py` | MODIFIED — `--wavelet-features` flag, `use_wavelet` param wires into `extract_features()` → CatBoost training |
| `scripts/benchmark_wavelet_features.py` | NEW — 4-model IS/OOS comparison |
| `scripts/analyze_wavelet_importance.py` | NEW — SHAP importance ranking + gate validation |

## Phase 27A — TTS-GAN Financial Data Augmentation (2026-05-26)

Transformer-based GAN for generating synthetic OHLCV data. Augments scarce
financial training data to improve downstream forecasting model generalization.

**Prerequisite:** `torch` (already installed). GPU recommended for full training.

### Training TTS-GAN

```bash
# Train TTS-GAN on SPY IS data
uv run scripts/train_tts_gan.py --symbol SPY --start 2016-01-01 --end 2021-12-31

# Long sequence (paper: K=120)
uv run scripts/train_tts_gan.py --symbol SPY --seq-len 120 --epochs 300

# Fast smoke test
uv run scripts/train_tts_gan.py --symbol SPY --fast --epochs 20
```

### GAN-Augmented ML Training

```bash
# Train CatBoost with TTS-GAN augmented data (inline GAN training)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --gan-augment --gan-epochs 50 --fast

# Use pre-trained GAN model (skip GAN training)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --gan-augment --gan-model models/gan/tts_gan_SPY.pt
```

### Benchmark GAN Augmentation (Gate Validation)

```bash
# Full gate test: ≥10% LSTM directional error reduction on SPY 2022 bear
uv run scripts/benchmark_gan_augmentation.py --symbol SPY --n-trials 3

# Fast gate check
uv run scripts/benchmark_gan_augmentation.py --symbol SPY --fast --n-trials 1
```

### TTS-GAN Paper Defaults

| Param | Generator | Discriminator |
|-------|-----------|---------------|
| Layers (D) | 3 | 3 |
| Heads (H) | 5 | 30 |
| Embed dim (M) | 10 | 90 |
| Patch size (P) | 15 | 15 |
| LR | 1e-4 | 1e-4 |

### Phase 27A File Changes

| File | Change |
|------|--------|
| `src/ml/gan_convergence.py` | NEW — DTW DeD-iMs convergence metric, Wasserstein distance, GANConvergenceMonitor |
| `src/ml/gan_data_augmentation.py` | NEW — TTSGAN, TTSGenerator, TTSDiscriminator, prepare_gan_samples, augment_dataset |
| `src/ml/__init__.py` | MODIFIED — +8 exports |
| `scripts/train_tts_gan.py` | NEW — CLI: train TTS-GAN, generate synthetic samples, save augmented data |
| `scripts/benchmark_gan_augmentation.py` | NEW — Gate validation: LSTM directional error with/without GAN augmentation |
| `scripts/train_ml_pipeline_v3.py` | MODIFIED — `--gan-augment`, `--gan-epochs`, `--gan-seq-len`, `--gan-aug-ratio`, `--gan-model` flags; `_gan_samples_to_dataframe()` helper; inline GAN training in Stage 1b |

## Phase 27C — TadGAN Regime Anomaly Detection (2026-05-26)

Cycle-consistent GAN for detecting market dislocations. Learns normal price
behavior manifold, flags crisis events as anomalies. Integrated as optional
risk gate in RulesFirstStrategy.

**Prerequisite:** `torch` (already installed). GPU recommended.

### Training TadGAN

```bash
# Train on pre-crisis data
uv run scripts/train_tadgan.py --symbol SPY --start 2010-01-01 --end 2019-12-31

# Fast smoke test
uv run scripts/train_tadgan.py --symbol SPY --fast --epochs 20
```

### Crisis Detection Gate Validation

```bash
# Full gate test: detect ≥4/4 crisis events at ≤5 FP/year
uv run scripts/benchmark_tadgan.py --symbol SPY --epochs 200

# Fast check
uv run scripts/benchmark_tadgan.py --symbol SPY --fast
```

### Anomaly Gate in Backtesting

```bash
# Block entries during TadGAN-detected anomalies
uv run scripts/backtest_rules_first.py SPY --start 2020-01-01 --end 2026-06-01 \
    --use-tadgan-gate --tadgan-model models/anomaly/tadgan_SPY.pt --fast
```

### GPU-Accelerated Commands (GPU Task Queue)

```bash
# Chronos-2 fine-tuning with LoRA
uv run scripts/finetune_chronos.py --symbol SPY --model chronos-2-small --lora --device cuda

# WaveletDiff generation model
uv run scripts/train_wavelet_diffusion.py --symbol SPY --epochs 500 --device cuda

# TTS-GAN full training (18 instruments)
for SYM in SPY QQQ XLK XLE GLD SLV; do
    uv run scripts/train_tts_gan.py --symbol $SYM --epochs 200 --device cuda
done
```

### Phase 27C File Changes

| File | Change |
|------|--------|
| `src/ml/anomaly_detection.py` | NEW — TadGAN (LSTM encoder/decoder + LSTM critic, cycle-consistent, α-calibrated anomaly scoring) |
| `scripts/train_tadgan.py` | NEW — CLI: train TadGAN, save .pt model |
| `scripts/benchmark_tadgan.py` | NEW — Gate validation: detect COVID/2022 bear/2025 tariff/2026 oil shock |
| `scripts/finetune_chronos.py` | NEW — CLI: Chronos-2 LoRA fine-tuning + evaluation |
| `scripts/train_wavelet_diffusion.py` | NEW — CLI: WaveletDiff training (DDIM sampling) |
| `src/strategies/rules_first_strategy.py` | MODIFIED — +`use_tadgan_gate`, +`tadgan_model_path`, +`_init_tadgan_gate()`, gate blocks entries during anomalies |
| `scripts/backtest_rules_first.py` | MODIFIED — +`--use-tadgan-gate`, +`--tadgan-model`, +`--tadgan-threshold-pct` flags |
| `src/ml/__init__.py` | MODIFIED — +11 exports (GAN + TadGAN) |
| `docs/GPU_TASK_QUEUE.md` | NEW — 7 GPU tasks with self-contained tutorials |

---

## Phase 25 — Component Wiring & End-to-End Integration (2026-05-27)

### NSGA2 Multi-Objective Optimizer
```bash
# Pareto-optimal parameter search via NSGA2
uv run scripts/run_nsga2_optimizer.py --ticker SPY --start 2018-01-01 --end 2024-12-31

# Custom population and generations
uv run scripts/run_nsga2_optimizer.py --ticker QQQ --population 50 --generations 30
```

### Dynamic GA Optimizer (Regime-Adaptive)
```bash
# Regime-adaptive GA with associative memory
uv run scripts/run_dynamic_ga.py --ticker SPY --window 252 --step 21

# Custom memory size
uv run scripts/run_dynamic_ga.py --ticker GLD --memory-size 20
```

### NLP Sentiment Pipeline
```bash
# Train SVM sentiment model on :) / :( distant supervision
uv run scripts/run_sentiment_pipeline.py --model svm --train

# Train BiLSTM sentiment model
uv run scripts/run_sentiment_pipeline.py --model bilstm --train

# Train ensemble model
uv run scripts/run_sentiment_pipeline.py --model ensemble --train

# Predict sentiment on text
uv run scripts/run_sentiment_pipeline.py --model svm --predict "Bullish quarter ahead"
```

### Fuzzy Logic + SVM Regime in Combined Strategy
```bash
# Backtest with fuzzy logic scoring blended at 30% weight
uv run scripts/backtest_combined.py SPY --start 2018-01-01 --use-fuzzy --fuzzy-weight 0.30

# Backtest with SVM regime detection gating entries
uv run scripts/backtest_combined.py SPY --start 2018-01-01 --use-svm-regime --svm-regime-window 100

# Skip entries during SVM-classified down markets
uv run scripts/backtest_combined.py SPY --start 2018-01-01 --use-svm-regime --no-svm-down-skip

# Full integration: fuzzy + SVM regime
uv run scripts/backtest_combined.py SPY --start 2018-01-01 --use-fuzzy --fuzzy-weight 0.25 --use-svm-regime
```

### Dual Alpha/Beta (Auto-Reported)
Dual alpha/beta decomposition runs automatically after every `run_ml_backtest.py` and `backtest_rules_first.py` backtest. Results saved to `reports/ml_backtest/dual_alpha_beta_{ticker}.json`.

## Phase 28 — Data Infrastructure & Universe Management (2026-05-27)

### Universal Symbol Query
```bash
# Get all US Information Technology equities
uv run python -c "from src.data import get_universe; print(get_universe(country='United States', sector='Information Technology')[:10])"

# Count symbols by US sector
uv run python -c "from src.data import count_by_sector; print(count_by_sector().to_string())"

# Search for symbols by name
uv run python -c "from src.data import search_symbols; print(search_symbols('Apple')[['name','exchange','sector']].head(5).to_string())"

# List available filter values
uv run python -c "from src.data import available_filter_values; import json; print(json.dumps({k: v[:5] for k, v in available_filter_values().items()}, indent=2))"
```

### Hierarchical Symbol Filtering
```bash
# Chain filter narrowing with step-by-step tracking
uv run python -c "
from src.data import filter_hierarchical
import financedatabase as fd
eq = fd.Equities()
result = filter_hierarchical(eq.data, {'country': 'United States', 'sector': 'Information Technology'})
print(result.summary)
print(f'Sample: {result.symbols[:10]}')
"

# Pre-compute filter pipeline by sector
uv run python -c "
from src.data import filter_pipeline
import financedatabase as fd
sectors = filter_pipeline(fd.Equities().data, level='sector')
for name, result in sectors.items():
    print(f'{name}: {result.final_count} symbols')
"
```

### Batch Data Loading
```bash
# Concurrent multi-symbol data load with progress
uv run python -c "
import yfinance as yf
from src.data import load_batch
result = load_batch(
    ['SPY', 'QQQ', 'XLK', 'XLE', 'GLD', 'SLV'],
    lambda sym: yf.download(sym, period='1y', progress=False),
    concurrency=4,
)
print(f'Loaded {len(result.data)}/{result.total} in {result.elapsed_seconds:.1f}s')
print(f'Failed: {result.failed}')
"
```

### Congressional Trade Signal Feed (P28-10)
```bash
# Fetch recent congressional trades + aggregate per-ticker signals
uv run python -c "
from src.data import fetch_congress_trades, get_congress_signals
trades = fetch_congress_trades()
print(f'Fetched {len(trades)} trades')
signals = get_congress_signals(min_trades=3)
print(signals.head(10).to_string())
"

# Filter to specific chamber
uv run python -c "
from src.data import fetch_congress_trades
house = fetch_congress_trades(chambers=['House'])
senate = fetch_congress_trades(chambers=['Senate'])
print(f'House: {len(house)}, Senate: {len(senate)}')
"
```

### Fundamental Analysis Pipeline (P28-7)
```bash
# Pipe tickers into FinanceToolkit for deep fundamental ratios
uv run python -c "
from src.data import get_universe, to_toolkit
tech = get_universe(country='United States', sector='Information Technology', market_cap='Large Cap')[:5]
batch = to_toolkit(tech, quarters=8)
for ticker, df in batch.success.items():
    print(f'{ticker}: {len(df)} ratios')
print(f'Coverage: {batch.coverage:.0f}%')
"

# ML-ready fundamental features
uv run python -c "
from src.data import fundamental_features_for_ml
feat = fundamental_features_for_ml(['AAPL', 'MSFT', 'GOOGL'], quarters=12)
print(feat.to_string())
"

# End-to-end: sector -> tickers -> fundamentals
uv run python -c "
from src.data import pipe_sector_fundamentals
batch = pipe_sector_fundamentals(sector='Energy', max_tickers=10, quarters=8)
for t, df in batch.success.items():
    print(f'{t}: {len(df)} ratios')
"
```

### US Stock Symbol Auto-Sync (P28-8)
```bash
# Download full US symbol list from rreichel3/US-Stock-Symbols
uv run python -c "
from src.data import fetch_all_us_symbols
result = fetch_all_us_symbols()
print(f'{result.meta.ticker_count} total US symbols')
print(f'NYSE: {len(result.by_exchange[\"nyse\"])}, NASDAQ: {len(result.by_exchange[\"nasdaq\"])}')
"

# Sync with diff against local cache
uv run python -c "
from src.data import sync_with_diff
current, diff = sync_with_diff()
print(f'Synced {len(current.tickers)} symbols')
if diff['added'] or diff['removed']:
    print(f'+{len(diff[\"added\"])} added, -{len(diff[\"removed\"])} removed')
"
```

### Fund Flow / Order Flow Signals (P28-13)
```bash
# Compute fund flow components from OHLCV data
uv run python -c "
import yfinance as yf
from src.signals.fund_flow import compute_flow_components, compute_flow_summary
df = yf.download('SPY', period='6mo', progress=False)
signals = compute_flow_components(df)
summary = compute_flow_summary(df['Close'].values, signals)
print(f'Net flow: {summary.net_flow:.0f}, SMI: {summary.smart_money_index:.3f}')
print(f'Accum: {summary.accumulation_days}d, Distr: {summary.distribution_days}d')
print(f'Large orders: {summary.large_order_count}')
"

# Generate ML-ready fund flow features
uv run python -c "
import yfinance as yf
from src.signals.fund_flow import fund_flow_to_ml_features
df = yf.download('SPY', period='6mo', progress=False)
features = fund_flow_to_ml_features(df)
print(f'{features.shape[1]} flow features over {len(features)} bars')
print(features.columns.tolist())
"
```

### Survivorship-Bias-Free Historical Universe
```bash
# Get investable US equities on a specific date (retains delisted)
uv run python -c "
from src.data import HistoricalUniverse
h = HistoricalUniverse()
symbols = h.universe_on('2020-01-15')
print(f'{len(symbols)} symbols investable on 2020-01-15')
"
```

### Anti-Overfitting Wiring Status
| Module | Wired To | Gate | CLI Flag |
|--------|----------|------|----------|
| `lock_box.py` | `train_ml_pipeline_v3.py` | Blind holdout, one-time access | `--use-lock-box` |
| `blind_analysis.py` | `train_ml_pipeline_v3.py` | Tune on scrambled labels | `--blind-analysis` |
| `label_shuffling.py` | `train_ml_pipeline_v3.py` | Verify model beats random | `--label-shuffling` |
| `nested_cv.py` | `train_ml_pipeline_v3.py` | Nested PurgedKFold CV | `--use-phase25-cv` |
| `overfitting_detector.py` | `train_ml_pipeline_v3.py` | KNN-DTW loss curve detection | `--dtw-overfit-detect` |
| `dual_alpha_beta.py` | `run_ml_backtest.py`, `backtest_rules_first.py` | Bull/bear alpha-beta + Chow test | *auto* |
| `fuzzy_system.py` | `combined_strategy.py` | 5-state Mamdani fuzzy logic | `--use-fuzzy` |
| `svm_regime.py` | `combined_strategy.py` | SVM regime gating entries | `--use-svm-regime` |
| `nsga2_optimizer.py` | `scripts/run_nsga2_optimizer.py` | Pareto multi-objective optimizer | *CLI script* |
| `dynamic_ga.py` | `scripts/run_dynamic_ga.py` | Regime-adaptive GA | *CLI script* |
| `sentiment_pipeline.py` | `scripts/run_sentiment_pipeline.py` | NLP sentiment ensemble | *CLI script* |

### Chart Pattern Similarity Search (P28-11)
```bash
# Find 10 most similar historical patterns to the last 20 bars
uv run python -c "
import yfinance as yf
from src.patterns import search_similar_patterns, format_result_table
df = yf.download('SPY', period='1y', progress=False)
result = search_similar_patterns(df, query_end_idx=len(df)-1, window=20, top_k=5)
print(format_result_table(result))
print(f'Bias: {result.positive_pct:.1f}% positive forward returns')
"

# Rolling scan every 5 bars
uv run python -c "
import yfinance as yf
from src.patterns import search_rolling, format_result_table
df = yf.download('SPY', period='1y', progress=False)
results = search_rolling(df, window=20, step=5, start_idx=100, top_k=3)
for i, r in enumerate(results):
    print(f'Step {i}: {r.n_matches} matches, bias={r.positive_pct:.0f}%')
"
```

### Patternity Pattern Comparison (P28-12)
```bash
# Compare patternity detections vs project detectors
uv run python -c "
import yfinance as yf
from src.patterns import PatternityWrapper
df = yf.download('SPY', period='6mo', progress=False)
pw = PatternityWrapper()
matches = pw.detect(df)
print(f'Patternity found {len(matches)} patterns')
for m in matches[:5]:
    print(f'  {m.pattern_name} at bars [{m.start_idx}-{m.end_idx}] conf={m.confidence:.2f}')
"
```

### Futures Inventory Signals (P28-14)
```bash
# Compute commodity supply/demand bias from inventory data
uv run python -c "
from datetime import datetime, timedelta
from src.data import FuturesInventory, InventoryRecord
fi = FuturesInventory()
base = datetime(2024, 1, 1)
# Load sample data (replace with real exchange data)
recs = [InventoryRecord(date=base+timedelta(weeks=i), symbol='GC=F', exchange='COMEX',
    warehouse_stocks=100000+5000*(i%10), registered_stocks=60000+3000*(i%10),
    eligible_stocks=40000+2000*(i%10), cancelled_warrants=5000+250*(i%10))
    for i in range(52)]
fi.load_records('GC=F', recs)
sig = fi.get_signals('GC=F')
print(f'Gold inventory: {sig.signal}, zscore={sig.stock_zscore:.1f}, bias={fi.supply_demand_bias(\"GC=F\"):.2f}')
print(fi.to_dataframe().to_string())
"
```

### skfolio Portfolio Optimization (P28-18)
```bash
# Compare 5 optimization methods side-by-side
uv run python -c "
import yfinance as yf
import pandas as pd
from src.optimization import compare_methods, weights_to_dataframe
tickers = ['SPY', 'QQQ', 'XLK', 'GLD', 'TLT']
df = yf.download(tickers, start='2022-01-01', progress=False)['Close']
comparison = compare_methods(df, max_weight=0.30, min_weight=0.01)
print(f'Best Sharpe: {comparison.best_by_sharpe.method} ({comparison.best_by_sharpe.expected_sharpe:.2f})')
print(f'Best Diversification: {comparison.best_by_diversification.method}')
print(f'Equal-weight Sharpe: {comparison.equal_weight_sharpe:.2f}')
print(weights_to_dataframe(comparison.methods, tickers).to_string())
"

# Single-method optimization
uv run python -c "
import yfinance as yf
from src.optimization import optimize_hrp
df = yf.download(['SPY','QQQ','XLK','GLD','TLT'], start='2022-01-01', progress=False)['Close']
w = optimize_hrp(df)
for ticker, weight in sorted(w.weights.items(), key=lambda x: -x[1]):
    print(f'{ticker}: {weight:.1%}')
"
```

### MCP Stock Server (P28-21)
```bash
# Start the MCP server (stdio — configure in Cursor/Claude MCP settings)
uv run python -c "from src.mcp import run_mcp_server; run_mcp_server()"

# Or test tools directly without server
uv run python -c "
from src.mcp import StockDataTools
import json
print(json.dumps(StockDataTools.get_price('SPY', start='2025-01-01'), indent=2))
print(json.dumps(StockDataTools.get_technicals('AAPL'), indent=2))
print(json.dumps(StockDataTools.get_multi('SPY,QQQ,XLK', start='2025-01-01'), indent=2))
"
```

### Daily Report Agent (P28-22)
```bash
# Generate daily production basket report
uv run python scripts/daily_report_agent.py

# Custom basket
uv run python -c "
from scripts.daily_report_agent import generate_daily_report
basket = {'S': ['SPY','QQQ','XLK'], 'A': ['GLD','SLV'], 'B': ['TLT']}
report = generate_daily_report(basket)
print(report)
"
```

### Options Chain & Greeks (P28-20)
```bash
# Fetch options chain and compute key metrics
uv run python -c "
from src.data import fetch_options_chain, options_chain_to_features, get_options_sentiment
chain = fetch_options_chain('SPY')
if chain:
    print(f'Spot: {chain.spot:.2f}, ATM IV: {chain.atm_iv:.3f}')
    print(f'P/C Volume: {chain.pc_ratio_volume:.3f}, P/C OI: {chain.pc_ratio_oi:.3f}')
    print(f'Max Pain: {chain.max_pain:.2f}')
    print(f'Features: {options_chain_to_features(chain)}')
    print(f'Sentiment: {get_options_sentiment(\"SPY\")}')
"

# Compute Black-Scholes Greeks
uv run python -c "
from src.data import compute_greeks
g = compute_greeks(S=450.0, K=455.0, T=30/365, r=0.05, sigma=0.25, option_type='call')
print(g.summary())
# delta=+0.412 gamma=0.0312 theta=-0.0543 vega=0.2134
"

# Scan for unusual options activity
uv run python -c "
from src.data import fetch_options_chain, detect_unusual_options_activity
chain = fetch_options_chain('SPY')
if chain:
    unusual = detect_unusual_options_activity(chain, volume_threshold=3.0)
    for u in unusual[:5]:
        print(f'{u[\"type\"]} {u[\"strike\"]:.0f}: vol/OI={u[\"vol_oi_ratio\"]:.1f}x ({u[\"volume\"]}/{u[\"open_interest\"]})')
"
```

### Risk-Profiling Onboarding (P28-25)
```bash
# Interactive CLI
uv run python scripts/user_profile.py

# Programmatic profile building
uv run python -c "
from scripts.user_profile import build_profile, generate_recommendation
profile = build_profile(risk_tolerance='moderate', time_horizon='medium', goal='growth')
rec = generate_recommendation(profile)
print(f'Instruments: {rec[\"total_instruments\"]}')
print(f'CLI: {rec[\"cli_flags\"]}')
print(f'Backtest: {rec[\"backtest_command\"]}')
"
```

### Docker Compose Production (P28-24)
```bash
# Start production services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f app

# Stop
docker compose -f docker-compose.prod.yml down
```

---

## New Ticker Screening & Backtesting (2026-05-27)

> **Script:** `scripts/screen_and_backtest_10.py`. Full 4-phase pipeline: fundamental screen (11 hard filters) → data download → Rules-First all-on backtest (IS=2016-2024, OOS=2025-2026) → tier classification (S/A/B/C).

### Quick Pipeline
```bash
# Run on default 10-candidate list
uv run scripts/screen_and_backtest_10.py
```

### Results Summary (2026-05-27)

| Tier | Tickers | N | Mean OOS Sharpe | Notes |
|------|---------|---|-----------------|-------|
| **S** | CHTR, LRCX | 2 | **+1.00** | CHTR +1.11 (phoenix, -59% BH), LRCX +0.90 (69% WR) |
| **A** | GD, ABT | 2 | **+0.73** | Both phoenix — IS negative, OOS positive. +0.83/+1.08 delta |
| **B** | NOC | 1 | +0.23 | Defense sector expansion. 62% WR, 16 trades. |
| **C** | DHI, URI, CTVA, APH, GE | 5 | -0.81 | Low trades or terrible WR. Skip. |

### Adding Custom Candidates
```bash
# Edit CANDIDATES list in script, then re-run
uv run scripts/screen_and_backtest_10.py
```

---

## Production Monitoring — Paper Trading & OOS Re-Run (Q2 2026)

> **Basket:** 18 instruments, 3 tiers. S: XLK/XLE/GLD/SPY/SLV/QQQ (60%), A: NUE/STLD/HAL/MPC/EOG (25%), B: INTC/AMD/LMT/JNJ/MRK/NEM (15%). CN_CATL (yfinance 404).

### Daily Paper Trading Signals
```bash
# Generate signals for all 18 instruments
$env:PYTHONIOENCODING = "utf-8"; uv run scripts/paper_trade_daily.py --basket XLK,XLE,GLD,SPY,SLV,QQQ,NUE,STLD,HAL,MPC,EOG,INTC,AMD,LMT,JNJ,MRK,NEM,CN_CATL

# Single symbol
uv run scripts/paper_trade_daily.py --symbol SPY

# Show recent paper trading activity
uv run scripts/paper_trade_daily.py --status

# Backfill 90 days
uv run scripts/paper_trade_daily.py --symbol SPY --days 90
```

### Daily Market Report
```bash
# Generate daily basket report (regime check, sentiment, tier summary)
$env:PYTHONIOENCODING = "utf-8"; uv run scripts/daily_report_agent.py
```

### Quarterly OOS Re-Run (Q2 2026)
```bash
# OOS re-run for 17-instrument production basket (per-instrument best params)
# CN_CATL excluded — yfinance 404 for Chinese ticker
$env:TQDM_DISABLE = "1"; uv run scripts/backtest_all_comprehensive.py \
    --tickers XLK,XLE,GLD,SPY,SLV,QQQ,NUE,STLD,HAL,MPC,EOG,INTC,AMD,LMT,JNJ,MRK,NEM \
    --period oos --use-best

# Full IS+OOS rerun
$env:TQDM_DISABLE = "1"; uv run scripts/backtest_all_comprehensive.py \
    --tickers XLK,XLE,GLD,SPY,SLV,QQQ,NUE,STLD,HAL,MPC,EOG,INTC,AMD,LMT,JNJ,MRK,NEM \
    --use-best
```

### Q2 2026 OOS Results (2026-05-27)
| Metric | Value |
|--------|-------|
| OOS Positive | **17/17 (100%)** |
| Mean OOS Sharpe | **1.135** |
| Median OOS Sharpe | 1.065 |
| OOS > IS improvement | 17/17 (100%) |
| IS→OOS Correlation | -0.388 |
| Top OOS | INTC +1.781, HAL +1.631, LMT +1.602 |
| Market Regime | BULLISH (71% above MA50, RSI 59.0) |
