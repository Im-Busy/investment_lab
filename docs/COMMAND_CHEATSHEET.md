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

## Phase 23 — RulesFirst Advanced Signal Wiring (2026-05-21)

> **New params (Phase 23 RF3):** GARCH dynamic ATR trail, options sentiment modifier, Kelly dynamic sizing, order book signals.
> All default OFF — opt-in via `--use-*` flags.

### Rules-First with Advanced Signals
```bash
# GARCH dynamic ATR trail: wider stops in low vol, tighter in high vol
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-16 \
    --entry-threshold 0.55 --min-reliability 0.70 --use-garch-atr

# Options sentiment: scale signals by put/call ratio + GEX proxy
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-16 \
    --entry-threshold 0.55 --use-options-sentiment --options-sentiment-weight 0.10

# Kelly dynamic position sizing: trade fractional equity by signal confidence
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-16 \
    --entry-threshold 0.55 --use-kelly-sizing --kelly-fraction 0.5

# Order book microstructure: bid-ask imbalance signal enhancement
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-16 \
    --entry-threshold 0.55 --use-order-book

# All four advanced signals together
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-16 \
    --entry-threshold 0.55 --min-reliability 0.70 \
    --use-garch-atr --use-options-sentiment --use-kelly-sizing --use-order-book
```

### Advanced Signal Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--use-garch-atr` | False | Use EGARCH forecast vol for dynamic trail width (RF3.1) |
| `--garch-model` | egarch | GARCH variant: garch, egarch, gjr-garch |
| `--use-options-sentiment` | False | Scale signals by PC ratio + GEX proxy (RF3.2) |
| `--options-sentiment-weight` | 0.10 | Sentiment modifier weight [0-1] |
| `--use-kelly-sizing` | False | Kelly-derived fraction-of-equity position sizing (RF3.3) |
| `--kelly-fraction` | 0.5 | Kelly fraction: 0.5=half-Kelly, 0.25=quarter |
| `--use-order-book` | False | Bid-ask imbalance signal modifier (RF3.4) |
| `--use-vix-regime-sizing` | False | P1.5: Cap position size by VIX regime (50% HIGH_VOL, 25% CRISIS) |
| `--vix-size-high-vol-cap` | 0.50 | Max size fraction in ELEVATED VIX regime |
| `--vix-size-crisis-cap` | 0.25 | Max size fraction in STRESS VIX regime |

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
