# Investment Trying Lab — Technical Introduction

> A deep-dive into the architecture, algorithms, and engineering decisions behind a 187,000-line research-grade quantitative trading system spanning pattern detection, machine learning, risk management, and AI-native infrastructure.

---

## Table of Contents

1. [Signal Generation: 54 Pattern Detectors](#1-signal-generation-54-pattern-detectors)
2. [Pattern Aggregation & Scoring](#2-pattern-aggregation--scoring)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
4. [Backtesting Engine](#4-backtesting-engine)
5. [Risk Management](#5-risk-management)
6. [Signal Quality & Research Infrastructure](#6-signal-quality--research-infrastructure)
7. [AI-Native Workflow & Observability](#7-ai-native-workflow--observability)
8. [Infrastructure & Dependencies](#8-infrastructure--dependencies)

---

## 1. Signal Generation: 54 Pattern Detectors

The system's primary alpha source is a catalog of **54 chart pattern detectors** across **10 categories**, each an implementation of `BasePattern` (`src/patterns/base.py:95-643`).

### 1.1 Pattern Detector Interface

Every detector is a `@dataclass` subclass of `BasePattern` and must implement:

```python
class BasePattern(ABC):
    name: str                          # e.g., "DoubleTop"
    pattern_type: PatternType          # REVERSAL | CONTINUATION | BREAKOUT | COUNTER_TREND | VOLATILITY
    signal_direction: SignalDirection  # LONG | SHORT | NEUTRAL
    min_bars_required: int             # Minimum lookback (default 5)

    @abstractmethod
    def detect(self, df, i) -> PatternResult: ...

    @abstractmethod
    def generate_signal(self, df, i) -> Optional[TradeSignal]: ...

    def detect_vectorized(self, df) -> np.ndarray: ...
```

**Key architectural decisions:**

- **Per-bar detection** — `detect(df, i)` judges pattern presence at bar `i` using only data up to and including bar `i`. No forward-looking bias possible.
- **Vectorized acceleration** — `detect_vectorized(df)` returns an `np.int8` array of `[-1, 0, +1]` for the entire dataset at once, delivering 50-100x speedup over per-bar iteration. Critical for backtesting 125+ instruments.
- **Precompute caching** — `precompute_signals(df)` calls `detect_vectorized()` once before any backtest and stores results in an O(1) dictionary lookup. The strategy's `_compute_score()` then reads from cache rather than re-running detection.
- **ARRAY CACHING BY `id(df)`** — `_extract_arrays(df)` caches OHLCV NumPy arrays by DataFrame identity so repeated pattern detection on the same data avoids redundant extraction.
- **IndicatorCache sharing** — `IndicatorCache` (`src/indicators/indicator_cache.py`) pre-computes shared indicators (ATR, RSI, MACD, ADX, EMAs) once and shares across all patterns, eliminating redundant computation in a 54-detector ensemble.
- **TradeSignal contract** — `detect()` returns a `PatternResult` with detection metadata; `generate_signal()` produces a `TradeSignal` dataclass with `entry_price`, `stop_loss`, `take_profit_{1,2,3}`, and `confidence` (0.0-1.0). The backtest engine consumes `TradeSignal` directly, decoupling detection from execution.

### 1.2 The 10 Categories

| # | Category | Detectors | Source |
|---|----------|-----------|--------|
| 1 | **Classic** (10) | Double Top/Bottom, Triple Top/Bottom, Ascending/Descending Triangle, Rectangle, Wedge, Dead Cat Bounce, Trader Vic 2B | NCFE, Duddella research |
| 2 | **Candlestick** (6) | Engulfing, Hammer, Doji, Harami, Dark Cloud Cover, Piercing Line | Japanese candlestick literature |
| 3 | **Harmonic** (9) | Gartley, ABC, Symmetric Triangle, Bollinger Bands, Butterfly, Bat, Crab, Cypher, Shark | Fibonacci retracement ratios (0.382/0.618/0.786) |
| 4 | **Complex** (6) | Head & Shoulders, Cup & Handle, Parabolic Arc, Spike & Ledge, Three Hills, Pipe | Composite multi-swing geometry |
| 5 | **Breakout** (2) | Gap Detection (breakaway/continuation/exhaustion), Donchian Channel | Turtle Trading system, gap hierarchy |
| 6 | **Continuation** (2) | Flag, Pennant | Pole-and-flag geometry |
| 7 | **Basic** (6) | Floor Pivot, MSL, NR7ID, N-Bar Decline, 2-Bar Reversal, Matching Lows | Mechanical price structure |
| 8 | **FMZ** (6) | AlphaBeast, MultiFactorTrend, MomentumZigZag, EMAMACDHF, AdaptiveBollinger, AIVolatilityBreakout | PineScript→Python from 5,807-strategy repo |
| 9 | **Technical** (4) | Ichimoku Cloud, Keltner Channel, Williams %R, CCI | Multi-component indicator systems |
| 10 | **Range Persistence** (3) | PersistentRange, ContractingRange, ExpandingRange | From "Hypotheses to Factors" (arXiv:2604.26747v1), Pure OOS Sharpe +2.41 |

### 1.3 Reliability Weights

Each detector carries a **reliability weight** sourced from academic research (NCFE Technical Analysis, Duddella's "Trade Chart Patterns Like the Pros") and empirically calibrated via the pattern evaluation gate:

| Tier | Weight Range | Example Patterns |
|------|-------------|------------------|
| Highest | 0.80-0.87 | Head & Shoulders, Cup & Handle, Ascending Triangle, Gartley |
| High | 0.72-0.78 | Flag, Pennant, Hammer, Piercing Line, Engulfing |
| Medium | 0.65-0.70 | Donchian Channel, Gap Detection, Doji, Dark Cloud Cover |
| Lower | 0.60-0.72 | Floor Pivot, MSL, NR7ID, Matching Lows |

These weights are the **base multiplier** in the scoring equation — see §2. A pattern with reliability 0.40 was excluded entirely (gated by `min_reliability`).

### 1.4 Regime Awareness

Every pattern declares `preferred_regimes` and `incompatible_regimes` via the `RegimeState` enum (`TRENDING`, `RANGING`, `VOLATILE`, `TRANSITION`). Patterns auto-inherit defaults from their `PatternType`:
- **REVERSAL** patterns → preferred in `TRENDING/TRANSITION`, avoided in weak-trend regimes
- **CONTINUATION** patterns → preferred in `TRENDING`, avoided in `RANGING`
- **BREAKOUT** patterns → preferred in `RANGING/VOLATILE`
- **COUNTER_TREND** → preferred in `TRENDING/TRANSITION`

The `RegimeDetector` (`src/indicators/regime_detector.py`) classifies each bar and the signal generator applies regime gating — filtering incompatible patterns and adjusting `confidence` by preference strength.

---

## 2. Pattern Aggregation & Scoring

The core of the production strategy (`src/strategies/rules_first_strategy.py:331-386`) is `_compute_score(idx)`, which transforms 54 binary/ternary pattern signals into a single **bipolar score** for each bar.

### 2.1 The Aggregation Formula

For each bar `idx`:

```
1. PER-PATTERN SUM
   score = Σ [ pattern_detected × reliability_weight × quality_multiplier × IR_scalar × signal_direction ]

2. CONFLUENCE BONUS
   if active_pattern_count ≥ 2:
       score += confluence_bonus × sign(score)        # default +0.10

3. VOLUME CONFIRMATION
   rel_vol = clamp(vol[-1] / mean(vol[-20:]), 0.5, 2.0)
   score *= rel_vol

4. MACRO GATING (optional)
   score *= VIX_regime_scalar      # STRESS=0.30, ELEVATED=0.75, NORMAL=1.0
   score *= yield_curve_scalar     # INVERTED=0.50, NEAR_INVERTED=0.75, NORMAL=1.0

5. FUNDAMENTAL OVERLAY (optional)
   score *= multi_factor_scalar

6. FINAL SQUASH
   score = tanh(score)             # maps (-∞, +∞) → (-1, +1)
```

The `tanh` squash is critical — it bounds the score to a meaningful `[-1, +1]` range where `+1` = unanimous strong bullish, `-1` = unanimous strong bearish, `0` = neutral. This avoids score drift from pattern count differences.

### 2.2 Configurable Weights and Gates

The aggregation is parameterized by 15+ toggles, each independently validated:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `entry_threshold` | 0.55 | Minimum `tanh` score for long entry |
| `exit_threshold` | 0.30 | Score below which to close long positions |
| `min_reliability` | 0.40 | Exclude patterns below this NCFE/Duddella weight |
| `confluence_bonus` | 0.10 | Boost when ≥2 patterns agree on direction |
| `use_ir_weights` | False | Enable 252-bar rolling IR weighting |
| `use_multi_tp` | True | Split exit into TP1 (partial) + trailing stop |
| `use_quality_registry` | True | Apply PASS/FAIL multipliers from evaluation gate |
| `use_vix_gate` | False | Scale scores by VIX stress regime |
| `use_yield_curve_gate` | False | Reduce scores in inverted yield curve |
| `use_short` | True | Enable bearish pattern short-side trading |
| `use_multi_factor` | False | Apply fundamental sector overlay |

### 2.3 Multi-Take-Profit Exit Logic

The strategy splits exit into two phases (`src/strategies/rules_first_strategy.py:531-571`):

```
TP1 at entry + tp1_atr × ATR(14)        → closes tp1_size portion (default 50%)
After TP1 hit:
    move stop-loss to breakeven          → locks in profit on remaining 50%
TP2 (remaining position)                  → managed by ATR trailing stop (default 3.0× ATR)
```

Symmetric for short side. This **split-exit** design improved OOS Sharpe from +0.54 to +1.21 (+124%) on SPY 2025.

### 2.4 Pattern Quality Registry

The `PatternQualityRegistry` (`src/signals/pattern_quality_registry.py`) gates patterns by empirical evaluation, preventing low-quality detectors from generating noise:

| Tier | Criteria | Weight Multiplier | Required Confluence |
|------|----------|-------------------|---------------------|
| **PASS** | 3-4 evaluation gate steps passed | 1.00× | 1 (trade independently) |
| **FAIL_STRONG** | 2 steps passed | 0.50× | 2 (need at least 1 companion) |
| **FAIL_WEAK** | 0-1 steps passed | 0.30× | 3 (need 2+ companions) |
| **ERROR** | Crash / no data | 0.00× (excluded) | — |

With quality registry ON, SPY OOS Sharpe improves from +1.21 to +2.00 with 100% win rate on 8 trades.

### 2.5 IR-Weighted Dynamic Reliability

`IRWeighting` (`src/signals/ir_weighting.py:1-247`) recomputes pattern reliability from recent empirical performance:

```
IR_p = mean(return_p[-252:]) / std(return_p[-252:])
weight_p = max(IR_p, 0) / Σ max(IR_k, 0)    # only positive-IR patterns contribute
```

Two modes:
- **Gate mode**: IR < 0 → pattern excluded entirely (weight = 0)
- **Scalar mode**: sigmoid(IR × 2) mapped to `[0.3, 0.7]` multiplier on base reliability

Look-ahead safety: returns are shifted forward by 1 bar, so `IR` at bar `idx` uses `return[idx-1]` — the pattern's *past* forward returns, not future.

---

## 3. Machine Learning Pipeline

### 3.1 Feature Engineering

`FeatureExtractor` (`src/ml/feature_engineering.py:1-463`) produces **100+ features per bar** across 6 categories:

| Category | Count | Features |
|----------|-------|----------|
| **Price** | ~30 | `return_N` (5/10/20/50/100), `log_return_N`, `highest_N`, `lowest_N`, `price_to_ma_N`, `ma_slope_N`, `ema_8/21/55`, `ema_spread`, `hl_range`, `oc_range`, `gap`, `price_position_N`, `dist_to_high_N`, `dist_to_low_N` |
| **Momentum** | ~20 | `rsi_N` (5/10/14/20), `roc_N`, `momentum_N`, `macd` + `signal` + `histogram` + `cross`, `stoch_k/d_N`, `atr_14/20` |
| **Volatility** | ~12 | `volatility_N`, `std_return_N`, `bb_pct/width/squeeze_N`, `volatility_regime`, `volatility_zscore` |
| **Volume** | ~12 | `volume_ratio_N`, `volume_zscore_N`, `obv` + `obv_change`, `vwap_N`, `close_to_vwap_N`, `cmf`, `volume_trend` |
| **Pattern Shape** | ~20 | `body_ratio`, `upper/lower_shadow`, `doji`, `long_body`, `engulfing`, `hh/hl_hc_lc`, `consecutive_dir`, `narrow_range_N`, `inside/outside_bar`, `close_percentile_N` |
| **Regime** | ~12 | `adx`, `plus/minus_di`, `di_diff`, `adx_trend`, `ma_slope_N`, `price_vs_ma_N`, `vol_regime`, `trend_strength`, `vol_regime_state` (tercile) |
| **Cross-Asset** | ~10 | Relative returns, `beta` to SPY/QQQ/TLT/GLD/XLE, cross-asset correlations |

ATR computation uses `prev_close.shift(1)` rather than current close to prevent look-ahead bias.

### 3.2 Triple-Barrier Labeling

Instead of naive binary labels ("profitable within N bars?"), the system uses **triple-barrier labeling** (`src/ml/triple_barrier.py:1-373`) from López de Prado (2018):

```
For each observation at bar t:
    Upper barrier  = entry_price + tp_atr_mult × ATR(t)
    Lower barrier  = entry_price - sl_atr_mult × ATR(t)
    Time barrier   = t + max_holding_bars

Label = +1  if upper barrier hit first
        -1  if lower barrier hit first
         0  if time limit expires first
```

This aligns training labels with actual trade mechanics — real trades exit when TP/SL levels are hit, not after an arbitrary N-bar horizon.

### 3.3 Model Architecture

`PatternClassifier` (`src/ml/pattern_classifier.py:1-556`) supports 4 backends:

| Backend | Implementation | Use Case |
|---------|---------------|----------|
| **CatBoost** (default) | `CatBoostClassifier` with GPU support | Primary — gradient boosting with ordered target encoding |
| **Chronos** | Amazon's foundation time-series model | Zero-shot forecasting, no training |
| **FinCast** | Financial time-series forecaster | Domain-specific zero-shot |
| **xLSTM** | Extended LSTM (128 hidden, 4 layers) | Sequence modeling |

**CatBoost hyperparameters** (shallow, regularized to resist overfitting):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `iterations` | 500 | Enough to converge, not enough to memorize |
| `depth` | 6 | Shallow — each split only uses 6 interactions |
| `learning_rate` | 0.03 | Slow learning for noisy financial data |
| `l2_leaf_reg` | 3.0 | Aggressive L2 regularization |
| `random_strength` | 1.0 | Random score diffusion for tree diversity |
| `bagging_temperature` | 1.0 | Bayesian bootstrap subsampling |
| `min_data_in_leaf` | 20 | Minimum samples per leaf — blocks single-sample memorization |
| `subsample` | 0.8 | Train each tree on 80% of rows |
| `colsample_bytree` | 0.8 | Train each tree on 80% of features |

### 3.4 Validation Pipeline

The system uses a **PurgedKFold-like time-series split**, not standard K-fold, to prevent temporal data leakage:

```
Data: [0 .... idx .... N-1]

Split:  X_train = X[0 : idx - purge_window]     ← purged boundary
        X_test  = X[idx : ]                      ← no overlap with train
        y_train = y[0 : idx - purge_window]
        y_test  = y[idx : ]
```

The `purge_window` (default 5 bars) excludes training samples near the boundary whose labels overlap with test-label forward-return windows. This prevents the most common form of overfitting in financial ML.

**Walk-forward validation** runs expanding-window backtests across 125+ instruments with per-ticker Information Coefficient tracking.

### 3.5 Isotonic Calibration

Raw CatBoost probabilities suffer from miscalibration — a model predicting 0.7 probability might actually be correct only 55% of the time. The pipeline wraps trained models in `sklearn.calibration.CalibratedClassifierCV` with `method="isotonic"` and `cv="prefit"`:

| Metric | Pre-Calibration | Post-Calibration |
|--------|----------------|------------------|
| ECE (Expected Calibration Error) | 0.197 | 0.070 |
| IS Sharpe | 0.59 | 0.85 |

The calibrator maps raw model probabilities to empirically accurate probabilities using a monotonically increasing piecewise-constant function.

### 3.6 Stability Selection

Rather than trusting single-run feature importance, the system runs **N=100 bootstrap iterations** of CatBoost training on subsampled data. Only features that appear in the top-K of >50% of bootstrap runs are retained. This eliminates noise features that CatBoost assigns high importance by chance in any single run.

### 3.7 Meta-Labeling

A secondary CatBoost classifier (`src/ml/meta_labeler.py`) filters primary model signals. It is trained only on **examples where the primary model predicted a trade** and learns:

> "Given that the primary model says BUY, what is the probability this specific BUY is actually profitable?"

This improved IS Sharpe from 0.69 to 0.94 (+36%).

### 3.8 OOS Failure & Honest Diagnostics

The ML model achieves IS Sharpe 0.85 but degrades to **Sharpe -1.24 OOS** (2025-2026). The diagnostic suite identified the root cause:

```
Feature:  vol_regime      → KS statistic = 0.81  (complete regime shift)
Feature:  ema_8_21_spread   → Correlation flipped sign
Feature:  volume_ratio_5     → Correlation flipped sign
Feature:  rsi_14             → Correlation flipped sign

3 of 38 features completely changed their relationship with the target
in the 2025-2026 regime — a regime the training data (2016-2024) never saw.
```

The diagnostic tools (`src/ml/model_validation.py`) include PSI (Population Stability Index), KS drift tests, KL divergence monitoring, and correlation-flip detection with automatic retrain recommendation.

---

## 4. Backtesting Engine

### 4.1 Custom Event-Driven Engine

`BacktestEngine` (`src/backtest/engine.py:1-1084`) implements a comprehensive event-driven backtest loop:

```python
for bar in range(min_bars, total_bars):
    equity = calculate_equity_from_open_positions()       # mark-to-market
    check_circuit_breaker()                                # halt if DD > 20%
    check_exits_for_all_positions()                        # always allowed for risk
    update_regime()                                        # optional regime detection
    check_dynamic_rebalancing()                            # skip if rebalance cadence not met
    signals = signal_generator.generate(df, bar)           # run pattern detectors
    apply_regime_gating(signals)                           # filter incompatible patterns
    apply_position_probability(signals)                    # estimate success probability
    position_manager.open_position(signals)                # execute with sizing
    record_equity(equity, regime, rebalance_flag)
```

**Post-loop analysis**:
- Close remaining positions at final bar
- Apply turnover penalty to final equity
- Compile trades, equity curve, metrics
- Apply friction-adjusted metrics (spread + slippage + commission)
- Run time-reversal overfitting check

### 4.2 Friction Modeling

The engine models realistic trading frictions:

| Friction | Default | Mechanism |
|----------|---------|-----------|
| **Commission** | 0.10% | Per-trade percentage of notional |
| **Slippage** | 0.05% | Per-trade percentage of notional |
| **Spread** | (from data) | Bid-ask spread from actual market data |
| **Market impact** | Almgren-Chriss model | `src/risk/market_impact.py` — temporary + permanent components |
| **Turnover penalty** | R1 module | `src/risk/turnover_penalty.py` — penalizes excessive trading |

### 4.3 Time-Reversal Overfitting Check

Based on Svozil (2026) "Against a Universal Trading Strategy" (`engine.py:764-897`):

1. Run backtest on forward data
2. Run backtest on time-reversed data (`df.iloc[::-1]`)
3. If strategy is profitable on **both** forward AND reversed data → **overfit flag raised**

The logic: a strategy that wins regardless of temporal ordering is exploiting noise, not signal. The time-reversal score `= 1 - 1/(1 + reversed_ratio)` quantifies overfitting severity (HIGH/MEDIUM/LOW).

---

## 5. Risk Management

### 5.1 Position Sizing (`src/risk/position_sizing.py:1-612`)

Seven sizing methods, each with different inputs and use cases:

| Method | Formula | When Used |
|--------|---------|-----------|
| **Fixed Fractional** | `shares = (equity × 2%) / (entry - stop)` | Default, simple |
| **Fixed Amount** | `shares = $10,000 / entry_price` | Fixed dollar positions |
| **Kelly Criterion** | `f* = W - (1-W)/R`, capped at 25% | Requires win_rate, avg_win/avg_loss |
| **Kelly Information** | Numeric optimization of `Σ p_i × log(1 + f × r_i)` | ML model probability available |
| **ATR-Based** | Stop from ATR multiplier, size from risk budget | Volatility-aware sizing |
| **Volatility-Adjusted** | `size *= target_vol / actual_vol`, bounded [0.1, 2.0] | Dynamic vol targeting |
| **Risk Parity** | `risk = equity × risk_pct / n_positions` | Multi-position portfolio |

All methods apply post-calculation constraints: max 20% equity per position, min $100, max 5% risk per trade.

### 5.2 Kelly Allocator (`src/risk/kelly_allocator.py:1-569`)

Based on Stiffelman "Investing Is Compression" (arXiv:2604.10758). Three allocation modes:

| Mode | Formula | Bounds |
|------|---------|--------|
| **Classic Kelly** | `f* = (p × b - q) / b` | 0 ≤ f* ≤ 0.25 (full Kelly), 0.125 (half Kelly) |
| **Information Kelly** | Numeric root-finding over 1000 candidates | 0 ≤ f* ≤ 0.25 |
| **Winner Fraction** | `f = win_rate × (1 - 0.3 × entropy_penalty)` | 0 ≤ f ≤ 0.25 |

Edge estimation uses expanding-window trade history (no look-ahead) with a minimum of 5 trades required.

### 5.3 Circuit Breakers (`src/risk/circuit_breakers.py:1-309`)

A 3-state finite state machine:

```
ACTIVE ──(DD > 20%)──► COOLDOWN ──(20 bars elapsed + DD recovered)──► ACTIVE
                        │
                        └──(DD still above limit)──► COOLDOWN (extended)
```

Multi-layer triggers:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Max Drawdown | 20% | Full halt |
| DD Warning | 10% | Warning only |
| VaR 95% gate | > 5% | Halt |
| VaR 99% gate | > 10% | Halt |
| CVaR 95% gate | > 8% | Halt |
| Anomaly score (HDBSCAN) | > 0.7 | Halt |

### 5.4 Daily / Weekly / Monthly Limits (`src/risk/daily_limits.py:1-733`)

| Limit | Threshold |
|-------|-----------|
| Max daily loss | 3% |
| Max daily drawdown | 5% |
| Max daily trades | 10 |
| Max consecutive losses | 5 |
| Max weekly loss | 6% |
| Max monthly loss | 10% |
| Portfolio heat (max) | 6% |
| Position reduction near limits | 50% |
| Cool-off | 24 hours |

### 5.5 Copula Tail-Risk (`src/risk/copula_risk.py:1-200+`)

Models joint tail dependence across positions using Gaussian and t-copula models. Instead of assuming independent positions (which underestimates crash risk), the copula captures the joint probability of multiple positions hitting their stop-losses simultaneously — the scenario that destroys portfolios in correlated selloffs.

### 5.6 Almgren-Chriss Market Impact (`src/risk/market_impact.py:1-160+`)

Models execution cost as two components:
- **Permanent impact**: Information leakage that permanently moves the price
- **Temporary impact**: Liquidity demand that temporarily moves the price but reverts

Enables the backtest engine to estimate realistic execution costs for larger position sizes.

---

## 6. Signal Quality & Research Infrastructure

### 6.1 4-Step Pattern Evaluation Gate (`src/signals/evaluation_gate.py:1-405`)

Formal entry requirement for any pattern entering the catalog, based on 华泰多因子系列1 §1.3:

| Step | Test | Method | Threshold |
|------|------|--------|-----------|
| 1 | **Factor Return t-stat** | OLS: `return_t+1 ~ signal_t + sector_dummies` | \|t\| > 2.0, p < 0.05 |
| 2 | **Factor Classification** | \|t\|>2 ratio across rolling windows | 50%+ windows pass, 60%+ positive |
| 3 | **Rank IC** | `Spearman(signal_t, forward_return_t+1)` | IC > 0.01, p < 0.05 |
| 4 | **Quantile Backtest** | Sector-neutral top vs bottom quantile | Spread > 0 |

**Gate PASS** requires success on steps 1, 3, and 4. Results feed into `PatternQualityRegistry` (§2.4).

### 6.2 5-Axis Signal Scoring (`src/signals/scoring.py:1-360`)

Decomposes pattern quality into orthogonal dimensions, replacing single-number confidence:

| Axis | Formula | Range | What It Measures |
|------|---------|-------|------------------|
| **IC** | `max(Spearman(signal, forward_ret), 0)` | [0, 1] | Predictive accuracy |
| **IR** | `clip(IC_mean / IC_std / 3, 0, 1)` | [0, 1] | Signal stability (high IC with low variance) |
| **Turnover** | `1 - fraction_direction_changes` | [0, 1] | Signal persistence (low turnover = high) |
| **Diversity** | `1 - max(|corr|, other_signals)` | [0, 1] | Uniqueness vs other active signals |
| **Overfit Risk** | `1 - |IS_IC - OOS_IC| / max(|IS_IC|, |OOS_IC|)` | [0, 1] | IS vs OOS consistency |
| **Composite** | Geometric mean of 5 axes | [0, 1] | Overall quality |

### 6.3 Collinearity Analysis (`src/signals/collinearity.py:1-338`)

Detects redundant signals via Variance Inflation Factor (VIF):
- VIF > 5 → flagged as redundant
- Within-category correlation → recommend IR-weight synthesis (combine correlated signals)
- Across-category correlation → recommend discard weaker (lower IC)

### 6.4 Factor Purification (`src/signals/factor_purification.py:1-246`)

Removes sector and size contamination from pattern signals:
```
signal_pure = signal - OLS(signal ~ sector_dummies + log_market_cap)
purity_ratio = var(ε) / var(y)
```

If `purity_ratio < 0.5`, more than half the signal's variance is explained by sector membership or size — it's not a pattern edge, it's a sector bet.

### 6.5 DSR / PSR / FDR (Multiple Testing Correction)

When testing 54+ patterns, random chance will produce false positives. The system applies three corrections:

- **DSR (Deflated Sharpe Ratio)**: Adjusts Sharpe for the expected maximum under multiple independent trials: `DSR = Φ((Sharpe - E[max]) / std[max])`
- **PSR (Probabilistic Sharpe Ratio)**: Probability that the true Sharpe exceeds a threshold, accounting for sample size and skewness/kurtosis: `PSR = Φ((Ŝr - S*₀) / σŝr × √(n-1))`
- **FDR (False Discovery Rate)**: Benjamini-Hochberg correction on p-values across 54+ simultaneous tests

### 6.6 VIF / PSI / KS / Gini — Model Validation (`src/ml/model_validation.py:1-300+`)

Production monitoring for trained models:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **PSI** (Population Stability Index) | > 0.25 → alert | Feature distribution shift between train and live |
| **KS statistic** | > 0.30 → alert | Maximum difference in CDFs between train and live distributions |
| **KL divergence** | > 0.50 → alert | Information-theoretic distribution distance |
| **Gini coefficient** | < 0.05 → alert | Model no longer discriminating profitable from unprofitable trades |
| **Correlation flips** | sign change → alert | Feature-target relationship inverted in new regime |

---

## 7. AI-Native Workflow & Observability

### 7.1 Kilo Agents

The project uses 6 specialized AI agents (defined in `.kilo/agent/`) for automated workflows:

| Agent | File | Purpose |
|-------|------|---------|
| **model-doctor** | `.kilo/agent/model-doctor.md` | Runs calibration audit, regime shift investigation, WFO comparison. Produces health report. |
| **backtest-runner** | `.kilo/agent/backtest-runner.md` | Executes backtests, updates `BESTS.md` leaderboard, interprets results. |
| **ml-trainer** | `.kilo/agent/ml-trainer.md` | Trains CatBoost models via 9-stage pipeline, knows overfitting thresholds. |
| **repo-syncer** | `.kilo/agent/repo-syncer.md` | Syncs curated files from private to public repo. |
| **housekeeper** | `.kilo/agent/housekeeper.md` | Audits file system, flags misplaced files. |
| **researcher** | `.kilo/agent/researcher.md` | Searches papers, repos, benchmarks. |

These agents encode project-specific knowledge — known baselines, overfitting thresholds, pitfall avoidance — so that any AI session can produce consistent results.

### 7.2 Slash Commands

| Command | Agent | Usage |
|---------|-------|-------|
| `/model-diagnose` | model-doctor | Full diagnostic suite |
| `/backtest` | backtest-runner | Run backtests with config flags |
| `/train-ml` | ml-trainer | Train CatBoost models |
| `/repo-sync` | repo-syncer | Sync to public repository |
| `/housekeeper` | housekeeper | File system audit |
| `/research` | researcher | Paper/repo search |
| `/knowledge-graph` | — | Paper-to-module cross-reference analysis |

### 7.3 MLflow Experiment Tracking

All model training runs are logged to MLflow (`mlflow.db`, `mlruns/`) with:
- **Parameters**: CatBoost/LightGBM hyperparameters, feature config, label parameters
- **Metrics**: Train/test AUC, accuracy, calibration error, Sharpe from backtest
- **Artifacts**: Feature importance plots, SHAP summary plots, confusion matrices
- **Model registry**: Versioned model storage with staging/production transitions

### 7.4 Knowledge Graph

`graphify` (`src/graphify-out/GRAPH_REPORT.md`) builds a knowledge graph of the entire codebase:
- **7,068 nodes** — functions, classes, modules
- **11,170 edges** — imports, calls, inheritance, references
- **498 communities** — automatically detected functional clusters (regime components, pattern detectors, backtest adapters, ensemble methods, etc.)
- **314 files** indexed
- **Natural language query**: `graphify query "how does rules_first_strategy work?"` traverses the graph

GitNexus (`gitnexus`) adds code intelligence:
- **28,316 symbols** indexed
- **43,736 relationships**
- **300 execution flows** (end-to-end process traces)
- **Impact analysis**: `gitnexus impact "PatternClassifier.train" --direction upstream` shows exact blast radius before any refactor

### 7.5 Factor DSL (`src/patterns/dsl/`)

Based on arXiv:2604.26747v1 ("From Hypotheses to Factors"), a constrained domain-specific language for expressing pattern/factor recipes:

```python
from src.patterns.dsl import rank, ma, log, linear_combo, evaluate

factor = rank(linear_combo([
    (-0.6, log("mcap")),
    (0.5, ma("hl_range", window=10)),
    (-0.2, ma(pct_chg("volume", lag=1), window=3))
]))
scores = evaluate(factor, ohlcv_df)
```

The DSL **enforces point-in-time constraints** — only operators available at bar `t` are permitted (lag, ma, rank, zscore, log, abs, clip, sqrt, diff, pct_chg, linear_combo). A validator (`src/patterns/dsl/validator.py`) rejects expressions that access future data. This eliminates the single most common source of backtest overfitting — forward-looking factor construction.

---

## 8. Infrastructure & Dependencies

### 8.1 Build & Quality

| Tool | Purpose | Config |
|------|---------|--------|
| **uv** (Astral) | Package management, environment isolation, workspace support | `pyproject.toml`, `uv.lock` |
| **ruff** | Linting + formatting | Line length 100, per-file E402 for scripts |
| **mypy** | Static type checking | `python_version=3.13`, `ignore_missing_imports=true` |
| **Bandit** | Security analysis | Skips B101/B608 for pre-existing assert patterns |
| **pre-commit** | Git hooks (ruff + mypy + Bandit) | `.pre-commit-config.yaml` |
| **pytest** | Testing (66 files) | `testpaths=["tests"]`, `python_files=["test_*.py"]` |
| **pytest-cov** | Coverage tracking | Via pytest |

### 8.2 Core Data Science Stack

| Package | Version | Role |
|---------|---------|------|
| `pandas` | ≥2.0.0 | OHLCV data structures, time series operations |
| `numpy` | ≥1.24.0 | Vectorized pattern detection, array math |
| `scipy` | ≥1.10.0 | Optimization, statistical tests, copula models |
| `statsmodels` | ≥0.14.6 | VAR models, Granger causality, OLS regression |
| `numba` | ≥0.59.0 | JIT-compiled pattern detection (50-100x speedup) |
| `arch` | ≥8.0.0 | GARCH/EGARCH/GJR-GARCH volatility forecasting |

### 8.3 Machine Learning Stack

| Package | Version | Role |
|---------|---------|------|
| `catboost` | ≥1.2.8 | Primary gradient boosting classifier |
| `lightgbm` | ≥4.6.0 | Alternative boosting backend |
| `scikit-learn` | ≥1.4.0 | Calibration, preprocessing, metrics, pipelines |
| `shap` | ≥0.51.0 | SHAP explainability (waterfall, beeswarm, bar, scatter, force, heatmap) |
| `optuna` | ≥4.8.0 | Bayesian TPE hyperparameter optimization with multi-objective Pareto front |
| `scikit-survival` | ≥0.26.0 | Survival analysis for time-to-exit prediction (C-index OOS 0.684) |
| `hmmlearn` | ≥0.3.3 | Hidden Markov Model regime detection |
| `ruptures` | ≥1.1.10 | Changepoint/structural break detection |
| `hdbscan` | ≥0.8.42 | Density-based anomaly detection for circuit breakers |

### 8.4 Backtesting & Data

| Package | Version | Role |
|---------|---------|------|
| `backtesting` | ≥0.6.5 | Strategy prototyping framework |
| `vectorbt` | ≥0.25.0 | Vectorized backtesting |
| `yfinance` | ≥1.2.0 | Yahoo Finance data provider |
| `ccxt` | ≥4.5.54 | Crypto exchange OHLCV (Binance, 8 symbols) |
| `pyportfolioopt` | ≥1.6.0 | Portfolio optimization (HRP, Efficient Frontier, CVaR, Black-Litterman) |
| `duckdb` | ≥1.5.2 | Analytical SQL queries on OHLCV data |

### 8.5 Deep Learning & GPU

| Package | Version | Role |
|---------|---------|------|
| `torch` | ≥2.0.0 | xLSTM, neural network backends |
| `cupy-cuda12x` | ≥12.0.0 | GPU-accelerated NumPy (Linux only) |
| `stable-baselines3` | ≥2.9.0a2 | PPO/SAC reinforcement learning for trade execution |
| `gymnasium` | ≥1.3.0 | RL environment for trade execution |
| `transformers` | ≥4.57.6 | FinBERT sentiment model |

### 8.6 MLOps & Automation

| Package | Version | Role |
|---------|---------|------|
| `mlflow` | ≥3.11.1 | Experiment tracking, model registry, versioning |
| `prefect` | ≥3.6.28 | Workflow orchestration pipelines |
| `ploomber` | ≥0.23.3 | Pipeline definition and execution |

### 8.7 Research & Paper Processing

| Package | Version | Role |
|---------|---------|------|
| `markitdown` | ≥0.1.5 | Microsoft's PDF-to-Markdown converter |
| `pymupdf4llm` | ≥1.27.2.3 | MuPDF-based LLM-ready extraction |
| `marker-pdf` | ≥1.6.1 | Table-focused PDF conversion |
| `surya-ocr` | ≥0.13.1 | OCR for scanned PDFs |
| `graphifyy` | ≥0.8.5 | Knowledge graph generation |
| `graphiti-core` | ≥0.29.0 | Graph database backend (KùzuDB) |

---

## System Scale

| Metric | Value |
|--------|-------|
| Python source files | 364 |
| Pattern detectors | 54 |
| Strategy files | 55 |
| CLI scripts | 119 |
| Test files | 66 |
| Model artifacts (.pkl + .json) | 265 |
| Total lines of code | ~187,000 |
| Research papers integrated | 97 |
| Research insights tracked | 77 |
| Kilo AI agents | 6 |
| Kilo skills | 40 |
| GitNexus symbols indexed | 28,316 |
| Knowledge graph nodes | 7,068 |
| Knowledge graph edges | 11,170 |
| Instruments backtested | 125+ |
| Pattern categories | 10 |
| Sizing methods | 7 |
| Regime detection methods | 8 |

---

*Document generated 2026-05-18. The system continues to evolve — this represents the state at time of writing.*
