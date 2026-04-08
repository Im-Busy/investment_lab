# Pattern & Strategy Audit Log

**Last updated:** 2026-04-02
**Purpose:** Track which patterns/strategies have been tested, on what data, and whether they are profitable enough to keep in production.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Profitable — kept in production |
| ❌ | Unprofitable — removed from active set |
| ⚠️ | Inconclusive — needs more data or different asset/timeframe |
| 🔧 | Being implemented / debugged |
| ⏳ | Queued for testing |

---

## 1. Existing Pattern Detectors (Geometric / Chart Patterns)

All tested via `PatternSelectionFramework` on **SPY_daily.csv** (full history, ~20+ years).
Thresholds: min_trades=5, min_sharpe=-1.0, min_pf=0.0, min_wr=0%, max_dd=100%.

| Pattern | Category | File | Trades | Win Rate | Sharpe | Profit Factor | Return | Max DD | Status | Notes |
|---------|----------|------|--------|----------|--------|---------------|--------|--------|--------|-------|
| Floor Pivot Breakout | basic | `basic/floor_pivot.py` | 134 | 29.1% | -0.49 | 0.84 | -2.24% | -3.66% | ❌ | Noise Generator — hurts portfolio Sharpe |
| Market Structure Low (MSL) | basic | `basic/msl.py` | 3 | 0% | 0.94 | 0.00 | +4.13% | -1.95% | ❌ | Insufficient trades (< 5) |
| n-Bar Decline | basic | `basic/n_bar_decline.py` | 1 | 0% | 1.68 | 0.00 | +10.38% | -2.27% | ❌ | Insufficient trades (< 5) |
| Matching Lows | basic | `basic/matching_lows.py` | 1 | 0% | -0.78 | 0.00 | -0.47% | -0.47% | ❌ | Insufficient trades (< 5) |
| NR7ID | basic | `basic/nr7id.py` | 12 | 0% | -1.32 | 0.00 | -1.60% | -1.67% | ❌ | Win rate CI too wide, low Sharpe |

### Summary: Basic Patterns
- **5 tested, 0 passed, 0 kept**
- Floor Pivot had enough trades (134) but was a **Noise Generator** (negative marginal contribution)
- All others failed on insufficient sample size
- **Action:** None of these should be in the active `SignalGenerator` pattern list for SPY daily. Code retained for potential re-testing on other assets.

---

## 2. Existing Complex / Classic / Harmonic Patterns

| Pattern | Category | File | Tested? | Status | Notes |
|---------|----------|------|---------|--------|-------|
| Double Top | classic | `classic/double_top.py` | ⏳ | Not yet backtested in selection framework | |
| Double Bottom | classic | `classic/double_bottom.py` | ⏳ | Not yet backtested | |
| Triple Top | classic | `classic/triple_top.py` | ⏳ | Not yet backtested | |
| Triple Bottom | classic | `classic/triple_bottom.py` | ⏳ | Not yet backtested | |
| Head & Shoulders | complex | `complex/head_shoulders.py` | ⏳ | Not yet backtested | |
| Cup & Handle | complex | `complex/cup_handle.py` | ⏳ | Not yet backtested | |
| Ascending Triangle | classic | `classic/ascending_triangle.py` | ⏳ | Not yet backtested | |
| Descending Triangle | classic | `classic/descending_triangle.py` | ⏳ | Not yet backtested | |
| Rectangle | classic | `classic/rectangle.py` | ⏳ | Not yet backtested | |
| Wedge | classic | `classic/wedge.py` | ⏳ | Not yet backtested | |
| Dead Cat Bounce | classic | `classic/dead_cat_bounce.py` | ⏳ | Not yet backtested | |
| Trader Vic 2B | classic | `classic/trader_vic_2b.py` | ⏳ | Not yet backtested | |
| Spike & Ledge | complex | `complex/spike_ledge.py` | ⏳ | Not yet backtested | |
| Three Hills | complex | `complex/three_hills.py` | ⏳ | Not yet backtested | |
| Parabolic Arc | complex | `complex/parabolic_arc.py` | ⏳ | Not yet backtested | |
| Flag | continuation | `continuation/flag.py` | ⏳ | Not yet backtested | |
| Pennant | continuation | `continuation/pennant.py` | ⏳ | Not yet backtested | |
| Gartley | harmonic | `harmonic/gartley.py` | ⏳ | Not yet backtested | |
| ABC | harmonic | `harmonic/abc.py` | ⏳ | Not yet backtested | |
| Bollinger Harmonic | harmonic | `harmonic/bollinger.py` | ⏳ | Not yet backtested | |
| Symmetric Triangle | harmonic | `harmonic/symmetric_triangle.py` | ⏳ | Not yet backtested | |
| Donchian Breakout | breakout | `breakout/donchian.py` | ⏳ | Not yet backtested | Rewritten from Eterna repo |
| Gap | breakout | `breakout/gap.py` | ⏳ | Not yet backtested | |
| Two Bar Reversal | basic | `basic/two_bar_reversal.py` | ⏳ | Not yet backtested | |

### Summary: Complex/Classic/Harmonic
- **23 patterns exist, none formally backtested in selection framework**
- These are geometric pattern detectors — likely to suffer same low-frequency signal problem on daily data
- **Decision:** Hold off on individual backtesting. Instead, migrate to Eterna indicator-based strategies first which generate higher-frequency signals.

---

## 3. EternaHybridExchange Strategies (Indicator-Based)

Source: `useful_resources/repomix-output-EternaHybridExchange-tradingview-strategies.git.xml`
Target asset for initial testing: **BTC_USD_1h.csv**

### 3.1 Trend Following (10 strategies)

| # | Strategy | Eterna Source | Win Rate* | R:R* | Implemented? | Tested? | Status | Notes |
|---|----------|---------------|-----------|------|-------------|---------|--------|-------|
| 1 | SMA Crossover 50/200 | `trend-following/sma-crossover/` | 45-55% | 1:2-3 | ✅ | ❌ | FAILED | 61 trades, 31.2% WR, Sharpe 0.07, PF 1.08, Return 4.6% (underperformed B&H 8.2%) |
| 2 | EMA Ribbon 9/21/55 | `trend-following/ema-ribbon/` | 55-65% | 1:1.5-2 | ✅ | ❌ | FAILED | 393 trades, 28.5% WR, Sharpe -1.12, PF 0.81, Return -49% |
| 3 | VWAP Bounce | `trend-following/vwap-bounce/` | 60-70% | 1:1.2-1.8 | ✅ | ❌ | FAILED | 474 trades, 19.2% WR, Sharpe -2.60, PF 0.62, -55% return on BTC 1H |
| 4 | Ichimoku Cloud | `trend-following/ichimoku-cloud/` | 50-60% | 1:2-3 | ⏳ | ⏳ | Queued | Japanese complete system |
| 5 | Parabolic SAR | `trend-following/parabolic-sar/` | 40-50% | 1:2-4 | ✅ | ❌ | FAILED | 599 trades, 33.7% WR, Sharpe -2.03, PF 0.77, -64% return. |
| 6 | ADX Trend Strength | `trend-following/adx-trend-strength/` | 55-65% | 1:2-3 | ✅ | ❌ | FAILED | 115 trades, 20.9% WR, Sharpe -1.67, PF 0.50, -37% return. |
| 7 | Donchian Channel | `trend-following/donchian-channel/` | 35-45% | 1:3-5 | ✅ | ⏳ | Already rewritten | Rewritten from this repo |
| 8 | Keltner Channel | `trend-following/keltner-channel/` | 60-70% | 1:1.5-2 | ✅ | ❌ | FAILED | 3 trades, 33% WR, Sharpe -0.26, PF 0.50. Bounce condition too restrictive on BTC 1H trend. |
| 9 | Linear Regression Channel | `trend-following/linear-regression-channel/` | 65-75% | 1:1.5-2 | ✅ | ❌ | FAILED | Tested on all 15 assets. All fail. Best: BTC 1H 25.5% WR, Sharpe -2.33. |
| 10 | Chandelier Exit | `trend-following/chandelier-exit/` | 40-50% | 1:3-5 | ✅ | ❌ | FAILED | 600 trades, 22.8% WR, Sharpe -1.95, PF 0.75, -64% return. |

### 3.2 Momentum (10 strategies)

| # | Strategy | Eterna Source | Win Rate* | R:R* | Implemented? | Tested? | Status | Notes |
|---|----------|---------------|-----------|------|-------------|---------|--------|-------|
| 11 | RSI Divergence | `momentum/rsi_divergence.pine` | 55-65% | 1:1.5-2 | ⏳ | ⏳ | Queued | Hidden/regular divergence |
| 12 | Stochastic RSI Crossover | `momentum/stoch_rsi_crossover.pine` | 50-60% | — | ✅ | ❌ | FAILED | 373 trades, 39.4% WR, Sharpe -2.28, PF 0.66, -70% return. |
| 13 | MACD Histogram | `momentum/momentum_combo.pine` | 54% | — | ⏳ | ⏳ | Queued | Part of momentum combo |
| 14 | CCI | `momentum/cci_strategy.pine` | 59% | — | ✅ | ❌ | FAILED | 495 trades, 54.9% WR, Sharpe -0.82, PF 0.88, -43% return. Best WR so far but still unprofitable. |
| 15 | MFI | `momentum/mfi_strategy.pine` | 56% | — | ⏳ | ⏳ | Queued | Money Flow Index (volume RSI) |
| 16 | Williams %R | `momentum/williams_r_reversal.pine` | 58% | — | ⏳ | ⏳ | Queued | Overbought/oversold reversals |
| 17 | TSI | `momentum/tsi_strategy.pine` | 53% | — | ✅ | ❌ | FAILED | Tested on all 15 assets. Best: SPY daily 60.7% WR but Sharpe -0.40. All fail. |
| 18 | Ultimate Oscillator | `momentum/ultimate_oscillator.pine` | 55% | — | ✅ | ❌ | FAILED | Tested on all 15 assets. All fail. Best: GC_F_1h 60% WR but only 5 trades, Sharpe -0.42. |
| 19 | Awesome Oscillator | `momentum/awesome_oscillator.pine` | 52% | — | ✅ | ❌ | FAILED | Tested on all 15 assets. All fail. Best: SPY daily 43.9% WR, Sharpe -0.41. |
| 20 | Chaikin Oscillator | `momentum/chaikin_oscillator.pine` | 54% | — | ✅ | ❌ | FAILED | Needs volume data. 0 trades on EURUSD (vol=0). On BTC 1H: 650 trades, 31.5% WR, Sharpe -2.43. |

*Win rates and R:R are from Eterna backtest docs — to be verified on our data.

### 3.3 SMC/ICT Strategy

| Strategy | File | Status | Notes |
|----------|------|--------|-------|
| SMC Reversal | `src/strategies/smc_reversal.py` | ⏳ | Not formally benchmarked against selection framework |

---

## 4. Testing Protocol

Each new strategy goes through:

1. **Implementation** — Port Pine Script logic to Python indicator + pattern detector
2. **Unit tests** — Verify indicator calculation matches TradingView values
3. **Solo backtest** — Run on `BTC_USD_1h.csv` (2020-2024) with `backtesting.py`
4. **Selection framework** — Run through `PatternSelectionFramework` with thresholds:
   - min_trades ≥ 10
   - min_sharpe ≥ 0.3
   - min_profit_factor ≥ 1.0
   - min_win_rate ≥ 35%
   - max_drawdown ≤ 25%
5. **Correlation check** — Ensure not redundant with existing kept strategies (r > 0.8)
6. **Ablation** — Measure marginal contribution to portfolio
7. **Decision** — ✅ Keep / ❌ Drop / ⚠️ Retest on different asset

---

## 5. Decision Log

| Date | Decision | Pattern/Strategy | Reason |
|------|----------|-----------------|--------|
| 2026-04-02 | ❌ Remove from active set | Floor Pivot, MSL, n-Bar Decline, Matching Lows, NR7ID | All failed pattern selection framework on SPY daily |
| 2026-04-02 | ⏳ Hold | All classic/complex/harmonic patterns | Not yet backtested; low expected frequency on daily data |
| 2026-04-02 | 🔧 Implement next | VWAP Bounce (Eterna #3) | Highest reported win rate (60-70%), institutional edge |
| 2026-04-06 | ❌ DROP | VWAP Bounce | 474 trades, 19.2% WR, Sharpe -2.60, PF 0.62, -55% return. Session-resetting VWAP confirmed. Strategy fails on BTC 1H. Eterna backtest numbers likely cherry-picked or use different parameters. |
| 2026-04-06 | ❌ DROP | SMA Crossover 50/200 | 61 trades, 31.2% WR, Sharpe 0.07, PF 1.08, Return 4.6% (underperformed B&H 8.2%). Too few signals on 1H, lagging nature of SMA makes it miss most of the move. |
| 2026-04-06 | ❌ DROP | EMA Ribbon 9/21/55 | 393 trades, 28.5% WR, Sharpe -1.12, PF 0.81, Return -49%. High frequency but terrible win rate. EMA alignment condition is too permissive, generates too many whipsaw signals in choppy markets. |

---

## 6. Key Insights

1. **Geometric patterns on SPY daily produce too few signals.** Most patterns generated < 5 trades over 20+ years.
2. **Floor Pivot was the only high-frequency pattern** (134 trades) but it was unprofitable (29% WR, PF 0.84).
3. **Indicator-based strategies are likely better suited** for this framework because they generate signals on every oscillation cycle, not just rare geometric formations.
4. **Asset matters:** SPY is a slow-moving ETF. BTC/ETH on 1H-4H will produce far more signals and better test conditions.
5. **The confluence system requires multiple patterns to work.** With 0 patterns passing, the entire signal generation pipeline is non-functional. Adding new strategies is not optional — it's required to unblock the system.
