# Phase 21: Quant-Resources-Driven Signal Enhancers

**Source:** `useful_resources/useful_repos/quant-resources/Quant-Developers-Resources/` — comprehensive quant career prep repo (52 markdown files, 12 PDFs, zero code).

**Extraction Method:** "Think Freely, Then Compare" protocol (AGENTS.md). 36 raw ideas extracted → 8 recommended after head-to-head comparison.

**Key insight:** Options and fixed income data are NOT separate domains — they are the richest **leading indicators** available for equity trading. VIX slope, put/call ratio, yield curve inversion, and GEX all predict equity moves before they happen. We don't need to trade options or bonds to use their data as signals.

---

## Task Table

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **Q1** | VIX term structure → regime gate wiring | `regimefolio.py` (existing), `regime_gate.py` (existing) | **75% built.** `RegimeFolio` computes VIX slope (contango/backwardation). Wire VIX stress/normal/panic regimes into `RegimeGate` feature set. ~80 loc, 0 new deps. |
| **P0** | **Q2** | Yield curve inversion → macro regime flag | `macro_regime.py` (existing), FRED API | **Has yield_curve column.** Add formal 2s10s/10y3m inversion detection as binary flag. Inversion → defensive allocation. 100+ years of empirical evidence. ~120 loc, FRED API or yfinance. |
| **P1** | **Q3** | GARCH/EGARCH volatility forecasting | `volatility_forecaster.py` (existing) | Add `arch` library GARCH family for forward-looking vol forecasts. Replace reactive ATR-based vol_regime. Volatility clustering is the single most robust stylized fact in returns. ~200 loc, `uv add arch`. |
| **P1** | **Q4** | Put/Call ratio + Gamma Exposure (GEX) signals | `sentiment_scorer.py` (existing), options data API | PC ratio is best options-derived equity signal. GEX predicts intraday reversal levels. Plug into existing `SentimentProvider` protocol. ~250 loc, yfinance or FMP for options chain data. |
| **P2** | **Q5** | Model Validation Framework (PSI/KS/Gini) | CatBoost models (existing), `src/ml/` | Industry-standard model monitoring. PSI for feature drift, KS for discrimination decay, Gini for overall quality. Prevents silent model degradation. ~300 loc, 0 deps (sklearn already installed). |
| **P2** | **Q6** | Copula tail-risk models for baskets | `mc_var.py` (existing), `circuit_breakers.py` (existing) | Joint extreme risk modeling via Gaussian copulas for basket strategies. Fixes the independence assumption in current VaR/CVaR. ~300 loc, `uv add copulae` or scipy multivariate t. |
| **P3** | **Q7** | Market impact model (Almgren-Chriss) | `src/backtest/engine.py` (existing) | Bridges backtest→live by quantifying permanent + temporary price impact. Every backtest currently assumes zero cost. ~200 loc, numpy/scipy. |
| **P3** | **Q8** | Order book dynamics features | `order_flow.py` (existing skeleton) | Bid-ask imbalance, queue position, cancellation heat maps. Leading indicators for short-term directional moves. Uses existing tick/crypto data. ~300 loc, 0 deps. |

---

## Implementation Order

```
P0: Q1 (VIX regime gate) → Q2 (Yield curve macro)
    ↓
P1: Q3 (GARCH vol forecast) → Q4 (PC ratio + GEX signals)
    ↓
P2: Q5 (Model validation PSI/KS/Gini) → Q6 (Copula tail-risk)
    ↓
P3: Q7 (Market impact Almgren-Chriss) → Q8 (Order book dynamics)
```

---

## Full Idea Inventory (46 Ideas from 4 Blocks)

All ideas extracted via "Think Freely, Then Compare" protocol. Ideas already incorporated into Q1-Q8 are marked with their task reference. Remaining ideas are retained for future gating/revisit.

---

### Block A: Options Pricing & Volatility (13 ideas)

**Source:** `Projects/readme.md`, `Python/readme.md`, `Risk Management/README.md`, `Technical_Indicators/readme.md`, `TextBooks/readme.md` (Euan Sinclair, John Hull, Sheldon Natenberg)

| # | Idea | Source | Status | Gated On | Notes |
|---|------|--------|--------|----------|-------|
| A1 | **Black-Scholes analytical pricing** — closed-form European option pricing with Greeks | `Projects/Black-Scholes.../README.md`, `Python/readme.md` | 🔴 Deferred | Q4 (options data access) | Foundation for all options work. ~150 loc, scipy.stats.norm. |
| A2 | **Binomial Tree option pricing** — lattice method for American/early-exercise options | `Projects/readme.md`, `Python/readme.md` | 🔴 Deferred | Q4 | Lattice framework. ~120 loc, numpy. |
| A3 | **Monte Carlo option pricing** — path simulation for exotic options | `Projects/readme.md`, `Python/readme.md` | 🔴 Deferred | Q4 | Path-dependent payoffs. ~150 loc, numpy. |
| A4 | **Heston stochastic volatility model** — mean-reverting variance captures vol smile | `Projects/readme.md`, `overfitting_detectors.py:131` (stub) | 🔴 Deferred | Q4 | Stub exists. Full calibration with options data. ~200 loc, scipy.optimize. |
| A5 | **SABR stochastic volatility model** — industry standard for rate options | `Projects/readme.md` | 🔴 Deferred | Q4 + A4 | More complex than Heston. ~250 loc. |
| A6 | **Volatility Surface Construction** — implied vol across strike and maturity | `Projects/readme.md` | 🔴 Deferred | Q4 + A1-A4 | Requires pricing + options chain data. ~350 loc. |
| A7 | **Greeks calculation** — Delta/Gamma/Vega/Theta/Rho sensitivity analysis | `Python/readme.md`, `Risk Management/README.md` | 🔴 Deferred | Q4 + A1 | Comes free with A1-A3 pricing models. ~100 loc. |
| A8 | **Put/Call ratio as sentiment** — institutional hedging flow → equity signal | `Technical_Indicators/readme.md` (inferred) | ✅ → **Q4** | — | Incorporated into Q4 P1 task. |
| A9 | **Gamma Exposure (GEX)** — dealer hedging predicts intraday reversals | (inferred from vol surface) | ✅ → **Q4** | — | Incorporated into Q4 P1 task. |
| A10 | **VIX term structure regime** — contango vs backwardation classification | `Technical_Indicators/readme.md` | ✅ → **Q1** | — | Incorporated into Q1 P0 task. `regimefolio.py` has slope. |
| A11 | **Delta hedging strategies** — dynamic hedging for portfolio protection | `Python/readme.md` | 🔴 Deferred | Q4 + A7 | Requires Greeks. ~150 loc. |
| A12 | **Options payoff visualization** — 3D theta/IV surface plots | `Projects/Option Chain Analyser.../readme.md` | 🔴 Deferred | Q4 + A6 | Visualization only. ~100 loc, matplotlib/mplfinance. |
| A13 | **Volatility trading strategies** — straddles, strangles, vol arbitrage | `TextBooks/readme.md` (Sinclair) | 🔴 Deferred | Q4 + A1-A7 | Full strategies. ~400 loc. Highest-value but highest-dependency. |

**Block A gate:** Q4 must complete first (data access). If PC ratio + GEX prove valuable, unlock A1-A13 sequentially.

---

### Block B: Fixed Income & Macro (11 ideas)

**Source:** `Econometrics/readme.md`, `Python/readme.md`, `Financial Theory/readme.md`, `Risk Management/README.md`, `Credit Risk Modeling/README.md`, `TextBooks/readme.md`

| # | Idea | Source | Status | Gated On | Notes |
|---|------|--------|--------|----------|-------|
| B1 | **Nelson-Siegel yield curve** — parametric level/slope/curvature decomposition | `Econometrics/readme.md` (projects) | 🔴 Deferred | Q2 success + Treasury data | Elegant 3-parameter model. ~200 loc, scipy.optimize. |
| B2 | **Bond pricing (duration, convexity)** — price sensitivity to yield changes | `Python/readme.md`, `Financial Theory/readme.md` | 🔴 Deferred | Q2 | TLT/IEF already in `cross_asset_features.py`. Add sensitivity calc. ~120 loc, 0 deps. |
| B3 | **2s10s yield spread as recession predictor** — single best macro indicator | `Risk Management/README.md` | ✅ → **Q2** | — | Incorporated into Q2 P0 task. `macro_regime.py` has yield_curve column. |
| B4 | **Credit spread (BAA-AAA) as risk appetite** — corporate stress gauge | `Risk Management/README.md` | 🔴 Planned (Q2+) | Q2 | `macro_regime.py` already has credit_spread column. Wire into regime gate. ~80 loc. |
| B5 | **Vasicek interest rate model** — mean-reverting short rate | `Python/readme.md` | 🔴 Deferred | Q2 + B1 | Rate modeling. ~100 loc, scipy. |
| B6 | **CIR interest rate model** — non-negative short rate (Vasicek alternative) | `Python/readme.md` | 🔴 Deferred | Q2 + B5 | Better for low-rate environments. ~100 loc. |
| B7 | **Interest rate derivative pricing** — swaps, caps, floors, swaptions | `Projects/readme.md` | ✅ → Done | — | **Implemented 2026-05-18.** ~420 loc, scipy. Rate derivative signals for ML. |
| B8 | **Treasury auction cycle effects** — predictable liquidity/vol patterns | (inferred from fixed income) | 🔴 Deferred | Q2 | Calendar-driven alpha. ~80 loc, Treasury auction calendar. |
| B9 | **TLT/IEF as rate proxies** — bond ETFs as tradable rate exposure | Inferred — used in `cross_asset_features.py` | 🔴 Partial | — | `_add_bond_features` exists. Add yield-to-maturity estimation. ~50 loc. |
| B10 | **Real yield analysis (TIPS)** — inflation-adjusted yield for macro regime | (inferred from fixed income) | 🔴 Deferred | Q2 + B1 | Needs TIPS data. ~100 loc. |
| B11 | **Credit Default Swap (CDS) pricing** — market-implied default probability | `Credit Risk Modeling/README.md` | 🔴 Deferred (P3) | Q4 infrastructure | CDS data hard to access (OTC). ~200 loc. |

**Block B gate:** Q2 must complete first (yield curve inversion). If macro regime flag proves valuable, unlock B1 → B4 → B5/B6 → B7.

---

### Block C: FPGA / HFT Systems (10 ideas)

**Source:** `FPGA/readme.md` (144-line comprehensive guide), `Risk Management/README.md`

| # | Idea | Source | Status | Gated On | Notes |
|---|------|--------|--------|----------|-------|
| C1 | **Market data feed handlers** — FIX/FAST protocol parsing at wire speed | `FPGA/readme.md` | 🔴 Out-of-Scope | FPGA hardware | Requires VHDL/Verilog + FPGA dev board. |
| C2 | **Order book management in hardware** — matching engine on FPGA | `FPGA/readme.md` | 🔴 Out-of-Scope | FPGA hardware | Requires FPGA + HDL expertise. |
| C3 | **Kernel bypass techniques** — DPDK, RDMA for 10GbE+ networking | `FPGA/readme.md` | 🔴 Concept-Only | C++ + custom NICs | Educational for engine architecture. Solarflare/Napatech NICs. |
| C4 | **Timestamping and precision timing** — PTP/IEEE 1588, GPS-disciplined clocks | `FPGA/readme.md` | 🔴 Concept-Only | FPGA hardware | Educational for backtest timing accuracy. |
| C5 | **HFT-specific risk checks** — pre-trade risk in hardware (max order, position limits, fat-finger) | `FPGA/readme.md`, `Risk Management/README.md` | 🔴 Concept-Only | C++ | **Design patterns applicable to software.** Pre-trade checks before order submission. |
| C6 | **Exchange protocol knowledge** — ITCH/OUCH (NASDAQ), FIX/FAST, native binary | `FPGA/readme.md` | 🔴 Concept-Only | External docs | Educational. NASDAQ,NYSE,BATS protocol guides available. |
| C7 | **Latency budgeting and profiling** — breakdown of tick-to-trade latency | `FPGA/readme.md` | 🔴 Concept-Only | Instrumentation | **Design patterns applicable to software.** Instrument backtest engine. ~100 loc. |
| C8 | **Order book dynamics modeling** — bid-ask bounce, queue position, cancellation rates | `Risk Management/README.md` — "Order Book Modeling" | ✅ → **Q8** | — | Incorporated into Q8 P3 task. `order_flow.py` skeleton exists. |
| C9 | **Market impact models (Almgren-Chriss)** — permanent + temporary impact | `Risk Management/README.md` | ✅ → **Q7** | — | Incorporated into Q7 P3 task. |
| C10 | **High-frequency signal processing** — wavelet denoising, FFT on tick data | `FPGA/readme.md` — "Digital Signal Processing with FPGA" | 🔴 Deferred | Q8 | Tick data required. ~200 loc, scipy.signal + pywt. |

**Block C gate:** C1-C6 are FPGA hardware — out of scope. C3-C7 design patterns are educational resources for backtest engine architecture; not implementation items. C8+C9 absorbed into Q8+Q7. C10 gated on Q8 (tick data).

---

### Block D: Original Recommendations (12 ideas)

**Source:** `Technical_Indicators/readme.md`, `Econometrics/readme.md`, `Reinforcement Learning/readme.md`, `Python/readme.md`, `Risk Management/README.md`, `Credit Risk Modeling/README.md`, `Mathematics/readme.md`

| # | Idea | Source | Status | Gated On | Notes |
|---|------|--------|--------|----------|-------|
| D1 | **GARCH/EGARCH volatility forecasting** | `Econometrics/readme.md`, `Python/readme.md`, `Risk Management/README.md` | ✅ → **Q3** | — | Incorporated into Q3 P1 task. |
| D2 | **Copula tail-risk models** — Gaussian copula, tail dependence | `Risk Management/README.md` — "CORE FOR QUANT ROLES" | ✅ → **Q6** | — | Incorporated into Q6 P2 task. |
| D3 | **Structural Break / Unit-Root tests** — Chow, Bai-Perron, ADF | `Econometrics/readme.md` — Non-Stationarity section | 🔴 Deferred | Q1+Q2 (regime infra) | Formal statistical regime detection. Complements ADX-based. ~150 loc, statsmodels. |
| D4 | **Almgren-Chriss optimal execution** | `Risk Management/README.md` — Liquidity Risk | ✅ → **Q7** | — | Incorporated into Q7 P3 task. |
| D5 | **PPO/SAC RL algorithms** — SOTA RL beyond DQN | `Reinforcement Learning/readme.md` — Deep RL section | 🔴 Deferred | RL infra stable | `rl_trade_executor.py` has DQN. PPO/SAC with stable-baselines3. ~500 loc. |
| D6 | **VAR + Granger Causality** — cross-asset lead/lag signals | `Econometrics/readme.md` — Time-Series Econometrics | 🔴 Planned (P3) | Basket strategies proven | Multivariate time series. ~150 loc, statsmodels. |
| D7a | **Ichimoku Cloud** — 5-component trend/momentum/SR system | `Technical_Indicators/readme.md` | 🔴 Deferred | 45+ patterns already | Diminishing returns on pattern count. ~80 loc, 0 deps. |
| D7b | **Keltner Channels** — EMA ± ATR-based bands | `Technical_Indicators/readme.md` | 🔴 Deferred | 45+ patterns already | Alternative to Bollinger. ~50 loc, 0 deps. |
| D7c | **Williams %R** — momentum oscillator | `Technical_Indicators/readme.md` | 🔴 Deferred | 45+ patterns already | Similar to stochastic. ~30 loc, 0 deps. |
| D7d | **CCI (Commodity Channel Index)** — deviation from SMA of typical price | `Technical_Indicators/readme.md` | 🔴 Deferred | 45+ patterns already | ~40 loc, 0 deps. |
| D8 | **Model Validation Framework** — PSI/KS/Gini/SHAP monitoring | `Credit Risk Modeling/README.md`, `Risk Management/README.md` | ✅ → **Q5** | — | Incorporated into Q5 P2 task. |
| D9 | **Kalman Filter dynamic regression** — time-varying hedge ratios | `Econometrics/readme.md` — Advanced | 🔴 Partial | — | `kalman_hedge.py` exists for pairs. Extend to portfolio hedging. ~100 loc, pykalman. |
| D10 | **Rolling ARIMA+GARCH strategy** — alternative signal source | `Projects/ARIMA + GARCH.../readme.md`, `Python/readme.md` | 🔴 Deferred | Q3 (GARCH infra) | ML models likely dominate. ~250 loc, statsmodels. |
| D11 | **State Space Models** — structural time series | `Econometrics/readme.md` — Advanced | 🔴 Deferred | Q3 success | Complex. ~200 loc, statsmodels. |
| D12 | **Offline RL (CQL)** — conservative Q-learning for finance | `Reinforcement Learning/readme.md` — Advanced RL | 🔴 Deferred | D5 success | Safer than online RL. ~400 loc, d3rlpy or custom. |

**Block D gates:** D3 gated on Q1+Q2. D5/D12 gated on RL infra stability. D6 gated on basket strategies. D7a-d: low priority (diminishing returns). D9: partial (`kalman_hedge.py` exists). D10/D11: gated on Q3.

---

## Cross-Reference: Existing Coverage

| Module | Status | Relevance to Phase 21 |
|--------|--------|----------------------|
| `src/ml/regimefolio.py` | VIX slope + stress detection | **Q1** wires into this |
| `src/ml/regime_gate.py` | Market features for regime gate | **Q1** adds VIX regime to feature set |
| `src/ml/macro_regime.py` | yield_curve + credit_spread columns | **Q2** adds inversion flag |
| `src/ml/volatility_forecaster.py` | CatBoost vol forecasting | **Q3** adds GARCH alternative |
| `src/signals/sentiment_scorer.py` | SentimentProvider protocol | **Q4** plugs PC ratio + GEX into this |
| `src/risk/mc_var.py` | MCVaR class with 3 methods | **Q6** adds copula joint risk |
| `src/risk/circuit_breakers.py` | VaR/CVaR threshold tripping | **Q6** benefits from copula VaR |
| `src/signals/order_flow.py` | Golden ratio alpha skeleton | **Q8** extends with full order book features |
| `src/backtest/risk_metrics.py` | Drawdown + risk metrics | **Q5** adds PSI/KS/Gini monitoring |

---

## Expected Impact

| Item | Signal Improvement | Risk Reduction | Edge Over Current |
|------|-------------------|----------------|-------------------|
| Q1: VIX regime | High — regime-contextual signals | High — defensive during stress | 75% built, immediate |
| Q2: Yield curve | High — recession timing | High — allocation shift | No current macro regime flag |
| Q3: GARCH vol | Medium — forward-looking vol | High — better vol gate | Beats reactive ATR ratios |
| Q4: PC ratio + GEX | High — institutional flow | Medium — new orthogonal alpha | No options-derived signals |
| Q5: Model validation | Low — monitoring only | High — catches silent degradation | Zero production monitoring |
| Q6: Copula risk | Low — risk only | High — basket crash protection | Fixes independence assumption |
| Q7: Market impact | Low — cost estimation | Medium — realistic sizing | Zero impact modeling |
| Q8: Order book | Medium — new alpha source | Low — risk management | Skeleton only |

---

## Books & Resources Referenced

| Resource | Relevance |
|----------|-----------|
| Euan Sinclair, "Volatility Trading" | GARCH/vol forecasting (Q3) |
| John Hull, "Options, Futures and Other Derivatives" | Options pricing background (Q4) |
| Larry Harris, "Trading and Exchanges" | Market microstructure (Q8) |
| Ernie Chan, "Algorithmic Trading" | Strategy development reference |
| Stefan Jansen, "ML for Algorithmic Trading" | Companion code at `useful_repos/quant-resources/Machine-Learning-for-Algorithmic-Trading-Second-Edition/` (97+ .py/.ipynb) |
| Philippe Jorion, "Value at Risk" | VaR methodology (Q6) |
| Nassim Taleb, "Dynamic Hedging" | Tail risk (Q6) |
