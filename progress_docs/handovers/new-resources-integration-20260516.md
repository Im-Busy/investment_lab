# Handover — 2026-05-16 (3 New Resources Analyzed)

> **State:** Three new knowledge resources ingested and analyzed. Zero code changes.
> **GPU:** None (nvidia-smi not found). **Paid data:** None.
> **Next target:** Cross-reference new factor/strategy knowledge against existing project modules. Decide which insights to operationalize.

---

## What Was Done This Session

1. Converted 2 PDFs to markdown via `markitdown`:
   - `Beyond Fama-French_ Integrating Proven Mathematical Models of Default, Liquidity, and Momentum into Modern Factor-Based Systems.pdf` → `useful_resources/papers_md/Beyond_Fama-French_Integrating_Factors.md`
   - `华泰多因子系列1：多因子模型体系初探.pdf` → `useful_resources/papers_md/华泰多因子系列1_多因子模型体系初探.md`

2. Analyzed the **FMZ strategies repository** at `useful_resources/useful_repos/trading-system/strategies/` (5,807 .md files — all TradingView/quant strategies).

3. For the 华泰 PDF, an external AI provided textual descriptions of its 7 figures and 3 tables (included below).

---

## Resource #1: Beyond Fama-French (Multilingual AI-Generated Survey)

**File:** `useful_resources/papers_md/Beyond_Fama-French_Integrating_Factors.md` (1037 lines)

**What it is:** An AI-generated survey/report extending multi-factor asset pricing models beyond Fama-French with additional risk factors.

### Key Insights

**1. Baseline: Fama-French Model Family**
- 3-factor: Market + SMB (size) + HML (value)
- 5-factor: adds RMW (profitability) + CMA (investment)
- 8-factor extension: adds momentum, liquidity, default risk

**2. Proposed Extensions**
| Factor | Math Basis | Implementation Proxy | Validation |
|--------|-----------|---------------------|------------|
| **Default Risk** | Merton Distance-to-Default | Naïve DtD (Factor Engine: Distress) | 8-factor model tests |
| **Liquidity** | Acharya & Pedersen | CEI = 12-mo Δ(market equity) − 12-mo return (Factor Engine) | r=0.9883 vs Stata |
| **Profitability (RMW)** | Novy-Marx JFE | Gross Profit / Assets (Factor Engine) | Fama-French canonical |
| **Investment (CMA)** | Hou, Xue, Zhang JFE | Δ(PPE+Inventory) / Lagged Assets | Factor Engine |

**3. Advanced Architectures**
- **MFIN (Multi-Factor Inception Networks):** End-to-end neural networks for systematic trading, processes multiple assets/factors simultaneously
- **KAN Autoencoders:** Kolmogorov-Arnold Networks replacing MLPs for factor models
- **Tensor Factor Models:** Higher-order generalization of matrix factorization

**4. Automated Factor Discovery**
- **LLM + MCTS framework:** Large Language Model generates candidate alpha formulas, Monte Carlo Tree Search evaluates them via backtests
- Multi-dimensional scoring: IC (predictive accuracy), RankIR (stability), turnover, diversity, overfitting risk
- **FSA (Frequent Subtree Avoidance):** Prevents rediscovery of known factor structures
- Results on Chinese A-shares: IR 1.1792 (vs 0.9337 GP baseline), IC 0.0549 (vs 0.0459)

**5. Recommended Implementation Toolkit**
| Tool | Role | Key Features |
|------|------|-------------|
| **Factor Engine** | Core factor computation | Python, modular decorator API, Polars backend, 11 validated mispricing factors |
| **QRAFTI** | Research workflow automation | Agentic quant research team, MCP servers, standardized diagnostics (Novy-Marx/Velikov 2023 protocol) |
| **QFin** | Numerical methods | Stochastic process simulation, Euler-Maruyama, option pricing |

**6. Strategic Recommendation (Tiered)**
1. **Immediate:** Add Default Risk + Liquidity factors (highest impact, lowest risk)
2. **Foundational:** Add RMW + CMA (watch for multicollinearity)
3. **Advanced:** Replace linear regression with non-linear architectures (MFIN, KAN)
4. **Future:** Deploy LLM+MCTS for automated alpha mining

**Relevance to this project:**
- The 8-factor framework provides an alternative return decomposition to pattern-based signals
- Factor Engine library could be used for systematic factor computation
- LLM+MCTS alpha discovery is applicable to generating new pattern/feature ideas
- QRAFTI's standardized evaluation protocol is worth adopting for model validation

---

## Resource #2: 华泰多因子系列1 (Huatai Securities Multi-Factor Model)

**File:** `useful_resources/papers_md/华泰多因子系列1_多因子模型体系初探.md` (~3100 lines)

**What it is:** Official Huatai Securities research report (2016-09-21) — comprehensive Chinese-language guide to multi-factor model construction. Authors: 林晓明, 陈烨.

### Key Conceptual Framework

**Core Philosophy:** Active quantitative management = statistical arbitrage. Focus on factors (commonalities), NOT individual stocks (idiosyncrasies).

**Multi-Factor Model Equation:**
```
r_j̃ = Σ(k=1 to K) X_jk * f_k̃ + u_j̃
X_jk = stock j's factor exposure (loading) on factor k
f_k̃ = factor k's return
u_j̃ = stock j's residual return
```

### 12 Categories of Style Factors (Table 3 — Master Catalog)

| # | Category (EN/CN) | Specific Factors | Count |
|---|-----------------|------------------|-------|
| 1 | **Value** (估值) | EP, EPcut, BP, SP, NCFP, OCFP, FCFP, DP | 8 |
| 2 | **Growth** (成长) | sales_growth, profit_growth, operationcashflow_growth (q/ttm/3y each) | 9 |
| 3 | **Financial Quality** (财务质量) | roe, roa, grossprofitmargin, profitmargin, assetturnover, operationcashflowratio (q/ttm) | 12 |
| 4 | **Leverage** (杠杆) | marketvalue_leverage, financial_leverage, debtequityratio, cashration, currentratio | 5 |
| 5 | **Size** (规模) | ln_capital (log market cap) | 1 |
| 6 | **Momentum** (动量) | HAlpha, relative_strength_1m/2m/3m/6m/12m | 6 |
| 7 | **Volatility** (波动率) | high_low_1m-12m, std_1m-12m, ln_price, beta_consistence | 14 |
| 8 | **Turnover** (换手率) | turnover_1m/2m/3m/6m/12m | 5 |
| 9 | **Modified Momentum** (改进动量) | weighted_strength_1m-12m (turnover-weighted returns) | 5 |
| 10 | **Sentiment** (分析师情绪) | rating_average, rating_change, rating_targetprice | 3 |
| 11 | **Shareholder** (股东) | holder_avgpct, holder_avgpctchange_half/1y | 3 |
| 12 | **Technical** (技术) | macd, dif, dea | 3 |
| | | **TOTAL** | ~74 factors |

### 4-Phase Model Construction Pipeline

```
Phase 1: PREPARATION
  1.1 Data collection → 1.2 Data standardization → 1.3 Effective factor identification

Phase 2: RETURN MODEL
  2.1 Factor category analysis → 2.2 Collinearity analysis → 2.3 Heteroscedasticity analysis
  → 2.4 Multiple linear regression → 2.5 Estimate expected factor returns → 2.6 Calculate expected stock returns

Phase 3: RISK MODEL
  3.1 Factor return covariance matrix → 3.2 Residual risk estimation

Phase 4: OPTIMIZATION MODEL
  4.1 Return target + 4.2 Risk target + 4.3 Sector weight constraints
  + 4.4 Factor exposure constraints + 4.5 Individual stock weight limits
  → 4.6 Quadratic programming (portfolio weight optimization) → 4.7 Simulated performance backtest
```

### Key Methodological Details

**Data Standardization (两种方法):**
1. Raw value standardization (after median-based outlier removal: `|x_i - x_M| > n * DMAD`)
2. Rank-based standardization (non-parametric, broader applicability)

**Effective Factor Identification (4-step process):**
1. **Single-factor regression** with industry dummies (controlling for sector effects)
2. **t-test on factor return series:** |t|-mean, |t|>2 ratio, directional t-test → classifies into *return factors* vs *risk factors*
3. **IC (Information Coefficient) analysis:** rank correlation between factor exposure at T and return at T+1. Factor purification (regress out sector/size effects first)
4. **Quantile backtest (分层回测):** Sort stocks by factor into N portfolios, sector-neutral or not, evaluate by Sharpe, max drawdown, win rate

**Factor Category Analysis (大类因子分析):**
- Within-category correlation testing → either discard less significant or synthesize
- Synthesis methods: equal-weight, historical return weighted, historical IR weighted (best — accounts for both return AND volatility), PCA

**Collinearity & Heteroscedasticity:**
- Collinearity: Same logic as category analysis but across different category factors → must discard, can't merge
- Heteroscedasticity: Breusch-Pagan or White test → Weighted Least Squares (sqrt of market cap as weight per BARRA)

**Factor Return Forecasting:**
1. Historical mean (36 or 60 months)
2. EWMA
3. AR/MA/ARMA/ARIMA
4. **HP Filter (recommended):** Extract trend from cumulative factor return, divide by sample length. Eliminates noise, minimal parameters, Huatai-validated.

**Risk Model — Key Equation:**
```
V_ij = Σ(k1,k2) X_i,k1 * F_k1,k2 * X_j,k2 + Δ_ij
```
N stocks, K factors: reduces from estimating N(N-1)/2 correlations to estimating K(K-1)/2.

**Portfolio Optimization:**
- Quadratic programming: `min H^T * Q * H + H^T * c` subject to constraints
- Sector-neutral via 0-1 dummy variable matrix S: `Σ h_PAj * s_ji = 0`
- Factor exposure bounds: `|Σ h_PAj * X_jk| ≤ x_k`
- Two modes: max return given risk cap OR min risk given return floor

**Performance Attribution:**
- Return attribution: `r_P(t) = Σ x_Pj(t) * b_j(t) + u_P(t)` → factor contributions + stock-specific alpha
- Risk attribution: FMCAR = `∂σ_P / ∂x_PA = (F * x_PA) / σ_P` → marginal risk contribution per factor

### Huatai Service Architecture
1. Sequential single-factor testing across all 12 categories
2. Category factor synthesis (IR-weighted)
3. Stock selection model via regression + HP filter forecasting
4. Backtest + performance analysis
5. Future: Alpha factor discovery, non-linear factor usage, ML-based stock selection

### FIGURES & TABLES (from external AI with visual capabilities)

**Figure 1 (p4):** Hierarchical tree — 投资组合管理 → 被动管理 / 主动管理 → 定性管理 / 定量管理

**Figure 2 (p6):** 4-component diagram (BARRA) — 收益预测, 风险控制, 过程控制, 成本控制 are the 4 pillars of excellent investment performance

**Figure 3 (p7):** Bell curve of return normal distribution with ±1σ/±2σ/±3σ markings, showing risk = dispersion

**Figure 4 (p12):** Risk decomposition tree — 整体风险 → 市场风险 / 行业风险 / 风格风险 (12 style factor leaves)

**Figure 5 (p13):** Multi-factor model construction flowchart — 4 phases as swimlanes with process boxes and decision diamonds

**Figure 6 (p25):** Return decomposition tree — 组合收益率 → 业绩基准收益率 / 主动收益率 → 特定主动收益率 / 因子主动收益率 → 市场/行业/风格

**Figure 7 (p31):** Performance attribution waterfall/stacked bar — Market Contribution, Industry Contribution, Style Factor Contributions, Specific Alpha

**Table 1 (p5):** Quantitative vs Qualitative management comparison matrix (6 advantages + 4 disadvantages each)

**Table 2 (p12):** Information Ratio percentile distribution (90th=1.0, 75th=0.5, 50th=0.0, 25th=-0.5, 10th=-1.0)

**Table 3 (pp15-16):** Master factor catalog — 12 categories × ~74 specific factors with formulas (see above)

**Relevance to this project:**
- The 12-category factor taxonomy is directly applicable to feature engineering for ML models
- The 4-phase pipeline mirrors this project's ML training pipeline conceptually
- Factor purification (regressing out sector/size before IC computation) is a technique worth adopting for signal quality
- HP filter for trend extraction could improve factor return forecasts
- The return/risk factor classification (方向性/非方向性) is relevant for understanding which patterns predict direction vs which just explain variance

---

## Resource #3: FMZ Strategies Repository

**Location:** `useful_resources/useful_repos/trading-system/strategies/`
**Size:** 5,807 .md files
**Origin:** [FMZ (Financial Magic Zone)](https://www.fmz.com) — a Chinese crypto quant trading platform
**Languages:** PineScript v5/v6 (~75%), JavaScript (~12%), Python (~8%), MyLanguage (~3%), C++ (<1%)
**README:** Contains 586 curated entries (not comprehensive — most strategies are in the file listing but not README)

### Categories of Strategies Found

| Category | Examples | Approx % |
|----------|----------|----------|
| **EMA/MA Crossover** | 9/21 EMA, golden cross, triple EMA, DEMA, TEMA, HMA | ~25% |
| **RSI-based** | RSI oversold/overbought, RSI divergence, RSI + Bollinger | ~12% |
| **MACD-based** | MACD crossover, MACD histogram, zero-lag MACD | ~10% |
| **Bollinger Bands** | BB squeeze, BB breakout, BB mean reversion | ~8% |
| **Multi-Indicator Composite** | EMA+RSI+MACD, SuperTrend+ADX+RSI | ~15% |
| **Ichimoku Cloud** | Ichimoku breaks, twists, crossover | ~5% |
| **Breakout** | Donchian channels, price channels, session breakouts | ~8% |
| **Martingale/Grid** | DCA grid, martingale with multipliers | ~5% |
| **Momentum** | Relative strength, momentum oscillators | ~5% |
| **Machine Learning** | Exactly 1 file: Random Forest trend strategy (Python, basic) | <1% |
| **Infrastructure/Utility** | Exchange API wrappers, notifications, data collection, account management | ~7% |

### Representative Strategy Deep-Dives (from 8 sampled files)

**#1: Adaptive Bollinger Bands Trend Following (PineScript v5)**
- **Mechanism:** BB breakout REVERSAL (fades the break). Previous candle closes above upper band bullish, then current candle bearish = short.
- **Risk:** 4-layer exit (trailing at BB middle, fixed $ SL, fixed $ TP, time-based). Each toggleable.
- **Notable:** Unusual mean-reversion approach within a "trend following" name. Robust exit layering.

**#2: AI Volatility Adaptive Breakout (PineScript v6)**
- **Mechanism:** 3 independent logics: (1) Gap fill mean reversion, (2) VWAP momentum crossover, (3) Volatility compression breakout.
- **Risk:** Dynamic position sizing = `1 / riskFactor` where riskFactor = SMA(ATR,10)/ATR(14). ATR change-rate for vol-spike warning.
- **Notable:** "AI" is rule-based adaptive sizing, not ML. Multi-strategy composite approach.

**#3: Alpha Beast (PineScript v6)**
- **Mechanism:** Triple confirmation — Supertrend (trend) + RSI>60/<40 (momentum) + volume>1.5x avg (participation).
- **Risk:** ATR-based SL=1.2×ATR, TP=SL×2.5 (2.5:1 R:R). 20% equity allocation.
- **Notable:** Cleanest, most minimalist design. ~50 lines. Survives at ~30% win rate due to R:R ratio.

**#4: Multi-Factor Trend Following (PineScript v5)**
- **Mechanism:** 4-indicator confirmation: Parabolic SAR + 2-EMA + RSI(6) + ADX(14)≥30.
- **Risk:** 2% risk cap per trade. Position size = (equity × risk%) / ATR SL distance.
- **Notable:** Ultra-short periods (EMA=2, RSI=6) for scalping. Tested on DOGE/USDT.

**#5: Momentum-based ZigZag (PineScript v5)**
- **Mechanism:** Custom momentum-driven ZigZag without repainting — uses MACD/MA/QQE flips for pivot detection.
- **"Force Detection":** Only enters reversal when prior move lacked RSI momentum (RSI didn't hit OB/OS). If prior leg had "force," the reversal is likely just a pullback.
- **Notable:** Most conceptually sophisticated in the set. QQE implementation is advanced.

**#6: EMA-MACD High-Frequency (PineScript v5)**
- **Mechanism:** 9/21 EMA crossover + MACD(6,13,4) confirmation.
- **Risk:** 1% risk per trade. Position sizing from SL distance.
- **Notable:** Ultra-short MACD parameters (vs standard 12,26,9). Arabic comments in code.

**#7: HFT Order Flow Alpha Factor (JavaScript)**
- **Mechanism:** Real-time WebSocket order flow analysis. Golden ratio (0.382) weighted window: only last 38.2% of trades counted for bull/bear ratio. Tanh normalization to [-1,1].
- **Purpose:** Signal generator for market-making spread pricing, NOT a trading strategy.
- **Notable:** Only tick-level strategy. Runs on FMZ cloud. Entry-level HFT concept.

**#8: Random Forest Trend Strategy (Python)**
- **Mechanism:** scikit-learn RandomForestClassifier. Features: last 7 ternary volatility labels (±0.5% threshold). Predicts 8th bar. Rolling 300-point window, retrains at 200 samples.
- **Risk:** NONE. Binary all-in/all-out. No SL/TP. No position sizing.
- **Notable:** ONLY ML strategy in 5,807 files. Extremely rudimentary demo/PoC. 1-month backtest only.

### Infrastructure/Utility Scripts of Interest
- **OKX/Binance WebSocket high-frequency templates** — ready-made for real-time data
- **Telegram/WeChat/DingTalk notification systems** — multiple implementations
- **Funding rate aggregation** — multi-exchange
- **Exchange API wrappers** — OKX V5, Binance, BitMEX, Deribit
- **Grid trading engines** — multiple variants (spot, futures, dynamic grids)
- **Order book imbalance calculators** — checksum validation included
- **Triangular arbitrage** — spot and futures implementations
- **Uniswap V3 trade template** — DeFi integration
- **K-line period conversion utilities** — converting between timeframes

### Key Patterns & Anti-Patterns Across the Repository

**Patterns:**
1. 90%+ of strategies are single-instrument, single-timeframe
2. Most use fixed parameters with no optimization or walk-forward validation
3. EMA crossover (especially 9/21, 5/13, 50/200) is the most common base mechanism
4. Risk management when present is almost always ATR-based with fixed multipliers
5. Very few include market regime filters (trending vs ranging detection)
6. Almost no multi-asset or portfolio-level strategies

**Anti-Patterns:**
1. Massive overfitting risk — most have no OOS testing
2. Martingale/grid strategies dominate the "beginner" section — dangerous without understanding
3. "AI" branding without actual ML (see #2 above)
4. Zero risk management in many strategies (including the only ML one)
5. Copy-pasta code with inconsistent naming conventions
6. Strategies that work only on specific crypto pairs (overfit to high-vol assets)

### Most Relevant to This Project

Given this project has: pattern detection (34+ chart patterns), backtesting engine (event-driven + backtesting.py), CatBoost ML pipeline, and signal aggregation...

| FMZ Strategy | Why Relevant |
|-------------|-------------|
| **Multi-Factor Trend Following (#4)** | Closest analog to this project's multi-signal confluence approach. 4-indicator confirmation pattern could inspire weighted signal scoring. |
| **Momentum ZigZag (#5)** | "Force detection" concept (avoiding reversals after strong momentum) parallels pattern validation logic. QQE implementation could supplement existing oscillators. |
| **Alpha Beast (#3)** | Triple-confirmation (trend+momentum+volume) is the cleanest example of how to combine independent signals. Minimalist design to emulate. |
| **HFT Order Flow (#7)** | If this project ever adds real-time execution, the golden-ratio weighted imbalance ratio is a novel signal. |
| **EMA-MACD HF (#6)** | Short-period MACD parameters (6,13,4) are a useful baseline for comparison with this project's signal generation speed. |

### Search Suggestions for Next Session

These strategies contain brief introductions only — the next AI session should search for:
1. **QQE (Quantitative Qualitative Estimation)** indicator — used in Momentum ZigZag. Search for mathematical definition and parameterization.
2. **Supertrend with volume filter** — Alpha Beast approach. Search for academic or practitioner validation of volume-confirmed Supertrend.
3. **Golden ratio weighting in order flow** — HFT strategy. Search for "golden ratio order flow imbalance" or "Fibonacci-weighted trade imbalance" to find the origin of this technique.
4. **FMZ platform API details** — some strategies use FMZ-specific `exchange.Buy/Sell`, `_C()` polling function. If you need to adapt these strategies, search `site:fmz.com API documentation`.
5. **Factor Engine Python library** — search `arxiv Factor Engine` or `github Factor Engine` for the full documentation.
6. **QRAFTI agentic framework** — search `arxiv QRAFTI agentic quantitative finance` for the paper describing the agent-based quant research workflow.
7. **LLM+MCTS alpha factor mining** — search `arxiv LLM MCTS formulaic factor mining` for the paper on automated factor discovery.

---

## Cross-Reference: How These Resources Connect

```
华泰 Multi-Factor Pipeline ──→ Provides rigorous factor construction methodology
    │                              (standardization → screening → synthesis → regression)
    │
    ├── Factor categories (12 types, 74 factors) ──→ Can inform feature engineering
    │
Beyond Fama-French ──→ Provides modern extensions
    │                    (default, liquidity, ML architectures, automated discovery)
    │
    ├── Factor Engine ──→ Ready-made Python library for factor computation
    ├── LLM+MCTS ──→ Automated alpha mining methodology
    └── QRAFTI ──→ Standardized evaluation protocol

FMZ Strategies (5,807 files) ──→ Massive collection of trading ideas
    │                               (trend, momentum, breakout, grid, ML)
    │
    ├── Practical implementations ──→ Code examples for common patterns
    ├── Risk management patterns ──→ ATR-based with fixed multipliers
    └── Gaps ──→ Almost no ML, no walk-forward validation, no portfolio-level
```

**Key Gap in This Project vs. These Resources:**
- This project has ML (CatBoost) which the FMZ repo lacks
- This project has proper backtesting validation which FMZ strategies lack
- The factor resources provide systematic frameworks this project could adopt for feature importance and signal quality assessment
- The FMZ repo provides practical implementation patterns this project could mine for new signal ideas

---

## Recommended Next Session Actions

1. **Read this handover file first** (you're reading it now)
2. **Read the converted markdown files:**
   - `useful_resources/papers_md/Beyond_Fama-French_Integrating_Factors.md`
   - `useful_resources/papers_md/华泰多因子系列1_多因子模型体系初探.md`
3. **Explore the strategies repo** at `useful_resources/useful_repos/trading-system/strategies/` — pick any strategy file by name and read it
4. **Search online** using the search suggestions above for any concept you want deeper understanding of
5. **Decide direction** using `.kilo/project-loop.md`:
   - **DEEPEN:** Implement one of the factor/strategy ideas into the existing codebase
   - **BROADEN:** Mine the strategies repo for more signal ideas
   - **PIVOT:** Adopt the systematic factor framework from 华泰/Beyond Fama-French
   - **CONCLUDE:** Document findings and move to next priority (see `to-do-next-session.md`)

---

## Quick Reference

| Resource | Location |
|----------|----------|
| Beyond Fama-French MD | `useful_resources/papers_md/Beyond_Fama-French_Integrating_Factors.md` |
| 华泰多因子 MD | `useful_resources/papers_md/华泰多因子系列1_多因子模型体系初探.md` |
| FMZ Strategies Repo | `useful_resources/useful_repos/trading-system/strategies/` |
| FMZ Strategies README | `useful_resources/useful_repos/trading-system/strategies/README.md` |
| MEMORY.md | Project root |
| Master plan | `progress_docs/plans/full.md` |
| Current session log | `progress_docs/current.md` |
| Previous handover (code tasks) | `progress_docs/handovers/to-do-next-session.md` |
| Project loop decision framework | `.kilo/project-loop.md` |
| Pattern knowledge base | `useful_resources/CHART_PATTERN_KNOWLEDGE_BASE.md` |
| Commands | `docs/COMMAND_CHEATSHEET.md` |
