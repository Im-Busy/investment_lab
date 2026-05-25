# Post-Backtest Next Steps: A Structured Roadmap
## Research Synthesis — May 2026

**Context:** Python-based Rules-First pattern trading system with 122 instruments, IS=2016-2024, OOS=2025-2026. This document synthesizes 40+ academic papers, industry deployment guides, and practitioner frameworks into a practical, sequenced roadmap.

---

## PHASE 1: Statistical Validation (Prevent Self-Deception)

Before any deployment decision, pass these statistical gates. These are the minimum bar.

### 1.1 Deflated Sharpe Ratio (DSR) & PBO

| Step | Description | Implementation |
|------|-------------|----------------|
| Compute DSR | Adjusts Sharpe for multiple testing (number of strategy variants tried) | `src/analysis/deflated_sharpe.py` |
| Compute PBO | Probability of Backtest Overfitting via Combinatorial Symmetric Cross-Validation (CSCV) | Split returns into N blocks, enumerate C(N, N/2) train/test splits |
| Threshold | DSR > 0.95 means <5% false positive probability | PBO < 0.10 preferred; PBO > 0.50 = reject |

**Sources:**
- Bailey, D.H. & Lopez de Prado, M. (2014). "The Probability of Backtest Overfitting." *Journal of Computational Finance*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- Lopez de Prado, M. "Pseudo-Mathematics and Financial Charlatanism." [PDF](https://obj.portfolioconstructionforum.edu.au/articles_perspectives/Pseudo-mathematics-and-financial-charlatanism.pdf)
- Reference implementation: [github.com/Aliipou/backtest-audit](https://github.com/Aliipou/backtest-audit)

### 1.2 Minimum Backtest Length (MinBTL)

```
MinBTL ≈ (E[SR_IS]^2 / Var(SR)) × ln(N)
```
For 122 instruments tested: if you tried 500+ configurations, you need ~5-7 years IS data to avoid selecting a strategy with E[SR_OOS] ≈ 0. Your IS=2016-2024 (9 years) passes this threshold.

**Source:** Lopez de Prado, M. "What to Look for in a Backtest." [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308682)

### 1.3 Harvey-Liu Protocol

The Harvey, Liu, Zhu (2016) backtesting protocol prescribes:
- Report number of strategy configurations tried (N)
- Use Bonferroni or Holm multiple-testing correction
- New t-stat threshold: |t| > 3.0 (not 2.0) for significance
- Require economic significance, not just statistical significance

**Source:** Harvey, C.R., Liu, Y., & Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *Review of Financial Studies*. [Duke PDF](https://people.duke.edu/~charvey/Research/Published_Papers/G138_A_backtesting_protocol.pdf)

---

## PHASE 2: Purged Walk-Forward Analysis (WFA)

### 2.1 Why Purged WFA

Standard OOS holdout is insufficient. Purged Walk-Forward Analysis:
- Splits history into sequential IS/OOS window pairs (e.g., 4-year IS, 1-year OOS)
- Re-optimizes parameters on each IS window
- Tests on strictly unseen OOS window
- **Purge gap:** Remove N bars between IS and OOS to prevent information leakage from overlapping features (e.g., 20-day MA crosses boundary)
- Chain all OOS segments into a single equity curve — THIS is your true performance estimate

### 2.2 Decision Gates (Pre-Committed)

| Gate | Criteria | Action if Fail |
|------|----------|----------------|
| Walk-Forward Efficiency (WFE) | WFE = OOS_return / IS_return > 0.5 | Strategy unstable → redesign |
| Majority-Pass | >50% of WFA windows have OOS Sharpe > 0 | Insufficient robustness |
| Catastrophic Veto | Any single window has OOS return < -30% | Hard fail — strategy has hidden tail risk |

**Sources:**
- AlgoXpert Alpha Research Framework (IS→WFA→OOS pipeline): [arXiv:2603.09219](https://arxiv.org/pdf/2603.09219)
- Arian, Norouzi Mobarekeh, & Seco. "Backtest Overfitting in the ML Era: Comparison of OOS Testing Methods." [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376) — CPCV outperforms Walk-Forward for PBO
- Trends and Breakouts. "Walk-Forward Analysis — The Backtesting Protocol That Catches Overfitting." [Link](https://trendsandbreakouts.com/walk-forward-analysis)

---

## PHASE 3: Regime Testing & Stress Scenarios

A strategy that looks good across an aggregate period may fail catastrophically in specific regimes.

### 3.1 Regime Audit

| Regime | Definition | What to Check |
|--------|-----------|---------------|
| LOW_VOL | VIX < 15 | Does strategy overtrade in quiet markets? |
| NORMAL | 15 ≤ VIX < 25 | Baseline performance |
| HIGH_VOL | 25 ≤ VIX < 35 | Do signals degrade? |
| CRISIS | VIX ≥ 35 | Max drawdown, position sizing |
| BEAR | SPY -20% from peak | Does strategy take counter-trend trades? |
| BULL | SPY +20% from trough | Is trend-following logic working? |

Run DSR + Monte Carlo **per regime**. Report per-regime Sharpe and consistency score.

**Implementation:** The `backtest-audit` library already has `run_regime_audit()`. Port to this codebase.

### 3.2 Stress Scenarios

- 2008-style liquidity freeze (bid-ask spreads triple)
- 2020 COVID flash crash (VIX spike to 85)
- 2022 rate-hiking cycle (bonds + equities correlated selloff)
- Slippage multiplier: run at 1x, 2x, 5x modeled costs

---

## PHASE 4: Monte Carlo Robustness

### 4.1 Methods

| Type | What It Does | Why |
|------|-------------|-----|
| **Return Reshuffling** | Randomly permute returns | Tests if sequence matters (autocorrelation test) |
| **Return Replacement** | Bootstrap returns with replacement | Sensitivity to drawdown ordering |
| **Synthetic Price Paths** | Generate paths from fitted distribution (Heston, Merton Jump-Diffusion) | Tests robustness under structural assumptions |
| **Parameter Perturbation** | Jitter parameters by ±10% | Sensitivity to parameter estimation error |

**Acceptance threshold:** 5th percentile of simulated returns > 0, ruin probability < 2%.

**Sources:**
- Bailey & Lopez de Prado. "Tactical Investment Algorithms." [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3459866)
- The Three Types of Backtests: Walk-Forward, Resampling, Monte Carlo. [Hillsdale](https://www.hillsdaleinv.com/uploads/The_Three_Types_of_Backtests.pdf)
- Futuretesting Engine: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4647103)

---

## PHASE 5: Multi-Asset & Cross-Instrument Robustness

### 5.1 Per-Instrument Audit

Not all 122 instruments should pass. Report:
- Pass rate by asset class (equities, ETFs, crypto, forex)
- Instruments where strategy fails → exclude from universe
- Correlation matrix of returns across instruments → diversification check

### 5.2 Portfolio Construction

Once individual strategies are validated:
1. **Equal-weight baseline** — simplest, hardest to beat
2. **Inverse-volatility weighting** — reward low-vol instruments
3. **Risk parity** — equal risk contribution
4. **Regime-aware allocation** — scale exposure by regime compatibility

**Source:** ArturSepp/OptimalPortfolios — ROSAA framework for multi-asset SAA/TAA: [GitHub](https://github.com/ArturSepp/OptimalPortfolios)

---

## PHASE 6: Regime-Adaptive Position Sizing

### 6.1 Key Findings from Research

| Approach | Finding | Source |
|----------|---------|--------|
| ATR-based sizing | 14-day ATR for position sizing, combined with regime filter | [arXiv:2601.19504](https://arxiv.org/pdf/2601.19504) |
| Volatility targeting | Scale positions inversely to rolling realized vol, target 15% annualized | [arXiv:2509.01393](https://arxiv.org/abs/2509.01393v2) |
| Dynamic capacity | Adjust exposure by volatility, correlation, sentiment | [Springer:10.1007/s41060-026-01066-0](https://link.springer.com/article/10.1007/s41060-026-01066-0) |
| Regime penalty | Soft penalty when position direction conflicts with regime direction | [arXiv:2509.01393](https://arxiv.org/abs/2509.01393v2) |
| Wasserstein HMM | Probabilistic regime inference → volatility-adjusted MVO allocation | [arXiv:2603.04441](https://arxiv.org/pdf/2603.04441) |
| Multi-module adaptive | Five integrated modules: dynamic sizing, multi-TF momentum, VIX timing, contrarian, regime transition alpha | [arXiv:2510.14986](https://arxiv.org/html/2510.14986v1) |

### 6.2 Recommended Implementation for This Project

Given the Rules-First system already has pattern detectors, add:
1. **VIX-based regime classifier** (LOW/NORMAL/HIGH/CRISIS) → adjust max position size
2. **ATR-based volatility scaler** → `position_size = base_size × (target_vol / ATR_vol)`
3. **Regime-penalty overlay** → reduce size 50% when pattern direction opposes regime

---

## PHASE 7: Paper Trading (Forward Performance Testing)

### 7.1 Industry Standard Sequence

| Stage | Duration | Capital | Criteria to Advance |
|-------|----------|---------|---------------------|
| Paper Trading | 30-90 days | $0 (simulated) | Stable logs, boring telemetry, no config errors |
| Micro-Live | 90 days, 60-100 trades | 10-25% of target | Expectancy within ±0.1R of backtest, drawdown in range |
| Scaled Live | Indefinite | Scale weekly | Fill rates >95%, slippage within assumptions |
| Full Production | Ongoing | 100% of target | 6+ months stable, all conditions met |

### 7.2 Paper Trading Monitoring Checklist

- [ ] Signal frequency matches OOS expectation (±20%)
- [ ] Order latency distribution stable
- [ ] Cancel/replace counts near zero
- [ ] API errors and partial fills tracked
- [ ] Position sizing logic verified under live fills
- [ ] Circuit breakers tested and confirmed functional

**Sources:**
- Kiploks Robustness Engine. "When is a trading strategy ready to deploy?" [Link](https://kiploks.com/research/when-is-a-trading-strategy-ready-to-deploy-a-framework-for-the-decision)
- RustyBT Production Deployment Guide. [Docs](https://rustybt.readthedocs.io/en/latest/api/live-trading/production-deployment/)
- DNS Research. "From Backtest to Live Trading." [Link](https://digitalninjasystems.wpcomstaging.com/2026/04/19/from-backtest-to-live-trading-a-systematic-process-for-successful-deployment/)
- Forex Mechanics. "Backtesting and Forward Testing." [Link](https://forexmechanics.com/traders-workshop/backtesting-forward-testing/)

---

## PHASE 8: Production Risk Controls

### 8.1 Three-Layer Risk Architecture

| Layer | Checks | Response |
|-------|--------|----------|
| **Pre-Trade** | Position limits, order size sanity, price reasonability, margin availability, fat-finger prevention | Block order |
| **In-Trade** | P&L velocity, correlation spikes, VaR breaches, daily/weekly loss limits | Reduce or halt |
| **Post-Trade** | Reconciliation, fill quality, slippage attribution, strategy drift vs benchmark | Flag for review |

### 8.2 Minimum Kill-Switch Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Max daily loss | Hard stop (auto) | Cease all trading |
| Max weekly loss | Hard stop (auto) | Cease all trading |
| Drawdown exceeds Monte Carlo 5th percentile | Alert → manual review | Evaluate strategy health |
| Win rate drops >15pp from historical | Alert | Reduce position size |
| Consecutive losses >1.5× historical max | Alert | Pause strategy |

**Sources:**
- HFT Advisory. "Trading Infrastructure Sequencing: Why Risk Controls Must Ship Before Execution." [Substack](https://hftadvisory.substack.com/p/trading-infrastructure-sequencing)
- Agent37. "2026 Guide: Building a Quantitative Trading System." [Link](https://www.agent37.com/blog/how-to-build-a-quantitative-trading-system)
- Jenacie AI. "How to Automate Systematic Trading Across Global Markets." [Link](https://www.jenacie.com/research/automate-systematic-trading-global-markets)
- MLOps for Trading (Stefan Jansen). "Multi-level circuit breakers." [ml4trading.io](https://ml4trading.io/third-edition/chapters/26_mlops_governance/)

---

## PHASE 9: Live Monitoring & Decay Detection

### 9.1 Concept Drift Monitoring

| Metric | How to Compute | Threshold |
|--------|---------------|-----------|
| **PSI (Population Stability Index)** | Compare feature distribution live vs. training | PSI > 0.25 = drift |
| **KS Statistic** | Two-sample KS test on signal values | p < 0.01 = investigate |
| **Rolling Sharpe** | 60-day rolling Sharpe ratio | Persistent decline = decay |
| **Win Rate Decay** | Last 20 trades vs. historical avg | >5pp drop = warning |
| **Payoff Ratio** | Avg win / avg loss | Compression = strategy fighting harder |
| **WFE Degradation** | OOS segments compared to most recent | New WFE < 0.3× baseline = stale |

### 9.2 Alpha Decay Benchmarks

- US equity signals decay at ~5.6% annually
- European signals decay at ~9.9% annually
- Momentum strategies typically lose edge within 10 months
- Crowding accelerates decay: monitor correlation with known factor returns

### 9.3 Operational Response Staging

| Stage | Condition | Response |
|-------|-----------|----------|
| **Fresh** (score 75-100) | Performance within expected band | No intervention |
| **Aging** (score 50-74) | Early signs of decay | Review parameters, check regime |
| **Stale** (score 25-49) | Meaningful degradation | Re-optimize on recent data |
| **Expired** (score 0-24) | Active underperformance | Pause strategy, full re-evaluation |

**Sources:**
- StockAlpha.ai. "Concept Drift Alarms for Quant Signals." [Link](https://stockalpha.ai/alpha-learning/concept-drift-alarms-for-quant-signals-detecting-when-alpha-decays)
- Anny Trade. "How I Detect When Your Strategy Stops Working." [Link](https://anny.trade/blog/how-i-detect-when-your-strategy-stops-working-before-you-do)
- Kx Systems. "Drift Detection's Blind Spot: Live TCA." [Link](https://kx.com/blog/drift-detections-blind-spot-how-live-tca-insights-help-firms-win-the-race-against-alpha-decay/)
- Vertox Quant. "Strategy Decay Detection: Building a Warning System." [Link](https://www.vertoxquant.com/p/strategy-decay-detection)
- Dan Analytics. "Understanding Alpha Decay." [Link](https://dananalytics.com/en/alpha-decay-en)

---

## PHASE 10: Deployment Pipeline (CI/CD for Strategies)

### 10.1 Strategy Lifecycle

```
Ideation → Offline Testing → Paper Trading → Micro-Live → Scaled Production → Retirement
   │            │                │              │              │               │
   │      Pass DSR/PBO     Pass WFA gates  30d stable    90d micro OK    Decay detected
   │      Pass MinBTL      Pass regime      ops verified   fill rates     Kill switch
   │      Pass MC          Pass MC robust                   confirmed      activated
```

### 10.2 Automation Checklist

- [ ] Strategy config is version-controlled (YAML/JSON)
- [ ] Parameter set is frozen before any forward testing
- [ ] Backtest regression tests run on every config change
- [ ] Deployment gate blocks promotion if paper-trading metrics violate controls
- [ ] Shadow mode: challenger models run on live data without executing trades
- [ ] Rollback plan: revert to previous model version within 5 minutes
- [ ] Audit trail: who, what, when, why for every deployment

**Sources:**
- Sharemarket.bot. "Operational Playbook for Deploying Bots." [Link](https://sharemarket.bot/operational-playbook-for-deploying-and-maintaining-bots-on-a)
- QuantHedgeFund Pre-Live Checklist. [GitHub](https://github.com/Ashutosh0x/QuantHedgeFund)
- Brenndoerfer, M. "Research Pipeline & Deployment: Strategy Lifecycle Guide." [Link](https://mbrenndoerfer.com/writing/research-pipeline-strategy-deployment-production-workflow)
- Quanto Architecture (research/execution boundary). [GitHub](https://github.com/skyliquid22/Quanto)

---

## Summary Priority Ranking (For This Project)

| Priority | Phase | Effort | Impact | Rationale |
|----------|-------|--------|--------|-----------|
| **P0** | 1.1 DSR/PBO calc | 1-2 days | Critical | Without this, you don't know if OOS results are real |
| **P0** | 2. Purged WFA | 2-3 days | Critical | Your single OOS holdout may be a lucky draw |
| **P1** | 3. Regime Audit | 1-2 days | High | OOS=2025-2026 is one regime; test across bull/bear/crisis |
| **P1** | 4. Monte Carlo | 1 day | High | Return reshuffling + parameter perturbation |
| **P1** | 6. Regime-Adaptive Sizing | 2-3 days | High | ATR + VIX-based sizing is the highest-ROI enhancement |
| **P2** | 5. Portfolio Construction | 2-4 days | Medium | Only needed when you have 3+ validated strategies |
| **P2** | 8. Production Risk Controls | 3-5 days | Medium | Required before ANY live capital |
| **P2** | 7. Paper Trading | 30-90 days (wall clock) | Medium | Operational, not analytical |
| **P3** | 9. Decay Monitoring | 2-3 days build + ongoing | Medium | Needed after live deployment |
| **P3** | 10. Deployment Pipeline | 5-10 days | Lower | Nice-to-have for repeatability |

---

## Key Academic References (Condensed)

1. **Bailey, D.H. & Lopez de Prado, M. (2014)** — PBO, CSCV, MinBTL. The canonical work on backtest overfitting.
2. **Lopez de Prado, M. (2018)** — *Advances in Financial Machine Learning*. Wiley. Covers purged K-fold, CPCV, triple-barrier labeling.
3. **Harvey, C.R., Liu, Y., & Zhu, H. (2016)** — T-stat threshold > 3.0, multiple testing correction, protocol for empirical finance.
4. **Arian, Norouzi, & Seco (2024)** — CPCV outperforms Walk-Forward and Purged K-Fold for PBO reduction.
5. **AlgoXpert Framework (2026)** — IS→WFA→OOS pipeline with majority-pass and catastrophic-veto gates.
6. **Pardo, R. (1992)** — The original walk-forward analysis methodology.
7. **Flint, E.J. (2015)** — "Systematic Testing of Systematic Trading Strategies." Best-practice comparison of WRC, MCP, StepM methods.

---

*Last updated: 2026-05-25. Synthesized from 40+ sources. Prioritize P0 items first; these are the gates between your OOS results and any deployment decision.*
