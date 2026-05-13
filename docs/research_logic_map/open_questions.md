# Open Questions

**Last Updated:** 2026-05-08
**Total Questions:** 9
**Unassigned:** 7
**Target Resolution:** 2026-05-15 to 2026-12-31

---

## Format

- **Question:** The unresolved question
- **Context:** Why this matters
- **Related Insights:** Links to insight registry
- **Research Needed:** What would answer this
- **Owner:** Who's responsible (or "Unassigned")
- **Target Date:** When to resolve by
- **Priority:** Based on impact × urgency

---

## Q1: What's the Right Mathematical Framework for Event-Type Signal Weighting?

- **Question:** How do we weight signals by event-type when P6 shows event-type signals outperform aggregated sentiment, but provides no formula?
- **Context:** P6 (Event-Based Trading: IE Tools) demonstrates event-type weighting superiority empirically, but no weighting formula given
- **Related Insights:** I6.1, I6.2, I6.3, R6
- **Research Needed:**
  - Review NLP event extraction literature (VIP platform, BERT-based classifiers)
  - Prototype Bayesian event-type weighting: `w(event) = P(return | event) / P(return)`
  - Backtest vs. equal-weight baseline on SPY 2020-2025
- **Owner:** Unassigned
- **Target Date:** 2026-05-15
- **Priority:** 🔴 High

---

## Q2: What's Our Portfolio's Effective Diversity Score (ρ)?

- **Question:** Jorion (P3) shows ρ=0.03 correlates to 6x capital reduction — what's our 34-pattern portfolio's effective ρ?
- **Context:** We have 34 patterns but unknown effective independent bets; ρ determines economic capital requirement
- **Related Insights:** I3.2, R7, H4
- **Research Needed:**
  - Calculate pattern signal correlation matrix (34×34)
  - Compute diversity score using BET formula: `D = N / (1 + (N-1)ρ)`
  - Sensitivity analysis on ρ assumption (0.00 to 0.10 range)
- **Owner:** Unassigned
- **Target Date:** 2026-05-01
- **Priority:** 🟠 High

---

## Q3: What's the Optimal Turnover Budget Threshold?

- **Question:** OOM-RL (P1) shows 6700% turnover destroyed alpha with 0.08% slippage — what's the optimal turnover budget for our system?
- **Context:** R1 implemented turnover penalty, but threshold (400% annual) chosen arbitrarily
- **Related Insights:** I1.1, I1.2, R1
- **Research Needed:**
  - Sweep turnover thresholds (100%, 200%, 400%, 800%)
  - Measure friction drag vs. return degradation
  - Find turnover "sweet spot" maximizing friction-adjusted Sharpe
- **Owner:** Unassigned
- **Target Date:** 2026-06-30
- **Priority:** 🟡 Medium

---

## Q4: Do Failure-Set Analyzers Prevent Blow-Ups or Add False Positives?

- **Question:** P4 proves every strategy has a failure set — can we detect failureSets without killing too many good trades?
- **Context:** R9 (failure-set analyzers) is backlogged; concern is over-filtering vs. risk prevention
- **Related Insights:** I4.2, I4.3, R9, H3
- **Research Needed:**
  - Build time-reversal, counter-trend, fat-tail tests for top 5 strategies
  - Backtest failure-set validation on 2008, 2020 stress periods
  - Measure prevented losses vs. missed opportunities
- **Owner:** Unassigned
- **Target Date:** 2026-07-31
- **Priority:** 🟠 High

---

## Q5: Is Divergence-in-Bits Actually Better than Sharpe for OOS Prediction?

- **Question:** P2 claims divergence-in-bits is "unit-independent strategy comparison metric" — does it actually predict OOS performance better than Sharpe?
- **Context:** R11 (divergence-in-bits metric) backlogged; theoretical appeal untested empirically
- **Related Insights:** I2.1, I2.3, R11, H9
- **Research Needed:**
  - Implement KL divergence estimator for strategy weight distributions
  - Compute divergence-in-bits for 20 strategies on IS data
  - Correlate with OOS Sharpe degradation; compare to IS Sharpe correlation
- **Owner:** Unassigned
- **Target Date:** 2026-08-31
- **Priority:** 🟡 Medium

---

## Q6: Should We Add Pairs Trading as 8th Pattern Category?

- **Question:** P5 survey shows DRL pairs trading outperforms threshold models — should we add as new category or stay chart-pattern-focused?
- **Context:** Current system has 34 chart patterns; P5 shows cointegration + ML hybrid dominates
- **Related Insights:** I5.1, I5.2, R13, H12
- **Research Needed:**
  - Implement cointegration screening (Engle-Granger, Johansen)
  - Build ML spread predictor (XGBoost/LSTM)
  - Backtest as overlay to existing 34 patterns
  - Measure diversification benefit (correlation to existing patterns)
- **Owner:** Unassigned
- **Target Date:** 2026-09-30
- **Priority:** 🟡 Medium

---

## Q7: What's the Marginal Benefit of 34 Patterns vs. Top 10?

- **Question:** Do we need all 34 patterns, or would top 10 achieve similar results with less complexity?
- **Context:** DEC-2026-04-15 chose "full catalog" over "top 10 only" —但未.experimental validation
- **Related Insights:** I4.1, I5.3, R4
- **Research Needed:**
  - Ablation study: backtest with 10, 20, 30, 34 patterns
  - Measure Sharpe, MDD, turnover, capacity
  - Calculate marginal contribution per pattern
- **Owner:** Unassigned
- **Target Date:** 2026-06-15
- **Priority:** 🟠 High

---

## Q8: Can Network-Derived Factors Replace/augment PCA?

- **Question:** P10 shows factors "emerge from asset interaction structure" — could this replace PCA for confluence scoring?
- **Context:** Current confluence scoring uses equal/PCR weighting; P10 proposes network-based factor extraction
- **Related Insights:** I10.1, I10.2, R14
- **Research Needed:**
  - Build coupling matrix from stock correlation structure
  - Extract emergent factors via coupled iterated maps
  - Compare factor loadings to PCA; test predictive power
- **Owner:** Unassigned
- **Target Date:** 2026-12-31
- **Priority:** 🟢 Low (research phase)

---

## Q9: What Optimal Instrument Universe Should We Trade?

- **Question:** Given our current universe is dominated by highly efficient instruments (SPY, QQQ, mega-cap tech), what is the optimal mix of efficient and inefficient instruments that maximizes risk-adjusted returns while maintaining tradability?
- **Context:** Research (Bartram 2019, DeMiguel 2024, Damodaran) shows alpha is inversely proportional to trading activity. Our current 10-instrument active universe is exclusively the most efficient, most arbitraged instruments. We need a data-driven answer on what to add and in what proportion.
- **Related Insights:** I14.1, I14.2, I16.1, I16.2, I17.1, I17.2
- **Related Hypotheses:** H13 (inefficient instruments yield higher ML alpha), H14 (cross-asset features improve SPY)
- **Research Needed:**
  - Quantify efficiency scores for all 85+ instruments referenced in the project
  - Rank by efficiency proxies (dollar volume, analyst coverage, institutional ownership, Amihud illiquidity)
  - Run H13 experiment (ML model on efficient vs. inefficient groups)
  - Determine minimum liquidity threshold for execution feasibility
  - Propose target universe with efficiency diversification (e.g., 30% efficient, 40% moderate, 30% inefficient)
- **Owner:** Unassigned
- **Target Date:** 2026-07-31
- **Priority:** 🔴 High
- **Discussion Document:** `docs/research_logic_map/market_efficiency_and_instrument_selection.md`

---

## Question Backlog by Priority

| Priority | Question | Impact | Target Date | Status |
|----------|----------|--------|-------------|--------|
| 🔴 High | Q1: Event-type weighting formula | Signal Quality | 2026-05-15 | Unassigned |
| 🟠 High | Q2: Portfolio diversity score (ρ) | Risk Management | 2026-05-01 | Unassigned |
| 🟠 High | Q4: Failure-set analyzer efficacy | Risk Management | 2026-07-31 | Unassigned |
| 🟠 High | Q7: 34 vs. 10 patterns ablation | System Design | 2026-06-15 | Unassigned |
| 🟡 Medium | Q3: Optimal turnover budget | Friction | 2026-06-30 | Unassigned |
| 🟡 Medium | Q5: Divergence-in-bits vs. Sharpe | Metrics | 2026-08-31 | Unassigned |
| 🟡 Medium | Q6: Pairs trading category | Strategy | 2026-09-30 | Unassigned |
| 🟢 Low | Q8: Network-derived factors | Research | 2026-12-31 | Unassigned |
| 🔴 High | Q9: Optimal instrument universe | Market.Efficiency | 2026-07-31 | Unassigned |

---

## Research Queue

**Ready Now (no dependencies, high priority):**
- Q2: Diversity score calculation — need correlation matrix + BET formula
- Q7: 34 vs. 10 patterns — can run ablation with existing `src/analysis/ablation_engine.py`

**Blocked (need prerequisites):**
- Q1: Blocked by event taxonomy (R6 dependency)
- Q4: Blocked by failure-set taxonomy (R9 dependency)
- Q5: Blocked by KL divergence implementation (R11 dependency)
- Q6: Blocked by cointegration infrastructure (R13 dependency)
- Q8: Research phase — low priority

---

## Related Documents

- **Insight Registry:** `docs/research_logic_map/insight_registry.md` (42 insights tracked)
- **Implementation Status:** `docs/research_logic_map/implementation_status.md` (29 recommendations tracked)
- **Hypothesis Backlog:** `docs/research_logic_map/hypothesis_backlog.md` (12 hypotheses, 8 ready for testing)
- **Decision Log:** `docs/research_logic_map/decision_log.md` (8 architectural decisions)

---

*Questions extracted from research_synthesis_report.md Tier 3 recommendations, contradiction analysis (Section 4), and implementation gaps.*
