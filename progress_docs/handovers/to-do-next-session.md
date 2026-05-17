# Handover -- 2026-05-16 (Phase 16 CONCLUDED. Next: Phase 6d pattern work + Phase 12c cleanup.)

> **State:** Phase 16 formally concluded (ACCEPT LIMITS). All 3 directions (C/E/A+B) previously concluded.
> **GPU:** None (nvidia-smi not found). **Paid data:** None. T+U permanently gated.
> **Production system:** Rules-First (OOS Sharpe +0.76). Pairs (CVX-XOM Sharpe 0.40) as orthogonal alpha.
> **Next target:** Unblock pattern improvements (C9-C17) now that B9-B14 dependency gates are met.

---

## Remaining Open Work (all dependencies satisfied)

### Tier P1 — Pattern Signal Quality (C11-C13)

| Item | What | Files to touch | Est. lines |
|------|------|----------------|-----------|
| **C11** | Volume/OI validation layer | `src/patterns/complex/head_shoulders.py`, `src/patterns/classic/double_top.py`, `src/patterns/classic/ascending_triangle.py` | ~100 |
| **C12** | Multi-TP exit logic (partial take-profit) | `src/strategies/rules_first_strategy.py` or `src/strategies/ml_strategy.py` | ~80 |
| **C13** | Gap pattern hierarchy + size filter (4-type classification) | `src/patterns/breakout/gap.py` | ~120 |

**C11 details:** 4 rules from 4 independent sources (NCFE, Fidelity, Duddella, Warrior Trading):
1. High volume on breakout = confirm (+0.02 bonus already in C8)
2. Declining volume during formation = normal (no penalty)
3. **Volume dissipating on Right Shoulder (H&S)** = required validation
4. **OI declining at Head (H&S)** = required validation for signal generation

**C12 details:** Add partial TP to strategy:
- TP1 at 50% of ATR target (exit 50% position)
- TP2 at 100% target (exit remainder)
- Move SL to breakeven after TP1 hit
- Estimated +0.05–0.15 Sharpe from volatility drag reduction
- backtesting.py supports partial exits via `self.position.close(portion=0.5)`

**C13 details:** Enhance gap detector with 4-type classification from Duddella:
1. **Breakaway** → trade direction (almost never fills)
2. **Continuation** → trade direction
3. **Exhaustion** → fade (reversal)
4. **Common** → skip entirely (low reliability)
- Gap size > 2.5× ATR(10) → skip bar (noise filter)

### Tier P2 — Production Readiness + Missing Detectors (C9-C10, C14-C16)

| Item | What | Files | Notes |
|------|------|-------|-------|
| **C9** | Walk-forward paper trading | `scripts/paper_trade_v3.py` | Re-run with stability-selected + per-sector model. Expanding window only. |
| **C10** | Portfolio-level backtest | New `scripts/backtest_portfolio.py` | Equal-weight portfolio of profitable tickers, monthly rebalance. Threshold: Sharpe > 0.5, DD < 15%, 50+ trades. |
| **C14** | 5 missing harmonic detectors | `src/patterns/harmonic/` (butterfly.py, bat.py, crab.py, cypher.py, shark.py) | Follow `gartley.py` structure. Fibonacci tables in `CHART_PATTERN_KNOWLEDGE_BASE.md` Parts 2, 9. |
| **C15** | Pipe pattern detector | `src/patterns/complex/pipe.py` | Two-bar, zero-param. L=max(pipe1,pipe2), T1=±L, T2=±2L |
| **C16** | Dead Cat Bounce ≥15% threshold fix | `src/patterns/classic/dead_cat_bounce.py` | Enforce 15% event-day move, 50-62% bounce retracement |

### Tier P2/P3 — Phase 12c/d Cleanup

| Item | What | Notes |
|------|------|-------|
| **P2-3** | Dynamic Ensemble Collapse | Investigate why DEL produced Sharpe 0.26 vs expected 0.91 |
| **P2-4** | Walk-Forward Cadence | Simulate retraining at 6mo/12mo/24mo intervals |
| **P3-1** | Paper-Trading Harness | Daily signal generation for 2026-05-15 onward |
| **P3-2** | Kelly Position Sizing | Compute Kelly fraction, half-Kelly, min capital estimate |
| **P3-3** | Survival Analysis | scikit-survival Cox/RandomSurvivalForest |
| **P3-4** | Regression Labels | Predict 5-day forward return via CatBoostRegressor |
| **P3-5** | HMM Regime Detection | hmmlearn GaussianHMM vs rule-based ADX/ATR |

**Note:** Phase 10a (T10a-2 through T10a-5) is functionally DONE per 2026-05-12 session (`src/ml/tuning/optuna_tuner.py`, `src/optimizer/pypfopt_integration.py` exist). Only T10a-6 (documentation) may remain — check `docs/COMMAND_CHEATSHEET.md` for Optuna/PyPortfolioOpt sections.

---

## Recommended Execution Order

```
Session 1: C11 (Volume/OI) → C12 (Multi-TP) → C13 (Gap hierarchy)
  Rationale: Highest impact-to-effort. All 3 are isolated changes to pattern detectors.
  Estimate: ~300 lines, 3 files touched. Free Sharpe improvement.

Session 2: C9 (WF paper trade) → C10 (Portfolio backtest)
  Rationale: Production readiness. Requires C9+C10 to complete the pattern line.
  Estimate: ~400 lines, 2 scripts.

Session 3: C14-C17 (Harmonic detectors, Pipe, DCB, calibration)
  Rationale: Coverage expansion. More alpha sources.
  Estimate: ~500 lines, 5 files.

Optional: Phase 12c/d items (P2-3, P3-1, etc.) — lower priority, marginal gains.
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/strategies/rules_first_strategy.py` | PRIMARY — 34 pattern detectors, ATR trail. C12/C13 target. |
| `src/patterns/harmonic/gartley.py` | Template for C14 (Butterfly/Bat/Crab/Cypher/Shark) |
| `src/patterns/breakout/gap.py` | C13 target — gap hierarchy + size filter |
| `src/patterns/complex/head_shoulders.py` | C11 target — volume validation on Right Shoulder |
| `src/patterns/classic/dead_cat_bounce.py` | C16 target — ≥15% threshold fix |
| `src/signals/pattern_boost.py` | C8 — PatternBoostFilter with reliability weights |
| `src/analysis/ablation_engine.py` | Used by C17 for empirical calibration |
| `useful_resources/CHART_PATTERN_KNOWLEDGE_BASE.md` | Fibonacci tables, pattern rules (12 parts) |
| `docs/COMMAND_CHEATSHEET.md` | Check for T10a-6 (Optuna/PyPortfolioOpt docs) |

---

## Verification Checklist (before next handover)

- [ ] C11 volume/OI validation builds and imports
- [ ] C12 multi-TP logic tested on SPY 2016-2024 (Sharpe should improve)
- [ ] C13 gap hierarchy classifies gaps into 4 types
- [ ] All new files ruff clean
- [ ] `BESTS.md` updated if any backtests produce new bests
- [ ] `docs/COMMAND_CHEATSHEET.md` updated for new commands

---

## Quick Reference

| Resource | Location |
|----------|----------|
| MEMORY.md | project root |
| Master plan | `progress_docs/plans/full.md` |
| Phase 6d detail | `progress_docs/plans/full.md#phase-6d-pdf-insight-integration` |
| Pattern knowledge base | `useful_resources/CHART_PATTERN_KNOWLEDGE_BASE.md` |
| Commands | `docs/COMMAND_CHEATSHEET.md` |
| Leaderboard | `BESTS.md` |
| Agent catalog | `AGENTS.md` § Agent-Centric Workflows |
| Project loop | `.kilo/project-loop.md` |
