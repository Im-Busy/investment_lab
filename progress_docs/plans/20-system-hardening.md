# Phase 20: System Hardening & Signal Quality

> **Planned:** 2026-05-17
> **Source:** Empirical session testing of all system modules (portfolio, multi-factor, pairs, mean reversion, signal quality, short-side, crypto).
> **Motivation:** Session revealed clear low-hanging fruit — multi-TP exit alone delivers +124% Sharpe boost. Pattern evaluation gate confirms individual patterns lack standalone edge and need quality-based gating. Sector-specific factor ICs diverge (tech ≠ healthcare ≠ financials).

## Evidence Base (2026-05-17 Session)

| Experiment | Key Finding |
|------------|-------------|
| Multi-TP (SPY OOS 2025) | Sharpe 0.54→1.21 (+124%), Win 67%→83%, MaxDD -13.3%→-4.7% |
| IR-Weights (SPY OOS 2025) | 12→2 trades, Sharpe 0.54→0.79. Ultra-selective but too few trades. |
| H&S Evaluation Gate (SPY) | FAIL on 3/4 steps (t-stat -0.59, IC -0.011, return/risk 0.00). No standalone edge. |
| Tech Multi-Factor (2020-2026) | +592% vs SPY +151%, Sharpe 1.31. ps_ratio IC 0.74, pe_ratio 0.68. |
| Healthcare Multi-Factor | +164% vs SPY +151%, Sharpe 1.00. profit_margin IC 0.90**. |
| Financials Multi-Factor | +120% vs SPY +151%, Sharpe 0.59. pb_ratio IC 0.95***. |
| GA Portfolio (8-asset) | Sharpe 1.38, GLD+XLK anchor. Only 2/8 RMT eigenvalues survive. |
| Pairs CVX-XOM (2016-2025) | +98.8%, PF 2.83, 40 trades, 57.5% win. |
| BTC IS rules-first | -9.3%, Sharpe -0.06. Diverges from batch Sharpe 0.77 — config needs audit. |

---

## Task Table

| Priority | # | Task | Source | Depends On | Notes |
|----------|---|------|--------|------------|-------|
| **P0** | **H1** | Multi-TP as system default | Multi-TP test (+124% Sharpe) | — | Set `use_multi_tp=True` in RulesFirstStrategy, CombinedStrategy, paper_trade scripts. Add to BESTS.md baseline config. 1 line change per strategy. |
| **P0** | **H2** | Pattern Quality Registry + Gate | H&S evaluation FAIL | H1 | Run 4-step gate on ALL 45 patterns. Build `src/signals/pattern_quality_registry.py` (json mapping pattern→{gate_result, weight_multiplier}). FAIL patterns get 0.5x weight or require 2+ confluence. ~150 loc + sweep script. |
| **P1** | **H3** | Sector-Specific Multi-Factor Scoring | Sector factor IC divergence | — | Build per-sector factor models (3 sectors tested). Tech uses ps_ratio/pe_ratio/ev_ebitda. HC uses profit_margin/revenue_growth. Fin uses pb_ratio/ps_ratio/pe_ratio. Feed sector-score into RulesFirstStrategy as `--use-multi-factor` modifier. ~200 loc in `src/signals/fundamental_scorer.py`. |
| **P1** | **H4** | Adaptive IR-Weighting (Scalar Mode) | IR test (Sharpe 0.79, too few trades) | — | Change IR-weighting from binary gate to scalar multiplier. Low-IR patterns get 0.3x–0.7x instead of being zeroed. Preserves trade count while filtering noise. ~30 loc in `src/signals/ir_weighting.py`. Add `--ir-weighting-mode scalar|gate` flag. |
| **P2** | **H5** | GA Portfolio Post-Backtest Step | GA optimizer (Sharpe 1.38) | H3 | Wire `optimize_portfolio.py --from-signals` as post-processing step after `backtest_rules_batch.py`. Auto-generate optimal allocation from batch results. Append to `reports/batch/INSIGHTS.md`. ~50 loc. |
| **P2** | **H6** | BTC Configuration Audit + Crypto Preset | BTC IS divergence (-0.06 vs batch 0.77) | — | Run systematic sweep on BTC: `--sweep-entry 0.45,0.55,0.65,0.75 --sweep-reliability 0.4,0.5,0.6,0.7`. Document optimal crypto preset. ~30 loc wrapper script. |
| **P3** | **H7** | Pattern Evaluation Sweep (All 45 Detectors) | Gate confirmed H&S FAIL | H2 | Run `evaluate_pattern.py` on every pattern detector. Output `reports/pattern_gate/all_patterns.json`. This feeds H2 (Pattern Quality Registry). Batch shell script or Python wrapper. ~50 loc. |

---

## Implementation Order

```
H1 (Multi-TP default) → H7 (Pattern sweep) → H2 (Quality registry)
    → H4 (IR scalar mode) → H3 (Sector-factor scoring)
        → H5 (GA post-step) → H6 (BTC audit)
```

```
P0: H1 + H2 — Immediate impact. Multi-TP is 1 line. Pattern gate eliminates ~30% noise.
P1: H3 + H4 — New alpha layers. Sector factors are uncorrelated with technical patterns.
P2: H5 + H6 — Automation + crypto fix.
P3: H7 — Enables H2. Run first as data-collection step.
```

## Expected Impact

| Task | Est. Sharpe Δ | Est. LOC | Risk |
|------|--------------|----------|------|
| H1 Multi-TP default | +0.20–0.40 | 3 | None (mechanical exit, no signal change) |
| H2 Pattern gate | +0.10–0.20 | 150 | Medium (may eliminate valid signals if calibrated wrong) |
| H3 Sector-factor | +0.05–0.15 | 200 | Low (additive layer, no removal) |
| H4 IR scalar | +0.05–0.10 | 30 | Low (preserves trades, only scales weights) |
| H5 GA post-step | 0 (convenience) | 50 | None (post-processing only) |
| H6 BTC audit | 0 (config fix) | 30 | None (diagnostic only) |

**Combined target:** OOS Sharpe 0.76 → 1.0+ with H1+H2+H4 alone.

## Files to Modify / Create

| File | Action | H# |
|------|--------|-----|
| `src/strategies/rules_first_strategy.py` | Change `use_multi_tp` default to True | H1 |
| `src/strategies/combined_strategy.py` | Change `use_multi_tp` default to True | H1 |
| `scripts/paper_trade_wf_honest.py` | Add `--use-multi-tp` flag (default True) | H1 |
| `scripts/paper_trade_daily.py` | Add `--use-multi-tp` flag (default True) | H1 |
| `src/signals/pattern_quality_registry.py` | **NEW** — Pattern→gate_result mapping, FAIL→0.5x or confluence gate | H2 |
| `scripts/sweep_pattern_gates.py` | **NEW** — Batch evaluate_pattern.py on all 45 detectors | H7 |
| `reports/pattern_gate/all_patterns.json` | **NEW** — Sweep output | H7 |
| `scripts/backtest_rules_first.py` | Add `--ir-weighting-mode` flag | H4 |
| `src/signals/ir_weighting.py` | Add scalar mode (0.3x–0.7x instead of zeroing) | H4 |
| `src/signals/fundamental_scorer.py` | **NEW** — Per-sector factor models, score→RulesFirst modifier | H3 |
| `scripts/backtest_rules_batch.py` | Add `--optimize-portfolio` post-step | H5 |
| `scripts/audit_btc_config.py` | **NEW** — Systematic sweep on BTC config | H6 |
| `src/strategies/rules_first_strategy.py` | Add `use_multi_factor` + `multi_factor_weight` params | H3 |
| `BESTS.md` | Update baseline config (multi-TP on), add Phase 20 sections | H1-H6 |
| `docs/COMMAND_CHEATSHEET.md` | Add Phase 20 commands | H1-H6 |
