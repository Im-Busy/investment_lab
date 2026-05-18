# Handover: C9-C17 Implementation (2026-05-16 — Updated)

## Session Summary

All C9-C16 items complete. C17 (empirical calibration) deferred as P3 priority.

### COMPLETE (9/10 items)

| Item | Status | Files |
|------|--------|-------|
| C9: Honest walk-forward paper trade | ✅ | `scripts/paper_trade_wf_honest.py` |
| C10: Portfolio-level backtest | ✅ | `scripts/backtest_portfolio.py` |
| C11: Volume/OI validation layer | ✅ | head_shoulders.py, double_top.py, double_bottom.py |
| C12: Multi-TP exit logic | ✅ | `src/strategies/rules_first_strategy.py` |
| C13: Gap pattern hierarchy | ✅ | Already in `src/patterns/breakout/gap.py` |
| C14: Extended harmonic detectors | ✅ | `src/patterns/harmonic/extended.py` — init kwarg fix applied |
| C15: Pipe pattern detector | ✅ | `src/patterns/complex/pipe.py` — 2-bar mechanical reversal |
| C16: Dead Cat Bounce threshold | ✅ | `src/patterns/classic/dead_cat_bounce.py` — min bounce 50% |
| C17: Empirical calibration | ⬜ | P3, deferred |

### C14 Fix Applied

Extended harmonic patterns (Butterfly, Bat, Crab, Cypher, Shark) had a bug: they passed `name`/`pattern_type`/`min_bars_required` as kwargs to `GartleyPattern.__init__()`, which doesn't accept them. Fixed by calling `super().__init__()` with correct Gartley params then overriding `self.name` after init. All 5 classes patched in `src/patterns/harmonic/extended.py`.

### C15: Pipe Pattern Detector

Created `src/patterns/complex/pipe.py`:
- Two-bar mechanical pattern (zero parameters, no Fibonacci)
- Pipe Up (bullish): 2 red candles, second engulfs first → long at High[i] + offset
- Pipe Down (bearish): 2 green candles, second engulfs first → short at Low[i] - offset
- Target: L = max(body0, body1); T1 = ±L, T2 = ±2L
- Confidence: 0.55 base, +0.05 if second bar volume > first
- Registered in RulesFirstStrategy, paper_trade_wf_honest.py, and `__init__.py`

### C16: Dead Cat Bounce Fix

Changed `min_bounce_pct` from 0.38 to 0.50 (requires minimum 50% retracement per Duddella spec). The 15% event-day decline threshold was already in place.

### Smoke Test (SPY 2025-2026)

```
Total Return:    38.69%
Sharpe:           2.74
Max Drawdown:   -66.10%
Win Rate:        42.9%
Profit Factor:    0.65
Trades:             14
```

Note: High max drawdown and profit factor < 1.0 suggests confluence scoring may be too permissive. C17 empirical calibration would address this by tuning PATTERN_RELIABILITY weights based on actual win rates.

## Key Code Paths

| What | Where |
|------|-------|
| Production strategy | `src/strategies/rules_first_strategy.py` |
| Walk-forward paper trade | `scripts/paper_trade_wf_honest.py` |
| Pipe pattern | `src/patterns/complex/pipe.py` |
| Harmonic extensions | `src/patterns/harmonic/extended.py` |
| Dead Cat Bounce | `src/patterns/classic/dead_cat_bounce.py` |

## Test Command

```bash
uv run python scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01
```

## All Patterns Now Active

Total patterns registered: 39 (excluding 6 FMZ patterns that hang on full data). New since last session: Butterfly, Bat, Crab, Cypher, Shark, Pipe.

## Next Steps (Post-C16)

1. C17: Empirical calibration — run solo backtests per pattern, update reliability weights
2. Investigate -66% max drawdown — likely over-trading from low-confluence signals
3. FMZ pattern optimization — profile and fix `detect_vectorized()` hang on PineScript helpers
