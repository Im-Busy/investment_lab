---
description: Run parameter tuning and backtests in parallelized batches across multiple instruments. Tunes Rules-First strategy (entry_threshold, min_reliability, trail_stop_atr, confluence_bonus) per instrument, finds universal best config. Uses IS/OOS split, multiprocessing, and auto-generates comparison reports.
---

# Batch Tune — Parameter Sweeps Across Instruments

Load the **batch-tuner** agent to run parameter sweeps and backtests in parallelized batches.

## Arguments

```
/batch-tune [--fast|--mini|--full] [--symbols SYM1,SYM2,...] [--workers N]
/batch-tune oos                 # Run OOS validation + universal best (after IS batches)
/batch-tune report              # Regenerate markdown report from existing JSON
```

## Quick Examples

```
# Full 16-instrument tuning (3 batches, sequentially)
/batch-tune

# Single instrument quick sweep
/batch-tune --symbols SPY --mini

# Custom basket
/batch-tune --symbols SPY,QQQ,GLD,TLT,BTC_USD --fast

# OOS validation phase (after IS batches complete)
/batch-tune oos
```

## Workflow

1. **Phase 1:** `/batch-tune` — Runs 3 batches of IS sweeps (5-6 instruments each)
2. **Phase 2:** `/batch-tune oos` — Validates best configs on OOS + finds universal best
3. **Phase 3:** Agent updates `BESTS.md` and `COMMAND_CHEATSHEET.md`

**Report:** `reports/parameter_tuning/RULES_TUNING.md`
