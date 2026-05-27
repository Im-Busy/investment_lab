# Project Rules: investment_trying

## Financial Data & Backtesting Guardrails
- NEVER use future data in backtests (look-ahead bias). Validate signal generation only uses data available at that point in time.
- Handle stock splits and dividends correctly — always use adjusted close prices for backtesting.
- Validate that backtest results are sane: check for impossible returns, negative prices, or trades on non-existent data.
- When using `backtesting.py` or custom engines: verify signal timing aligns with trade execution (signals at close execute next bar open).
- For crypto: use correct ticker format `CRYPTO::SYMBOL//USD`. 24/7 trading has no market hours constraints.
- For equities: check market hours before executing trades. Use `get_market_hours` tool.

## Trading System Context
- This is a rule-based multi-pattern trading system with 54 chart pattern detectors across 10 categories (7 original + fmz, technical, range-persistence).
- Pattern detectors live in `src/patterns/` across 10 categories (basic, breakout, candlestick, classic, complex, continuation, harmonic, fmz, technical, range-persistence).
- Strategy wrappers for `backtesting.py` live in `src/strategies/`.
- The custom event-driven backtest engine is in `src/backtest/engine.py`.
- Signal aggregation and confluence scoring is in `src/signals/`.
- Risk management (position sizing, loss limits, circuit breakers) is in `src/risk/`.
- PineScript→Python conversion helpers in `src/indicators/pinescript_helpers.py` (17 functions: supertrend, sar, dmi, macd, qqe, vwap_simple, etc.).
- Research insights tracked in `docs/research_logic_map/insight_registry.md` (77 insights from 21 sources).

## Backtesting Standards
- Every strategy must produce valid signal logs before being considered complete.
- Report key metrics in comparison tables: return, Sharpe, max drawdown, win rate, profit factor.
- Compare against SPY baseline (buy and hold) and BTC baseline for context.
- A strategy is considered overfit if OOS performance is significantly worse than training performance.
- Minimum viable thresholds: 30+ trades, Sharpe > 0.5, profit factor > 1.2 (unless user specifies otherwise).

## Backtest Leaderboard — BESTS.md (CRITICAL)
- **After every backtest run that produces a result, update `BESTS.md`** in the project root.
- If a new result beats the existing best in its condition category (trail, baseline, vol-gate, etc.), update the table and bump the old entry down.
- If a new result introduces a new condition/config not previously tracked, add a new section.
- Record: config string, return%, Sharpe, trades, win%, profit factor, max drawdown, exposure%, and annualized return.
- Update the "Last updated" timestamp at the bottom of the file.
- Never remove old records — the leaderboard is a cumulative history of bests.

## Command Documentation Protocol (CRITICAL — for all AI sessions)

**Every time you create, modify, or delete a script/CLI command in this project, you MUST also update its documentation.** This is how future AI sessions and human developers discover and correctly use these commands.

### Required Locations (update ALL of these)

| Priority | File | What to record |
|----------|------|---------------|
| **P0 (always)** | `docs/COMMAND_CHEATSHEET.md` | Every user-facing CLI command with flags, examples, and one-line description |
| **P1 (if complex)** | `.useful_commands/<category>_commands.txt` | Detailed workflow steps, multiple variations, expected output |
| **P2 (if module-specific)** | `src/<module>/AI_COMMANDS.txt` | Quick reference specific to that module |
| **P3 (if new category)** | `AGENTS.md` | If a new command category or documentation convention emerges |

### What to Include for Each Command

- **What it does** — 1-line description
- **When to use it** — decision criteria (e.g., "use `--trail-stop` for bull markets, skip for ranging")
- **Full CLI example** — copy-paste ready, with realistic flags
- **Expected output** — key metrics or behavior to expect
- **Related commands** — cross-reference sibling/alternative commands

### When Modifying or Deleting

- **Modified**: Update existing entries to reflect new flags, changed defaults, removed options
- **Deleted**: Remove all references from cheatsheet and useful_commands files. Do NOT leave orphaned commands.
- **Renamed**: Update all references. Use `grep` to find every occurrence before renaming.

### Validation Checklist

Before marking a script/command change as complete, verify:
- [ ] `docs/COMMAND_CHEATSHEET.md` entry exists and is current
- [ ] All CLI flags and options are documented with examples
- [ ] Dependencies or prerequisites are noted if needed
- [ ] Related `.useful_commands/` files updated if workflow is complex
- [ ] `AGENTS.md` updated if documentation structure changed
- [ ] `BESTS.md` updated if the change affects backtest results

### Examples of Good Documentation

```markdown
# Run ML strategy backtest with custom entry threshold
# Use when: model probabilities are too conservative (entry signals too rare).
# Lower threshold = more trades, higher threshold = more selective.
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --entry-threshold 0.45
```

## End-to-End Wiring Protocol (CRITICAL — MANDATORY CHECK BEFORE MARKING COMPLETE)

**Every new parameter, feature flag, detector, indicator, risk module, or CLI argument MUST be wired end-to-end into the execution path.** A feature that compiles but doesn't affect scoring/entry/exit is invisible debt.

### The Rule: Trace Every Parameter

After implementing any new feature, you MUST trace its parameter from declaration → init → scoring → entry/exit gates. Use this checklist:

| Check | Question |
|-------|----------|
| **Declaration** | Is the param on the class with a default? |
| **Init gate** | Does `init()` conditionally precompute based on this param? |
| **Scoring** | Does `_compute_score()` or equivalent read this param or its arrays? |
| **Entry/Exit** | Does `next()` conditionally gate on this param's value? |
| **CLI exposure** | Is there an argparse flag that passes to this param? |
| **CLI-in-kwargs** | Does the script's kwargs dict include the flag? |

### Anti-Patterns That Passed Code Review But Were Dead

| Anti-Pattern | Real Example (2026-05-21 audit) | Consequence |
|-------------|------|-------------|
| Param declared but never read in scoring | `min_confluence=3` in SMC | Users think they're gating entries but zero effect |
| Feature precomputes data but never scored | `use_smc_phl=True` computes 4 arrays, none read | Wasted CPU + user confusion |
| Function exists but never called | `_calculate_size()` | Positions stay hardcoded 0.95 |
| CLI flag not in kwargs dict | `--no-volume-pressure` flag declared but not mapped | Flag silently ignored |
| Toggle gates precompute but not scoring | `use_breaker_blocks` blocks init but scoring always reads arrays | Misleading — appears wired when it's not |

### Mandatory End-to-End Validation Script

After implementing any new parameter, run this audit (or equivalent):

```bash
# Check if param is referenced in scoring
uv run python -c "
import inspect
from src.strategies.smc_strategy import SMCStrategy
# For each param with default, grep for usage in _compute_score and next()
# Any param not found is a wiring gap
"

# Verify CLI flag maps into kwargs dict
grep -n 'my_new_flag' scripts/backtest_smc.py  # should appear in BOTH argparse AND kwargs dict
```

**A feature is NOT complete until it produces a documented, measurable change in backtest output when toggled ON vs OFF.** If `--my-feature` and its absence produce identical backtest metrics, the feature is either dead or not contributing signal — fix or remove it.
