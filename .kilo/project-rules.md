# Project Rules: investment_trying

## Financial Data & Backtesting Guardrails
- NEVER use future data in backtests (look-ahead bias). Validate signal generation only uses data available at that point in time.
- Handle stock splits and dividends correctly — always use adjusted close prices for backtesting.
- Validate that backtest results are sane: check for impossible returns, negative prices, or trades on non-existent data.
- When using `backtesting.py` or custom engines: verify signal timing aligns with trade execution (signals at close execute next bar open).
- For crypto: use correct ticker format `CRYPTO::SYMBOL//USD`. 24/7 trading has no market hours constraints.
- For equities: check market hours before executing trades. Use `get_market_hours` tool.

## Trading System Context
- This is a rule-based multi-pattern trading system with 45+ chart pattern detectors (34 original + 6 FMZ PineScript/JS conversions + expansion in Phase 17).
- Pattern detectors live in `src/patterns/` across 8 categories (7 original + `fmz/`).
- Strategy wrappers for `backtesting.py` live in `src/strategies/`.
- The custom event-driven backtest engine is in `src/backtest/engine.py`.
- Signal aggregation and confluence scoring is in `src/signals/`.
- Risk management (position sizing, loss limits, circuit breakers) is in `src/risk/`.
- PineScript→Python conversion helpers in `src/indicators/pinescript_helpers.py` (17 functions: supertrend, sar, dmi, macd, qqe, vwap_simple, etc.).
- Research insights tracked in `docs/research_logic_map/insight_registry.md` (65 insights from 20 sources).

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
