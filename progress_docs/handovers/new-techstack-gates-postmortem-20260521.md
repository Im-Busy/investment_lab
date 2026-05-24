# Handover — 2026-05-21 (New Techstack Gates Postmortem)

> **State:** All 24 phases COMPLETE. Phase 07 paper trading continues.
> **Key finding:** VIX+yield gates defaulted ON destroy returns without improving risk-adjusted performance.
> **Immediate action needed:** Revert VIX+yield gates to default OFF in all production scripts.

---

## What Happened

Ran full "new techstack" backtests — VIX regime gate + yield curve macro gate both ON by default — across:

1. **RulesFirst Multipattern — 16 instruments** (3 small batches via `backtest_all_comprehensive.py`)
2. **SMC/ICT — 5 instruments** (via `backtest_smc.py --all`)

## Results

### RulesFirst — 16 instruments (VIX+yield gates ON)

| Metric | IS (2016-2024) | OOS (2025-2026) |
|--------|---------------|------------------|
| Positive Sharpe | 8/16 (50%) | 7/16 (44%) |
| Mean Sharpe (traded) | +0.040 | +0.110 |
| Returns | 0.00-0.11% (except BTC +25% IS) | 0.00-0.09% |

| Symbol | OOS Sharpe | OOS Ret% | Pre-Gate Ret% | Gate Impact |
|--------|-----------|----------|--------------|-------------|
| XLV | +1.180 | +0.01% | +3.50% | Returns destroyed |
| GLD | +1.103 | +0.09% | +6.60% | Returns destroyed |
| XLE | +0.832 | +0.01% | +9.30% | Returns destroyed |
| SPY | +0.705 | +0.04% | +9.20% | Returns destroyed |
| BTC_USD | 0.000 | 0.00% | — | 0 OOS trades |
| EURUSD_X | 0.000 | 0.00% | — | 0 OOS trades |

### SMC/ICT — 5 instruments (VIX+yield gates ON)

All 5 negative Sharpe. NQ=F: +0.34→-1.13 (Δ -1.47).

### Root Cause

The VIX gate treats 2025's structurally elevated VIX (25-30) as "ELEVATED" or "STRESS", applying 0.75x-0.30x multipliers to all signals. This filters trades that would otherwise be profitable in a trending bull market. The Sharpe ratio appears decent (+0.7 on SPY) only because it's calculated on microscopic returns — a mathematical artifact of gate-induced trade scarcity.

## Action Items

### P0 — Revert Gate Defaults (est. 30 min)

1. `scripts/backtest_all_comprehensive.py` — change PRODUCTION_CONFIG: `use_vix_gate=False, use_yield_curve_gate=False`
2. `scripts/backtest_smc.py` — change parser defaults: `--use-vix-gate`→False, `--use-yield-curve-gate`→False
3. `src/strategies/rules_first_strategy.py` — change strategy class defaults for `use_vix_gate`, `use_yield_curve_gate` to False
4. `src/strategies/smc_strategy.py` — change strategy class defaults to False
5. Re-run SPY OOS to confirm +9.2% return is restored
6. Update COMMAND_CHEATSHEET.md gate flag descriptions

### P1 — Gate as Opt-In (est. 20 min)

1. Update BESTS.md with recommendation: gates = opt-in for specific regime conditions
2. Update MEMORY.md completed tasks
3. Verify `--use-vix-gate` / `--use-yield-curve-gate` flags work as opt-in across all scripts

## Files Modified in This Session

| File | Change |
|------|--------|
| `BESTS.md` | +2 new sections (RulesFirst New Techstack, SMC/ICT New Techstack) with full results, comparison tables, insights |
| `docs/COMMAND_CHEATSHEET.md` | +"New Techstack Backtests" section with batch commands, gate impact table |
| `MEMORY.md` | Updated objective, system state, OOS comparison table, completed tasks |
| `progress_docs/current.md` | Session log entry |

## Commands Used

```bash
# RulesFirst batches
uv run scripts/backtest_all_comprehensive.py --tickers SPY,QQQ,IWM,XLK,XLF --output-dir reports/comprehensive_batch --period both
uv run scripts/backtest_all_comprehensive.py --tickers XLE,XLV,GLD,TLT,KO --output-dir reports/comprehensive_batch --period both
uv run scripts/backtest_all_comprehensive.py --tickers JPM,XOM,JNJ,SO,BTC_USD,EURUSD_X --output-dir reports/comprehensive_batch --period both

# SMC all instruments
uv run scripts/backtest_smc.py --all --interval 1h --start 2025-01-01 --json outputs/smc_all_new_tech_20260521.json
```

## System State

- **Production system:** Rules-First Strategy (primary). SMC/ICT (experimental).
- **Current config:** mr=0.70, et=0.55, multi-TP=ON, quality-registry=ON, VIX-gate=**ON**, yield-gate=**ON**
- **Recommended config:** mr=0.70, et=0.55, multi-TP=ON, quality-registry=ON, VIX-gate=**OFF**, yield-gate=**OFF**
- **Paper trading:** Active (Phase 07), awaiting 14-day evaluation period.
