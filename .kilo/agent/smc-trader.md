# SMC Trader Agent

> **Purpose:** Execute SMC strategy backtests, interpret results, auto-update BESTS.md.
> **Module:** `src/strategies/smc_strategy.py` (Phase 6-12 enhanced, ~1,700 loc)
> **CLI:** `scripts/backtest_smc.py`
> **Last updated:** 2026-05-21 — Conviction grading fix (scores now continuous, not binary ±0.905)

## Quick Commands

```bash
# Default config (sweep-only with conviction grading + killzones + multi-TP)
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --use-multi-tp \
  --sweep-entry "0.40,0.45,0.50,0.55,0.60"

# With Phase 6 blocks (breaker/mitigation/rejection) — opt-in for equities
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --use-multi-tp \
  --use-breaker-blocks --use-mitigation-blocks --use-rejection-blocks \
  --entry-threshold 0.45

# Sweep entry thresholds (verify score granularity)
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --use-multi-tp \
  --sweep-entry "0.40,0.45,0.50,0.55,0.60,0.65,0.70" --json

# Sweep buffer multipliers (key tuning parameter)
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --use-multi-tp \
  --entry-threshold 0.45 --sweep-buffer "0.50,0.75,1.00,1.20,1.50"

# Full enchilada (Phase 6 + order blocks + volume pressure)
uv run scripts/backtest_smc.py --symbol NQ=F --interval 1h --use-multi-tp \
  --use-breaker-blocks --use-mitigation-blocks --use-rejection-blocks \
  --order-blocks --volume-pressure \
  --entry-threshold 0.45 --trail-stop-atr 2.0 --sweep-buffer-mult 1.20
```

## Component Reference

| Flag | Param | Group | Phase |
|------|-------|-------|-------|
| `--use-breaker-blocks` | `use_breaker_blocks` | SMC Core | 6 |
| `--use-mitigation-blocks` | `use_mitigation_blocks` | SMC Core | 6 |
| `--use-rejection-blocks` | `use_rejection_blocks` | SMC Core | 6 |
| `--use-ir-weights` | `use_ir_weights` | SMC Core | 5 |
| `--use-dow-gate` | `use_dow_gate` | Time | 12 |
| `--use-90min-cycle` | `use_90min_cycle` | Time | 12 |
| `--use-frankfurt-gate` | `use_frankfurt_gate` | Time | 12 |
| `--use-smc-sessions` | `use_smc_sessions` | Library | 9 |
| `--use-smc-phl` | `use_smc_phl` | Library | 9 |
| `--use-smc-retrace` | `use_smc_retrace` | Library | 9 |
| `--use-vol-gate` | `use_vol_gate` | Risk | 5 |
| `--use-session-gate` | `use_session_gate` | Risk | 5 |
| `--use-crash-gate` | `use_crash_gate` | Risk | 5 |
| `--volume-pressure` | `use_volume_pressure` | Risk | 5 |
| `--order-blocks` | `use_order_blocks` | SMC | 5 |
| `--use-multi-tp` | `use_multi_tp` | Exit | 2 |
| `--trail-stop-atr` | `trail_stop_atr` | Exit | 2 |
| `--tp1-atr` | `tp1_atr` | Exit | 2 |
| `--entry-threshold` | `entry_threshold` | Entry | 1 |
| `--sweep-buffer-mult` | `sweep_buffer_mult` | Entry | 1 |
| `--use-short` | `use_short` | Entry | 1 |
| `--msl-msh-lookback` | `msl_msh_lookback` | SMC | Core |
| `--crp-lookback` | `crp_lookback` | Risk | 5 |
| `--ob-lookback` | `ob_lookback` | SMC | 5 |
| `--htf-bias-weight` | `htf_bias_weight` | Trend | Bias |
| `--crypto-mode` | `crypto_mode` | Risk | 5 |
| `--min-confluence` | `min_confluence` | Entry | 1 |

## Component Weights (in scoring)

| Component | Default Weight | Mandatory? |
|-----------|---------------|------------|
| sweep_reversal | 1.50 | YES (score=0 without sweep) |
| breaker_block | 0.70 | No |
| mss_bos_choch | 0.65 | No |
| mitigation_block | 0.60 | No |
| fvg_proximity | 0.55 | No |
| rejection_block | 0.55 | No |
| msl_msh | 0.15 | No |
| bos_choch | 0.05 | No |
| order_block | 0.10 | No |

## Conviction Grading (2026-05-21 Fix)

Previously scores were binary (±0.905) because sweep signals were ±1 and most component features defaulted OFF.
Tanh(±1.5) = ±0.905 with no granularity, making entry threshold sweeps meaningless.

**Fix:** Sweeps are now graded continuously:
- `sweep_depth`: How far beyond the session level (ATR-normalized, 0-3x)
- `reversal_strength`: How strongly the candle reversed (close position within candle)
- `conviction = clip(depth × reversal, 0.1, 2.0)` → base_score = conviction × 1.5
- Scores now range continuously from ~0.15 to ~3.0 before tanh, ~0.15 to ~0.995 after tanh

## Default Feature States

| Feature | Default | Justification |
|---------|---------|---------------|
| `use_smc_sessions` | ON | Killzone detection adds temporal context |
| `use_multi_tp` | ON | Multi-TP improves risk/reward |
| `volume_confirm` | ON | Volume confirmation filters low-conviction signals |
| `use_breaker/mitigation/rejection` | OFF | Help equities (NQ=F Sharpe +0.32), harm forex/crypto |
| `use_order_blocks` | OFF | Adds noise (degraded NQ=F Sharpe from 0.32 → 0.12) |
| `use_volume_pressure` | OFF | Adds noise, expensive precomputation |
| `use_vix_gate`/`use_yield_curve_gate` | OFF | Hard gates, OFF per gate fix postmortem |
| `use_killzone_gate`/`use_htf_gate` | OFF | Hard filters prevent trades |

## Interpreting Results

| Metric | Good | OK | Poor |
|--------|------|----|------|
| Sharpe | > 0.5 | 0.0-0.5 | < 0 |
| Win Rate | > 50% | 40-50% | < 40% |
| Profit Factor | > 1.2 | 0.8-1.2 | < 0.8 |
| Max Drawdown | < 10% | 10-20% | > 20% |
| Trade Count | > 30 | 10-30 | < 10 |
| IS→OOS Sharpe gap | < 0.3 | 0.3-0.5 | > 0.5 (overfit) |

## Known Baselines (2026-05-21, conviction grading)

**NQ=F 1h (full history, sweep-only default config):**
- et=0.40: Sharpe 0.12, 34 trades, 55.9% WR, PF 1.97
- et=0.45: Sharpe 0.12, 33 trades, 54.5% WR, PF 1.97

**NQ=F 1h (Phase 6 blocks opt-in, best config):**
- et=0.45: Sharpe 0.32, 44 trades, 54.5% WR, PF 1.98

**BTC-USD 1h (sweep-only default):**
- Negative across all thresholds (SMC sweep edge is weak on crypto)

## Pitfalls

1. **Scores are now continuous** — entry threshold sweeps produce meaningful trade count variation. Test 0.40-0.70 in 0.05 steps.
2. **Conviction grading is always ON** — no flag to disable. Baked into `_compute_sweep_signals()`.
3. **Phase 6 blocks are instrument-dependent** — help NQ=F (+0.20 Sharpe), degrade BTC/forex. Use opt-in.
4. **Forex win rates < 10%** — SMC sweep detection is unreliable on low-volatility forex pairs.
5. **Crypto vs equities:** Crypto 24/7 — session/killzone gates meaningless. Use `--crypto-mode`.
6. **smartmoneyconcepts startup banner:** Suppressed via `_import_smc_quietly()`.
7. **Gate cascade:** Each gate reduces signals. Running all gates simultaneously may yield 0 trades.

## When to Use This Agent

| Situation | Agent |
|-----------|-------|
| SMC backtest with Phase 6+ detectors | smc-trader |
| Rules-First backtest with pattern detectors | backtest-runner |
| ML model training | ml-trainer |
| Batch parameter tuning | batch-tuner |
| Model health diagnostic | model-doctor |
