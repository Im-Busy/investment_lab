---
type: phase
phase: "03"
name: "Regime Detection & Adaptation"
status: complete
started: 2026-04-19
completed: 2026-04-19
sub_phases:
  - name: "Regime Detector"
    status: complete
  - name: "Adaptive Strategy Router"
    status: complete
  - name: "Parameter Optimization (RSI/MACD)"
    status: complete
---

# Phase 03: Regime Detection & Adaptation

## Overview

Built ADX/ATR-based regime classifier and adaptive strategy router that enables/disables strategies based on detected market regime. Added parameter optimization scripts for RSI and MACD using vectorbt.

## Key Deliverables

### Files Created
- `src/indicators/regime_detector.py` — ADX/ATR regime classifier (260 lines)
- `src/strategies/adaptive_router.py` — Strategy routing by regime (200 lines)
- `tests/test_regime_components.py` — 13 tests (all passing)
- `scripts/optimize_rsi.py` — RSI parameter sweep via vectorbt
- `scripts/optimize_macd.py` — MACD parameter sweep via vectorbt

### Regime Classification
| Regime | Condition | Enabled Strategies | Disabled Strategies |
|--------|-----------|-------------------|---------------------|
| Trending | ADX(14) > 25 | EMA Ribbon, SMA Crossover, ADX, Parabolic SAR, TSI | RSI Divergence, Williams %R, Stoch RSI |
| Ranging | ADX(14) < 20, low ATR | RSI Divergence, Williams %R, Stoch RSI, CCI, MFI | EMA Ribbon, SMA Crossover, ADX |
| Volatile | ATR > 80th %ile | Chandelier Exit, Bollinger, Keltner, VWAP Bounce | All trend-following |
| Transition | ADX 20-25 | Maintains previous regime | — |

### Test Results
- 209 total tests pass, 2 skipped
- 13 new tests for regime components

## Notes
- Regime detector classifies bars using ADX + ATR rolling statistics
- AdaptiveRouter filters signals by regime-appropriateness
- Full time series output with regime labels, ADX, ATR, active strategy counts
- RSI/MACD optimization scripts sweep parameter grids for optimal thresholds
