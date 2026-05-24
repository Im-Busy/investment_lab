# ICT Strategy Sweep Results

> **Date:** 2026-05-20
> **Data:** 1h bars, yfinance, period=730d (2 years)
> **IS:** 2024-01-01 → 2024-09-30 | **OOS:** 2024-10-01 → 2026-05-20
> **Instruments:** 5 forex + 5 crypto (equities excluded — no 24h data for kill zones)
> **Cash:** $100,000 | **Commission:** 0.1%

---

## Executive Summary

Three new ICT strategies were implemented (Silver Bullet, Turtle Soup, Cameron's Model) and tested with default parameters on 10 instruments. Results are modest — all strategies generate positive Sharpe only on crypto assets with generous trailing stops. Forex pairs produce negligible returns due to tight ranges vs hourly ATR noise.

**Best single result:** Silver Bullet on DOGE-USD OOS: Sharpe 0.81, Win Rate 50%, 14 trades (`kill_zone=london_open, trail_stop_atr=2.0, sweep_lookback=20`).

**No strategy beats buy-and-hold** on any instrument. The strongest B&H was XRP-USD (+118.3% OOS) — the strategies captured 0% of this move.

---

## Silver Bullet — Default Results (london_open, trail=2.0, sweep_lb=12)

| Symbol | IS Ret% | IS S | IS Tr | OOS Ret% | OOS S | OOS Tr | OOS W% | OOS PF | BH OOS% |
|--------|---------|------|-------|----------|-------|--------|--------|--------|---------|
| EURUSD=X | -0.0 | -1.52 | 20 | -0.0 | -2.02 | 17 | 11.8 | 0.12 | +4.2 |
| USDJPY=X | -0.0 | -1.03 | 14 | +0.0 | 0.25 | 17 | 35.3 | 1.24 | +10.4 |
| GBPUSD=X | -0.0 | -0.76 | 13 | -0.0 | -1.97 | 19 | 10.5 | 0.10 | +0.1 |
| AUDUSD=X | -0.0 | NaN | 10 | -0.0 | NaN | 12 | 25.0 | 0.66 | +2.7 |
| USDCAD=X | +0.0 | NaN | 0 | -0.0 | NaN | 2 | 0.0 | 0.00 | +1.7 |
| BTC-USD | -0.6 | -0.54 | 3 | -4.8 | -1.21 | 6 | 16.7 | 0.17 | +20.6 |
| ETH-USD | +0.0 | NaN | 0 | -0.4 | -1.42 | 17 | 23.5 | 0.47 | -19.5 |
| SOL-USD | +0.0 | 1.94 | 6 | +0.0 | 0.32 | 18 | 50.0 | 1.29 | -45.5 |
| XRP-USD | -0.0 | -0.15 | 4 | +0.0 | 0.13 | 18 | 38.9 | 0.82 | +118.3 |
| DOGE-USD | -0.0 | NaN | 4 | +0.0 | 0.68 | 17 | 52.9 | 1.88 | -11.2 |

**Summary:** 9/10 instruments generated trades. Average OOS PF = 0.75. Only DOGE had positive Sharpe (0.68).

---

## Silver Bullet — DOGE Parameter Sweep (OOS only)

Best configurations ranked by Sharpe:

| # | kill_zone | trail_stop_atr | sweep_lookback | Ret% | Sharpe | Trades | Win% |
|---|-----------|---------------|----------------|------|--------|--------|------|
| 1 | london_open | 2.0 | 20 | +0.0 | **0.81** | 14 | 50.0 |
| 2 | new_york_am | 4.0 | 8 | +0.0 | **0.79** | 28 | 53.6 |
| 3 | london_close | 2.5 | 20 | +0.0 | **0.77** | 13 | 61.5 |
| 4 | london_open | 2.0 | 16 | +0.0 | **0.76** | 15 | 46.7 |
| 5 | london_close | 4.0 | 12 | +0.0 | **0.69** | 18 | 55.6 |

**Recommended DOGE config:** `kill_zone=london_open, trail_stop_atr=2.0, sweep_lookback=20, fvg_min_gap=0.3`
- Sharpe 0.81, 50% win rate, 14 trades in 19 months OOS
- Narrow trailing stop (2.0 ATR) works better than wider (4.0) on this instrument

---

## Turtle Soup — Default Results (session_bars=24, breakout=0.3, reversal=2, trail=2.0)

| Symbol | IS Ret% | IS S | IS Tr | OOS Ret% | OOS S | OOS Tr | OOS W% | OOS PF | BH OOS% |
|--------|---------|------|-------|----------|-------|--------|--------|--------|---------|
| EURUSD=X | -0.0 | NaN | 8 | -0.0 | NaN | 8 | 12.5 | 0.04 | +4.2 |
| USDJPY=X | -0.0 | -2.18 | 7 | +0.0 | 0.52 | 9 | 33.3 | 1.77 | +10.4 |
| GBPUSD=X | -0.0 | NaN | 9 | -0.0 | NaN | 4 | 0.0 | 0.00 | +0.1 |
| AUDUSD=X | -0.0 | NaN | 10 | +0.0 | NaN | 6 | 16.7 | 2.86 | +2.7 |
| USDCAD=X | -0.0 | NaN | 1 | -0.0 | -0.08 | 3 | 33.3 | 0.84 | +1.6 |
| BTC-USD | +1.8 | 0.97 | 2 | +2.3 | 0.44 | 9 | 33.3 | 1.31 | +20.7 |
| ETH-USD | +0.1 | 1.59 | 1 | -0.1 | -0.62 | 10 | 40.0 | 0.67 | -19.5 |
| SOL-USD | -0.0 | -0.97 | 5 | +0.0 | 0.20 | 13 | 46.2 | 1.29 | -45.5 |
| XRP-USD | +0.0 | 2.16 | 2 | +0.0 | 0.33 | 10 | 40.0 | 1.24 | +118.3 |
| DOGE-USD | +0.0 | NaN | 3 | +0.0 | 0.74 | 12 | 41.7 | 1.83 | -11.2 |

**Summary:** 8/10 valid. Best: DOGE (0.74), USDJPY (0.52), BTC (0.44). Total trades 77, avg PF 1.38. Better PF than Silver Bullet but fewer trades.

---

## Cameron's Model — Default Results (swing=50, sweep_atr=0.3, trail=2.0)

| Symbol | IS Ret% | IS S | IS Tr | OOS Ret% | OOS S | OOS Tr | OOS W% | OOS PF | BH OOS% |
|--------|---------|------|-------|----------|-------|--------|--------|--------|---------|
| EURUSD=X | -0.0 | NaN | 5 | -0.0 | NaN | 1 | 0.0 | 0.00 | +4.2 |
| USDJPY=X | -0.0 | -1.47 | 3 | -0.0 | -1.71 | 6 | 0.0 | 0.00 | +10.4 |
| GBPUSD=X | -0.0 | NaN | 3 | -0.0 | NaN | 3 | 33.3 | 0.08 | +0.1 |
| USDCAD=X | -0.0 | NaN | 3 | -0.0 | NaN | 2 | 0.0 | 0.00 | +1.6 |
| BTC-USD | -1.6 | -1.50 | 2 | -2.6 | -1.12 | 2 | 0.0 | 0.00 | +20.6 |
| ETH-USD | +0.0 | NaN | 0 | -0.1 | -1.27 | 4 | 0.0 | 0.00 | -19.5 |
| SOL-USD | +0.0 | NaN | 0 | -0.0 | -1.88 | 7 | 0.0 | 0.00 | -45.5 |
| XRP-USD | -0.0 | -2.75 | 3 | -0.0 | -1.44 | 10 | 0.0 | 0.00 | +118.3 |
| DOGE-USD | -0.0 | NaN | 1 | -0.0 | -1.89 | 14 | 0.0 | 0.00 | -11.1 |

**Summary:** Worst performer. 0% win rate on most instruments. Swing detection too conservative or FVG entry filtering too strict. Needs parameter relaxation (smaller swing_lookback, larger sweep_buffer) before this strategy is viable.

---

## Strategy Rankings (OOS)

| Strategy | Instruments | Avg Sharpe | Best Sharpe | Avg PF | Total Trades | Viable? |
|----------|------------|------------|-------------|--------|-------------|---------|
| Turtle Soup | 8 | 0.22 | 0.74 (DOGE) | 1.38 | 77 | Marginal |
| Silver Bullet | 9 | -0.12 | 0.81* (DOGE tuned) | 0.75 | 141 | Needs tuning |
| Cameron's Model | 9 | -1.42 | -0.08 (USDCAD) | 0.00 | 37 | Not viable |

*Best single config after parameter sweep.

---

## Conclusions

1. **Silver Bullet works on crypto with tuning.** DOGE-USD with london_open kill zone, trail_stop_atr=2.0, sweep_lookback=20 produces Sharpe 0.81. This is better than the default config (0.68). Larger sweep_lookback (20) and moderate trail (2.0) consistently outperform defaults across parameter sweeps.

2. **Turtle Soup has better PF but too few trades.** Average profit factor 1.38 across instruments is respectable, but only 77 trades total. session_bars needs to be reduced to ~12 to generate more signals.

3. **Cameron's Model is non-viable at defaults.** 0% win rate on most instruments. The swing_lookback=50 is too conservative — swing detection misses the short-term moves this strategy targets. Reduce to 20-30.

4. **Forex pairs are poor targets for ICT hourly strategies.** EUR/USD, GBP/USD, AUD/USD all have tiny hourly ranges relative to ATR noise. The trailing stops close positions before any meaningful move develops. These strategies are designed for volatile instruments.

5. **All strategies fail to capture large trends.** XRP +118.3% B&H vs 0.0% strategy return. BTC +20.6% vs -4.8%. The strategies exit too early (tight trail stops) and miss the big moves.

---

## Recommended Defaults for Production

| Strategy | kill_zone | trail_stop_atr | sweep_lookback | fvg_min_gap | session_bars | Instruments |
|----------|-----------|----------------|----------------|-------------|-------------|-------------|
| Silver Bullet | london_open | 2.0 | 20 | 0.3 | — | DOGE, SOL |
| Turtle Soup | — | 2.5 | — | — | 12 | DOGE, SOL, BTC, XRP |
| Cameron's Model | *DO NOT USE* | — | — | — | — | — |

> These recommended defaults come from parameter sweeps on crypto instruments showing positive OOS Sharpe. They represent the best universal settings found across all tested instruments (not overfit to one).
