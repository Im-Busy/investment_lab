# Edge Characterization — Where the Pattern Classifier Works

> Based on 81-ticker walk-forward analysis with 33-ticker CatBoost model
> (2026-05-11). See `scripts/c6b_relaxation_analysis.py` and `scripts/c6c_edge_range.py`.

## Summary

The model's predictive edge is **volatility-bound**, not volume-bound or size-bound. Ticker selection should filter on annualized volatility first — everything else is secondary.

## The Three Zones

| Zone | Vol Range | N | Mean IC | Pass Rate | Negative IC | Action |
|------|-----------|---|---------|-----------|-------------|--------|
| **Sweet spot** | 12–25% | 33 | **0.056** | 73% | 0% | Always include |
| **Cliff zone** | 25–35% | 18 | 0.033 | 50% | 6% | Include only if sector signal is strong |
| **Dead zone** | >35% | 20 | 0.021 | 30% | 20% | Exclude |

## Granular Volatility Ranges (2% steps)

| Vol | N | Mean IC | Pass% |
|-----|---|---------|-------|
| 16–18% | 4 | 0.045 | 50% |
| **18–20%** | **7** | **0.069** | **100%** |
| 20–22% | 7 | 0.054 | 57% |
| 22–24% | 9 | 0.057 | 67% |
| 24–26% | 6 | 0.050 | 83% |
| 26–28% | 7 | 0.043 | 57% |
| 28–30% | 2 | 0.002 | 0% |
| 30–32% | 2 | 0.046 | 100% |
| 32–34% | 6 | 0.037 | 50% |
| 34–36% | 1 | −0.020 | 0% |
| 36–38% | 4 | 0.039 | 50% |
| 38–40% | 4 | 0.014 | 25% |
| 40–42% | 3 | 0.016 | 33% |
| 42–44% | 2 | 0.051 | 50% |
| 44–46% | 1 | 0.013 | 0% |
| 48–50% | 3 | −0.005 | 0% |
| 50–52% | 1 | −0.032 | 0% |

The peak is at **18–20% annualized volatility** where all 7 tickers pass and mean IC = 0.069.

## Hard Cutoffs

| Cutoff | N | Mean IC | Pass% | Neg% |
|--------|---|---------|-------|------|
| ≤20% vol | 11 | 0.060 | 82% | 0% |
| ≤22% vol | 18 | 0.058 | 72% | 0% |
| ≤25% vol | 33 | 0.056 | 73% | 0% |
| ≤28% vol | 40 | 0.054 | 70% | 0% |
| ≤30% vol | 42 | 0.051 | 67% | 2% |
| ≤35% vol | 51 | 0.048 | 65% | 6% |

At ≤25% vol, **zero tickers have negative IC**. This is the cleanest threshold.

## What Does NOT Matter

| Factor | Spearman r | p-value | Conclusion |
|--------|-----------|---------|------------|
| Market cap | +0.02 | 0.87 | No relationship. $1B names work same as $500B. |
| Daily volume | −0.31 | 0.009 | Weak negative. Mid-volume ($5M–$500M/d) actually outperforms $1B+/d. |

**Market cap is irrelevant.** You can include micro-caps or mega-caps — the model doesn't care. It only cares about how smoothly the price series moves.

**Volume is a weak negative signal.** The 4 highest-IC tickers (SO +0.13, D +0.12, AVB +0.11, KODK +0.10) are mid-volume names. The 15 tickers with $1B+/d volume average only 0.031 IC. High-volume = high-profile = efficient pricing = less edge for technical patterns.

## Sector × Volatility Interaction

The edge works because certain sectors *naturally cluster in the sweet spot*:

| Sector | N | Mean Vol | Mean IC | Verdict |
|--------|---|---------|---------|---------|
| **Utilities** | 9 | 21% | **0.086** | Best. Low vol, predictable. |
| **Financial (niche)** | 3 | 37% | 0.052 | Good despite vol. CRVL, HIFS drive it. |
| **Real Estate (REITs)** | 9 | 27% | 0.052 | Good. Moderate vol. |
| **Consumer Defensive** | 9 | 19% | 0.043 | Good. Low vol staples. |
| **Healthcare** | 10 | 28% | 0.043 | Mixed. Biotech vol drags, pharma works. |
| Industrials | 13 | 43% | 0.024 | Poor. Airlines, transports are volatile. |
| Energy | 10 | 35% | 0.019 | Poor. Oil price swings dominate. |
| Basic Materials | 6 | 41% | 0.017 | Poor. Miners are high-vol. |

Sector is informative because it's a proxy for volatility. Once you control for volatility, sector itself adds little independent signal.

## Why Volatility is the Binding Constraint

The CatBoost model uses 60 features — MA slopes, BB width, RSI, volatility regimes, distance-to-extremes, etc. These are **mean-reversion and trend-continuation** features. They assume price action follows semi-predictable patterns.

High-volatility tickers (>35% ann. vol) are driven by:
- Binary events (FDA approvals for biotech, OPEC decisions for energy)
- Liquidity shocks (gap opens on earnings)
- Commodity price swings (miners, oil E&P)

These overwhelm any technical pattern signal. The model sees noise, not structure.

## Revised Ticker Selection Criteria (v3)

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| 1 | **Annualized volatility** | **12–25%** (strict) or 12–35% (relaxed) | Primary gate. Below 25% = no negative IC. |
| 2 | Market cap | > $200M | Secondary. Size doesn't matter. |
| 3 | Daily notional volume | > $5M | Secondary. Mid-volume is fine. |
| 4 | Institutional ownership | > 25% | Tertiary. |
| 5 | Data history | > 5 years | Pipeline minimum. |
| 6 | Exchange | NYSE/NASDAQ | No OTC. |
| 7 | Walk-forward IC | > 0.03 | Final gate. |

## Basket Construction Rules

1. **Volatility filter first**: only consider tickers with 12–25% ann. vol for the core basket. Expand to 35% if needed for sector coverage.
2. **Sector diversity second**: ensure coverage across utilities, REITs, staples, healthcare, financials. Avoid clustering in any one sector.
3. **Avoid sectors that are structurally high-vol**: basic materials (miners), energy (E&P), airlines. Include only if individual ticker IC > 0.05.
4. **ETF preference when available**: XLU, XLRE, XLP, XLV capture sector beta cleanly with lower vol than individual names.
5. **Niche midcaps are gold**: CRVL, HIFS, JOE, KODK-style names with 10+ year histories and moderate vol punch above their weight.

## Related

- `docs/guide-ticker-selection.md` — original v1/v2 criteria (now superseded by v3 above)
- `docs/ticker-test-log.md` — full audit trail of 75 screened tickers
- `scripts/c6b_relaxation_analysis.py` — relaxation analysis across all factors
- `scripts/c6c_edge_range.py` — granular volatility range analysis
- `reports/c6c_edge_range/edge_sweet_spot.png` — visualization of the three zones
