# Ticker Selection Criteria for Basket Training

## Why this matters

The CatBoost pattern classifier trains on pooled data from multiple tickers. Adding the wrong ticker dilutes the signal (C3 ablation confirmed: 5 winners alone = worse IC than 12 diverse tickers). Adding the right ticker improves generalization. This guide defines the screening criteria.

## Criteria

| # | Criterion | v1 (strict) | v2 (relaxed) | Rationale |
|---|-----------|-------------|---------------|-----------|
| 1 | **Market cap** | > $500M | > $200M | Mid-cap floor. Below $200M: price action is dominated by binary events |
| 2 | **Daily notional volume** | > $10M | > $5M | Reduced from v1 since mid-caps may trade ~$5M/d |
| 3 | **Institutional ownership** | > 40% | > 25% | Relaxed to include more mid-cap and sector ETFs |
| 4 | **Data history** | > 8 years | > 5 years | Pipeline minimum: 3y train + steps. 5y gives buffer |
| 5 | **Annualized volatility** | 15-50% | 12-55% | Widened both ends. 12% floor includes utilities; 55% ceiling includes materials |
| 6 | **Sector decorrelation** | r < 0.50 vs SPY/QQQ/XLK | r < 0.70 | Relaxed. r < 0.50 preferred but r < 0.70 still adds some diversification |
| 7 | **Exchange** | NYSE/NASDAQ | NYSE/NASDAQ | No OTC, no pink sheets |
| 8 | **Bid-ask spread** | < 0.15% | < 0.25% | Relaxed for mid-cap names |

### Search Priorities (sectors under-represented in basket)

| Priority | Sector | In Basket? | Target Examples |
|----------|--------|-----------|-----------------|
| HIGH | Utilities | No | XLU (ETF), individual utility names |
| HIGH | Real Estate (non-JOE) | No ETF | XLRE, VNQ, individual REITs |
| HIGH | Transportation | No | XTN (ETF), airlines, rails |
| HIGH | Metals/Mining (non-gold) | No | FCX, individual miners |
| MEDIUM | Consumer Staples | No | XLP (ETF), COST, WMT |
| MEDIUM | Mid-cap niche stocks | KODK/JOE only | Individual $200M-$5B names with 10+yr history |

## Screening Process

1. **Fundamental screen** — check criteria 1-8 via Yahoo Finance
2. **Data download** — fetch to `data/raw/{TICKER}_daily.csv`
3. **Walk-forward IC** — run `walk_forward_per_ticker` with the latest 12-all model. Must show mean IC > 0.03
4. **Correlation check** — compare prediction correlations against the 5-winner cluster (SPY/QQQ/XLK/KODK/GLD). Must show r < 0.50

## Current Basket (2026-05-11)

| Ticker | Sector | Market Cap | Walk-forward IC |
|--------|--------|-----------|----------------|
| SPY | Broad market | $600B | +0.074 |
| QQQ | Tech | $300B | +0.087 |
| XLK | Tech | $75B | +0.112 |
| KODK | Materials | $500M | +0.074 |
| GLD | Gold | $75B | +0.048 |
| IWM | Small cap | $70B | +0.001 |
| JOE | Real estate | $3B | +0.032 |
| XLF | Financials | $40B | -0.043 |
| XLE | Energy | $40B | +0.002 |
| XLV | Healthcare | $40B | +0.006 |
| EEM | Emerging mkts | $25B | -0.034 |
| TLT | Treasuries | $50B | -0.039 |

## Recently Evaluated

| Ticker | Sector | Walk-forward IC | Verdict | Issue |
|--------|--------|----------------|---------|-------|
| CAR (Avis Budget) | Industrials | +0.005 | REJECTED | IC below threshold |
| REPL (Replimune) | Healthcare | — | REJECTED | Market cap $336M, beta 0.12 |
| LMT (Lockheed) | Defense | +0.009 | REJECTED | IC below threshold |
| COST (Costco) | Consumer | +0.001 | REJECTED | IC below threshold |
| HD (Home Depot) | Consumer | +0.024 | REJECTED | IC below threshold (borderline) |
| FCX (Freeport) | Materials | -0.045 | REJECTED | Negative IC |
| CAT (Caterpillar) | Industrials | +0.004 | REJECTED | IC below threshold |
| JPM (JPMorgan) | Financials | -0.003 | REJECTED | Negative IC |

Key finding: Even blue-chip S&P 500 stocks with strong fundamentals fail the walk-forward IC screen. The model works on ETFs (captures macro patterns) and niche stocks (KODK, JOE), not on individual large-cap names. Sector ETFs (XLF, XLE, XLV) should be preferred over individual stocks from the same sector.
