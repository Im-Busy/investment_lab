# Phase 16 Next Steps — Execution Plan & Data Source Roadmap

> **Date:** 2026-05-15 | **Status:** N+P+O+Q+R complete (5/9). V+S next.
> **Constraint:** No paid subscriptions. Build bases with pluggable providers.

---

## Current Position

| Done | Phase | Gate |
|------|-------|------|
| N | LM Dictionary NLP Foundation | PASS |
| P | Mean Reversion + RegimeRouter | PASS (Sharpe 0.68) |
| O | Sentiment Lead/Lag Gate | BLOCKED (no infra) |
| Q | Multi-Factor Fundamentals | PASS (14 factors) |
| R | FinBERT + SEC + Fusion | FAIL (AUC -0.019) |

Available without new subscriptions: **~120 price series** (US/CN/HK equities, BTC_USD, ETH_USD, EURUSD_X), **SEC EDGAR** (free text), **yfinance** (free snapshot fundamentals), **CCXT** (free crypto OHLCV, not installed yet).

---

## Recommended Execution Order

### Tier 1 — Build Now (zero new data needed)

**Phase S: Pairs Trading** → then **Phase V: Strategy v2**

Why this order: Phase V's V3 (correlation-aware portfolio) builds on pair correlation concepts from Phase S. Phase S is a genuinely independent alpha source — it doesn't correlate with trend-following. Phase V is structural improvements to what already works.

### Tier 2 — After one $10-30/mo data subscription

**Phase N Enhancement:** Real news text for FinBERT (Tiingo News API)
**Phase Q Enhancement:** Point-in-time quarterly fundamentals (FMP Starter)

### Tier 3 — Gated on GPU or higher-cost data

**Phase T:** Deep Learning (GPU gate, $0+)
**Phase U:** Alternative Data (paid gate, $100+/mo)

---

## Data Source Recommendations

### Table: Best Sources by Category

| Category | Best Free | Best Cheap | Monthly Cost | What You Get |
|----------|-----------|------------|-------------|--------------|
| Equity prices (EOD) | yfinance | Tiingo | $0→$30 | 30yr history, 86K securities |
| Crypto OHLCV | CCXT | Tiingo | $0→$30 | 100+ exchanges, 24/7 |
| Fundamentals (snapshot) | yfinance | FMP Free | $0 | Current P/E, P/B, ROE |
| Fundamentals (time-series) | FMP Free tier | FMP Starter | $0→$22 | 5yr quarterly statements |
| News text | SEC EDGAR (filings) | Tiingo | $0→$30 | 50M+ articles, 1990s+ |
| Real-time quotes | None reliable free | Polygon.io Basic | $29 | Delayed, unlimited requests |
| Alternative data | Google Trends (pytrends) | — | $0 | Search volume trends |

### Recommended First Subscription: Tiingo ($30/mo individual)

**Why Tiingo over FMP Starter ($22/mo)?**
1. Flat-rate pricing — no per-call limits, no bandwidth caps to worry about
2. News API included — 3 months queryable history + forward, 50M+ articles
3. 30+ years EOD price history — replace yfinance entirely
4. Fundamental data add-on available — contact sales for exact pricing
5. 103K symbols/month — covers entire US market

**When to pick FMP instead:** If you specifically need quarterly point-in-time fundamental statements (income statement, balance sheet, cash flow by quarter). FMP's free tier (250 req/day) is already enough for daily ML feature extraction — you can start with free tier immediately. Upgrade to Starter ($22/mo) when you need 5+ years history.

### Crypto: CCXT (free, install now)

```bash
uv add ccxt
```

CCXT is a pure-Python library supporting 100+ exchanges (Binance, Coinbase, Kraken, Bybit). No API key needed for public OHLCV data. We already have BTC_USD and ETH_USD daily files in `data/raw/`. CCXT would let us expand to SOL, AVAX, DOGE, etc. and fetch fresh data on demand.

---

## Phase S: Pairs Trading — Execution Plan

### What We Build

A market-neutral pairs trading engine. This is alpha that cannot be obtained from trend-following or mean-reversion — it's genuinely orthogonal.

### S1: Pairs Trading Engine (`src/strategies/pairs_trading_strategy.py`)

```
Pipeline:
1. Correlation screen: 252d rolling correlation for all pairs in universe
2. Cointegration test: Engle-Granger (statsmodels) — only trade cointegrated pairs
3. Hedge ratio: OLS regression or Kalman filter (statsmodels UnobservedComponents)
4. Spread: spread = log(P_A) - hedge_ratio * log(P_B)
5. Entry: Z-score > 2.0 (short spread) or < -2.0 (long spread)
6. Exit: Z-score crosses 0, or stop-loss at 3.0
```

Backtesting.py integration: `self.buy()` for pair A + `self.sell()` for pair B simultaneously. Position sizing: 50% capital per leg.

### S2: Pre-Built Sector Pairs (no data needed — all tickers in data/raw/)

| Sector | Pair | Rationale |
|--------|------|-----------|
| Consumer Staples | KO-PEP | Beverage duopoly |
| Energy | CVX-XOM | Supermajors |
| Big Oil | OXY-COP | US oil producers |
| Miners | NEM-GOLD | Gold miners |
| Railroads | UNP-CSX | Class I railroads |
| Tech ETFs | XLK-QQQ | Tech concentration |
| Broad Market | SPY-IWM | Large vs small cap |
| Healthcare | JNJ-MRK | Pharma giants |
| Defense | LMT-NOC | Defense contractors |
| Utilities | DUK-SO | Electric utilities |

All tickers are in `data/raw/`. No new data needed.

### S3: CLI (`scripts/backtest_pairs.py`)

```bash
uv run scripts/backtest_pairs.py KO-PEP --start 2016-01-01
uv run scripts/backtest_pairs.py --auto-pairs --min-correlation 0.7 --n-pairs 10
uv run scripts/backtest_pairs.py --compare  # all pre-built pairs
```

### Data Pluggability Design for Pairs

```python
class PairUniverseProvider(Protocol):
    """Protocol for providing candidate pairs."""
    def get_pairs(self) -> list[tuple[str, str]]: ...
    def get_prices(self, pair: tuple[str, str]) -> tuple[pd.Series, pd.Series]: ...
```

Default implementation: `CSVPriceProvider` (reads `data/raw/{ticker}_daily.csv`). Can later add `CCXTPriceProvider`, `TiingoPriceProvider`, etc.

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Cointegration test works | p < 0.05 for KO-PEP |
| Spread Z-score computed | Finite values |
| Pairs backtest completes | >= 10 trades |
| Pairs Sharpe vs Rules-First corr | < 0.3 (orthogonal alpha) |

---

## Phase V: Strategy Architecture v2 — Execution Plan

### V1: Short-Side Activation (`src/strategies/rules_first_strategy.py`)

Add `use_short: bool = True` toggle. Bearish patterns (Head & Shoulders, Double Top, Bearish Engulfing, Bearish Harmonic) trigger `self.sell()`. Default: short via `self.sell(size=size)` with same ATR trailing stop logic but inverted.

Gate: Test on SPY 2022 bear market. If short trades fire on the way down, V1 passes.

### V2: Strategy-Aware Position Sizing (`src/risk/strategy_aware_sizing.py`)

Kelly-derived sizing per strategy type:

```python
STRATEGY_KELLY = {
    "trend_following": 0.5,    # Half-Kelly — reliable in trends
    "mean_reversion": 0.25,    # Quarter-Kelly — fragile
    "pairs_trading": 0.5,      # Half-Kelly — market-neutral
    "ml_signal": 0.25,         # Quarter-Kelly — overconfidence risk
}
```

Computes Kelly fraction from rolling win rate and avg_win/avg_loss.

### V3: Correlation-Aware Portfolio (`src/portfolio/multi_asset_allocator.py`)

1. Compute NxN correlation from rolling 60d returns
2. Cluster correlated instruments (r > 0.7)
3. Equal weight to clusters (not individual instruments)
4. Within cluster: signal-strength allocation

Prevents triple-bet on tech (SPY+QQQ+XLK all go same direction).

### V4: Crypto Expansion — Pluggable Data Provider Architecture

**Design: `CryptoDataProvider` Protocol**

```python
class CryptoDataProvider(Protocol):
    """Protocol for crypto OHLCV data sources."""
    def fetch_ohlcv(
        self,
        symbol: str,       # e.g. "BTC/USDT"
        timeframe: str,     # "1d", "4h", "1h"
        since: str | None,
        limit: int = 1000,
    ) -> pd.DataFrame: ...
    def available_symbols(self) -> list[str]: ...
```

**Default implementation: `CCXTCryptoProvider`** — wraps CCXT, fetches from Binance (no API key for public data). Falls back to CSV files in `data/raw/{symbol}_daily.csv`.

**Install now (free):**
```bash
uv add ccxt
```

Then add `--asset-class crypto` flag to existing scripts. Crypto tickers formatted as `BTC_USD`, `ETH_USD`, etc. matching our existing convention.

### V4 Crypto Strategy Differences

| Aspect | Equity | Crypto |
|--------|--------|--------|
| Trading hours | 0930-1600 ET | 24/7 |
| ATR multiplier for stops | 1.5-3.0x | 2.5-5.0x |
| Pattern reliability | Literature-validated | Unknown (uncharted) |
| Commission | 0.1% | 0.1-0.2% (exchange-dependent) |
| Slippage | 0.01% | 0.05-0.15% |

---

## Architecture: Unified Data Provider Pattern

Every external data source follows the same pluggable protocol pattern we already use:

```
src/
  data_ingestion/
    providers/
      __init__.py           # Protocol exports
      _protocols.py          # PriceDataProvider, NewsDataProvider, FundamentalDataProvider, CryptoDataProvider
      yfinance_provider.py   # yfinance (free) — already working
      ccxt_provider.py       # CCXT crypto (free) — to build
      sec_edgar_provider.py  # SEC EDGAR text (free) — Phase R built scraper
      fmp_provider.py        # FMP (free tier first) — placeholder
      tiingo_provider.py     # Tiingo ($30/mo) — placeholder
```

Each provider implements its protocol. The system picks the best available at runtime:

```python
def get_price_provider(symbol: str) -> PriceDataProvider:
    # Try: Tiingo (if API key set) > CSV/yfinance (always available)
    if TII NGO_API_KEY and symbol not in CRYPTO_SYMBOLS:
        return TiingoProvider()
    if symbol in CRYPTO_SYMBOLS:
        return CCXTProvider()
    return CSVProvider(f"data/raw/{symbol}_daily.csv")
```

No provider? Fallback gracefully with a clear log message and degraded functionality — never crash.

---

## Immediate Actions (This Session)

### Do Now (zero new deps):
1. **Phase S1**: Build `src/strategies/pairs_trading_strategy.py` — full pipeline
2. **Phase S2**: Pre-built sector pairs from existing `data/raw/` tickers
3. **Phase S3**: `scripts/backtest_pairs.py` — CLI with `--compare`, `--auto-pairs`

### Install Once (free):
4. `uv add ccxt` — crypto OHLCV data
5. `uv add statsmodels` — already installed, cointegration test confirmed

### Phase V follows Phase S:
6. **V1**: Short-side activation in rules_first_strategy.py
7. **V2**: Strategy-aware Kelly sizing
8. **V3**: Correlation-aware multi-asset allocator
9. **V4**: CCXT crypto provider + `--asset-class crypto` flag

### Defer (until subscription):
- `src/data_ingestion/providers/tiingo_provider.py` — placeholder, implements protocol
- `src/data_ingestion/providers/fmp_provider.py` — placeholder, implements protocol
- Real news text pipeline for FinBERT (Tiingo News API)
- Point-in-time quarterly fundamentals (FMP Starter)

---

## Cost Summary

| When | What | Cost |
|------|------|------|
| **Now** | Phase S + V (build) | $0 |
| **Now** | CCXT crypto data | $0 |
| **Anytime** | FMP Free tier (fundamentals) | $0 (250 req/day) |
| **When ready** | Tiingo individual (prices + news) | $30/mo |
| **If needed** | FMP Starter (quarterly statements) | $22/mo |
| **Deferred** | Polygon.io (real-time) | $29/mo+ |
| **Deferred** | Alternative data (Phase U) | $100+/mo |

**Recommended path:** Build Phase S+V now ($0). When ready for real news text + better fundamentals, get Tiingo at $30/mo. Total cost to production-readiness: $0 now, $0-30/mo later.
