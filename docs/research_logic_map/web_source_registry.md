# Web Source Registry

> Counterpart to `insight_registry.md` for non-paper sources: blogs, tutorials, documentation,
> forum posts, YouTube, reference implementations, and other web content.

**Last Updated:** 2026-05-27
**Total Sources:** 0 (being populated)

---

## Tag Legend

### Topics
- **Risk**: Risk management, position sizing, drawdown control
- **Friction**: Transaction costs, turnover, slippage
- **Regime**: Market regime detection, adaptive strategies
- **Signal.Quality**: Signal aggregation, confluence scoring, filtering
- **Pattern**: Chart pattern detection, recognition algorithms
- **Implementation**: Reference implementations, code patterns, API usage
- **Concept**: Theoretical concepts, mental models
- **Data**: Data sourcing, ingestion patterns
- **ICT**: Inner Circle Trader / SMC concepts

### Status
- `active` — Referenced in code; attribution is current
- `deprecated` — Code referencing it has been removed
- `orphaned` — Source exists in registry but has no code references (for audit)

---

## Sources

### W1: awesome-ai-in-finance
- **URL:** https://github.com/georgezouq/awesome-ai-in-finance
- **Type:** Curated repository index
- **Topic:** Implementation
- **Status:** active
- **Used by:**
  - `src/signals/fund_flow.py` — fund flow data feeds
  - `src/data/congressional_signals.py` — congressional trade tracking
  - `src/patterns/similarity_search.py` — chart library pattern
  - `src/patterns/patternity_wrapper.py` — patternity wrapper
  - `src/optimization/skfolio_optimizer.py` — skfolio optimizer
  - `src/nlp/adanos_sentiment.py` — Adanos sentiment
  - `src/ml/label_shuffling.py` — Contingency RNG

### W2: FinanceDatabase
- **URL:** https://github.com/JerBouma/FinanceDatabase
- **Type:** Python library
- **Topic:** Data
- **Status:** active
- **Used by:**
  - `src/data/symbol_filter.py` — GICS-categorized symbols
  - `src/data/historical_universe.py` — survivorship bias handling
  - `src/data/fundamental_pipe.py` — fundamental data + FinanceToolkit
  - `src/data/financedb_layer.py` — wrapper layer
  - `src/data/symbol_sync.py` — symbol sync
  - `src/cli/option_discovery.py` — query pattern

### W3: stock-sdk (chengzuopeng)
- **URL:** https://github.com/chengzuopeng/stock-sdk
- **Type:** Python library
- **Topic:** Data
- **Status:** active
- **Used by:**
  - `src/data/options_data.py` — CFFEX/SSE options
  - `src/data/futures_inventory.py` — futures inventory
  - `src/data/batch_loader.py` — API pattern
  - `src/mcp/stock_server.py` — MCP server pattern

### W4: ICT / SMC Online Resources
- **URL:** toaz.info ICT glossary, ICT YouTube tutorials
- **Type:** Educational content
- **Topic:** ICT
- **Status:** active
- **Used by:**
  - `src/indicators/power_of_3.py` — ICT MMXM Model, Power of 3
  - `src/indicators/ote.py` — ICT Optimal Trade Entry
  - `src/indicators/judas_swing.py` — ICT Day 9/16 tutorial
  - `src/indicators/cisd.py` — ICT Institutional SMC Trading
  - `src/indicators/crt.py` — ICT Candle Range Theory
  - `src/patterns/smc/pd_array_matrix.py` — toaz.info glossary (David Woods)
  - `src/signals/smc_divergence.py` — toaz.info glossary

### W5: Eigenvesting (srome.github.io)
- **URL:** https://srome.github.io/Eigenvesting-I/
- **Type:** Blog post
- **Topic:** Implementation, Concept
- **Status:** active
- **Used by:**
  - `src/portfolio/eiten_adapters/eigen_portfolio.py`

### W6: hudson-and-thames/meta-labeling
- **URL:** https://github.com/hudson-and-thames/meta-labeling
- **Type:** Reference implementation
- **Topic:** Implementation
- **Status:** active
- **Used by:**
  - `src/ml/meta_labeler_v2.py`

### W7: Fidelity Technical Analysis (website)
- **URL:** https://www.fidelity.com/learning-center/trading-investing/technical-analysis
- **Type:** Educational content
- **Topic:** Pattern
- **Status:** active
- **Used by:**
  - `src/patterns/candlestick/shooting_star.py`
  - `src/patterns/volatility/nr4_inside_bar.py`

### W8: U.S. TreasuryDirect
- **URL:** https://www.treasurydirect.gov/
- **Type:** Official data source
- **Topic:** Data
- **Status:** active
- **Used by:**
  - `src/signals/treasury_auctions.py`

### W9: eiten (tomgrek)
- **URL:** https://github.com/tomgrek/eiten
- **Type:** Reference implementation
- **Topic:** Implementation
- **Status:** active
- **Used by:**
  - `src/portfolio/eiten_adapters/__init__.py`
  - `src/portfolio/eiten_adapters/eigen_portfolio.py`
  - `src/portfolio/eiten_adapters/maximum_sharpe.py`
  - `src/portfolio/eiten_adapters/rmt_filtering.py`

---

## How to Add a New Web Source

1. Assign next available W-ID (W10, W11, ...)
2. Fill in URL, type, topic, and files using it
3. Add `Source: WXX` or `Reference: WXX` to the relevant code file(s)
4. Update `SOURCE_MANIFEST.md` with the same entry

## How to Remove a Source

1. Mark it `~~strikethrough~~` in `SOURCE_MANIFEST.md`
2. Mark status as `deprecated` here
3. Run `/sync-attributions` to purge from code
