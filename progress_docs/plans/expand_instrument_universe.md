# Expand Instrument Universe with Cross-Asset Features — Plan

**Created:** 2026-05-09
**Prerequisite:** Cross-asset feature experiment on 6 instruments (completed)
**Estimated effort:** 1-2 AI sessions

---

## 1. Rationale

The 6-instrument experiment showed cross-asset features help most for smaller/less-efficient stocks:
- JOE (small-cap real estate): +0.15 AUC
- KODK (mid-cap chemicals): +0.07 AUC

Large efficient ETFs (SPY, QQQ) showed little benefit — they ARE the market.

This pattern suggests cross-asset features unlock alpha in instruments where the market beta alone explains less of the return. The question: does this generalize?

## 2. Candidate Instruments

### Already have data (in `data/raw/`):
- None remaining — all 6 instruments tested

### Download from Yahoo Finance:

| Ticker | Name | Type | Why interesting |
|--------|------|------|----------------|
| **Small/Mid Cap** | | | |
| SF | Stifel Financial | Mid-cap financial | Like CRVL but different sub-sector |
| OMF | OneMain Holdings | Mid-cap consumer finance | Different business model from banks |
| MTG | MGIC Investment | Mid-cap insurance | Interest-rate sensitive like HIFS |
| **Sector ETFs** | | | |
| XLB | Materials Select Sector | Sector ETF | Commodity/cyclical exposure |
| XLI | Industrial Select Sector | Sector ETF | Economic cycle proxy |
| XLP | Consumer Staples | Sector ETF | Defensive sector |
| XLY | Consumer Discretionary | Sector ETF | Cyclical sector |
| XLU | Utilities Select Sector | Sector ETF | Bond-proxy sector |
| **International** | | | |
| EFA | iShares MSCI EAFE | Developed international | Non-US exposure |
| VWO | Vanguard FTSE Emerging | Emerging markets | High beta, different drivers |

### Download command:
```bash
uv run python -c "
import yfinance as yf
for sym in ['SF','OMF','MTG','XLB','XLI','XLP','XLY','XLU','EFA','VWO']:
    df = yf.download(sym, start='2015-01-01', end='2025-12-31', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(f'data/raw/{sym}_daily.csv')
    print(f'{sym}: {len(df)} bars')
"
```

## 3. Experiment Design

### Phase 1: Train all new instruments
```bash
for sym in SF OMF MTG XLB XLI XLP XLY XLU EFA VWO; do
    uv run scripts/train_ml_model_v2.py --symbol data/raw/${sym}_daily.csv \
        --horizon 5 --suffix v3_ca
    uv run scripts/train_ml_model_v2.py --symbol data/raw/${sym}_daily.csv \
        --horizon 5 --suffix v3_baseline --no-cross-asset
done
```

### Phase 2: Analyze patterns
- Do small/mid-cap stocks consistently benefit from cross-asset features?
- Do sector ETFs show similar patterns to XLF/XLK (already tested)?
- Is there a relationship between market cap and cross-asset benefit?

### Phase 3: Meta-analysis across all 16 instruments
- Group by: market cap, sector, ETF vs individual stock
- Report AUC delta by group
- Identify characteristics of instruments that benefit most

## 4. What to Measure

| Instrument | NoCA AUC | CA AUC | AUC Δ | Gap Δ | Cross-Asset in Top 10 |
|-----------|----------|--------|-------|-------|----------------------|
| SF | ? | ? | ? | ? | ? |
| OMF | ? | ? | ? | ? | ? |
| ... | | | | | |

### Aggregate patterns to look for:
- **Market cap effect:** Smaller stocks → larger cross-asset benefit
- **Sector effect:** Financial/cyclical sectors → more cross-asset benefit than defensive
- **ETF vs stock:** ETFs already capture market beta → less cross-asset benefit

## 5. Success Criteria

- At least 3 new instruments show AUC improvement > 0.03
- Pattern generalizes: same instrument types benefit across the board
- No new instrument shows Scenario D (worse) without an explainable reason

## 6. Risks

- **Data quality:** Some tickers may have less history or survivorship bias
- **Overfitting to 6 instruments:** The pattern observed on 6 may not generalize
- **Time cost:** 10 instruments × 2 modes = 20 training runs (~10 minutes)

## 7. Next Step After This

If pattern generalizes → we have a robust finding: cross-asset features are a net positive, especially for small/mid-cap stocks. Proceed to building a production pipeline with per-instrument cross-asset configuration.

If pattern fails to generalize → the original 6 may have been cherry-picked lucky. Reduce scope to only JOE and KODK.
