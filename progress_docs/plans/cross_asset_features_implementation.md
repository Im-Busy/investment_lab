# Cross-Asset Feature Enrichment — Implementation Plan

**Created:** 2026-05-09
**Hypothesis:** H14 — Cross-asset features improve instrument-specific alpha detection by providing market context
**Prerequisite:** V2 overfitting fixes (completed)
**Estimated effort:** 3-5 AI sessions worth of work

---

## 1. Rationale

### 1.1 What We Learned from V2 Retraining

All 6 instruments converged on the same feature family: `volatility_50`, `rsi_10`, `std_return_100`, `price_to_ma_10`. The model learned one signal — "markets that are trending continue trending" — and applied it identically to every instrument. This is the momentum premium, not instrument-specific alpha.

Without market context, the model cannot distinguish between:
- A CRVL drop caused by broad financial sector rotation (mean-reverts → profitable fade)
- A CRVL drop caused by firm-specific bad news (trends → profitable momentum)

Both scenarios look identical from CRVL's own price history alone.

### 1.2 What Cross-Asset Features Provide

Cross-asset features answer "why is this happening?" by situating each instrument in its market context. They enable the model to learn conditional predictions: "CRVL's 5-day return when SPY is above its 50-day MA and XLF is underperforming vs when both are rising."

The key mechanism is that **market beta explains ~30-40% of any single stock's daily return**. By providing the model with the market component, we force it to learn on the residual — instrument-specific alpha — rather than memorizing the market trend.

### 1.3 Anti-Leakage Constraint (CRITICAL)

Every cross-asset feature must be computable using **only information available at timestamp t**. This means:

- **CORRECT**: `SPY_return_5d` = SPY's return from `t-5` to `t` (lookback, not forward)
- **WRONG**: `SPY_return_5d` = SPY's return from `t` to `t+5` (uses future SPY data to predict a stock's future return)

The "Time Travel is Cheating" paper (2505.11065v2) explicitly documents that leakage through temporal misalignment is the #1 source of inflated backtest results in financial ML. The Sasse et al. 2025 leakage survey confirms this.

Implementation rule: **all rolling computations use `.shift(1)` before the window** so the last data point in the window is `t-1`, never `t`.

---

## 2. Data Acquisition

### 2.1 Required Market Data

| Ticker | Role | Source | Already exists? |
|---|---|---|---|
| SPY | Broad equity market benchmark | `yfinance` or `data/raw/SPY_daily.csv` | Yes |
| QQQ | Tech/growth equity benchmark | `data/raw/QQQ_daily.csv` | Yes |
| TLT | Bond/rate proxy (inverse to rates) | `data/raw/TLT_daily.csv` | Yes |
| GLD | Gold / risk-off proxy | `data/raw/GLD_daily.csv` | Yes |
| BTC-USD | Crypto / risk appetite extreme | `data/raw/BTC_USD_daily.csv` | Yes |
| IWM | Small-cap equity | `yfinance` download | No — download |
| XLF | Financial sector ETF | `yfinance` download | No — download |
| XLE | Energy sector ETF | `yfinance` download | No — download |
| XLK | Technology sector ETF | `yfinance` download | No — download |
| XLV | Healthcare sector ETF | `yfinance` download | No — download |
| EEM | Emerging markets | `yfinance` download | No — download |

**Download command template** (run once to populate `data/raw/`):
```bash
uv run python -c "
import yfinance as yf
for sym in ['IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'EEM']:
    df = yf.download(sym, start='2015-01-01', end='2025-12-31', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    path = f'data/raw/{sym}_daily.csv'
    df.to_csv(path)
    print(f'Saved {path}: {len(df)} bars')
"
```

### 2.2 Data Alignment

All cross-asset DataFrames must share the same `DatetimeIndex` with the instrument DataFrame. The market was not open for 2515 days — SPY has ~2515 bars. Individual stocks may have fewer bars if they were listed later.

**Alignment strategy**: After loading all market data, `pd.concat` on index and forward-fill any missing values (rare, only for individual stock trading halts), then reindex to the instrument's index and drop any NaN rows.

### 2.3 Date Range

Use `2015-01-01` to `2025-12-31` for the initial data pull. The training pipeline's `--start` / `--end` flags handle subsetting. This gives us the same date range used in V2 for comparability.

---

## 3. Feature Design

### 3.1 Category 1: Relative Returns (6 features)

Captures whether this instrument is outperforming benchmarks.

```
relative_return_SPY_5d    = instrument_return_5d - SPY_return_5d
relative_return_SPY_20d   = instrument_return_20d - SPY_return_20d
relative_return_QQQ_5d    = instrument_return_5d - QQQ_return_5d
relative_return_QQQ_20d   = instrument_return_20d - QQQ_return_20d
relative_return_TLT_5d    = instrument_return_5d - TLT_return_5d
relative_return_IWM_5d    = instrument_return_5d - IWM_return_5d
```

**Leakage check**: Returns use windows ending at `t`, shifted by 1 so the last close in the window is `t-1`. No forward information.

### 3.2 Category 2: Rolling Beta (4 features)

Captures systematic exposure to market risk. High-beta instruments amplify market moves; low-beta instruments have more idiosyncratic return.

```
beta_SPY_60d   = Cov(instrument, SPY) / Var(SPY) over rolling 60-day window
beta_SPY_20d   = Same over 20-day window
beta_QQQ_60d   = Cov(instrument, QQQ) / Var(QQQ) over 60-day window
beta_QQQ_20d   = Same over 20-day window
```

Computed as: `rolling(60).apply(lambda x: np.polyfit(market[x.index], instr[x.index], 1)[0])`. This is computationally expensive but only done once per instrument. Cache results as a CSV to avoid recomputation.

### 3.3 Category 3: Residual Volatility (2 features)

Volatility of the component the market CANNOT explain. High residual vol = more instrument-specific noise = potentially more alpha (or more risk).

```
residual_vol_SPY_60d  = std(instrument_return - beta_SPY * SPY_return) over 60d
residual_vol_SPY_20d  = Same over 20d
```

### 3.4 Category 4: Market Correlation (2 features)

How closely the instrument tracks the market right now. Low correlation = decoupled = instrument is moving on its own.
```
corr_SPY_60d  = rolling Pearson r(instrument daily returns, SPY daily returns) over 60d
corr_SPY_20d  = Same over 20d
```

### 3.5 Category 5: Market Regime (5 features)

Describes the state of the broad market at the time of prediction.

```
SPY_trend_50d    = SPY price / SPY MA(50) - 1  (above/below trend)
SPY_trend_200d   = SPY price / SPY MA(200) - 1
SPY_vol_percentile_60d = rank(SPY current vol among past 60 days)  (0-1)
VIX_proxy         = SPY atr_20 / SPY_close  (inverse VIX approximation)
market_breadth    = (SPY > SPY_MA_50 for past 5 days).mean()  (0-1, how many days above MA)
```

### 3.6 Category 6: Sector Rotation (3 features)

Captures money flowing between sectors. This helps the model understand whether a stock's move is sector-wide or idiosyncratic.

```
XLF_relative_5d   = XLF_return_5d - SPY_return_5d
XLK_relative_5d   = XLK_return_5d - SPY_return_5d
sector_spread_5d  = XLF_return_5d - XLK_return_5d  (value vs growth rotation)
```

Since individual stocks don't all belong to XLF/XLK, these are noisy but informative — the model can learn to weight them appropriately.

### 3.7 Category 7: Cross-Asset Correlation Matrix (2 features)

Captures broad market stress. When all assets become correlated (crisis), alpha generation is harder.

```
avg_cross_corr_20d   = mean of all pairwise correlations between SPY, QQQ, TLT, GLD returns over 20d
corr_dispersion_20d  = std of same pairwise correlations
```

### 3.8 Feature Count Summary

| Category | Count |
|---|---|
| Relative returns | 6 |
| Rolling beta | 4 |
| Residual volatility | 2 |
| Market correlation | 2 |
| Market regime | 5 |
| Sector rotation | 3 |
| Cross-asset correlation | 2 |
| **Total new features** | **24** |
| Existing features (after IC filter) | ~54-93 |
| **Total after merge** | ~78-117 |

---

## 4. Code Architecture

### 4.1 New Module: `src/ml/cross_asset_features.py`

This module is standalone — it operates on instrument + market data and produces a feature DataFrame. It does NOT modify `FeatureExtractor` directly. This keeps concerns separate and allows independent testing.

```python
# src/ml/cross_asset_features.py

"""
Cross-Asset Feature Extraction for ML Pipeline.

Adds market-context features to instrument-level features.
All features are backward-looking only — no future information leakage.

Usage:
    from src.ml.cross_asset_features import CrossAssetFeatureExtractor

    extractor = CrossAssetFeatureExtractor(market_data=market_dict)
    cross_features = extractor.extract(instrument_df)
    combined = pd.concat([instrument_features, cross_features], axis=1)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class CrossAssetFeatureExtractor:
    """Extract cross-asset features for a single instrument.

    market_data is a dict of {ticker: OHLCV_DataFrame} keyed by ticker
    (e.g. {"SPY": spy_df, "QQQ": qqq_df}). All DataFrames must have
    compatible DatetimeIndex with the instrument DataFrame.
    """

    def __init__(self, market_data: Dict[str, pd.DataFrame]) -> None:
        self._market = market_data
        self._validate_market_data()

    def _validate_market_data(self) -> None:
        for ticker, df in self._market.items():
            if "Close" not in df.columns:
                raise ValueError(f"Market data for {ticker} missing 'Close' column")

    def extract(self, instrument_df: pd.DataFrame) -> pd.DataFrame:
        """Extract all cross-asset features for the instrument.

        Args:
            instrument_df: OHLCV DataFrame for the target instrument.

        Returns:
            DataFrame indexed by instrument_df.index with cross-asset features.
        """
        features = pd.DataFrame(index=instrument_df.index)
        close = instrument_df["Close"]
        returns = close.pct_change()

        features = self._add_relative_returns(features, returns)
        features = self._add_rolling_beta(features, returns)
        features = self._add_residual_vol(features, returns)
        features = self._add_market_correlation(features, returns)
        features = self._add_market_regime(features)
        features = self._add_sector_rotation(features)
        features = self._add_cross_asset_corr(features)

        return features
```

### 4.2 Detailed Implementation of Each Method

#### `_add_relative_returns`

```python
def _add_relative_returns(
    self, features: pd.DataFrame, returns: pd.Series
) -> pd.DataFrame:
    windows = [5, 20]
    benchmarks = ["SPY", "QQQ", "TLT", "IWM"]

    for bench in benchmarks:
        if bench not in self._market:
            continue
        bench_ret = self._market[bench]["Close"].pct_change()
        for w in windows:
            inst_cum = returns.rolling(w).sum().shift(1)
            bench_cum = bench_ret.rolling(w).sum().shift(1)
            features[f"rel_ret_{bench}_{w}d"] = inst_cum - bench_cum
            # .shift(1) ensures last data point in window is t-1, not t

    return features
```

**IMPORTANT**: The `.shift(1)` after `rolling().sum()` is the key leakage prevention. Without it, the window `[t-w+1, t]` includes the very bar we're trying to predict. With it, the window is `[t-w, t-1]`.

#### `_add_rolling_beta`

```python
def _add_rolling_beta(
    self, features: pd.DataFrame, returns: pd.Series
) -> pd.DataFrame:
    windows = [20, 60]

    for bench in ["SPY", "QQQ"]:
        if bench not in self._market:
            continue
        bench_ret = self._market[bench]["Close"].pct_change()
        for w in windows:
            # Efficient rolling beta using running sums
            def _rolling_beta(instr, mkt, window):
                cov = instr.rolling(window).cov(mkt)
                var = mkt.rolling(window).var()
                return (cov / var).shift(1)

            beta = _rolling_beta(returns, bench_ret, w)
            features[f"beta_{bench}_{w}d"] = beta

    return features
```

#### `_add_residual_vol`

```python
def _add_residual_vol(
    self, features: pd.DataFrame, returns: pd.Series
) -> pd.DataFrame:
    if "SPY" not in self._market:
        return features

    spy_ret = self._market["SPY"]["Close"].pct_change()

    for w in [20, 60]:
        # Compute rolling beta first
        cov = returns.rolling(w).cov(spy_ret)
        var = spy_ret.rolling(w).var()
        beta = cov / var

        # Residual = instrument return - beta * market return
        residual = returns - beta * spy_ret

        # Volatility of residual, shifted
        resid_vol = residual.rolling(w).std().shift(1)
        features[f"resid_vol_SPY_{w}d"] = resid_vol

    return features
```

#### `_add_market_correlation`

```python
def _add_market_correlation(
    self, features: pd.DataFrame, returns: pd.Series
) -> pd.DataFrame:
    if "SPY" not in self._market:
        return features

    spy_ret = self._market["SPY"]["Close"].pct_change()

    for w in [20, 60]:
        corr = returns.rolling(w).corr(spy_ret).shift(1)
        features[f"corr_SPY_{w}d"] = corr

    return features
```

#### `_add_market_regime`

```python
def _add_market_regime(self, features: pd.DataFrame) -> pd.DataFrame:
    if "SPY" not in self._market:
        return features

    spy_close = self._market["SPY"]["Close"]
    spy_ret = spy_close.pct_change()

    # Trend indicators
    ma_50 = spy_close.rolling(50).mean()
    ma_200 = spy_close.rolling(200).mean()
    features["SPY_trend_50d"] = (spy_close / ma_50 - 1).shift(1)
    features["SPY_trend_200d"] = (spy_close / ma_200 - 1).shift(1)

    # Volatility regime
    vol_20 = spy_ret.rolling(20).std()
    vol_rank = vol_20.rolling(60).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean()
    ).shift(1)
    features["SPY_vol_percentile_60d"] = vol_rank

    # VIX proxy (SPY ATR / SPY close)
    if "High" in self._market["SPY"] and "Low" in self._market["SPY"]:
        spy_high = self._market["SPY"]["High"]
        spy_low = self._market["SPY"]["Low"]
        tr = pd.concat([
            spy_high - spy_low,
            (spy_high - spy_close.shift(1)).abs(),
            (spy_low - spy_close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_20 = tr.rolling(20).mean()
        features["SPY_atr_ratio"] = (atr_20 / spy_close).shift(1)

    # Breadth
    above_ma = (spy_close > ma_50).astype(float)
    features["SPY_breadth_5d"] = above_ma.rolling(5).mean().shift(1)

    return features
```

#### `_add_sector_rotation`

```python
def _add_sector_rotation(self, features: pd.DataFrame) -> pd.DataFrame:
    if "SPY" not in self._market:
        return features

    spy_ret = self._market["SPY"]["Close"].pct_change()

    for sector in ["XLF", "XLK"]:
        if sector not in self._market:
            continue
        sector_ret = self._market[sector]["Close"].pct_change()
        relative = (sector_ret.rolling(5).sum() - spy_ret.rolling(5).sum()).shift(1)
        features[f"{sector}_relative_5d"] = relative

    # Value vs growth spread
    if "XLF" in self._market and "XLK" in self._market:
        xlf_ret = self._market["XLF"]["Close"].pct_change().rolling(5).sum()
        xlk_ret = self._market["XLK"]["Close"].pct_change().rolling(5).sum()
        features["value_growth_spread_5d"] = (xlf_ret - xlk_ret).shift(1)

    return features
```

#### `_add_cross_asset_corr`

```python
def _add_cross_asset_corr(self, features: pd.DataFrame) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    available = [t for t in tickers if t in self._market]
    if len(available) < 2:
        return features

    returns_dict = {
        t: self._market[t]["Close"].pct_change() for t in available
    }
    returns_df = pd.DataFrame(returns_dict)

    w = 20
    # Compute pairwise correlations (upper triangle) and average
    n = len(available)
    corr_series_list = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_corr = returns_df.iloc[:, i].rolling(w).corr(returns_df.iloc[:, j])
            corr_series_list.append(pair_corr)

    if corr_series_list:
        avg_corr = pd.concat(corr_series_list, axis=1).mean(axis=1)
        disp_corr = pd.concat(corr_series_list, axis=1).std(axis=1)
        features["avg_cross_corr_20d"] = avg_corr.shift(1)
        features["corr_dispersion_20d"] = disp_corr.shift(1)

    return features
```

### 4.3 Integration into `train_ml_model_v2.py`

The v2 pipeline needs one new function and a modified `_run_training` flow:

```python
def load_market_data(
    instrument_df: pd.DataFrame,
    market_tickers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load market data aligned to instrument's index.

    Args:
        instrument_df: The instrument's OHLCV DataFrame (for index alignment).
        market_tickers: List of tickers to load. Defaults to SPY, QQQ, TLT, GLD.

    Returns:
        Dict of {ticker: DataFrame} aligned to instrument_df.index.
    """
    if market_tickers is None:
        market_tickers = ["SPY", "QQQ", "TLT", "GLD"]

    market_data = {}
    for ticker in market_tickers:
        path = Path(f"data/raw/{ticker}_daily.csv")
        if path.exists():
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            # Align to instrument index
            aligned = df.reindex(instrument_df.index, method="ffill")
            # Drop rows where we have instrument but no market data
            aligned = aligned.dropna(subset=["Close"])
            market_data[ticker] = aligned
        else:
            logger.warning(f"Market data not found for {ticker}, skipping")

    return market_data
```

**Integration point in `_run_training`** (in `scripts/train_ml_model_v2.py`):

After feature extraction (~line 509 in original), before IC filtering:

```python
# New section 2b: Cross-Asset Feature Extraction
if not args.no_cross_asset:
    logger.info("=" * 60)
    logger.info("2b. Cross-Asset Feature Extraction")
    logger.info("=" * 60)
    market_data = load_market_data(df)
    ca_extractor = CrossAssetFeatureExtractor(market_data=market_data)
    cross_features = ca_extractor.extract(df)

    # Merge with instrument features
    features = features.join(cross_features, how="inner")
    logger.info(f"Added {cross_features.shape[1]} cross-asset features")
    logger.info(f"Total features: {features.shape[1]}")
```

### 4.4 New CLI Flag

Add to `train_ml_model_v2.py`'s `argparse`:

```python
parser.add_argument(
    "--no-cross-asset",
    action="store_true",
    help="Skip cross-asset feature extraction (for baseline comparison)",
)
```

This allows running with and without cross-asset features using the same script:

```bash
# Without cross-asset (current baseline)
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv \
    --horizon 5 --suffix baseline_no_ca --no-cross-asset

# With cross-asset
uv run scripts/train_ml_model_v2.py --symbol data/raw/CRVL_daily.csv \
    --horizon 5 --suffix with_ca_features
```

---

## 5. Experiment Design

### 5.1 What to Train

Train all 6 instruments (SPY, QQQ, CRVL, KODK, HIFS, JOE) with cross-asset features enabled. Compare against the V2 baseline we just produced.

### 5.2 What to Measure

| Metric | V2 Baseline | V2 + Cross-Asset | Interpretation |
|---|---|---|---|
| Train AUC | 0.76-0.87 | Expected: lower (harder to memorize) | If unchanged, features add no distinguishable signal |
| Test AUC | 0.47-0.59 | Expected: 0.55+ for all | If CRVL/QQQ improve and KODK/JOE don't, cross-asset features help efficient instruments more |
| Overfit Gap | 0.17-0.38 | Expected: < 0.15 for 3+ instruments | If gap drops but test AUC unchanged, features reduce memorization |
| CV AUC Std | 0.008-0.028 | Expected: similar or slightly higher | More features → more variance per fold |
| Top Features | All volatility/momentum | Expected: cross-asset features appear in top 10 | If they don't appear, the model is ignoring them |

### 5.3 What to Look For

**Scenario A — Strong improvement (best case):**
- 2-3 instruments show Test AUC > 0.60 AND overfit gap < 0.15
- Cross-asset features appear in top 10 importance
- These instruments have learnable alpha once market context is provided
- **Next step**: Expand to full instrument universe, try sector-specific features

**Scenario B — Gap reduction without AUC improvement:**
- Overfit gap drops to < 0.15 for more instruments
- Test AUC stays similar or drops slightly
- Cross-asset features absorb the redundant volatility/momentum signal
- **Next step**: The model is now honest but the signal is still weak → try meta-labeling

**Scenario C — No change:**
- Same results as baseline
- Cross-asset features don't appear in top 10 importance
- **Next step**: The information in market context is already captured by instrument-level features (e.g., `price_to_ma_50` already encodes SPY trend implicitly)

**Scenario D — Worse:**
- Test AUC drops (cross-asset features add noise)
- **Next step**: Reduce to only SPY and one sector ETF; the full set may overwhelm the model

---

## 6. Implementation Sequence

### Phase A: Market Data Download (5 min)
1. Run the `yfinance` download script for IWM, XLF, XLE, XLK, XLV, EEM
2. Verify all files have consistent date ranges with SPY_daily.csv
3. Check for any NaN gaps > 5 days

### Phase B: Create `cross_asset_features.py` (20-30 min)
1. Write the `CrossAssetFeatureExtractor` class with all 7 `_add_*` methods
2. Write standalone test: load SPY, compute cross-asset features for CRVL, verify no NaN columns, verify all features are backward-looking
3. Spot-check: for a date where SPY dropped 3% in 5 days, `SPY_trend_50d` should be negative

### Phase C: Integrate into `train_ml_model_v2.py` (10 min)
1. Add `load_market_data()` function
2. Add feature merge in `_run_training()`
3. Add `--no-cross-asset` CLI flag
4. Verify the pipeline runs end-to-end on CRVL with cross-asset features

### Phase D: Run Full Experiment (60-90 min)
1. Train all 6 instruments with cross-asset features enabled
2. Train all 6 instruments with `--no-cross-asset` for comparison
3. Compare results in a table

### Phase E: Analysis and Next Steps (15 min)
1. Classify each instrument into Scenarios A-D
2. Determine next action based on classification

---

## 7. Edge Cases and Risks

### 7.1 Data Alignment Failures
- **Risk**: Instrument has bars on days when market was closed (e.g., foreign stocks)
- **Fix**: `reindex` with `method="ffill"` fills missing market data from prior day
- **Detection**: Any row where `market_data[ticker].isna().any()` after alignment

### 7.2 Sector ETF Data Limits
- **Risk**: XLF, XLK etc. may not have data back to 2015 (some ETFs launched later)
- **Fix**: Check start date after download; if < 2015, use SPY/QQQ only and disable sector features
- **Detection**: `min(market_data[ticker].index)` < `instrument_df.index.min()`

### 7.3 Feature Explosion
- **Risk**: 24 new features could push total well over 100, increasing risk of spurious correlation
- **Mitigation**: IC filtering (already in v2 pipeline) will remove low-IC cross-asset features automatically
- **Check**: After IC filtering, `n_features` should be < 120 regardless

### 7.4 Computational Cost
- **Risk**: Rolling beta with `.apply(np.polyfit)` is O(n * window) per feature, could be slow for large datasets
- **Mitigation**: Use the running covariance/variance formula instead of `np.polyfit` (as shown in code above)
- **Benchmark**: 2500 bars × 4 beta features should complete in < 2 seconds

### 7.5 Sector Applicability
- **Risk**: XLF features are relevant for CRVL (financial analytics) but meaningless for KODK (chemicals)
- **Mitigation**: The model will learn to weight sector features differently per instrument via feature importance. The IC filter will also drop sector features that have no correlation with the instrument's returns.

---

## 8. Success Criteria

The hypothesis is supported if **any** of the following hold:

1. At least 2 instruments show Test AUC ≥ 0.60 with overfit gap ≤ 0.15 (clear alpha signal)
2. All 6 instruments show overfit gap ≤ 0.15 with cross-asset features appearing in top 10 importance for at least 3 instruments (features explain variance)
3. At least 4 instruments show improvement in Test AUC compared to baseline (directional improvement)

The hypothesis is rejected if:

- No instrument shows Test AUC improvement > 0.02 over baseline
- No cross-asset feature appears in top 10 importance for any instrument

---

## 9. Files to Create/Modify

| File | Action | Lines |
|---|---|---|
| `src/ml/cross_asset_features.py` | **Create** | ~250 lines |
| `scripts/train_ml_model_v2.py` | Modify: add `load_market_data()`, cross-asset merge, `--no-cross-asset` flag | +~50 lines |
| `data/raw/IWM_daily.csv` | Download | N/A |
| `data/raw/XLF_daily.csv` | Download | N/A |
| `data/raw/XLE_daily.csv` | Download | N/A |
| `data/raw/XLK_daily.csv` | Download | N/A |
| `data/raw/XLV_daily.csv` | Download | N/A |
| `data/raw/EEM_daily.csv` | Download | N/A |
| `plans/cross_asset_features_implementation.md` | This file | N/A |
