"""
R6: Liquidity Factor (CEI) and other advanced factor features.

Acharya-Pedersen Capital Efficiency Index and related factor computations
for ML pipeline integration. Complements the 14 fundamental factors in
fundamental_features.py with price-derived factor features.

Source: Beyond Fama-French §2, 华泰多因子 §1.2

CEI Formula:
    CEI = (ME_t / ME_{t-252}) - 1 - (P_t / P_{t-252} - 1)
    where ME = price * shares_outstanding

CEI captures the 12-month change in market equity minus the 12-month
cumulative return. Positive CEI means market cap grew faster than price
alone (share issuance; for ETFs: net creations). Negative CEI means
market cap grew slower (buybacks/delistings).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CEI config for integration with FUNDAMENTAL_FACTOR_CONFIG pattern
CEI_FACTOR_CONFIG: dict[str, dict[str, str]] = {
    "cei": {"field": "cei", "category": "Liquidity", "direction": "higher"},
}

# Window for CEI computation (252 trading days = ~1 year)
CEI_WINDOW: int = 252


def _fetch_shares_yfinance(ticker: str) -> Optional[pd.Series]:
    """Fetch historical shares outstanding from yfinance.

    Tries get_shares_full() first (historical), then falls back to
    info['sharesOutstanding'] (current snapshot).

    Args:
        ticker: Ticker symbol.

    Returns:
        pd.Series indexed by date, or None if unavailable.
    """
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        shares = t.get_shares_full()
        if shares is not None and len(shares) > 0:
            shares = shares.sort_index()
            return shares
    except Exception:
        pass

    try:
        t = yf.Ticker(ticker)
        info = t.info
        so = info.get("sharesOutstanding")
        if so and so > 0:
            logger.debug(
                "yfinance: using current sharesOutstanding=%d for %s (no history)",
                int(so),
                ticker,
            )
            return pd.Series({pd.Timestamp.now().normalize(): float(so)})
    except Exception:
        pass

    return None


def _fetch_shares_fmp(
    ticker: str,
    api_key: Optional[str] = None,
) -> Optional[pd.Series]:
    """Fetch historical shares outstanding from FMP API.

    FMP endpoint: /historical-shares-outstanding
    May require premium subscription for historical data.

    Args:
        ticker: Ticker symbol.
        api_key: FMP API key. Reads FMP_API_KEY env var if None.

    Returns:
        pd.Series indexed by date, or None if unavailable.
    """
    import os
    import urllib.request
    import json

    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        return None

    url = (
        f"https://financialmodelingprep.com/stable/"
        f"historical-shares-outstanding?symbol={ticker}&apikey={key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")["sharesOutstanding"].sort_index()
        return df.astype(float)
    except Exception:
        return None


def compute_cei(
    close_prices: np.ndarray,
    shares_outstanding: Optional[np.ndarray] = None,
    window: int = CEI_WINDOW,
) -> np.ndarray:
    """Compute Acharya-Pedersen Capital Efficiency Index.

    CEI_t = (ME_t / ME_{t-window}) - 1 - (P_t / P_{t-window} - 1)

    If shares_outstanding is None (constant shares), CEI defaults to 0
    for all bars, reflecting no market cap change beyond price.

    Args:
        close_prices: (N,) array of close prices.
        shares_outstanding: (N,) array of shares outstanding, aligned
            with close_prices. If None, assumes constant shares.
        window: Lookback window in bars (default 252 for 1 year).

    Returns:
        (N,) array of CEI values. Values before bar ``window`` are NaN.
    """
    close = np.asarray(close_prices, dtype=np.float64)
    n = len(close)
    cei = np.full(n, np.nan, dtype=np.float64)

    if shares_outstanding is None:
        cei[window:] = 0.0
        return cei

    so = np.asarray(shares_outstanding, dtype=np.float64)
    if len(so) != n:
        raise ValueError(f"shares_outstanding length {len(so)} != close_prices length {n}")

    me = close * so
    for t in range(window, n):
        t0 = t - window
        if me[t0] <= 0 or close[t0] <= 0:
            continue
        me_change = me[t] / me[t0] - 1.0
        price_change = close[t] / close[t0] - 1.0
        cei[t] = me_change - price_change

    return cei


def compute_liquidity_factor(
    close_prices: np.ndarray,
    shares_outstanding: Optional[np.ndarray] = None,
    window: int = CEI_WINDOW,
) -> np.ndarray:
    """Convenience alias for compute_cei().

    Returns the same CEI array. Provided for API clarity when used
    alongside other factor feature functions.
    """
    return compute_cei(close_prices, shares_outstanding, window)


@lru_cache(maxsize=32)
def _load_shares_for_ticker(ticker: str) -> Optional[pd.Series]:
    """Cached loader for shares outstanding data.

    Tries yfinance first, then FMP if API key is available.
    Results are cached to avoid repeated API calls.

    Args:
        ticker: Ticker symbol.

    Returns:
        pd.Series of shares outstanding indexed by date, or None.
    """
    shares = _fetch_shares_yfinance(ticker)
    if shares is not None and len(shares) > 1:
        return shares
    shares = _fetch_shares_fmp(ticker)
    return shares


def align_shares_to_prices(
    shares_series: Optional[pd.Series],
    price_index: pd.DatetimeIndex,
) -> Optional[np.ndarray]:
    """Align historical shares outstanding to a price index.

    Forward-fills shares from the latest known date to each trading day.
    If shares_series is None or has only one entry, returns None
    (caller should treat as constant shares).

    Args:
        shares_series: Series of shares outstanding indexed by date.
        price_index: DatetimeIndex of trading days.

    Returns:
        (N,) array aligned to price_index, or None if insufficient data.
    """
    if shares_series is None or len(shares_series) < 2:
        return None

    reindexed = shares_series.reindex(price_index, method="ffill")
    if reindexed.isna().all():
        return None

    reindexed = reindexed.ffill().bfill()
    return reindexed.to_numpy(dtype=np.float64)


def compute_cei_dataframe(
    ohlcv: pd.DataFrame,
    ticker: str = "",
    window: int = CEI_WINDOW,
) -> pd.Series:
    """Compute CEI for a DataFrame of OHLCV data.

    Loads shares outstanding for the ticker, aligns to the price index,
    and computes CEI. If shares are unavailable, returns all zeros (CEI
    of 0 = constant shares assumption).

    Args:
        ohlcv: DataFrame with 'Close' column and DatetimeIndex.
        ticker: Ticker symbol for fetching shares data.
        window: Lookback window in bars.

    Returns:
        pd.Series of CEI values aligned with ohlcv index.
    """
    close = ohlcv["Close"].values
    index = ohlcv.index

    if ticker:
        shares_series = _load_shares_for_ticker(ticker)
        shares_aligned = align_shares_to_prices(shares_series, index)
    else:
        shares_aligned = None

    cei_array = compute_cei(close, shares_aligned, window)
    return pd.Series(cei_array, index=index, name="cei")


def compute_amihud_illiquidity(
    ohlcv: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """Compute Amihud (2002) illiquidity measure.

    ILLIQ = mean(|daily_return| / dollar_volume) over rolling window.
    Higher values = less liquid. A standard microstructure liquidity proxy.

    Args:
        ohlcv: DataFrame with 'Close', 'Volume' columns.
        window: Rolling window in bars.

    Returns:
        pd.Series of Amihud illiquidity values.
    """
    close = ohlcv["Close"].values
    volume = ohlcv["Volume"].values
    n = len(close)

    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0

    dollar_volume = close * volume
    daily_illiq = np.abs(returns) / np.maximum(dollar_volume, 1e-10)

    result = pd.Series(np.nan, index=ohlcv.index, dtype=np.float64)
    for i in range(window, n):
        result.iloc[i] = daily_illiq[i - window : i].mean()

    return result


def compute_roll_spread(
    ohlcv: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """Compute Roll (1984) effective bid-ask spread estimator.

    Roll = 2 * sqrt(-cov(daily_return_t, daily_return_{t-1}))
    Only computed when first-order autocovariance is negative.
    Higher values = wider spreads = less liquid.

    Args:
        ohlcv: DataFrame with 'Close' column.
        window: Rolling window in bars.

    Returns:
        pd.Series of Roll spread estimates.
    """
    close = ohlcv["Close"].values
    n = len(close)

    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0

    result = pd.Series(np.nan, index=ohlcv.index, dtype=np.float64)
    for i in range(window + 1, n):
        r_window = returns[i - window : i]
        if len(r_window) < 2:
            continue
        autocov = np.cov(r_window[:-1], r_window[1:])[0, 1]
        if autocov < 0:
            result.iloc[i] = 2.0 * np.sqrt(-autocov)
        else:
            result.iloc[i] = 0.0

    return result


def add_liquidity_features(
    ohlcv: pd.DataFrame,
    ticker: str = "",
) -> pd.DataFrame:
    """Add all liquidity factor features to an OHLCV DataFrame.

    Returns a new DataFrame with additional columns: cei, amihud_illiq,
    roll_spread. Original columns are preserved.

    Args:
        ohlcv: DataFrame with 'Close', 'Volume' columns.
        ticker: Ticker symbol for shares data (CEI computation).

    Returns:
        DataFrame with liquidity feature columns added.
    """
    result = ohlcv.copy()

    if "Close" in result.columns:
        cei = compute_cei_dataframe(result, ticker=ticker)
        result["cei"] = cei

        if "Volume" in result.columns:
            result["amihud_illiq"] = compute_amihud_illiquidity(result)
        result["roll_spread"] = compute_roll_spread(result)

    return result


# ── R11: Merton Distance-to-Default ──────────────────────────────────────────


_MERTON_DTD_T = 1.0  # 1 year horizon
_US_TREASURY_RATE = 0.045  # 4.5% risk-free rate (approximate 2025-2026 average)


def _estimate_asset_volatility(
    market_cap: float,
    total_debt: float,
    equity_volatility: float,
) -> float:
    """Estimate asset volatility from equity volatility using a simple
    Merton model approximation.

    σ_asset ≈ (E / (E + D)) * σ_equity

    This is a first-order approximation; full Merton requires iterative
    solution of simultaneous equations.  Acceptable for factor features
    where relative ranking matters more than absolute precision.

    Args:
        market_cap: Current market capitalization.
        total_debt: Total debt (book value).
        equity_volatility: Annualized equity return volatility.

    Returns:
        Annualized asset volatility estimate.
    """
    total_value = market_cap + total_debt
    if total_value <= 0 or market_cap <= 0:
        return equity_volatility
    equity_ratio = market_cap / total_value
    return equity_ratio * equity_volatility


def compute_dtd(
    market_cap: float,
    total_debt: float,
    equity_volatility: float,
    risk_free_rate: float = _US_TREASURY_RATE,
    horizon: float = _MERTON_DTD_T,
) -> float:
    """Compute Merton Distance-to-Default.

    DtD = (ln(V/D) + (r − σ²/2) · T) / (σ √T)

    where:
        V = market_cap + total_debt (asset value)
        D = total_debt
        r = risk-free rate
        σ = asset volatility
        T = time horizon in years

    Positive DtD = distance above default threshold (healthy).
    DtD < 0 = firm is insolvent (negative distance to default).
    DtD < 1.5-2.0 = elevated default risk (typical thresholds).

    Args:
        market_cap: Current market capitalization.
        total_debt: Total debt (book value).
        equity_volatility: Annualized equity return volatility.
        risk_free_rate: Annualized risk-free rate (default 4.5%).
        horizon: Time horizon in years (default 1.0).

    Returns:
        Distance-to-Default value. Positive = healthy, negative = insolvent.
    """
    if total_debt <= 0:
        return float("inf")
    if market_cap <= 0:
        return -float("inf")

    asset_value = market_cap + total_debt
    asset_vol = _estimate_asset_volatility(market_cap, total_debt, equity_volatility)

    if asset_vol <= 0:
        return float("inf") if asset_value > total_debt else -float("inf")

    sqrt_t = horizon**0.5
    numerator = np.log(asset_value / total_debt) + (risk_free_rate - asset_vol**2 / 2) * horizon
    denominator = asset_vol * sqrt_t

    return float(numerator / denominator)


def _fetch_fmp_metrics(
    ticker: str,
    api_key: Optional[str] = None,
) -> dict[str, float]:
    """Fetch market cap, total debt, and beta from FMP API.

    Args:
        ticker: Ticker symbol.
        api_key: FMP API key. Reads FMP_API_KEY env var if None.

    Returns:
        Dict with keys market_cap, total_debt, beta, or empty if unavailable.
    """
    import os
    import json
    import urllib.request

    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        return {}

    result: dict[str, float] = {}

    # Profile (market cap)
    try:
        url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={key}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]
            result["market_cap"] = float(profile.get("mktCap", 0))
            result["beta"] = float(profile.get("beta", 1.0))
    except Exception:
        pass

    # Balance sheet (total debt)
    try:
        url = (
            f"https://financialmodelingprep.com/stable/"
            f"balance-sheet-statement-as-list?symbol={ticker}"
            f"&period=annual&limit=1&apikey={key}"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data and isinstance(data, list) and len(data) > 0:
            bs = data[0]
            total_debt = float(bs.get("totalDebt", 0)) or float(bs.get("shortTermDebt", 0)) + float(
                bs.get("longTermDebt", 0)
            )
            result["total_debt"] = total_debt
    except Exception:
        pass

    return result


def compute_dtd_dataframe(
    ohlcv: pd.DataFrame,
    ticker: str = "",
    api_key: Optional[str] = None,
    volatility_window: int = 252,
    risk_free_rate: float = _US_TREASURY_RATE,
) -> pd.Series:
    """Compute Merton Distance-to-Default for each bar in a DataFrame.

    Uses FMP API for current market cap and total debt (snapshot).
    Estimates equity volatility from rolling 252-day returns.
    Asset volatility is derived via Merton approximation.

    Args:
        ohlcv: DataFrame with 'Close' column and DatetimeIndex.
        ticker: Ticker symbol for FMP fundamentals lookup.
        api_key: FMP API key. Reads FMP_API_KEY env var if None.
        volatility_window: Rolling window for equity volatility (default 252).
        risk_free_rate: Annualized risk-free rate (default 4.5%).

    Returns:
        pd.Series of DtD values aligned with ohlcv index.
    """
    close = ohlcv["Close"].astype(float)
    returns = close.pct_change().dropna()
    n = len(close)
    dtd = pd.Series(np.nan, index=ohlcv.index, name="dtd")

    metrics: dict[str, float] = {}
    if ticker:
        metrics = _fetch_fmp_metrics(ticker, api_key)
        if metrics:
            logger.info(
                "FMP: mcap=%.1fB debt=%.1fB for %s",
                metrics.get("market_cap", 0) / 1e9,
                metrics.get("total_debt", 0) / 1e9,
                ticker,
            )

    mcap = metrics.get("market_cap", 0.0)
    debt = metrics.get("total_debt", 0.0)

    if mcap <= 0 or debt <= 0:
        logger.warning("dtd: missing mcap/debt for %s, returning NaN", ticker or "unknown")
        return dtd

    for i in range(volatility_window, n):
        window_returns = returns.iloc[max(0, i - volatility_window) : i - 1]
        if len(window_returns) < 20:
            continue
        eq_vol = float(window_returns.std() * np.sqrt(252))
        dtd.iloc[i] = compute_dtd(mcap, debt, eq_vol, risk_free_rate)

    return dtd


def add_dtd_features(
    ohlcv: pd.DataFrame,
    ticker: str = "",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Add Distance-to-Default feature to an OHLCV DataFrame.

    Returns a new DataFrame with additional column: dtd.

    Args:
        ohlcv: DataFrame with 'Close' column and DatetimeIndex.
        ticker: Ticker symbol for fundamentals lookup.
        api_key: FMP API key.

    Returns:
        DataFrame with dtd column added.
    """
    result = ohlcv.copy()
    if "Close" in result.columns:
        result["dtd"] = compute_dtd_dataframe(result, ticker=ticker, api_key=api_key)
    return result
