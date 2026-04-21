from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

_TALIB_AVAILABLE = False
try:
    import talib

    _TALIB_AVAILABLE = True
except ImportError:
    pass


def compute_bbands(
    prices: pd.Series,
    period: int = 21,
    std_dev: float = 2.0,
    ma_type: int = 0,
) -> pd.DataFrame:
    """Compute Bollinger Bands (upper, middle, lower).

    Args:
        prices: Asset price series.
        period: Look-back window for moving average.
        std_dev: Number of standard deviations for band width.
        ma_type: TA-Lib MA type (0=SMA, 1=EMA, 2=WMA, etc.)

    Returns:
        DataFrame with columns ['upper', 'middle', 'lower'].
        Returns NaN-filled DataFrame if TA-Lib is not available.
    """
    if not _TALIB_AVAILABLE:
        return pd.DataFrame(
            {
                "upper": np.nan * len(prices),
                "middle": np.nan * len(prices),
                "lower": np.nan * len(prices),
            },
            index=prices.index,
        )

    upper, middle, lower = talib.BBANDS(
        prices.values,
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev,
        matype=ma_type,
    )
    return pd.DataFrame(
        {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        },
        index=prices.index,
    )


def compute_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Relative Strength Index.

    Args:
        prices: Asset price series.
        period: RSI look-back period.

    Returns:
        Series of RSI values (0-100 scale).
    """
    if not _TALIB_AVAILABLE:
        return pd.Series(np.nan * len(prices), index=prices.index, name="rsi")

    rsi = talib.RSI(prices.values, timeperiod=period)
    return pd.Series(rsi, index=prices.index, name="rsi")


def compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence).

    Args:
        prices: Asset price series.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.
        signal_period: Signal line period.

    Returns:
        DataFrame with columns ['macd', 'signal', 'histogram'].
    """
    if not _TALIB_AVAILABLE:
        return pd.DataFrame(
            {
                "macd": np.nan * len(prices),
                "signal": np.nan * len(prices),
                "histogram": np.nan * len(prices),
            },
            index=prices.index,
        )

    macd, signal, histogram = talib.MACD(
        prices.values,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period,
    )
    return pd.DataFrame(
        {
            "macd": macd,
            "signal": signal,
            "histogram": histogram,
        },
        index=prices.index,
    )


def compute_all_talib(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    indicators: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute a battery of TA-Lib technical indicators across multiple assets.

    Applies specified indicators column-wise across the price/volume DataFrames.

    Args:
        prices: DataFrame with date index and ticker columns (close prices).
        volume: Optional DataFrame of volume data (same structure as prices).
        indicators: List of indicator names to compute. If None, computes:
            ['bbands', 'rsi', 'macd', 'atr', 'adx', 'cci', 'willr'].

    Returns:
        Dictionary mapping indicator name to a 3D structure: {ticker: DataFrame}.
    """
    if indicators is None:
        indicators = ["bbands", "rsi", "macd", "atr", "adx", "cci", "willr"]

    results: Dict[str, Dict[str, pd.DataFrame]] = {ind: {} for ind in indicators}

    for ticker in prices.columns:
        close = prices[ticker].dropna().values.astype(float)

        if "bbands" in indicators and _TALIB_AVAILABLE:
            up, mid, low = talib.BBANDS(close, timeperiod=21)
            results["bbands"][ticker] = pd.DataFrame(
                {"upper": up, "middle": mid, "lower": low}, index=prices[ticker].dropna().index
            )

        if "rsi" in indicators and _TALIB_AVAILABLE:
            rsi = talib.RSI(close, timeperiod=14)
            results["rsi"][ticker] = pd.Series(
                rsi, index=prices[ticker].dropna().index, name=ticker
            )

        if "macd" in indicators and _TALIB_AVAILABLE:
            macd_line, signal_line, hist = talib.MACD(close)
            results["macd"][ticker] = pd.DataFrame(
                {"macd": macd_line, "signal": signal_line, "histogram": hist},
                index=prices[ticker].dropna().index,
            )

        if "atr" in indicators and _TALIB_AVAILABLE and volume is not None:
            high = prices[ticker].dropna().values.astype(float) * 1.01
            low = prices[ticker].dropna().values.astype(float) * 0.99
            atr = talib.ATR(high, low, close, timeperiod=14)
            results["atr"][ticker] = pd.Series(
                atr, index=prices[ticker].dropna().index, name=ticker
            )

        if "adx" in indicators and _TALIB_AVAILABLE and volume is not None:
            high = prices[ticker].dropna().values.astype(float) * 1.01
            low = prices[ticker].dropna().values.astype(float) * 0.99
            adx = talib.ADX(high, low, close, timeperiod=14)
            results["adx"][ticker] = pd.Series(
                adx, index=prices[ticker].dropna().index, name=ticker
            )

        if "cci" in indicators and _TALIB_AVAILABLE:
            high = prices[ticker].dropna().values.astype(float) * 1.01
            low = prices[ticker].dropna().values.astype(float) * 0.99
            cci = talib.CCI(high, low, close, timeperiod=14)
            results["cci"][ticker] = pd.Series(
                cci, index=prices[ticker].dropna().index, name=ticker
            )

        if "willr" in indicators and _TALIB_AVAILABLE:
            high = prices[ticker].dropna().values.astype(float) * 1.01
            low = prices[ticker].dropna().values.astype(float) * 0.99
            willr = talib.WILLR(high, low, close, timeperiod=14)
            results["willr"][ticker] = pd.Series(
                willr, index=prices[ticker].dropna().index, name=ticker
            )

    return {ind: d for ind, d in results.items()}
