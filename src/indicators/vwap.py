"""
VWAP (Volume Weighted Average Price) Indicator

Calculates session-resetting VWAP and optional deviation bands for mean-reversion trading.
TradingView's ta.vwap() resets at each new session (typically daily).
"""

import pandas as pd


def compute_vwap(
    df: pd.DataFrame,
    deviation_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate session-resetting VWAP with optional upper/lower deviation bands.

    VWAP resets at the start of each calendar day (like TradingView's ta.vwap).

    VWAP = cumsum(typical_price * volume) / cumsum(volume)  [resetting each day]
    where typical_price = (high + low + close) / 3

    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns,
            with a DatetimeIndex (timezone-aware or naive).
        deviation_pct: Percentage for upper/lower bands (default 0.5%).

    Returns:
        DataFrame with added columns: 'vwap', 'vwap_upper', 'vwap_lower'.
    """
    df_out = df.copy()
    typical_price = (df_out["High"] + df_out["Low"] + df_out["Close"]) / 3.0
    tp_vol = typical_price * df_out["Volume"]

    # Group by date to reset VWAP each session
    idx = pd.DatetimeIndex(df_out.index)
    if idx.tz is not None:
        df_out["_session"] = idx.tz_localize(None).date
    else:
        df_out["_session"] = idx.date

    cum_tp_vol = tp_vol.groupby(df_out["_session"]).cumsum()
    cum_vol = df_out["Volume"].groupby(df_out["_session"]).cumsum()

    vwap = cum_tp_vol / cum_vol

    df_out["vwap"] = vwap
    df_out["vwap_upper"] = vwap * (1 + deviation_pct / 100.0)
    df_out["vwap_lower"] = vwap * (1 - deviation_pct / 100.0)

    # Clean up helper column
    df_out.drop(columns=["_session"], inplace=True)

    return df_out
