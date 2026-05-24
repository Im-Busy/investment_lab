"""Diagnose why SMC fails on non-NQ instruments."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fetch_data(symbol: str, interval: str = "1h") -> pd.DataFrame:
    safe_symbol = symbol.replace("=", "_").replace("/", "_")
    for alt in [safe_symbol, symbol.replace("-", "_")]:
        path = Path(f"data/raw/{alt}_{interval}.csv")
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    df[col] = 0 if col == "Volume" else df.iloc[:, 0]
            df.columns = [c.capitalize() for c in df.columns]
            return df
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period="730d", interval=interval, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def compute_features(df: pd.DataFrame) -> dict:
    close = df["Close"]
    returns = close.pct_change().dropna()
    vol = returns.std() * 100
    annual_vol = vol * np.sqrt(252 * 6.5)  # hourly

    high = df["High"]
    low = df["Low"]
    atr_raw = (high - low).rolling(14).mean()
    avg_atr = float(atr_raw.mean())
    avg_atr_pct = float((atr_raw / close).mean() * 100)
    avg_price = float(close.mean())

    trend_up = float(close.iloc[-1] / close.iloc[0] - 1)
    max_dd = float((close / close.cummax() - 1).min())

    n_bars = len(df)
    date_start = str(df.index[0].date())
    date_end = str(df.index[-1].date())

    swing_highs = 0
    swing_lows = 0
    for i in range(3, len(df) - 3):
        if (
            high.iloc[i] > high.iloc[i - 3 : i].max()
            and high.iloc[i] > high.iloc[i + 1 : i + 4].max()
        ):
            swing_highs += 1
        if low.iloc[i] < low.iloc[i - 3 : i].min() and low.iloc[i] < low.iloc[i + 1 : i + 4].min():
            swing_lows += 1

    session_bars = 24
    rolling_high = high.rolling(session_bars).max().shift(1)
    rolling_low = low.rolling(session_bars).min().shift(1)

    bull_sweeps = (low < rolling_low).sum()
    bear_sweeps = (high > rolling_high).sum()
    total_sweeps = bull_sweeps + bear_sweeps
    sweep_rate = total_sweeps / n_bars * 100

    sweep_bars = df[low < rolling_low]
    if len(sweep_bars) > 0:
        bull_sweep_fwd_ret = close.shift(-5) / close - 1
        bull_fwd = bull_sweep_fwd_ret[low < rolling_low].dropna()
        bull_sweep_wr = (bull_fwd > 0).mean() * 100 if len(bull_fwd) > 0 else 0
        bull_sweep_avg_ret = float(bull_fwd.mean() * 100) if len(bull_fwd) > 0 else 0
    else:
        bull_sweep_wr = 0
        bull_sweep_avg_ret = 0

    sweep_bars_bear = df[high > rolling_high]
    if len(sweep_bars_bear) > 0:
        bear_sweep_fwd_ret = close.shift(-5) / close - 1
        bear_fwd = bear_sweep_fwd_ret[high > rolling_high].dropna()
        bear_sweep_wr = (bear_fwd < 0).mean() * 100 if len(bear_fwd) > 0 else 0
        bear_sweep_avg_ret = float(-bear_fwd.mean() * 100) if len(bear_fwd) > 0 else 0
    else:
        bear_sweep_wr = 0
        bear_sweep_avg_ret = 0

    atr_14 = (high - low).rolling(14).mean()
    atr_available = atr_14.dropna()
    n_bos = 0
    bos_bull_wr = 0
    bos_bear_wr = 0
    if len(atr_available) > 0:
        atr_thresh = atr_14 > atr_14.mean() * 0.5
        for i in range(20, len(df) - 5):
            if atr_thresh.iloc[i]:
                lookback = df.iloc[i - 20 : i]
                prev_high = lookback["High"].max()
                prev_low = lookback["Low"].min()
                if close.iloc[i] > prev_high:
                    n_bos += 1
                elif close.iloc[i] < prev_low:
                    n_bos += 1
        bos_rate = n_bos / n_bars * 100
    else:
        bos_rate = 0

    range_days = (date_end, date_start)
    days_diff = len(close)

    return {
        "symbol": df.index.name if df.index.name else "N/A",
        "bars": n_bars,
        "start": date_start,
        "end": date_end,
        "avg_price": round(avg_price, 2),
        "avg_atr": round(avg_atr, 2),
        "avg_atr_pct": round(avg_atr_pct, 3),
        "vol_hourly_pct": round(vol, 3),
        "trend_pct": round(trend_up * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "bull_sweeps": int(bull_sweeps),
        "bear_sweeps": int(bear_sweeps),
        "total_sweeps": int(total_sweeps),
        "sweep_rate_pct": round(sweep_rate, 2),
        "bull_sweep_wr_pct": round(bull_sweep_wr, 1),
        "bull_sweep_avg_ret_pct": round(bull_sweep_avg_ret, 3),
        "bear_sweep_wr_pct": round(bear_sweep_wr, 1),
        "bear_sweep_avg_ret_pct": round(bear_sweep_avg_ret, 3),
        "bos_rate_pct": round(bos_rate, 2) if bos_rate else 0,
    }


def main():
    symbols = ["BTC-USD", "GC=F", "NQ=F", "EURUSD=X", "GBPJPY=X"]
    all_results = []
    for sym in symbols:
        try:
            df = fetch_data(sym)
            r = compute_features(df)
            r["symbol"] = sym
            all_results.append(r)
        except Exception as e:
            print(f"FAILED {sym}: {e}")

    print(
        f"\n{'Symbol':>10} {'Bars':>6} {'Start':>12} {'End':>12} {'Price':>8} {'ATR':>7} {'ATR%':>6} {'Vol%':>6} {'Trend%':>7} {'MaxDD%':>7} {'SwHi':>5} {'SwLo':>5}"
    )
    print("-" * 115)
    for r in all_results:
        print(
            f"{r['symbol']:>10} {r['bars']:>6} {r['start']:>12} {r['end']:>12} {r['avg_price']:>8.2f} {r['avg_atr']:>7.2f} {r['avg_atr_pct']:>6.3f} {r['vol_hourly_pct']:>6.3f} {r['trend_pct']:>7.1f} {r['max_dd_pct']:>7.1f} {r['swing_highs']:>5} {r['swing_lows']:>5}"
        )

    print(
        f"\n{'Symbol':>10} {'BullSwp':>8} {'BearSwp':>8} {'TotSwp':>7} {'SwpRate%':>9} {'B_SwpWR%':>9} {'B_SwpRet%':>10} {'B_BearWR%':>10} {'B_BearRet%':>10} {'BOSRate%':>9}"
    )
    print("-" * 105)
    for r in all_results:
        print(
            f"{r['symbol']:>10} {r['bull_sweeps']:>8} {r['bear_sweeps']:>8} {r['total_sweeps']:>7} {r['sweep_rate_pct']:>9.2f} {r['bull_sweep_wr_pct']:>9.1f} {r['bull_sweep_avg_ret_pct']:>10.3f} {r['bear_sweep_wr_pct']:>10.1f} {r['bear_sweep_avg_ret_pct']:>10.3f} {r['bos_rate_pct']:>9.2f}"
        )


if __name__ == "__main__":
    main()
