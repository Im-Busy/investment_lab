"""
P1-4: Regime-Conditional Return Decomposition.

Runs backtest on SPY 2020-2026 and decomposes returns by regime:
Trending/Ranging/Transition (ADX-based) and Bull/Bear (20d return).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime labels to OHLCV dataframe."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ADX-based regime
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_smooth = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(14).mean()

    regime = pd.Series("Transition", index=df.index)
    regime[adx > 25] = "Trending"
    regime[adx < 20] = "Ranging"

    # Volatility regime
    atr_pct = atr / close
    vol_80 = atr_pct.rolling(252).quantile(0.8)
    vol_regime = pd.Series("Normal", index=df.index)
    vol_regime[atr_pct > vol_80] = "High"

    # Bull/Bear
    ret_20d = close.pct_change(20)
    direction = pd.Series("Bear", index=df.index)
    direction[ret_20d > 0] = "Bull"

    result = pd.DataFrame(
        {
            "adx_regime": regime,
            "vol_regime": vol_regime,
            "direction": direction,
        },
        index=df.index,
    )
    return result.dropna()


def main() -> None:
    MODEL_PATH = "models/pattern_classifier_v3_SPY_20260514_124612.pkl"
    START = "2020-01-01"
    END = "2026-05-14"

    print("Loading SPY data...")
    df = pd.read_csv("data/raw/SPY_daily.csv", parse_dates=True, index_col=0)
    df.columns = [c.capitalize() for c in df.columns]
    df = df[(df.index >= START) & (df.index <= END)]

    # Get ML signals
    print("Generating ML signals...")
    from src.ml.feature_engineering import FeatureExtractor
    from src.ml.pattern_classifier import PatternClassifier

    model = PatternClassifier()
    model.load(MODEL_PATH)

    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, include_forward_returns=False)
    features = features.ffill().bfill().fillna(0)
    common = features.index.intersection(df.index)
    features = features.loc[common]
    df = df.loc[common]

    # Fill missing features (cross-asset not available for single ticker)
    for col in model.feature_names_:
        if col not in features.columns:
            features[col] = 0.0

    preds = model.predict(features[model.feature_names_])
    probs = preds["probability_profitable"]
    signals = (probs >= 0.45).astype(int)

    # Regime classification
    regimes = classify_regimes(df)
    common_idx = df.index.intersection(regimes.index)
    df = df.loc[common_idx]
    regimes = regimes.loc[common_idx]
    probs = probs.loc[common_idx]
    signals = signals.loc[common_idx]

    # Forward returns (next bar)
    fwd_ret = df["Close"].pct_change().shift(-1).loc[common_idx]

    # Entry returns: return on bars where signal=1
    entry_returns = fwd_ret[signals == 1]

    # Combine
    data = pd.DataFrame(
        {
            "fwd_ret": fwd_ret,
            "signal": signals,
            "prob": probs,
            "adx_regime": regimes["adx_regime"],
            "vol_regime": regimes["vol_regime"],
            "direction": regimes["direction"],
        }
    ).dropna()

    print()
    print("=" * 80)
    print(f"Regime-Conditional Return Decomposition: SPY {START} -> {END}")
    print("=" * 80)
    print(f"Total bars: {len(data)}, Signals: {data['signal'].sum()}")
    print(f"Buy & Hold return: {df['Close'].iloc[-1] / df['Close'].iloc[0] - 1:.1%}")
    print()

    # By ADX regime
    for reg in ["Trending", "Transition", "Ranging"]:
        subset = data[data["adx_regime"] == reg]
        n_bars = len(subset)
        n_sigs = subset["signal"].sum()
        if n_sigs > 0:
            sig_rets = fwd_ret.loc[subset[subset["signal"] == 1].index].dropna()
            mean_ret = sig_rets.mean() * 100
            win_rate = (sig_rets > 0).mean() * 100
        else:
            mean_ret = 0
            win_rate = 0
        print(
            f"  {reg:<12} Bars={n_bars:>5}  Signals={n_sigs:>4}  "
            f"AvgReturn={mean_ret:>6.2f}%  WinRate={win_rate:>5.1f}%"
        )

    print()

    # By direction
    for d in ["Bull", "Bear"]:
        subset = data[data["direction"] == d]
        n_bars = len(subset)
        n_sigs = subset["signal"].sum()
        if n_sigs > 0:
            sig_rets = fwd_ret.loc[subset[subset["signal"] == 1].index].dropna()
            mean_ret = sig_rets.mean() * 100
            win_rate = (sig_rets > 0).mean() * 100
        else:
            mean_ret = 0
            win_rate = 0
        print(
            f"  {d:<12} Bars={n_bars:>5}  Signals={n_sigs:>4}  "
            f"AvgReturn={mean_ret:>6.2f}%  WinRate={win_rate:>5.1f}%"
        )

    print()

    # By volume regime
    for reg in ["Normal", "High"]:
        subset = data[data["vol_regime"] == reg]
        n_bars = len(subset)
        n_sigs = subset["signal"].sum()
        if n_sigs > 0:
            sig_rets = fwd_ret.loc[subset[subset["signal"] == 1].index].dropna()
            mean_ret = sig_rets.mean() * 100
            win_rate = (sig_rets > 0).mean() * 100
        else:
            mean_ret = 0
            win_rate = 0
        print(
            f"  {reg:<12} Bars={n_bars:>5}  Signals={n_sigs:>4}  "
            f"AvgReturn={mean_ret:>6.2f}%  WinRate={win_rate:>5.1f}%"
        )

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Find best/worst
    trending = data[data["adx_regime"] == "Trending"]
    ranging = data[data["adx_regime"] == "Ranging"]
    bull = data[data["direction"] == "Bull"]
    bear = data[data["direction"] == "Bear"]

    trending_ret = fwd_ret.loc[trending[trending["signal"] == 1].index].dropna().mean() * 100
    ranging_ret = fwd_ret.loc[ranging[ranging["signal"] == 1].index].dropna().mean() * 100
    bull_ret = fwd_ret.loc[bull[bull["signal"] == 1].index].dropna().mean() * 100
    bear_ret = fwd_ret.loc[bear[bear["signal"] == 1].index].dropna().mean() * 100

    print(f"  Trending avg return: {trending_ret:+.2f}%")
    print(f"  Ranging  avg return: {ranging_ret:+.2f}%")
    print(f"  Bull     avg return: {bull_ret:+.2f}%")
    print(f"  Bear     avg return: {bear_ret:+.2f}%")

    if bear_ret < 0:
        print("  WARNING: Model loses money in Bear markets — capital protection thesis FAILED.")


if __name__ == "__main__":
    main()
