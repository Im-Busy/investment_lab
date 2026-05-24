# -*- coding: utf-8 -*-
"""
Market Microstructure Analytics

Estimates bid-ask spread and liquidity conditions from OHLCV data using
empirically validated microstructure models. Combines into a precision-weighted
composite with regime detection and Liquidity Stress Index.

Origin: EdgeTools, PineScript v6

Models:
- Roll (1984): covariance of consecutive price changes
- Corwin-Schultz (2012): two-period high-low decomposition
- Abdi-Ranaldo (2017): close deviation from geometric bar midpoint
- Amihud (2002): illiquidity ratio
- Kyle lambda (1985): price-impact coefficient
- Parkinson (1980): range-based volatility
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MicrostructureResult:
    """Market microstructure analysis results."""

    spread_roll: np.ndarray
    spread_cs: np.ndarray
    spread_ar: np.ndarray
    eff_spread: np.ndarray
    spread_comp: np.ndarray
    amihud_sma: np.ndarray
    kyle_lambda: np.ndarray
    park_vol: np.ndarray
    lsi: np.ndarray
    lsi_scaled: np.ndarray
    spread_rz: np.ndarray
    amihud_rz: np.ndarray
    kyle_rz: np.ndarray
    regime_strs: list
    weights: Tuple[float, float, float]


def _rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = float(np.mean(arr[i - window + 1 : i + 1]))
    return result


def _rolling_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(arr)
    result = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    result[0] = float(arr[0]) if not np.isnan(arr[0]) else 0.0
    for i in range(1, n):
        if not np.isnan(arr[i]) and not np.isnan(result[i - 1]):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        elif not np.isnan(arr[i]):
            result[i] = arr[i]
    return result


def _rolling_mad(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling median and MAD."""
    n = len(arr)
    med = np.full(n, np.nan)
    mad = np.full(n, np.nan)

    for i in range(window - 1, n):
        win = arr[i - window + 1 : i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            med[i] = float(np.median(valid))
            mad[i] = float(np.median(np.abs(valid - med[i])))

    return med, mad, np.abs(arr - med)


def compute_microstructure(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    open_: np.ndarray,
    lookback: int = 20,
    smooth: int = 5,
    regime_win: int = 100,
    w_cap: float = 0.70,
    winsor: float = 3.0,
) -> MicrostructureResult:
    """
    Compute full market microstructure analytics.

    Args:
        high, low, close, volume, open_: OHLCV arrays
        lookback: Estimation window
        smooth: EMA smoothing period
        regime_win: Regime window for z-scores
        w_cap: Max composite weight
        winsor: Winsorization cap for LSI

    Returns:
        MicrostructureResult
    """
    n = len(close)
    dp = np.zeros(n)
    dp[1:] = close[1:] - close[:-1]
    dp_lag = np.roll(dp, 1)
    dp_lag[0] = 0

    r = np.zeros(n)
    mask = (close > 0) & (np.roll(close, 1) > 0)
    r[mask] = np.log(close[mask] / np.roll(close, 1)[mask])

    hl_valid = (high > 0) & (low > 0) & (high > low)

    tick_dir = np.ones(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            tick_dir[i] = 1
        elif close[i] < close[i - 1]:
            tick_dir[i] = -1
        else:
            tick_dir[i] = tick_dir[i - 1]

    q = tick_dir * volume

    cov_dp = _rolling_sma(dp * dp_lag, lookback) - _rolling_sma(dp, lookback) * _rolling_sma(
        dp_lag, lookback
    )
    roll_price = 2.0 * np.sqrt(np.maximum(0.0, -cov_dp))
    roll_pct = np.where(close > 0, roll_price / close * 100.0, 0.0)
    spread_roll = _rolling_ema(roll_pct, smooth)

    cs_k = 3.0 - 2.0 * np.sqrt(2.0)
    log_hl = np.log(high / low)
    beta_cs = np.where(hl_valid, log_hl**2 + np.roll(log_hl, 1) ** 2, np.nan)
    hi2 = np.maximum(high, np.roll(high, 1))
    lo2 = np.minimum(low, np.roll(low, 1))
    gamma_cs = np.where(hl_valid, np.log(hi2 / lo2) ** 2, np.nan)

    alpha_cs = np.where(
        ~np.isnan(beta_cs) & ~np.isnan(gamma_cs),
        np.maximum(
            0.0, (np.sqrt(2.0 * beta_cs) - np.sqrt(beta_cs)) / cs_k - np.sqrt(gamma_cs / cs_k)
        ),
        np.nan,
    )
    cs_raw = np.where(
        ~np.isnan(alpha_cs), 2.0 * (np.exp(alpha_cs) - 1.0) / (1.0 + np.exp(alpha_cs)) * 100.0, 0.0
    )
    spread_cs = _rolling_ema(np.nan_to_num(cs_raw, 0.0), smooth)
    cs_active = ~np.isnan(cs_raw)

    ar_c = np.where(hl_valid, np.log(close) - (np.log(high) + np.log(low)) / 2.0, np.nan)
    pair_valid = hl_valid & np.roll(hl_valid, 1)
    ar_prod = np.where(
        pair_valid, np.nan_to_num(ar_c, 0.0) * np.nan_to_num(np.roll(ar_c, 1), 0.0), 0.0
    )
    n_valid = _rolling_sma(pair_valid.astype(float), lookback) * lookback
    prod_sum = _rolling_sma(ar_prod, lookback) * lookback
    min_valid = max(3, int(np.ceil(lookback * 0.5)))
    ar_sufficient = n_valid >= min_valid
    mean_prod = np.where(ar_sufficient, prod_sum / np.maximum(n_valid, 1), 0.0)
    ar_raw = np.where(ar_sufficient, 2.0 * np.sqrt(np.maximum(0.0, -mean_prod)) * 100.0, 0.0)
    spread_ar = _rolling_ema(ar_raw, smooth)
    ar_active = hl_valid & ar_sufficient

    mid_price = (high + low) / 2.0
    eff_bar = np.where(close > 0, 2.0 * tick_dir * (close - mid_price) / close * 100.0, 0.0)
    eff_spread = _rolling_sma(eff_bar, lookback)

    dollar_vol = close * volume
    amihud_inst = np.where(dollar_vol > 0, np.abs(r) / dollar_vol, 0.0)
    amihud_sma = _rolling_sma(amihud_inst, lookback) * 1e8

    cov_kl = _rolling_sma(dp * q, lookback) - _rolling_sma(dp, lookback) * _rolling_sma(q, lookback)
    var_kl = _rolling_sma(q**2, lookback) - _rolling_sma(q, lookback) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_raw = np.where(var_kl > 1e-20, cov_kl / var_kl, 0.0)
    kyle_lambda = _rolling_ema(kl_raw * 1e6, smooth)

    park_k = 1.0 / (4.0 * np.log(2.0))
    park_inst = np.where(hl_valid, park_k * log_hl**2, 0.0)
    park_var = _rolling_sma(park_inst, lookback)
    park_vol = np.sqrt(np.maximum(0.0, park_var)) * 100.0

    roll_med_v, roll_mad_v, _ = _rolling_mad(roll_pct, lookback)
    var_roll = (1.4826 * roll_mad_v) ** 2

    cs_med_v, cs_mad_v, _ = _rolling_mad(cs_raw, lookback)
    var_cs = (1.4826 * cs_mad_v) ** 2

    ar_med_v, ar_mad_v, _ = _rolling_mad(ar_raw, lookback)
    var_ar = (1.4826 * ar_mad_v) ** 2

    n_comps = np.where(cs_active & ar_active, 3.0, np.where(cs_active | ar_active, 2.0, 1.0))
    var_act_sum = var_roll + np.where(cs_active, var_cs, 0.0) + np.where(ar_active, var_ar, 0.0)
    reg_eps = np.where(var_act_sum > 0, 0.05 * var_act_sum / n_comps, 1e-10)

    ivar_roll = 1.0 / (var_roll + reg_eps)
    ivar_cs = np.where(cs_active, 1.0 / (var_cs + reg_eps), 0.0)
    ivar_ar = np.where(ar_active, 1.0 / (var_ar + reg_eps), 0.0)
    ivar_sum = ivar_roll + ivar_cs + ivar_ar

    w_roll_u = np.where(ivar_sum > 0, ivar_roll / ivar_sum, 1.0 / n_comps)
    w_cs_u = np.where(ivar_sum > 0, ivar_cs / ivar_sum, np.where(cs_active, 1.0 / n_comps, 0.0))
    w_ar_u = np.where(ivar_sum > 0, ivar_ar / ivar_sum, np.where(ar_active, 1.0 / n_comps, 0.0))

    w_roll_c = np.minimum(w_roll_u, w_cap)
    w_cs_c = np.minimum(w_cs_u, w_cap)
    w_ar_c = np.minimum(w_ar_u, w_cap)
    w_sum = w_roll_c + w_cs_c + w_ar_c

    w_roll = np.where(w_sum > 0, w_roll_c / w_sum, 1.0 / n_comps)
    w_cs = np.where(w_sum > 0, w_cs_c / w_sum, np.where(cs_active, 1.0 / n_comps, 0.0))
    w_ar = np.where(w_sum > 0, w_ar_c / w_sum, np.where(ar_active, 1.0 / n_comps, 0.0))

    spread_comp = w_roll * spread_roll + w_cs * spread_cs + w_ar * spread_ar

    sp_med, sp_mad, _ = _rolling_mad(spread_comp, regime_win)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_rz = np.where(sp_mad > 1e-10, (spread_comp - sp_med) / (1.4826 * sp_mad), 0.0)

    ah_med, ah_mad, _ = _rolling_mad(amihud_sma, regime_win)
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud_rz = np.where(ah_mad > 1e-10, (amihud_sma - ah_med) / (1.4826 * ah_mad), 0.0)

    kl_abs = np.abs(kyle_lambda)
    kl_med, kl_mad, _ = _rolling_mad(kl_abs, regime_win)
    with np.errstate(divide="ignore", invalid="ignore"):
        kyle_rz = np.where(kl_mad > 1e-10, (kl_abs - kl_med) / (1.4826 * kl_mad), 0.0)

    spread_rz_w = np.clip(spread_rz, -winsor, winsor)
    amihud_rz_w = np.clip(amihud_rz, -winsor, winsor)
    kyle_rz_w = np.clip(kyle_rz, -winsor, winsor)
    lsi = (spread_rz_w + amihud_rz_w + kyle_rz_w) / 3.0

    lsi_clamped = np.clip(lsi, -40, 40)
    lsi_e2x = np.exp(lsi_clamped)
    lsi_scaled = 50.0 * (1.0 + (lsi_e2x - 1.0) / (lsi_e2x + 1.0))

    regime_strs = []
    for i in range(n):
        if lsi[i] >= 2.0:
            regime_strs.append("STRESS")
        elif lsi[i] >= 1.0:
            regime_strs.append("ELEVATED")
        elif lsi[i] <= -1.0:
            regime_strs.append("COMPRESSED")
        else:
            regime_strs.append("NORMAL")

    w_roll_last = float(w_roll[-1]) if n > 0 else 0.0
    w_cs_last = float(w_cs[-1]) if n > 0 else 0.0
    w_ar_last = float(w_ar[-1]) if n > 0 else 0.0

    return MicrostructureResult(
        spread_roll=spread_roll,
        spread_cs=spread_cs,
        spread_ar=spread_ar,
        eff_spread=eff_spread,
        spread_comp=spread_comp,
        amihud_sma=amihud_sma,
        kyle_lambda=kyle_lambda,
        park_vol=park_vol,
        lsi=lsi,
        lsi_scaled=lsi_scaled,
        spread_rz=spread_rz,
        amihud_rz=amihud_rz,
        kyle_rz=kyle_rz,
        regime_strs=regime_strs,
        weights=(w_roll_last, w_cs_last, w_ar_last),
    )
