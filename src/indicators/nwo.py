# -*- coding: utf-8 -*-
"""
Neural Weight Oscillator (Zeiierman)

Combines Trend, Mean Reversion, and Momentum into an adaptive multi-factor
oscillator (0-100 scale). Uses Best-Worst Method (BWM) for structured factor
weighting and an adaptive learning layer that amplifies features that recently
produced strong directional behavior.

Origin: TradingView indicator by Zeiierman, PineScript v6
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class NWOConfig:
    """Neural Weight Oscillator configuration."""

    len_fast: int = 20
    len_slow: int = 100
    smooth_len: int = 5
    best_criterion: str = "Trend"
    worst_criterion: str = "Momentum"
    bo_trend: float = 1.0
    bo_mean: float = 3.0
    bo_momentum: float = 6.0
    ow_trend: float = 6.0
    ow_mean: float = 3.0
    ow_momentum: float = 1.0
    use_training: bool = True
    learn_influence: float = 0.30
    line_influence: float = 0.25
    signal_len: int = 9
    atr_len: int = 14
    rsi_len: int = 14
    momentum_len: int = 20
    memory_size: int = 150
    batch_size: int = 20
    target_len: int = 3
    huber_d: float = 0.01
    ai_center_len: int = 100


def _criterion_index(name: str) -> int:
    """Map criterion name to index."""
    mapping = {"Trend": 0, "Mean Reversion": 1, "Momentum": 2}
    return mapping.get(name, 0)


def _bwm_solve(bo: np.ndarray, ow: np.ndarray, best_idx: int, worst_idx: int) -> np.ndarray:
    """Solve BWM weights using geometric mean approximation."""
    a_bw = bo[worst_idx]
    weights = np.zeros(3)

    for i in range(3):
        bo_val = bo[i]
        ow_val = ow[i]
        weights[i] = np.sqrt((a_bw / max(bo_val, 1e-10)) * ow_val) if bo_val > 0 else 0.0

    total = np.sum(weights)
    if total > 0:
        weights /= total
    return weights


def compute_nwo(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    config: Optional[NWOConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Neural Weight Oscillator.

    Args:
        high, low, close: Price arrays
        config: NWO configuration

    Returns:
        Tuple of (oscillator, signal_line, histogram, trend_direction) arrays
    """
    if config is None:
        config = NWOConfig()

    cfg = config
    n = len(close)

    alpha_fast = 2.0 / (cfg.len_fast + 1)
    alpha_slow = 2.0 / (cfg.len_slow + 1)
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    ema_fast[0] = float(close[0])
    ema_slow[0] = float(close[0])
    for i in range(1, n):
        ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i - 1]
        ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i - 1]

    atr = np.zeros(n)
    if n > 1:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr[0] = float(high[0] - low[0]) if n > 0 else 1.0
        atr_alpha = 1.0 / cfg.atr_len
        for i in range(1, n):
            atr[i] = atr_alpha * tr[i - 1] + (1 - atr_alpha) * atr[i - 1]

    rsi = np.zeros(n)
    gain = np.zeros(n)
    loss = np.zeros(n)
    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gain[i] = delta
            loss[i] = 0
        else:
            gain[i] = 0
            loss[i] = -delta
    if n > cfg.rsi_len:
        for i in range(cfg.rsi_len, n):
            avg_gain = float(np.mean(gain[i - cfg.rsi_len + 1 : i + 1]))
            avg_loss = float(np.mean(loss[i - cfg.rsi_len + 1 : i + 1]))
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    rsi[0] = 50.0

    atr_safe = np.maximum(atr, 1e-10)
    trend_spread = (ema_fast - ema_slow) / atr_safe
    trend_slope = np.zeros(n)
    trend_slope[1:] = (ema_fast[1:] - ema_fast[:-1]) / atr_safe[1:]
    trend_raw = trend_spread + trend_slope
    trend_score = np.clip((trend_raw - (-2.5)) / (2.5 - (-2.5)) * 100.0, 0, 100)

    basis = np.zeros(n)
    dev = np.zeros(n)
    for i in range(cfg.momentum_len, n):
        basis[i] = float(np.mean(close[i - cfg.momentum_len + 1 : i + 1]))
        dev[i] = float(np.std(close[i - cfg.momentum_len + 1 : i + 1], ddof=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = np.where(dev > 1e-10, (close - basis) / dev, 0.0)
    rsi_reversion = 100.0 - rsi
    z_reversion = np.clip((-z_score - (-2.5)) / (2.5 - (-2.5)) * 100.0, 0, 100)
    mean_score = rsi_reversion * 0.5 + z_reversion * 0.5

    roc = np.zeros(n)
    roc[cfg.momentum_len :] = close[cfg.momentum_len :] / close[: -cfg.momentum_len] - 1.0
    roc_norm = np.clip((roc - (-0.05)) / (0.05 - (-0.05)) * 100.0, 0, 100)
    ema_vel = np.zeros(n)
    ema_vel[1:] = (ema_fast[1:] - ema_fast[:-1]) / atr_safe[1:]
    ema_momentum = np.clip((ema_vel - (-0.5)) / (0.5 - (-0.5)) * 100.0, 0, 100)
    momentum_score = roc_norm * 0.45 + rsi * 0.35 + ema_momentum * 0.20

    trend_feature = (trend_score - 50.0) / 50.0
    mean_feature = (mean_score - 50.0) / 50.0
    momentum_feature = (momentum_score - 50.0) / 50.0

    bo = np.array([cfg.bo_trend, cfg.bo_mean, cfg.bo_momentum], dtype=np.float64)
    ow = np.array([cfg.ow_trend, cfg.ow_mean, cfg.ow_momentum], dtype=np.float64)
    best_idx = _criterion_index(cfg.best_criterion)
    worst_idx = _criterion_index(cfg.worst_criterion)
    bo[best_idx] = 1.0
    ow[worst_idx] = 1.0

    bwm_weights = _bwm_solve(bo, ow, best_idx, worst_idx)
    bwm_trend = bwm_weights[0]
    bwm_mean = bwm_weights[1]
    bwm_momentum = bwm_weights[2]

    tw_trend = 0.01
    tw_mean = 0.01
    tw_momentum = 0.01
    tbias = 0.0

    osc = np.full(n, 50.0)

    for i in range(n):
        feature_amplification_trend = 1.0
        feature_amplification_mean = 1.0
        feature_amplification_momentum = 1.0
        ai_osc = 50.0
        line_blend = 0.0

        if cfg.use_training and i >= cfg.target_len + 1:
            target = close[i] / max(close[i - cfg.target_len], 1e-10) - 1.0
            target_dir = 1.0 if target > 0 else -1.0 if target < 0 else 0.0

            if target_dir != 0:
                pred = (
                    tw_trend * trend_feature[i - cfg.target_len]
                    + tw_mean * mean_feature[i - cfg.target_len]
                    + tw_momentum * momentum_feature[i - cfg.target_len]
                    + tbias
                )
                err = pred - target_dir

                abs_err = abs(err)
                loss = (
                    0.5 * err * err
                    if abs_err <= cfg.huber_d
                    else cfg.huber_d * (abs_err - 0.5 * cfg.huber_d)
                )
                grad = err if abs_err <= cfg.huber_d else cfg.huber_d * np.sign(err)

                lr = 0.01
                tw_trend -= lr * grad * trend_feature[i - cfg.target_len]
                tw_mean -= lr * grad * mean_feature[i - cfg.target_len]
                tw_momentum -= lr * grad * momentum_feature[i - cfg.target_len]
                tbias -= lr * grad

            max_w = max(abs(tw_trend), abs(tw_mean), abs(tw_momentum), 1e-4)
            lt = tw_trend / max_w
            lm = tw_mean / max_w
            lmo = tw_momentum / max_w

            blend = cfg.learn_influence
            feature_amplification_trend = 1.0 + lt * blend
            feature_amplification_mean = 1.0 + lm * blend
            feature_amplification_momentum = 1.0 + lmo * blend

            ai_pred = (
                tw_trend * trend_feature[i]
                + tw_mean * mean_feature[i]
                + tw_momentum * momentum_feature[i]
                + tbias
            )
            ai_osc = 50.0 + np.clip(ai_pred * 50.0, -50, 50)
            ai_strength = np.clip(abs(ai_pred) * 3.0, 0.0, 1.0)
            line_blend = blend * ai_strength * cfg.line_influence

        trend_pressure = (trend_score[i] - 50.0) * bwm_trend * feature_amplification_trend
        mean_pressure = (mean_score[i] - 50.0) * bwm_mean * feature_amplification_mean
        momentum_pressure = (
            (momentum_score[i] - 50.0) * bwm_momentum * feature_amplification_momentum
        )

        raw_osc = 50.0 + trend_pressure + mean_pressure + momentum_pressure

        if i == 0:
            base_osc = raw_osc
        else:
            alpha_s = 2.0 / (cfg.smooth_len + 1)
            base_osc = alpha_s * raw_osc + (1 - alpha_s) * osc[i - 1]

        osc[i] = np.clip(base_osc * (1.0 - line_blend) + ai_osc * line_blend, 0, 100)

    sig_alpha = 2.0 / (cfg.signal_len + 1)
    signal = np.zeros(n)
    signal[0] = osc[0]
    for i in range(1, n):
        signal[i] = sig_alpha * osc[i] + (1 - sig_alpha) * signal[i - 1]

    hist_raw = osc - signal
    hist_smooth_alpha = 2.0 / 4.0
    hist = np.zeros(n)
    hist[0] = hist_raw[0]
    for i in range(1, n):
        hist[i] = hist_smooth_alpha * hist_raw[i] + (1 - hist_smooth_alpha) * hist[i - 1]

    trend_dir = np.zeros(n, dtype=np.int8)
    trend_dir[osc > signal] = 1
    trend_dir[osc < signal] = -1

    return osc, signal, hist, trend_dir.astype(np.float64)


def compute_nwo_signals(
    osc: np.ndarray,
    signal: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sweep_lookback: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NWO trading signals with price sweep confirmation.

    Returns:
        Tuple of (bull_signal, bear_signal) boolean arrays
    """
    n = len(osc)
    bull_sig = np.zeros(n, dtype=bool)
    bear_sig = np.zeros(n, dtype=bool)

    for i in range(1, n):
        crossover = osc[i] > signal[i] and osc[i - 1] <= signal[i - 1]
        crossunder = osc[i] < signal[i] and osc[i - 1] >= signal[i - 1]

        sweep_bull = (
            low[i] < low[max(0, i - sweep_lookback)] and close[i] > low[max(0, i - sweep_lookback)]
        )
        sweep_bear = (
            high[i] > high[max(0, i - sweep_lookback)]
            and close[i] < high[max(0, i - sweep_lookback)]
        )

        bull_sig[i] = crossover and osc[i] < 30 and sweep_bull
        bear_sig[i] = crossunder and osc[i] > 70 and sweep_bear

    return bull_sig, bear_sig
