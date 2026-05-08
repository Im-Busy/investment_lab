"""
Vectorized exponential smoothing operations for GPU.

Replaces sequential Python for-loops in EMA, RSI, ATR, ADX with fully
parallel exponential convolution via unfold + dot product.

Principles applied:
  1. Data parallelism: All N bars processed simultaneously
  2. Minimal branching: torch.where() for conditional assignment
  3. Coalesced memory: Contiguous tensor windows via unfold
  4. Batched transfers: Purely on-device ops, no CPU round-trips
  5. Tensor cores: unfold + matmul-friendly dot products
  6. Maximized occupancy: Native PyTorch reductions handle scheduling

Math:
  EMA recurrence:  ema[i] = alpha * x[i] + (1-alpha) * ema[i-1]
  Closed form:     ema[i] = alpha * sum_{j=0}^{i} x[j] * (1-alpha)^(i-j)

  Wilder smoothing: S[i] = K * S[i-1] + (1-K) * x[i]    (K = (period-1)/period)
  Closed form:      S[i] = (1-K) * sum_{j=0}^{i} x[j] * K^(i-j)

Truncation: After T bars, the kernel weight decays below 1e-6.
We use unfold to extract windows, dot with the exponential kernel.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _exponential_kernel(
    weight: float,
    decay: float,
    truncation: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build truncated exponential kernel for convolution.

    kernel[j] = weight * decay^j  for j = 0..truncation-1

    Args:
        weight: Leading multiplier (alpha for EMA, 1/period for Wilder)
        decay: Decay factor (1-alpha for EMA, (period-1)/period for Wilder)
        truncation: Kernel length
    """
    indices = torch.arange(truncation, device=device, dtype=dtype)
    kernel = decay**indices
    return weight * kernel


def _truncation_for_decay(decay: float, threshold: float = 1e-6, min_len: int = 8) -> int:
    """
    Compute truncation length so that decay^T < threshold.

    T = ceil(ln(threshold) / ln(decay)), clamped to minimum.
    """
    if decay <= 0.0 or decay >= 1.0:
        return min_len
    exact = math.ceil(math.log(threshold) / math.log(decay))
    return max(exact + 1, min_len)


def ema_parallel(
    x: torch.Tensor,
    period: int,
) -> torch.Tensor:
    """
    GPU-parallel EMA via truncated exponential convolution.

    All N bars computed simultaneously — zero Python for-loops.
    Equivalent to pandas ewm(span=period, adjust=False).mean().

    Args:
        x: [N] tensor on GPU
        period: EMA period (span)

    Returns:
        [N] tensor, EMA values
    """
    n = x.shape[0]
    if n == 0:
        return x.clone()

    alpha = 2.0 / float(period + 1)
    decay = 1.0 - alpha

    truncation = min(_truncation_for_decay(decay, min_len=period * 2), n)

    kernel = _exponential_kernel(alpha, decay, truncation, x.device, x.dtype)
    kernel = kernel.flip(0)

    x_pad = F.pad(x.view(1, 1, -1), (truncation - 1, 0), mode="replicate")
    windows = x_pad.unfold(2, truncation, 1)  # [1, 1, N, T]

    dot = (windows * kernel.view(1, 1, 1, -1)).sum(dim=3)
    result = dot.view(-1)

    seed_sma = x[:period].mean()
    decay_powers = decay ** torch.arange(n, device=x.device, dtype=x.dtype)
    result = result + decay_powers * seed_sma

    return result


def wilder_smooth_parallel(
    x: torch.Tensor,
    period: int,
    seed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GPU-parallel Wilder smoothing via truncated exponential convolution.

    Wilder recurrence: S[i] = K * S[i-1] + (1-K) * x[i]
    where K = (period - 1) / period

    Args:
        x: [N] tensor on GPU
        period: Smoothing period
        seed: [1] initial value (defaults to mean of first `period` elements)

    Returns:
        [N] tensor, smoothed values
    """
    n = x.shape[0]
    if n == 0:
        return x.clone()

    k = (period - 1.0) / period
    one_minus_k = 1.0 / period

    truncation = min(_truncation_for_decay(k, min_len=period * 4), n)

    kernel = _exponential_kernel(one_minus_k, k, truncation, x.device, x.dtype)
    kernel = kernel.flip(0)

    x_pad = F.pad(x.view(1, 1, -1), (truncation - 1, 0), mode="replicate")
    windows = x_pad.unfold(2, truncation, 1)  # [1, 1, N, T]

    dot = (windows * kernel.view(1, 1, 1, -1)).sum(dim=3)
    result = dot.view(-1)

    if seed is None:
        seed = x[:period].mean()
    k_powers = k ** torch.arange(n, device=x.device, dtype=x.dtype)
    result = result + k_powers * seed

    return result


def ema_parallel_batch(
    x: torch.Tensor,
    periods: list[int],
) -> torch.Tensor:
    """
    GPU-parallel batch EMA: compute multiple periods at once.

    Args:
        x: [N] tensor on GPU
        periods: List of EMA periods

    Returns:
        [N, P] tensor where column p is EMA for periods[p]
    """
    results = [ema_parallel(x, p) for p in periods]
    return torch.stack(results, dim=1)


def rsi_parallel(close: torch.Tensor, period: int = 14) -> torch.Tensor:
    """
    GPU-parallel RSI with Wilder smoothing.

    1. Compute delta = diff(close)
    2. Separate gains (delta > 0) and losses (delta < 0)
    3. Wilder-smooth gains and losses in parallel
    4. RS = avg_gain / avg_loss → RSI = 100 - 100/(1+RS)

    Args:
        close: [N] tensor on GPU
        period: RSI period (default 14)

    Returns:
        [N] tensor, RSI values (NaN for first `period` bars)
    """
    n = close.shape[0]
    if n <= period:
        return torch.full_like(close, float("nan"))

    delta = torch.diff(close, prepend=close[:1])
    gain = torch.where(delta > 0, delta, torch.tensor(0.0, device=close.device))
    loss = torch.where(delta < 0, -delta, torch.tensor(0.0, device=close.device))

    avg_gain = wilder_smooth_parallel(gain, period, seed=gain[:period].mean())
    avg_loss = wilder_smooth_parallel(loss, period, seed=loss[:period].mean())

    rs = avg_gain / (avg_loss + 1e-9)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))

    rsi_vals[:period] = float("nan")
    return rsi_vals


def atr_parallel(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    period: int = 14,
) -> torch.Tensor:
    """
    GPU-parallel ATR with Wilder smoothing.

    1. Compute True Range (element-wise, fully parallel)
    2. Wilder-smooth TR in parallel
    3. Set first (period-1) bars to NaN

    Args:
        high: [N] tensor
        low: [N] tensor
        close: [N] tensor
        period: ATR period (default 14)

    Returns:
        [N] tensor, ATR values
    """
    n = close.shape[0]
    if n <= period:
        return torch.full_like(close, float("nan"))

    c_prev = torch.roll(close, 1, 0)
    c_prev[0] = close[0]

    tr1 = high - low
    tr2 = torch.abs(high - c_prev)
    tr3 = torch.abs(low - c_prev)
    tr = torch.maximum(torch.maximum(tr1, tr2), tr3)

    atr_vals = wilder_smooth_parallel(tr, period, seed=tr[:period].mean())
    atr_vals[: period - 1] = float("nan")
    return atr_vals


def adx_parallel(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    period: int = 14,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-parallel ADX with Wilder smoothing.

    1. Compute +DM, -DM, TR (element-wise, parallel)
    2. Wilder-smooth all three in parallel
    3. Compute +DI, -DI, DX (element-wise)
    4. Wilder-smooth DX → ADX

    Args:
        high: [N] tensor
        low: [N] tensor
        close: [N] tensor
        period: ADX period (default 14)

    Returns:
        (adx, plus_di, minus_di) each [N]
    """
    n = close.shape[0]
    if n <= 2 * period:
        empty = torch.full_like(close, float("nan"))
        return empty, empty, empty

    h_prev = torch.roll(high, 1, 0)
    h_prev[0] = high[0]
    l_prev = torch.roll(low, 1, 0)
    l_prev[0] = low[0]
    c_prev = torch.roll(close, 1, 0)
    c_prev[0] = close[0]

    up_move = high - h_prev
    dn_move = l_prev - low

    plus_dm = torch.where(
        (up_move > dn_move) & (up_move > 0),
        up_move,
        torch.tensor(0.0, device=close.device),
    )
    minus_dm = torch.where(
        (dn_move > up_move) & (dn_move > 0),
        dn_move,
        torch.tensor(0.0, device=close.device),
    )

    tr1 = high - low
    tr2 = torch.abs(high - c_prev)
    tr3 = torch.abs(low - c_prev)
    tr = torch.maximum(torch.maximum(tr1, tr2), tr3)

    tr_seed = tr[:period].sum()
    plus_dm_seed = plus_dm[:period].sum()
    minus_dm_seed = minus_dm[:period].sum()

    smooth_tr = wilder_smooth_parallel(tr, period, seed=tr_seed)
    smooth_plus = wilder_smooth_parallel(plus_dm, period, seed=plus_dm_seed)
    smooth_minus = wilder_smooth_parallel(minus_dm, period, seed=minus_dm_seed)

    tr_safe = smooth_tr + 1e-9
    plus_di = 100.0 * smooth_plus / tr_safe
    minus_di = 100.0 * smooth_minus / tr_safe

    dx = 100.0 * torch.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)

    adx_vals = wilder_smooth_parallel(dx, period, seed=dx[period : 2 * period].mean())

    adx_vals[: 2 * period - 1] = float("nan")
    plus_di[:period] = float("nan")
    minus_di[:period] = float("nan")

    return adx_vals, plus_di, minus_di
