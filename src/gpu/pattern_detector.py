"""
GPU-Accelerated Pattern Detection

Applies 6 GPU-friendly principles to all pattern detection:
1. Data parallelism: All bars processed simultaneously via tensor operations
2. Minimal branching: Mask-based selection replaces if/else chains
3. Coalesced memory: Contiguous [N,5] OHLCV tensor layout
4. Batched transfers: Single CPU->GPU transfer, all results on-device
5. Tensor core usage: Convolution/matmul for windowed operations
6. Maximized occupancy: PyTorch native kernels handle thread scheduling

Patterns implemented:
- Candlestick: Doji, Hammer, Engulfing, Harami, Dark Cloud Cover
- Basic: MSL/MSH, Two-Bar Reversal, NR7ID
- Breakout: Donchian Channel, Gap
- Pivot: Swing Highs/Lows (parallel max_pool1d)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .utils import get_device, to_numpy, to_tensor


@dataclass
class GPUSignalBatch:
    """Batch of detection signals across all bars for a pattern."""
    pattern_name: str
    detected: torch.Tensor  # [N] bool
    direction: torch.Tensor  # [N] int8: 0=neutral, 1=long, -1=short
    confidence: torch.Tensor  # [N] float32
    entry_price: torch.Tensor
    stop_loss: torch.Tensor
    take_profit_1: torch.Tensor
    take_profit_2: Optional[torch.Tensor] = None
    take_profit_3: Optional[torch.Tensor] = None
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_numpy(self) -> Dict[str, np.ndarray]:
        result = {}
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if field_name == "metadata":
                result[field_name] = {k: to_numpy(v) for k, v in val.items()}
            elif val is not None:
                result[field_name] = to_numpy(val)
        return result


class GPUPatternDetector:
    """
    GPU-accelerated pattern detector for 14+ chart patterns.

    Design: Single OHLCV tensor [N, 5] transferred to GPU once.
    All pattern detections operate on this tensor in-place on device.
    Results accumulated and batch-copied back to CPU.

    Principle 1 (Data Parallelism): Every detection is a tensor operation
    that processes all N bars simultaneously — no Python for-loops.

    Principle 2 (Minimal Branching): Boolean masks replace if/else.
    torch.where() handles conditional assignment on GPU.

    Principle 4 (Batched Transfers): ohlcv tensor stays on device.
    Only final signal arrays are copied back to CPU.
    """

    _DEFAULT_LOOKBACK = 5

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._ohlcv: Optional[torch.Tensor] = None
        self._n_bars: int = 0
        self._signals_cache: Dict[str, GPUSignalBatch] = {}

    def load_data(self, df: pd.DataFrame) -> None:
        """
        Transfer OHLCV data to GPU. Single transfer per backtest.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
        """
        ohlcv_np = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
        self._ohlcv = to_tensor(ohlcv_np, device=self.device, dtype=torch.float32)
        self._n_bars = self._ohlcv.shape[0]
        self._signals_cache.clear()

    @property
    def ohlcv(self) -> torch.Tensor:
        if self._ohlcv is None:
            raise RuntimeError("load_data() must be called before detection")
        return self._ohlcv

    @property
    def open(self) -> torch.Tensor:
        return self.ohlcv[:, 0]

    @property
    def high(self) -> torch.Tensor:
        return self.ohlcv[:, 1]

    @property
    def low(self) -> torch.Tensor:
        return self.ohlcv[:, 2]

    @property
    def close(self) -> torch.Tensor:
        return self.ohlcv[:, 3]

    @property
    def volume(self) -> torch.Tensor:
        return self.ohlcv[:, 4]

    def _shifted(self, col: torch.Tensor, offset: int) -> torch.Tensor:
        """Return column shifted by offset bars (NaN-padded at front)."""
        result = torch.full_like(col, float("nan"))
        if offset > 0:
            result[offset:] = col[:-offset]
        elif offset < 0:
            result[:offset] = col[-offset:]
        else:
            result.copy_(col)
        return result

    # ------------------------------------------------------------------
    # Candlestick Pattern Detections
    # ------------------------------------------------------------------

    def detect_doji(self, body_ratio_threshold: float = 0.1) -> GPUSignalBatch:
        """
        GPU Doji detection: body <= body_ratio_threshold of total range.

        Uses element-wise ops on [N] tensors — fully parallel.
        """
        body = torch.abs(self.close - self.open)
        total_range = self.high - self.low
        valid_range = total_range > 0

        body_ratio = torch.where(valid_range, body / total_range, torch.tensor(1.0, device=self.device))
        is_doji = body_ratio <= body_ratio_threshold

        upper_shadow = self.high - torch.maximum(self.open, self.close)
        lower_shadow = torch.minimum(self.open, self.close) - self.low

        shadow_sum = upper_shadow + lower_shadow
        valid_shadow = shadow_sum > 0

        dragonfly = torch.where(valid_range & valid_shadow,
                                (upper_shadow / total_range) <= 0.1,
                                torch.tensor(False, device=self.device))
        gravestone = torch.where(valid_range & valid_shadow,
                                 (lower_shadow / total_range) <= 0.1,
                                 torch.tensor(False, device=self.device))

        direction = torch.where(
            is_doji & dragonfly,
            torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(
                is_doji & gravestone,
                torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(self._n_bars, dtype=torch.int8, device=self.device),
            ),
        )

        confidence = torch.where(is_doji, torch.tensor(0.30, device=self.device), torch.zeros(self._n_bars, device=self.device))
        entry_price = torch.where(direction > 0, self.high + 0.01,
                          torch.where(direction < 0, self.low - 0.01,
                              torch.zeros(self._n_bars, device=self.device)))

        return GPUSignalBatch(
            pattern_name="Doji",
            detected=is_doji,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=torch.where(direction > 0, self.low - 0.01,
                          torch.where(direction < 0, self.high + 0.01,
                              torch.zeros(self._n_bars, device=self.device))),
            take_profit_1=entry_price * (1 + 0.02 * direction.float()),
        )

    def detect_engulfing(self, body_ratio_min: float = 0.3) -> GPUSignalBatch:
        """
        GPU Engulfing detection: 2-bar body engulfment with trend context.

        Uses shifted tensors for candle[i-1] vs candle[i] comparison.
        No loops — all bars processed in parallel via tensor ops.
        """
        o, h, l, c = self.open, self.high, self.low, self.close  # noqa: E741

        body = torch.abs(c - o)
        candle_range = h - l
        is_green = c > o
        is_red = c < o

        valid = candle_range > 0
        body_ratio = torch.where(valid, body / candle_range, torch.tensor(0.0, device=self.device))

        o_prev = torch.roll(o, 1, 0)
        c_prev = torch.roll(c, 1, 0)
        body_prev = torch.abs(c_prev - o_prev)
        is_green_prev = torch.roll(is_green, 1, 0)

        substantial_current = body_ratio >= body_ratio_min
        body_larger = body >= body_prev
        opposite_color = is_green != is_green_prev

        body_high = torch.maximum(o, c)
        body_low = torch.minimum(o, c)
        body_high_prev = torch.maximum(o_prev, c_prev)
        body_low_prev = torch.minimum(o_prev, c_prev)

        engulfs = (body_high > body_high_prev) & (body_low < body_low_prev)
        is_engulfing = substantial_current & body_larger & opposite_color & engulfs

        bull_engulf = is_engulfing & is_green
        bear_engulf = is_engulfing & is_red

        direction = torch.where(
            bull_engulf, torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(bear_engulf, torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(self._n_bars, dtype=torch.int8, device=self.device)),
        )

        confidence_base = torch.where(bull_engulf | bear_engulf,
                                      torch.tensor(0.65, device=self.device),
                                      torch.zeros(self._n_bars, device=self.device))

        return GPUSignalBatch(
            pattern_name="Engulfing",
            detected=is_engulfing,
            direction=direction,
            confidence=confidence_base,
            entry_price=torch.where(bull_engulf, h + 0.01,
                            torch.where(bear_engulf, l - 0.01,
                                torch.zeros(self._n_bars, device=self.device))),
            stop_loss=torch.where(bull_engulf, l - 0.01,
                          torch.where(bear_engulf, h + 0.01,
                              torch.zeros(self._n_bars, device=self.device))),
            take_profit_1=torch.where(direction != 0,
                                      self.close + (self.close * 0.03 * direction.float()),
                                      torch.zeros(self._n_bars, device=self.device)),
        )

    def detect_hammer(
        self, body_ratio_max: float = 0.3, lower_shadow_min: float = 0.6, trend_lookback: int = 20
    ) -> GPUSignalBatch:
        """GPU Hammer/Hanging Man detection."""
        body = torch.abs(self.close - self.open)
        total_range = self.high - self.low
        valid = total_range > 0

        body_ratio = torch.where(valid, body / total_range, torch.tensor(1.0, device=self.device))
        upper_shadow = self.high - torch.maximum(self.open, self.close)
        lower_shadow = torch.minimum(self.open, self.close) - self.low
        lower_ratio = torch.where(valid, lower_shadow / total_range, torch.tensor(0.0, device=self.device))

        is_hammer_body = body_ratio <= body_ratio_max
        is_hammer_shadow = lower_ratio >= lower_shadow_min
        is_hammer = is_hammer_body & is_hammer_shadow & valid

        trend_mask = self._trend_mask(lookback=trend_lookback)

        hammer_bull = is_hammer & trend_mask  # downtrend -> bull
        hammer_bear = is_hammer & ~trend_mask  # uptrend -> bear

        direction = torch.where(
            hammer_bull, torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(hammer_bear, torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(self._n_bars, dtype=torch.int8, device=self.device)),
        )

        return GPUSignalBatch(
            pattern_name="Hammer",
            detected=is_hammer,
            direction=direction,
            confidence=torch.where(is_hammer, torch.tensor(0.45, device=self.device),
                                   torch.zeros(self._n_bars, device=self.device)),
            entry_price=torch.where(hammer_bull, self.high + 0.01,
                            torch.where(hammer_bear, self.low - 0.01,
                                torch.zeros(self._n_bars, device=self.device))),
            stop_loss=torch.where(hammer_bull, self.low - 0.01,
                          torch.where(hammer_bear, self.high + 0.01,
                              torch.zeros(self._n_bars, device=self.device))),
            take_profit_1=torch.where(direction != 0,
                                      self.close * (1 + 0.04 * direction.float()),
                                      torch.zeros(self._n_bars, device=self.device)),
        )

    def detect_harami(self, body_ratio_min: float = 0.3) -> GPUSignalBatch:
        """GPU Harami detection: small body inside previous large body."""
        o, h, l, c = self.open, self.high, self.low, self.close

        body = torch.abs(c - o)
        body_high = torch.maximum(o, c)
        body_low = torch.minimum(o, c)
        is_green = c > o
        is_red = c < o

        body_prev = torch.roll(body, 1, 0)
        body_high_prev = torch.roll(body_high, 1, 0)
        body_low_prev = torch.roll(body_low, 1, 0)
        is_green_prev = torch.roll(is_green, 1, 0)

        range_prev = torch.roll(h - l, 1, 0)
        valid = range_prev > 0
        body_ratio_prev = torch.where(valid, body_prev / range_prev, torch.tensor(0.0, device=self.device))

        large_prev = body_ratio_prev >= body_ratio_min
        contained = (body_high <= body_high_prev) & (body_low >= body_low_prev)
        opposite_color = is_green != is_green_prev
        is_harami = large_prev & contained & opposite_color

        bull_harami = is_harami & is_green  # green inside red prev
        bear_harami = is_harami & is_red    # red inside green prev

        direction = torch.where(
            bull_harami, torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(bear_harami, torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(self._n_bars, dtype=torch.int8, device=self.device)),
        )

        return GPUSignalBatch(
            pattern_name="Harami",
            detected=is_harami,
            direction=direction,
            confidence=torch.where(is_harami, torch.tensor(0.40, device=self.device),
                                   torch.zeros(self._n_bars, device=self.device)),
            entry_price=torch.where(bull_harami, h + 0.01,
                            torch.where(bear_harami, l - 0.01,
                                torch.zeros(self._n_bars, device=self.device))),
            stop_loss=torch.where(bull_harami, body_low_prev - 0.01,
                          torch.where(bear_harami, body_high_prev + 0.01,
                              torch.zeros(self._n_bars, device=self.device))),
            take_profit_1=torch.where(direction != 0,
                                      self.close * (1 + 0.03 * direction.float()),
                                      torch.zeros(self._n_bars, device=self.device)),
        )

    def detect_dark_cloud_cover(self, body_ratio_min: float = 0.3, penetration_min: float = 0.5) -> GPUSignalBatch:
        """GPU Dark Cloud Cover / Piercing Line detection."""
        o, c = self.open, self.close
        o_prev = torch.roll(o, 1, 0)
        c_prev = torch.roll(c, 1, 0)
        h_prev = torch.roll(self.high, 1, 0)
        l_prev = torch.roll(self.low, 1, 0)

        is_green_prev = c_prev > o_prev
        is_red_curr = c < o
        is_red_prev = c_prev < o_prev
        is_green_curr = c > o

        body_prev = torch.abs(c_prev - o_prev)
        range_prev = h_prev - l_prev
        valid_prev = range_prev > 0

        large_prev = torch.where(valid_prev, body_prev / range_prev >= body_ratio_min,
                                 torch.tensor(False, device=self.device))

        dark_cloud = is_green_prev & is_red_curr & large_prev
        open_above_prev_high = o > h_prev
        close_below_mid = c < (o_prev + c_prev) / 2 * (1 - penetration_min) + (o_prev + c_prev) / 2 * penetration_min
        dark_cloud_confirmed = dark_cloud & open_above_prev_high & close_below_mid

        piercing = is_red_prev & is_green_curr & large_prev
        open_below_prev_low = o < l_prev
        close_above_mid = c > (o_prev + c_prev) / 2
        piercing_confirmed = piercing & open_below_prev_low & close_above_mid

        is_detected = dark_cloud_confirmed | piercing_confirmed

        direction = torch.where(
            dark_cloud_confirmed, torch.tensor(-1, dtype=torch.int8, device=self.device),
            torch.where(piercing_confirmed, torch.tensor(1, dtype=torch.int8, device=self.device),
                torch.zeros(self._n_bars, dtype=torch.int8, device=self.device)),
        )

        return GPUSignalBatch(
            pattern_name="DarkCloudCover",
            detected=is_detected,
            direction=direction,
            confidence=torch.where(is_detected, torch.tensor(0.55, device=self.device),
                                   torch.zeros(self._n_bars, device=self.device)),
            entry_price=torch.where(dark_cloud_confirmed, o - 0.01,
                            torch.where(piercing_confirmed, c + 0.01,
                                torch.zeros(self._n_bars, device=self.device))),
            stop_loss=torch.where(dark_cloud_confirmed, h_prev + 0.01,
                          torch.where(piercing_confirmed, l_prev - 0.01,
                              torch.zeros(self._n_bars, device=self.device))),
            take_profit_1=torch.where(direction != 0,
                                      self.close * (1 + 0.03 * direction.float().abs()),
                                      torch.zeros(self._n_bars, device=self.device)),
        )

    # ------------------------------------------------------------------
    # Basic Pattern Detections
    # ------------------------------------------------------------------

    def detect_msl(self, confirmation_offset: float = 0.01) -> GPUSignalBatch:
        """GPU MSL detection: 3-bar structure + confirmation at bar i."""
        c = self.close
        n = self._n_bars

        c_m3 = torch.roll(c, 3, 0)
        c_m2 = torch.roll(c, 2, 0)
        c_m1 = torch.roll(c, 1, 0)

        cond1 = c_m2 < c_m3
        cond2 = (c_m1 > c_m2) & (c_m1 < c_m3)
        pattern_form = cond1 & cond2

        max_close = torch.maximum(torch.maximum(c_m3, c_m2), c_m1)
        cond3 = c > max_close
        confirmed = pattern_form & cond3

        valid_range = torch.arange(n, device=self.device) >= 3

        direction = torch.where(confirmed & valid_range,
                                torch.tensor(1, dtype=torch.int8, device=self.device),
                                torch.zeros(n, dtype=torch.int8, device=self.device))

        entry_price = torch.where(confirmed & valid_range,
                                  max_close + confirmation_offset,
                                  torch.zeros(n, device=self.device))
        stop_loss = torch.where(confirmed & valid_range,
                                torch.roll(self.low, 1, 0) - confirmation_offset,
                                torch.zeros(n, device=self.device))
        risk = entry_price - stop_loss

        return GPUSignalBatch(
            pattern_name="MSL",
            detected=confirmed & valid_range,
            direction=direction,
            confidence=torch.where(confirmed & valid_range, torch.tensor(0.60, device=self.device),
                                   torch.zeros(n, device=self.device)),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=entry_price + risk * 2,
            take_profit_2=entry_price + risk * 3,
        )

    def detect_msh(self, confirmation_offset: float = 0.01) -> GPUSignalBatch:
        """GPU MSH detection: inverse of MSL."""
        c = self.close
        n = self._n_bars

        c_m3 = torch.roll(c, 3, 0)
        c_m2 = torch.roll(c, 2, 0)
        c_m1 = torch.roll(c, 1, 0)

        cond1 = c_m2 > c_m3
        cond2 = (c_m1 < c_m2) & (c_m1 > c_m3)
        pattern_form = cond1 & cond2

        min_close = torch.minimum(torch.minimum(c_m3, c_m2), c_m1)
        cond3 = c < min_close
        confirmed = pattern_form & cond3

        valid_range = torch.arange(n, device=self.device) >= 3

        direction = torch.where(confirmed & valid_range,
                                torch.tensor(-1, dtype=torch.int8, device=self.device),
                                torch.zeros(n, dtype=torch.int8, device=self.device))

        entry_price = torch.where(confirmed & valid_range,
                                  min_close - confirmation_offset,
                                  torch.zeros(n, device=self.device))
        stop_loss = torch.where(confirmed & valid_range,
                                torch.roll(self.high, 1, 0) + confirmation_offset,
                                torch.zeros(n, device=self.device))
        risk = stop_loss - entry_price

        return GPUSignalBatch(
            pattern_name="MSH",
            detected=confirmed & valid_range,
            direction=direction,
            confidence=torch.where(confirmed & valid_range, torch.tensor(0.60, device=self.device),
                                   torch.zeros(n, device=self.device)),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=entry_price - risk * 2,
            take_profit_2=entry_price - risk * 3,
        )

    def detect_two_bar_reversal(
        self, range_mult: float = 1.2, trend_lookback: int = 10
    ) -> GPUSignalBatch:
        """GPU Two-Bar Reversal: expanded range + reversal close positioning."""
        h, l, o, c = self.high, self.low, self.open, self.close
        n = self._n_bars

        bar_range = h - l
        avg_range = self._rolling_mean(bar_range, trend_lookback)
        valid_range = avg_range > 0

        bar1_range = torch.roll(bar_range, 1, 0)
        bar2_range = bar_range
        is_wide1 = bar1_range > avg_range * range_mult
        is_wide2 = bar2_range > avg_range * range_mult
        is_wide = is_wide1 & is_wide2 & valid_range

        bar1_bear = torch.roll(c < o, 1, 0)
        bar1_l = torch.roll(l, 1, 0)
        bar1_c = torch.roll(c, 1, 0)
        bar1_range_val = torch.roll(bar_range, 1, 0)

        bar1_close_ratio = torch.where(bar1_range_val > 0,
                                       (bar1_c - bar1_l) / bar1_range_val,
                                       torch.tensor(0.5, device=self.device))
        bar2_close_ratio = torch.where(bar_range > 0,
                                       (c - l) / bar_range,
                                       torch.tensor(0.5, device=self.device))

        pipe_bottom = is_wide & bar1_bear & (bar1_close_ratio < 0.3) & (bar2_close_ratio > 0.5)
        pipe_top = is_wide & (~bar1_bear) & (bar1_close_ratio > 0.7) & (bar2_close_ratio < 0.5)

        direction = torch.where(
            pipe_bottom, torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(pipe_top, torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(n, dtype=torch.int8, device=self.device)),
        )

        is_detected = pipe_bottom | pipe_top

        return GPUSignalBatch(
            pattern_name="TwoBarReversal",
            detected=is_detected,
            direction=direction,
            confidence=torch.where(is_detected, torch.tensor(0.50, device=self.device),
                                   torch.zeros(n, device=self.device)),
            entry_price=torch.where(direction > 0, h + 0.01,
                            torch.where(direction < 0, l - 0.01,
                                torch.zeros(n, device=self.device))),
            stop_loss=torch.where(direction > 0, l - 0.01,
                          torch.where(direction < 0, h + 0.01,
                              torch.zeros(n, device=self.device))),
            take_profit_1=torch.where(direction != 0,
                                      self.close * (1 + 0.04 * direction.float()),
                                      torch.zeros(n, device=self.device)),
        )

    def detect_nr7id(self, lookback: int = 50) -> GPUSignalBatch:
        """GPU NR7ID detection: narrowest range in 7 bars + highest volume in 50 bars."""
        bar_range = self.high - self.low
        n = self._n_bars

        range_window = F.pad(bar_range.unsqueeze(0).unsqueeze(0), (6, 0), mode="replicate")
        range_min7 = range_window.unfold(2, 7, 1).min(dim=3).values.squeeze(0).squeeze(0)

        is_nr7 = (bar_range == range_min7) & (bar_range > 0)
        is_nr7[:6] = False

        vol_window = F.pad(self.volume.unsqueeze(0).unsqueeze(0), (lookback - 1, 0), mode="replicate")
        vol_max = vol_window.unfold(2, lookback, 1).max(dim=3).values.squeeze(0).squeeze(0)

        is_id = self.volume >= vol_max

        combined = is_nr7 & is_id
        valid = torch.arange(n, device=self.device) >= max(6, lookback - 1)

        return GPUSignalBatch(
            pattern_name="NR7ID",
            detected=combined & valid,
            direction=torch.zeros(n, dtype=torch.int8, device=self.device),
            confidence=torch.where(combined & valid,
                                   torch.tensor(0.35, device=self.device),
                                   torch.zeros(n, device=self.device)),
            entry_price=self.close,
            stop_loss=self.close * 0.98,
            take_profit_1=self.close * 1.04,
        )

    # ------------------------------------------------------------------
    # Breakout Pattern Detections
    # ------------------------------------------------------------------

    def detect_donchian_breakout(self, period: int = 20) -> Tuple[GPUSignalBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU Donchian Channel Breakout: parallel rolling max/min via unfolding.

        Returns:
            (signal_batch, upper_band, lower_band, middle_band)
        """
        h, l, c = self.high, self.low, self.close
        n = self._n_bars

        h_pad = F.pad(h.unsqueeze(0).unsqueeze(0), (period - 1, 0), mode="replicate")
        l_pad = F.pad(l.unsqueeze(0).unsqueeze(0), (period - 1, 0), mode="replicate")

        upper = h_pad.unfold(2, period, 1).max(dim=3).values.squeeze(0).squeeze(0)
        lower = l_pad.unfold(2, period, 1).min(dim=3).values.squeeze(0).squeeze(0)
        middle = (upper + lower) / 2

        prev_upper = torch.roll(upper, 1, 0)
        prev_lower = torch.roll(lower, 1, 0)

        breakout_up = c > prev_upper
        breakdown_dn = c < prev_lower
        is_breakout = breakout_up | breakdown_dn

        direction = torch.where(
            breakout_up, torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(breakdown_dn, torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(n, dtype=torch.int8, device=self.device)),
        )

        vol_ratio = self.volume / (self._rolling_mean(self.volume, 20) + 1e-9)
        vol_confirmed = vol_ratio > 1.5

        confidence = torch.where(is_breakout & vol_confirmed,
                                 torch.tensor(0.65, device=self.device),
                                 torch.where(is_breakout,
                                             torch.tensor(0.50, device=self.device),
                                             torch.zeros(n, device=self.device)))

        return GPUSignalBatch(
            pattern_name="DonchianBreakout",
            detected=is_breakout,
            direction=direction,
            confidence=confidence,
            entry_price=torch.where(direction > 0, prev_upper,
                            torch.where(direction < 0, prev_lower,
                                torch.zeros(n, device=self.device))),
            stop_loss=torch.where(direction > 0, middle,
                          torch.where(direction < 0, middle,
                              torch.zeros(n, device=self.device))),
            take_profit_1=torch.where(direction > 0, c + (c - lower) * 2,
                              torch.where(direction < 0, c - (upper - c) * 2,
                                  torch.zeros(n, device=self.device))),
        ), upper, lower, middle

    def detect_gap(self, gap_threshold_pct: float = 0.005) -> GPUSignalBatch:
        """GPU Gap detection: opening gap vs previous close."""
        o = self.open
        c_prev = torch.roll(self.close, 1, 0)
        n = self._n_bars

        gap_pct = (o - c_prev) / (c_prev + 1e-9)

        gap_up = gap_pct > gap_threshold_pct
        gap_dn = gap_pct < -gap_threshold_pct

        is_gap = gap_up | gap_dn
        valid = torch.arange(n, device=self.device) >= 1

        direction = torch.where(
            gap_up, torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.where(gap_dn, torch.tensor(-1, dtype=torch.int8, device=self.device),
                torch.zeros(n, dtype=torch.int8, device=self.device)),
        )

        return GPUSignalBatch(
            pattern_name="Gap",
            detected=is_gap & valid,
            direction=direction,
            confidence=torch.where(is_gap & valid, torch.tensor(0.45, device=self.device),
                                   torch.zeros(n, device=self.device)),
            entry_price=o,
            stop_loss=torch.where(direction > 0, c_prev,
                          torch.where(direction < 0, c_prev,
                              torch.zeros(n, device=self.device))),
            take_profit_1=o * (1 + 0.02 * direction.float()),
        )

    # ------------------------------------------------------------------
    # Pivot Detection (Swing Highs/Lows)
    # ------------------------------------------------------------------

    def detect_swing_points(self, lookback: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU swing high/low detection using parallel max_pool1d.

        Principle 1 (Data Parallelism): All bars evaluated simultaneously
        via max_pool1d comparison instead of nested for-loops.

        Returns:
            (swing_highs [N] bool, swing_lows [N] bool)
        """
        h = self.high.view(1, 1, -1)
        l = self.low.view(1, 1, -1)

        kernel_size = 2 * lookback + 1
        local_max = F.max_pool1d(h, kernel_size=kernel_size, stride=1, padding=lookback)
        local_min = -F.max_pool1d(-l, kernel_size=kernel_size, stride=1, padding=lookback)

        h_local = h.squeeze(0).squeeze(0)
        l_local = l.squeeze(0).squeeze(0)
        max_val = local_max.squeeze(0).squeeze(0)
        min_val = local_min.squeeze(0).squeeze(0)

        swing_highs = (h_local == max_val) & (h_local > 0)
        swing_lows = (l_local == min_val) & (l_local > 0)

        swing_highs[:lookback] = False
        swing_lows[:lookback] = False

        n = self._n_bars
        swing_highs[n - lookback:] = False
        swing_lows[n - lookback:] = False

        return swing_highs, swing_lows

    # ------------------------------------------------------------------
    # Signal Summary & Reporting
    # ------------------------------------------------------------------

    def detect_all(self) -> List[GPUSignalBatch]:
        """Run all GPU pattern detections and return the results."""
        results: List[GPUSignalBatch] = []

        results.append(self.detect_msl())
        results.append(self.detect_msh())
        results.append(self.detect_doji())
        results.append(self.detect_hammer())
        results.append(self.detect_engulfing())
        results.append(self.detect_harami())
        results.append(self.detect_dark_cloud_cover())
        results.append(self.detect_two_bar_reversal())
        results.append(self.detect_nr7id())
        dc_signal, _, _, _ = self.detect_donchian_breakout()
        results.append(dc_signal)
        results.append(self.detect_gap())

        self._signals_cache = {r.pattern_name: r for r in results}
        return results

    def get_signal_matrix(self) -> torch.Tensor:
        """
        Aggregate all pattern signals into a unified matrix.

        Returns:
            [N, num_patterns] int8 tensor: -1=short, 0=neutral, 1=long
        """
        if not self._signals_cache:
            self.detect_all()

        directions = []
        for _, signal_batch in self._signals_cache.items():
            directions.append(signal_batch.direction)
        return torch.stack(directions, dim=1)

    def get_confidence_matrix(self) -> torch.Tensor:
        """
        Aggregate all pattern confidences into a matrix.

        Returns:
            [N, num_patterns] float32 tensor
        """
        if not self._signals_cache:
            self.detect_all()

        confidences = []
        for _, signal_batch in self._signals_cache.items():
            confidences.append(signal_batch.confidence)
        return torch.stack(confidences, dim=1)

    def to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Export all GPU-detected signals to a DataFrame for backtesting.

        Args:
            df: Original DataFrame (for timestamps/index alignment)

        Returns:
            DataFrame with columns: date, pattern_name, direction, confidence,
            entry, stop, tp1, tp2, tp3
        """
        if not self._signals_cache:
            self.detect_all()

        rows = []
        for pattern_name, signal_batch in self._signals_cache.items():
            detected_np = to_numpy(signal_batch.detected)
            dir_np = to_numpy(signal_batch.direction)
            conf_np = to_numpy(signal_batch.confidence)
            entry_np = to_numpy(signal_batch.entry_price)
            stop_np = to_numpy(signal_batch.stop_loss)
            tp1_np = to_numpy(signal_batch.take_profit_1)

            for i in range(self._n_bars):
                if detected_np[i]:
                    rows.append({
                        "date": df.index[i],
                        "bar_index": i,
                        "pattern_name": pattern_name,
                        "direction": "Long" if dir_np[i] > 0 else "Short",
                        "confidence": conf_np[i],
                        "entry_price": entry_np[i],
                        "stop_loss": stop_np[i],
                        "take_profit_1": tp1_np[i],
                    })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------------

    def _rolling_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Parallel rolling mean via padding + unfold + mean."""
        if window <= 1:
            return x.clone()
        pad = F.pad(x.unsqueeze(0).unsqueeze(0), (window - 1, 0), mode="replicate")
        return pad.unfold(2, window, 1).mean(dim=3).squeeze(0).squeeze(0)

    def _trend_mask(self, lookback: int = 20) -> torch.Tensor:
        """
        Compute uptrend mask using linear regression slope over lookback.

        Returns:
            [N] bool: True = uptrend, False = downtrend
        """
        c = self.close

        indices = torch.arange(lookback, device=self.device, dtype=torch.float32)
        x_mean = indices.mean()
        denom = ((indices - x_mean) ** 2).sum()

        if denom == 0:
            denom = torch.tensor(1.0, device=self.device)

        c_pad = F.pad(c.unsqueeze(0).unsqueeze(0), (lookback - 1, 0), mode="replicate")
        windows = c_pad.unfold(2, lookback, 1).squeeze(0).squeeze(0)  # [N, lookback]

        y_mean = windows.mean(dim=1)
        numerator = ((windows - y_mean.unsqueeze(1)) * (indices - x_mean)).sum(dim=1)
        slope = numerator / denom

        return slope > 0
