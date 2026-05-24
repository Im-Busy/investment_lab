"""SMC-aware risk management — structural stops, killzone limits, OB-based sizing.

Bridges SMC pattern detection with risk management, replacing generic
ATR-based stops and fixed R-multiples with SMC structural levels.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SMCStructuralRisk:
    """Risk parameters derived from SMC structural context."""

    entry_price: float
    stop_price: float
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    risk_per_unit: float = 0.0
    reward_risk_ratio: float = 0.0
    position_risk_pct: float = 1.0
    killzone_active: bool = False


def compute_smc_structural_stop(
    entry_price: float,
    entry_direction: int,
    ob_levels: np.ndarray,
    ob_directions: np.ndarray,
    breaker_levels: np.ndarray,
    breaker_directions: np.ndarray,
    atr: float,
    min_atr_buffer: float = 0.5,
) -> float:
    """Place stop loss behind the nearest SMC structural level.

    ICT rules:
    - Long: SL below nearest Bullish OB (demand zone) or Bullish Breaker
    - Short: SL above nearest Bearish OB (supply zone) or Bearish Breaker
    - If no structural level nearby, fall back to ATR-based stop
    """
    if entry_direction == 1:
        candidates: list[float] = []
        for i in range(len(ob_levels)):
            if ob_directions[i] == 1 and ob_levels[i] < entry_price:
                candidates.append(ob_levels[i])
        for i in range(len(breaker_levels)):
            if breaker_directions[i] == 1 and breaker_levels[i] < entry_price:
                candidates.append(breaker_levels[i])
        if candidates:
            return max(candidates) - min_atr_buffer * atr
        return entry_price - 3.0 * atr
    else:
        candidates = []
        for i in range(len(ob_levels)):
            if ob_directions[i] == -1 and ob_levels[i] > entry_price:
                candidates.append(ob_levels[i])
        for i in range(len(breaker_levels)):
            if breaker_directions[i] == -1 and breaker_levels[i] > entry_price:
                candidates.append(breaker_levels[i])
        if candidates:
            return min(candidates) + min_atr_buffer * atr
        return entry_price + 3.0 * atr


def compute_smc_partial_exit(
    entry_price: float,
    entry_direction: int,
    fvg_top: np.ndarray,
    fvg_bottom: np.ndarray,
    fvg_direction: np.ndarray,
    liquidity_levels: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    """Compute TP1 (first FVG fill) and TP2 (first liquidity level) per ICT MMXM."""
    tp1 = None
    tp2 = None

    if entry_direction == 1:
        for i in range(len(fvg_top)):
            if fvg_direction[i] == -1 and fvg_top[i] > entry_price:
                if tp1 is None or fvg_top[i] < tp1:
                    tp1 = fvg_top[i]
        for level in liquidity_levels:
            if level > entry_price:
                if tp2 is None or (tp1 is not None and level < tp2 and level > tp1):
                    tp2 = level
    else:
        for i in range(len(fvg_bottom)):
            if fvg_direction[i] == 1 and fvg_bottom[i] < entry_price:
                if tp1 is None or fvg_bottom[i] > tp1:
                    tp1 = fvg_bottom[i]
        for level in liquidity_levels:
            if level < entry_price:
                if tp2 is None or (tp1 is not None and level > tp2 and level < tp1):
                    tp2 = level

    return tp1, tp2


def compute_ob_based_position_size(
    base_risk_pct: float = 1.0,
    ob_strength: float = 50.0,
    killzone_active: bool = False,
    confluence_count: int = 1,
) -> float:
    """Scale position size based on OB strength and SMC context.

    SMC rules:
    - Base: 1% per setup
    - Strong OB (strength > 70%): up to 2%
    - Multiple confluence: +0.25% per additional component
    - Active killzone: +0.25%
    - Max: 2% per trade
    """
    risk = base_risk_pct

    if ob_strength > 70:
        risk += 0.5
    elif ob_strength > 50:
        risk += 0.25

    risk += 0.25 * max(0, confluence_count - 1)

    if killzone_active:
        risk += 0.25

    return min(risk, 2.0)


def compute_daily_smc_limits(
    current_pnl_pct: float,
    daily_profit_target_pct: float = 5.0,
    smc_consecutive_failures: int = 0,
    max_consecutive_losses: int = 3,
) -> dict[str, object]:
    """SMC-specific daily limits.

    From ICT 1 Year Trading Plan:
    - Daily profit target reached → stop for the day
    - 3 consecutive losses → step away
    - Maximum daily loss = 3%
    """
    limits: dict[str, object] = {"should_stop_trading": False, "reason": ""}
    if current_pnl_pct >= daily_profit_target_pct:
        limits["should_stop_trading"] = True
        limits["reason"] = f"Daily profit target {daily_profit_target_pct}% reached"
    elif smc_consecutive_failures >= max_consecutive_losses:
        limits["should_stop_trading"] = True
        limits["reason"] = f"{smc_consecutive_failures} consecutive SMC failures"
    return limits


class SMCCircuitBreaker:
    """SMC-specific circuit breakers.

    - If 3+ consecutive sweeps fail → halt SMC signals
    - If BOS structure invalidated → halt
    """

    def __init__(self, cooldown_bars: int = 20):
        self.consecutive_sweep_failures = 0
        self.consecutive_bos_failures = 0
        self.cooldown_until = -1
        self.cooldown_bars = cooldown_bars

    def check(self, bar_idx: int, sweep_failed: bool, bos_failed: bool) -> bool:
        """Check if SMC signals should be halted. Returns True if HALT."""
        if bar_idx < self.cooldown_until:
            return True

        if sweep_failed:
            self.consecutive_sweep_failures += 1
        else:
            self.consecutive_sweep_failures = 0

        if bos_failed:
            self.consecutive_bos_failures += 1
        else:
            self.consecutive_bos_failures = 0

        if self.consecutive_sweep_failures >= 3:
            self.cooldown_until = bar_idx + self.cooldown_bars
            self.consecutive_sweep_failures = 0
            return True

        if self.consecutive_bos_failures >= 3:
            self.cooldown_until = bar_idx + self.cooldown_bars
            self.consecutive_bos_failures = 0
            return True

        return False
