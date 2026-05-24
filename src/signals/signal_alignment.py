"""P24-16: Signal alignment rule — fundamental direction MUST match technical.

Conflict → no trade. Ensures fundamental and technical signals reinforce,
filtering out contrarian entries that degrade win rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class AlignmentResult:
    """Result of signal alignment check."""

    technical_direction: int  # +1 long, -1 short, 0 neutral
    fundamental_bias: int  # +1 bullish, -1 bearish, 0 neutral
    aligned: bool
    qualifiers: List[str] = field(default_factory=list)


def check_signal_alignment(
    technical_score: float,
    fundamental_bias: float,
    min_technical_magnitude: float = 0.05,
    min_fundamental_magnitude: float = 0.10,
) -> AlignmentResult:
    """Check whether technical and fundamental signals agree on direction.

    Args:
        technical_score: Composite technical signal (positive = long bias).
        fundamental_bias: Fundamental signal (VIX slope, yield curve, etc.).
        min_technical_magnitude: Threshold to consider technical signal meaningful.
        min_fundamental_magnitude: Threshold to consider fundamental bias meaningful.

    Returns:
        AlignmentResult with alignment status.
    """
    tech_dir = (
        1
        if technical_score > min_technical_magnitude
        else -1
        if technical_score < -min_technical_magnitude
        else 0
    )
    fund_dir = (
        1
        if fundamental_bias > min_fundamental_magnitude
        else -1
        if fundamental_bias < -min_fundamental_magnitude
        else 0
    )

    qualifiers: List[str] = []

    if tech_dir == 0:
        qualifiers.append("technical_neutral")
    if fund_dir == 0:
        qualifiers.append("fundamental_neutral")

    aligned = tech_dir == fund_dir or fund_dir == 0

    return AlignmentResult(
        technical_direction=tech_dir,
        fundamental_bias=fund_dir,
        aligned=aligned,
        qualifiers=qualifiers,
    )


def compute_combined_fundamental_bias(
    vix_slope: Optional[float] = None,
    yield_curve: Optional[float] = None,
    pc_ratio: Optional[float] = None,
) -> float:
    """Aggregate fundamental signals into a single bias score [-1, 1].

    +VIX slope = bearish, +yield curve = bullish, high PC ratio = bearish.
    """
    bias = 0.0
    count = 0

    if vix_slope is not None:
        bias -= float(np.clip(vix_slope / 5.0, -1.0, 1.0))
        count += 1
    if yield_curve is not None:
        bias += float(np.clip(yield_curve / 2.0, -1.0, 1.0))
        count += 1
    if pc_ratio is not None:
        bias -= float(np.clip((pc_ratio - 0.7) / 0.5, -1.0, 1.0))
        count += 1

    if count > 0:
        bias /= count

    return float(np.clip(bias, -1.0, 1.0))
