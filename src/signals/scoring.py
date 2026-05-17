"""
R7: Multi-Dimensional Signal Scoring.

Replaces single confidence score with 5-axis scoring:
1. IC (predictive accuracy) — rank correlation signal vs forward return
2. IR (stability) — IC mean / IC std over rolling window
3. Turnover impact — how often the signal changes direction
4. Diversity — correlation with other active signals (want low)
5. Overfitting risk — IS IC / OOS IC ratio

Source: Beyond Fama-French §4, 华泰多因子 §1.3

Architecture:
    consumed by SignalGenerator, ConfluenceScorer, RulesFirstStrategy._compute_score()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ── Score thresholds ──
_IC_FLOOR = 0.0
_IR_FLOOR = 0.0
_TURNOVER_EXPECTED = 0.15  # expected daily turnover for a reasonable signal


@dataclass
class MultiAxisScore:
    """Five-axis decomposed signal quality score.

    Attributes:
        ic: Predictive accuracy (Spearman rank correlation with forward return).
            0-1 scale, higher = more predictive.  Floor-clamped to 0.
        ir: Information Ratio stability (IC mean / IC std over rolling window).
            0-1 scale, higher = more stable. Raw IR capped at 3 then ÷3.
        turnover: Signal stability (1 - turnover_rate). 0-1 scale,
            higher = more stable (less direction flipping).
        diversity: Signal uniqueness (1 - max_correlation with other signals).
            0-1 scale, higher = less redundant.
        overfit_risk: Overfitting risk (1 - |IS_IC - OOS_IC| / max(IS_IC, OOS_IC, eps)).
            0-1 scale, higher = less overfit. 1.0 when IS and OOS match perfectly.
        composite: Geometric mean of all five axes. 0-1 scale.
        breakdown: Per-axis raw values (IC, IR, turnover_rate, max_corr, OOS_IC).
            Raw values before normalization for audit trail.
    """

    ic: float
    ir: float
    turnover: float
    diversity: float
    overfit_risk: float
    composite: float
    breakdown: Dict[str, float] = field(default_factory=dict)


class MultiAxisScorer:
    """Score trading signals across five quality dimensions.

    Replaces single ``confidence`` with decomposed scoring. Each axis
    is normalized to [0, 1] where higher is consistently better. The
    composite score is the geometric mean of all five axes.

    Parameters:
        ir_window: Rolling window size for IR stability computation.
        is_oos_split: Index where IS period ends and OOS begins (for overfit).

    Usage:
        scorer = MultiAxisScorer(ir_window=60, is_oos_split=400)
        score = scorer.score(signal_arr, forward_ret, other_signals)
        print(f"Composite: {score.composite:.3f}, IC: {score.ic:.3f}")
    """

    def __init__(
        self,
        ir_window: int = 60,
        is_oos_split: Optional[int] = None,
    ) -> None:
        self._ir_window = ir_window
        self._is_oos_split = is_oos_split

    def score(
        self,
        signal: np.ndarray,
        forward_returns: np.ndarray,
        other_signals: Optional[Dict[str, np.ndarray]] = None,
    ) -> MultiAxisScore:
        """Compute 5-axis score for a signal array.

        Args:
            signal: (N,) array of signal values (-1/0/1 for directional,
                or continuous for probability signals).
            forward_returns: (N,) array of 1-bar forward returns aligned
                with signal. forward_returns[t] = return from t to t+1.
            other_signals: Dict of other active signal name → (N,) array.
                Used for diversity scoring. If None, diversity = 0.5.

        Returns:
            MultiAxisScore with all five axes and composite.
        """
        signal = np.asarray(signal, dtype=np.float64)
        forward_returns = np.asarray(forward_returns, dtype=np.float64)
        n = len(signal)

        # ── Axis 1: IC (predictive accuracy) ──
        mask = ~(np.isnan(signal) | np.isnan(forward_returns))
        if mask.sum() < 3:
            return self._na_score()
        ic_raw, _ = spearmanr(signal[mask], forward_returns[mask])
        if np.isnan(ic_raw):
            ic_raw = 0.0
        ic_norm = max(float(ic_raw), 0.0)  # floor at 0

        # ── Axis 2: IR (stability) ──
        ir_norm = self._compute_ir(signal, forward_returns)

        # ── Axis 3: Turnover ──
        turnover_norm = self._compute_turnover(signal)

        # ── Axis 4: Diversity ──
        diversity_norm = self._compute_diversity(signal, other_signals)

        # ── Axis 5: Overfitting risk ──
        overfit_norm = self._compute_overfit(signal, forward_returns)

        # ── Composite: geometric mean ──
        axes = np.array([ic_norm, ir_norm, turnover_norm, diversity_norm, overfit_norm])
        axes = np.clip(axes, 1e-10, 1.0)
        composite = float(np.exp(np.mean(np.log(axes))))

        return MultiAxisScore(
            ic=ic_norm,
            ir=ir_norm,
            turnover=turnover_norm,
            diversity=diversity_norm,
            overfit_risk=overfit_norm,
            composite=composite,
            breakdown={
                "ic_raw": ic_raw,
                "ir_raw": self._raw_ir,
                "turnover_rate": self._raw_turnover,
                "max_corr": self._raw_max_corr,
                "oos_ic": self._raw_oos_ic,
                "is_ic": self._raw_is_ic,
            },
        )

    def _compute_ir(self, signal: np.ndarray, forward_returns: np.ndarray) -> float:
        """Compute normalized IR stability score."""
        n = len(signal)
        winsize = min(self._ir_window, n)
        if winsize < 10:
            self._raw_ir = 0.0
            return 0.0

        ics = []
        for start in range(0, n - winsize + 1, max(1, winsize // 4)):
            win_sig = signal[start : start + winsize]
            win_ret = forward_returns[start : start + winsize]
            mask = ~(np.isnan(win_sig) | np.isnan(win_ret))
            if mask.sum() < 5:
                continue
            ic, _ = spearmanr(win_sig[mask], win_ret[mask])
            if not np.isnan(ic):
                ics.append(ic)

        if len(ics) < 2:
            self._raw_ir = 0.0
            return 0.0

        mu = np.mean(ics)
        sigma = np.std(ics, ddof=1)
        ir_raw = mu / max(sigma, 1e-10)
        self._raw_ir = ir_raw

        ir_norm = np.clip(max(ir_raw, 0.0) / 3.0, 0.0, 1.0)
        return float(ir_norm)

    def _compute_turnover(self, signal: np.ndarray) -> float:
        """Compute normalized turnover (stability) score.

        Turnover = fraction of bars where signal changes direction.
        Lower turnover = higher score.
        """
        sig = signal[~np.isnan(signal)]
        if len(sig) < 2:
            self._raw_turnover = 0.0
            return 0.5

        changes = (sig[1:] != sig[:-1]).sum()
        turnover_rate = changes / (len(sig) - 1)
        self._raw_turnover = float(turnover_rate)

        # Score: 1 - turnover_rate, with floor. Scales relative to expected.
        stability = 1.0 - turnover_rate
        return float(np.clip(stability, 0.0, 1.0))

    def _compute_diversity(
        self,
        signal: np.ndarray,
        other_signals: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """Compute normalized diversity score.

        Diversity = 1 - max(|correlation|) with other signals.
        If no other signals, returns 0.5 (neutral).
        """
        if other_signals is None or len(other_signals) == 0:
            self._raw_max_corr = 0.0
            return 0.5

        max_corr = 0.0
        mask = ~np.isnan(signal)
        for name, other in other_signals.items():
            other = np.asarray(other, dtype=np.float64)
            joint = mask & ~np.isnan(other)
            if joint.sum() < 5:
                continue
            corr = np.abs(np.corrcoef(signal[joint], other[joint])[0, 1])
            if not np.isnan(corr):
                max_corr = max(max_corr, corr)

        self._raw_max_corr = float(max_corr)
        diversity = 1.0 - min(max_corr, 1.0)
        return float(np.clip(diversity, 0.0, 1.0))

    def _compute_overfit(self, signal: np.ndarray, forward_returns: np.ndarray) -> float:
        """Compute normalized overfitting risk score.

        Compares IS IC vs OOS IC. Score = 1 - |IS - OOS| / max(|IS|, |OOS|, eps).
        If no OOS split defined, returns 0.5 (neutral).
        """
        split = self._is_oos_split
        is_ic = 0.0
        oos_ic = 0.0

        # Always compute full-sample IC for reference
        mask = ~(np.isnan(signal) | np.isnan(forward_returns))
        if mask.sum() >= 3:
            full_ic, _ = spearmanr(signal[mask], forward_returns[mask])
            if not np.isnan(full_ic):
                is_ic = float(full_ic)
                oos_ic = float(full_ic)

        if split is not None and 0 < split < len(signal):
            # IS
            mask_is = mask.copy()
            mask_is[split:] = False
            is_sig = signal[mask_is]
            is_ret = forward_returns[mask_is]
            if len(is_sig) >= 5:
                ic_is, _ = spearmanr(is_sig, is_ret)
                if not np.isnan(ic_is):
                    is_ic = float(ic_is)

            # OOS
            mask_oos = mask.copy()
            mask_oos[:split] = False
            oos_sig = signal[mask_oos]
            oos_ret = forward_returns[mask_oos]
            if len(oos_sig) >= 5:
                ic_oos, _ = spearmanr(oos_sig, oos_ret)
                if not np.isnan(ic_oos):
                    oos_ic = float(ic_oos)

        self._raw_is_ic = is_ic
        self._raw_oos_ic = oos_ic

        if split is None:
            return 0.5

        max_abs = max(abs(is_ic), abs(oos_ic), 1e-10)
        gap = abs(is_ic - oos_ic)
        overfit_score = 1.0 - min(gap / max_abs, 1.0)
        return float(np.clip(overfit_score, 0.0, 1.0))

    def _na_score(self) -> MultiAxisScore:
        return MultiAxisScore(
            ic=0.0,
            ir=0.0,
            turnover=0.0,
            diversity=0.0,
            overfit_risk=0.0,
            composite=0.0,
            breakdown={
                "ic_raw": 0.0,
                "ir_raw": 0.0,
                "turnover_rate": 0.0,
                "max_corr": 0.0,
                "oos_ic": 0.0,
                "is_ic": 0.0,
            },
        )

    def score_many(
        self,
        signals: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> Dict[str, MultiAxisScore]:
        """Score multiple signals simultaneously with proper diversity.

        Each signal's diversity is computed against all other signals.

        Args:
            signals: Dict of signal_name -> (N,) array.
            forward_returns: (N,) array of forward returns.

        Returns:
            Dict of signal_name -> MultiAxisScore.
        """
        results: Dict[str, MultiAxisScore] = {}
        for name, sig in signals.items():
            others = {k: v for k, v in signals.items() if k != name}
            results[name] = self.score(sig, forward_returns, others)
        return results

    def rank_by_composite(
        self,
        signals: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> list[tuple[str, MultiAxisScore]]:
        """Score and rank multiple signals by composite score.

        Returns list of (name, score) sorted descending by composite.
        """
        scored = self.score_many(signals, forward_returns)
        ranked = sorted(scored.items(), key=lambda x: x[1].composite, reverse=True)
        return ranked


def quick_5axis_report(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    other_signals: Optional[Dict[str, np.ndarray]] = None,
    name: str = "signal",
) -> str:
    """Generate a one-line 5-axis report string.

    Args:
        signal: Signal array.
        forward_returns: Forward returns array.
        other_signals: Other active signals for diversity.
        name: Signal name for display.

    Returns:
        Formatted report string.
    """
    scorer = MultiAxisScorer()
    s = scorer.score(signal, forward_returns, other_signals)
    return (
        f"{name:20s} | composite={s.composite:.4f} | "
        f"IC={s.ic:.4f} IR={s.ir:.4f} TO={s.turnover:.4f} "
        f"DIV={s.diversity:.4f} OF={s.overfit_risk:.4f}"
    )
