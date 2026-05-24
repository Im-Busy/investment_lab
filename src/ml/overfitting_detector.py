"""P24-9: Training history overfitting detector via KNN-DTW on loss curves.

Classifies overfitting from validation loss curve shape using Dynamic Time
Warping distance to known overfit/normal templates. Paper reports F1=0.91
and 32% earlier detection than epoch-count heuristics.

Depends on: dtaidistance (DTW computation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from dtaidistance import dtw

    HAS_DTAI = True
except ImportError:
    HAS_DTAI = False
    logger.warning("dtaidistance not installed — KNN-DTW overfitting detection disabled")


@dataclass
class LossCurveSample:
    """A labeled training history sample for template matching."""

    loss_curve: np.ndarray
    label: str  # "normal" or "overfit"
    source: str = ""


@dataclass
class OverfittingDetection:
    """Result of training history overfitting check."""

    is_overfit: bool
    confidence: float  # 0-1
    nearest_template: str
    dtw_distance: float
    loss_curve: np.ndarray
    early_detection_bar: int  # bar index where overfit first detected
    notes: List[str] = field(default_factory=list)


def _smooth_curve(curve: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing of loss curve."""
    if len(curve) < window:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode="same")


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = curve.min(), curve.max()
    if mx - mn < 1e-10:
        return np.zeros_like(curve)
    return (curve - mn) / (mx - mn)


def _build_templates() -> List[LossCurveSample]:
    """Build synthetic overfit/normal template curves.

    Normal: monotonically decreasing loss, plateau at ~40% of epochs.
    Overfit: decreasing then sharply rising divergence between train/val loss.
    Divergence-only curves focus on val_loss - train_loss gap.
    """
    n = 100
    x = np.linspace(0, 1, n)

    normal_val = 1.0 / (1.0 + np.exp(10 * (x - 0.15))) * 0.3 + 0.05
    normal_val += np.random.default_rng(42).normal(0, 0.01, n)
    normal_val = np.clip(normal_val, 0, 1)

    overfit_val = 1.0 / (1.0 + np.exp(10 * (x - 0.15))) * 0.15 + 0.05
    overfit_val += x * 0.4
    overfit_val += np.random.default_rng(43).normal(0, 0.01, n)
    overfit_val = np.clip(overfit_val, 0, 1)

    templates: List[LossCurveSample] = []
    for scale in [0.8, 1.0, 1.2]:
        templates.append(LossCurveSample(normal_val * scale, "normal", f"template_normal_{scale}"))
        templates.append(
            LossCurveSample(overfit_val * scale, "overfit", f"template_overfit_{scale}")
        )

    return templates


class TrainingHistoryOverfitDetector:
    """KNN-DTW classifier for overfitting from loss curves.

    Args:
        k: Number of nearest neighbors for classification.
        templates: Custom template curves (default: synthetic normal + overfit).
        dtw_window: Warping window constraint (percentage of curve length).
    """

    def __init__(
        self,
        k: int = 3,
        templates: Optional[List[LossCurveSample]] = None,
        dtw_window: float = 0.1,
    ):
        if not HAS_DTAI:
            raise ImportError(
                "dtaidistance required for KNN-DTW overfitting detection. "
                "Install with: uv add dtaidistance"
            )
        self.k = k
        self.templates = templates if templates is not None else _build_templates()
        self.dtw_window = dtw_window
        self._template_curves = [_normalize_curve(t.loss_curve) for t in self.templates]
        self._template_labels = [t.label for t in self.templates]

    def detect(
        self,
        val_loss: np.ndarray,
        train_loss: Optional[np.ndarray] = None,
    ) -> OverfittingDetection:
        """Detect overfitting from validation loss curve.

        Args:
            val_loss: Validation loss per epoch/bar.
            train_loss: Training loss per epoch/bar (optional — divergence-based).

        Returns:
            OverfittingDetection with classification result.
        """
        if train_loss is not None and len(train_loss) == len(val_loss):
            curve = val_loss - train_loss
        else:
            curve = val_loss.copy()

        curve = _smooth_curve(curve)
        curve_norm = _normalize_curve(curve)
        n = len(curve_norm)

        distances: List[Tuple[float, int]] = []
        for i, template_curve in enumerate(self._template_curves):
            t_upsampled = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(template_curve)),
                template_curve,
            )
            window = max(2, int(self.dtw_window * n))
            try:
                dist = dtw.distance(curve_norm, t_upsampled, window=window)
            except Exception:
                dist = 1e10
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        k_dist = distances[: self.k]

        overfit_votes = sum(1 for _, i in k_dist if self._template_labels[i] == "overfit")
        is_overfit = overfit_votes > self.k // 2
        confidence = float(min(1.0, overfit_votes / self.k + (1.0 / (1.0 + k_dist[0][0]))))

        early_bar = n
        if train_loss is not None and len(train_loss) == len(val_loss):
            divergence = val_loss - train_loss
            slope = np.gradient(_smooth_curve(divergence), 1)
            for i in range(10, len(slope)):
                if (slope[i - 10 : i + 1] > 0).sum() >= 8 and divergence[i] > divergence[
                    i - 10
                ] * 1.2:
                    early_bar = i
                    break

        nearest_label = self._template_labels[k_dist[0][1]]

        return OverfittingDetection(
            is_overfit=is_overfit,
            confidence=confidence,
            nearest_template=nearest_label,
            dtw_distance=k_dist[0][0],
            loss_curve=curve,
            early_detection_bar=early_bar if early_bar < n else n - 1,
            notes=(
                [f"Overfit detected at epoch {early_bar}" if is_overfit else "Loss curve normal"]
                if n > 0
                else ["Insufficient data"]
            ),
        )


def quick_overfitting_check(
    val_loss: np.ndarray,
    train_loss: Optional[np.ndarray] = None,
) -> bool:
    """Quick check: returns True if overfitting detected."""
    if not HAS_DTAI:
        logger.warning("dtaidistance not available — returning False")
        return False
    detector = TrainingHistoryOverfitDetector()
    result = detector.detect(val_loss, train_loss)
    return result.is_overfit
