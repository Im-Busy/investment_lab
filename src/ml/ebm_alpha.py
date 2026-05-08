"""EBM Shape Function Alpha Research Pipeline.

Extracts per-feature contribution curves from a trained EBM (Explainable
Boosting Machine) regime classifier and identifies quantifiable alpha signals.

Pipeline:
  1. Load fitted EBMRegimeClassifier
  2. Extract per-feature shape functions via explain_global()
  3. Classify contributions by shape (linear+/-, U-shaped, threshold, flat)
  4. Rank features by alpha potential
  5. Generate tradeable alpha signals from top-ranked features

Reference: Nori et al., "InterpretML", 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class AlphaShape(Enum):
    LINEAR_INCREASING = "linear_increasing"
    LINEAR_DECREASING = "linear_decreasing"
    U_SHAPED = "u_shaped"
    INVERTED_U = "inverted_u"
    THRESHOLD = "threshold"
    MONOTONIC_NONLINEAR = "monotonic_nonlinear"
    FLAT = "flat"
    COMPLEX = "complex"


@dataclass
class AlphaFeature:
    feature_name: str
    alpha_shape: AlphaShape
    alpha_score: float
    monotonicity: float
    magnitude: float
    consistency: float
    direction: float
    threshold_value: Optional[float]
    signal_description: str
    contribution_range: Tuple[float, float]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "alpha_shape": self.alpha_shape.value,
            "alpha_score": round(self.alpha_score, 4),
            "monotonicity": round(self.monotonicity, 4),
            "magnitude": round(self.magnitude, 4),
            "consistency": round(self.consistency, 4),
            "direction": round(self.direction, 4),
            "threshold_value": (round(self.threshold_value, 4) if self.threshold_value else None),
            "signal_description": self.signal_description,
            "contribution_range": [
                round(self.contribution_range[0], 4),
                round(self.contribution_range[1], 4),
            ],
            "recommendation": self.recommendation,
        }


@dataclass
class AlphaResult:
    model_name: str
    n_features_total: int
    n_alpha_features: int
    alpha_features: List[AlphaFeature]
    summary: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if self.table.empty and self.alpha_features:
            self.table = pd.DataFrame([f.to_dict() for f in self.alpha_features])

    def top_features(self, n: int = 10) -> pd.DataFrame:
        if self.table.empty:
            return pd.DataFrame()
        return self.table.head(n)

    def __repr__(self) -> str:
        return (
            f"AlphaResult(model={self.model_name}, "
            f"alpha_features={self.n_alpha_features}/{self.n_features_total})"
        )


class EBMAlphaPipeline:
    """Pipeline for extracting alpha signals from EBM shape functions.

    Takes a trained EBMRegimeClassifier, extracts per-feature contribution
    curves, and ranks features by their potential as alpha signals.

    Usage:
        >>> pipeline = EBMAlphaPipeline(ebm_model)
        >>> result = pipeline.run()
        >>> print(result.table[["feature_name", "alpha_score", "alpha_shape"]])
    """

    _SHAPE_WEIGHTS: Dict[AlphaShape, float] = {
        AlphaShape.LINEAR_INCREASING: 0.9,
        AlphaShape.LINEAR_DECREASING: 0.9,
        AlphaShape.U_SHAPED: 0.7,
        AlphaShape.INVERTED_U: 0.7,
        AlphaShape.THRESHOLD: 0.75,
        AlphaShape.MONOTONIC_NONLINEAR: 0.6,
        AlphaShape.FLAT: 0.1,
        AlphaShape.COMPLEX: 0.3,
    }

    def __init__(
        self,
        ebm_model: Any,
        alpha_score_threshold: float = 0.3,
    ) -> None:
        if not getattr(ebm_model, "is_trained", False):
            raise ValueError("EBM model must be trained before running alpha pipeline.")
        self._ebm = ebm_model
        self._alpha_score_threshold = alpha_score_threshold

    def run(self) -> AlphaResult:
        global_exp = self._ebm.explain_global()
        intercept = global_exp.get("intercept", 0.0)
        feat_names = self._ebm.feature_names_
        contributions = global_exp.get("feature_importance", {})

        if not contributions:
            return AlphaResult(
                model_name="EBM",
                n_features_total=len(feat_names),
                n_alpha_features=0,
                alpha_features=[],
                summary="No feature contributions extracted from EBM model.",
            )

        normalized = self._normalize_contributions(contributions, intercept)
        shape_data = global_exp.get("shape_functions", {})

        alpha_features: List[AlphaFeature] = []
        for feat_name in feat_names:
            contrib_val = contributions.get(feat_name, 0.0)
            contrib_data = shape_data.get(feat_name, [])
            norm_val = normalized.get(feat_name, 0.0)

            shape_info = self._classify_shape(contrib_data)
            mono = self._compute_monotonicity(contrib_data)
            mag = abs(norm_val)
            direction = 1.0 if contrib_val > 0 else -1.0

            alpha_score = self._compute_alpha(shape_info["shape"], mono, mag)

            if alpha_score >= self._alpha_score_threshold:
                desc, rec = self._gen_signal_desc(feat_name, shape_info, direction)
                crange = self._contribution_range(contrib_data)
                alpha_features.append(
                    AlphaFeature(
                        feature_name=feat_name,
                        alpha_shape=shape_info["shape"],
                        alpha_score=alpha_score,
                        monotonicity=mono,
                        magnitude=mag,
                        consistency=0.8,
                        direction=direction,
                        threshold_value=shape_info.get("threshold"),
                        signal_description=desc,
                        contribution_range=crange,
                        recommendation=rec,
                    )
                )

        alpha_features.sort(key=lambda x: x.alpha_score, reverse=True)

        return AlphaResult(
            model_name="EBM",
            n_features_total=len(feat_names),
            n_alpha_features=len(alpha_features),
            alpha_features=alpha_features,
            summary=self._gen_summary(alpha_features, len(feat_names)),
        )

    def _classify_shape(self, contrib_data: list) -> Dict[str, Any]:
        if not contrib_data:
            return {"shape": AlphaShape.FLAT, "threshold": None}

        values = []
        for item in contrib_data:
            if isinstance(item, dict):
                values.append(float(item.get("contribution", 0.0)))
            else:
                values.append(0.0)

        if len(values) < 3:
            return {"shape": AlphaShape.FLAT, "threshold": None}

        arr = np.array(values)
        if float(np.ptp(arr)) < 1e-6:
            return {"shape": AlphaShape.FLAT, "threshold": None}

        diffs = np.diff(arr)
        sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))
        is_inc = bool(np.all(diffs >= 0))
        is_dec = bool(np.all(diffs <= 0))

        if is_inc:
            shape = (
                AlphaShape.LINEAR_INCREASING
                if self._is_linear(arr)
                else AlphaShape.MONOTONIC_NONLINEAR
            )
        elif is_dec:
            shape = (
                AlphaShape.LINEAR_DECREASING
                if self._is_linear(arr)
                else AlphaShape.MONOTONIC_NONLINEAR
            )
        elif sign_changes == 1:
            mid = len(arr) // 2
            if arr[mid] < arr[0] and arr[mid] < arr[-1]:
                shape = AlphaShape.U_SHAPED
            elif arr[mid] > arr[0] and arr[mid] > arr[-1]:
                shape = AlphaShape.INVERTED_U
            else:
                shape = AlphaShape.THRESHOLD
        elif sign_changes == 2:
            shape = AlphaShape.THRESHOLD
        else:
            shape = AlphaShape.COMPLEX

        threshold = None
        if shape == AlphaShape.THRESHOLD:
            t_idx = int(np.argmax(np.abs(np.diff(arr))))
            threshold = float(t_idx) / max(len(arr), 1)

        return {"shape": shape, "threshold": threshold}

    def _is_linear(self, arr: np.ndarray) -> bool:
        if len(arr) < 3:
            return True
        x = np.arange(len(arr), dtype=np.float64)
        slope, intercept = np.polyfit(x, arr, 1)
        predicted = slope * x + intercept
        ss_res = float(np.sum((arr - predicted) ** 2))
        ss_tot = float(np.sum((arr - np.mean(arr)) ** 2))
        if ss_tot < 1e-10:
            return True
        return (1.0 - ss_res / ss_tot) > 0.85

    def _compute_monotonicity(self, contrib_data: list) -> float:
        if not contrib_data:
            return 0.0
        values = []
        for item in contrib_data:
            if isinstance(item, dict):
                values.append(float(item.get("contribution", 0.0)))
            else:
                values.append(0.0)
        if len(values) < 2:
            return 0.0
        diffs = np.diff(np.array(values))
        if len(diffs) == 0:
            return 0.0
        dominant = max(int(np.sum(diffs >= 0)), int(np.sum(diffs <= 0)))
        return float(dominant / len(diffs))

    def _compute_alpha(self, shape: AlphaShape, mono: float, mag: float) -> float:
        shape_score = self._SHAPE_WEIGHTS.get(shape, 0.3)
        alpha = 0.4 * shape_score + 0.35 * mono + 0.25 * min(mag * 5, 1.0)
        return float(np.clip(alpha, 0.0, 1.0))

    def _normalize_contributions(
        self, contributions: Dict[str, float], intercept: float
    ) -> Dict[str, float]:
        if abs(intercept) < 1e-8:
            return {k: 1.0 for k in contributions}
        return {k: abs(v / intercept) for k, v in contributions.items()}

    def _gen_signal_desc(
        self, name: str, shape_info: Dict[str, Any], direction: float
    ) -> Tuple[str, str]:
        shape = shape_info["shape"]
        clean = name.replace("_", " ").title()
        desc_map = {
            AlphaShape.LINEAR_INCREASING: f"{clean} rises with value",
            AlphaShape.LINEAR_DECREASING: f"{clean} falls with value",
            AlphaShape.U_SHAPED: f"{clean} optimal at extremes",
            AlphaShape.INVERTED_U: f"{clean} optimal at mid-range",
            AlphaShape.THRESHOLD: f"{clean} shows threshold behavior",
            AlphaShape.MONOTONIC_NONLINEAR: f"{clean} non-linear monotonic",
            AlphaShape.FLAT: f"{clean} minimal variation",
            AlphaShape.COMPLEX: f"{clean} complex pattern",
        }
        desc = desc_map.get(shape, f"{clean} contribution pattern")

        rec_map = {
            (AlphaShape.LINEAR_INCREASING, True): f"long when {name} above 50th pctile",
            (AlphaShape.LINEAR_DECREASING, True): f"long when {name} below 50th pctile",
            (AlphaShape.U_SHAPED, True): f"long when {name} in top/bottom decile",
            (AlphaShape.INVERTED_U, True): f"long when {name} in 40-60th pctile",
            (AlphaShape.THRESHOLD, True): f"long when {name} crosses above threshold",
            (AlphaShape.LINEAR_INCREASING, False): f"reduce when {name} above 50th pctile",
            (AlphaShape.LINEAR_DECREASING, False): f"reduce when {name} below 50th pctile",
        }
        rec = rec_map.get((shape, direction > 0), f"Monitor {name} for signal")
        return desc, rec

    def _contribution_range(self, contrib_data: list) -> Tuple[float, float]:
        if not contrib_data:
            return (0.0, 0.0)
        values = [
            float(item.get("contribution", 0.0)) if isinstance(item, dict) else 0.0
            for item in contrib_data
        ]
        if not values:
            return (0.0, 0.0)
        return (float(min(values)), float(max(values)))

    def _gen_summary(self, features: List[AlphaFeature], n_total: int) -> str:
        if not features:
            return (
                f"No features passed alpha threshold ({self._alpha_score_threshold}). "
                "Consider lowering threshold or improving EBM fit."
            )
        shapes: Dict[str, int] = {}
        for f in features:
            shapes[f.alpha_shape.value] = shapes.get(f.alpha_shape.value, 0) + 1
        shape_desc = ", ".join(f"{k}:{v}" for k, v in sorted(shapes.items()))
        top = features[0]
        return (
            f"Identified {len(features)} alpha features (of {n_total}). "
            f"Shapes: {shape_desc}. "
            f"Top: {top.feature_name} (score={top.alpha_score:.3f}, "
            f"shape={top.alpha_shape.value})."
        )

    def export_to_csv(self, result: AlphaResult, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(out, index=False)
