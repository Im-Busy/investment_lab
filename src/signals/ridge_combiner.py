"""
Ridge-Regularized Signal Combiner.

Post-search factor aggregation using Ridge regression, as described in
"From Hypotheses to Factors" (arXiv:2604.26747v1, §4.5 and §5.2).

Workflow:
    1. Standardize each factor cross-sectionally by date (z-score within date)
    2. Stack factors into a matrix S_t (n_observations × n_factors)
    3. Fit Ridge(alpha=1.0) on training period only
    4. Generate composite scores for all periods
    5. Evaluate composite signal quality (IC, Sharpe, etc.)

The Ridge combiner is simpler and more interpretable than complex ML ensembles.
The paper achieved Sharpe 1.55 with this approach vs. more complex LightGBM alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


@dataclass
class RidgeCombineResult:
    """Result of Ridge factor aggregation."""

    coefficients: dict[str, float]
    intercept: float
    alpha: float
    n_factors: int
    n_train_obs: int
    train_r2: float
    composite_scores: pd.DataFrame | None = None
    composite_name: str = "composite"
    diagnostics: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"Ridge Combiner: {self.composite_name}",
            f"  N factors:   {self.n_factors}",
            f"  N train obs: {self.n_train_obs}",
            f"  Alpha:       {self.alpha}",
            f"  Train R²:    {self.train_r2:.4f}",
            "  Coefficients:",
        ]
        for name, coef in sorted(self.coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"    {name:<30} {coef:+.6f}")
        if self.intercept != 0:
            lines.append(f"  Intercept:   {self.intercept:.6f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "alpha": self.alpha,
            "n_factors": self.n_factors,
            "n_train_obs": self.n_train_obs,
            "train_r2": self.train_r2,
            "composite_name": self.composite_name,
            "diagnostics": self.diagnostics,
        }


class RidgeSignalCombiner:
    """Aggregate multiple factor signals into a composite score using Ridge regression.

    The combiner:
        1. Standardizes each factor cross-sectionally by date
        2. Fits Ridge(alpha) on the training period only
        3. Generates composite scores for all periods

    Parameters:
        alpha: L2 regularization strength (default 1.0, as in the paper).
        fit_intercept: Whether to fit an intercept term.
        standardize: Whether to z-score each factor cross-sectionally before fitting.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = False,
        standardize: bool = True,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self._model: Ridge | None = None
        self._factor_names: list[str] = []

    def fit(
        self,
        factor_scores: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        train_mask: np.ndarray | pd.Series | None = None,
        composite_name: str = "composite",
    ) -> RidgeCombineResult:
        """Fit Ridge regression on training data.

        Args:
            factor_scores: Dict of {factor_name: DataFrame(n_dates, n_tickers)}.
            forward_returns: DataFrame(n_dates, n_tickers) of forward returns.
            train_mask: Boolean mask for training observations. If None, uses all data.
            composite_name: Name for the composite signal.

        Returns:
            RidgeCombineResult with coefficients and diagnostics.
        """
        self._factor_names = sorted(factor_scores.keys())
        n_factors = len(self._factor_names)

        if n_factors == 0:
            raise ValueError("No factors provided")
        if n_factors == 1:
            logger.warning("Only one factor provided. Ridge reduces to scaling.")

        ref_shape = next(iter(factor_scores.values())).shape
        for name, scores in factor_scores.items():
            if scores.shape != ref_shape:
                raise ValueError(f"Shape mismatch for '{name}': {scores.shape} vs {ref_shape}")

        if forward_returns.shape != ref_shape:
            raise ValueError(f"Returns shape {forward_returns.shape} != factor shape {ref_shape}")

        X_parts: list[pd.DataFrame] = []
        for name in self._factor_names:
            scores = factor_scores[name].copy()
            if self.standardize:
                scores = scores.sub(scores.mean(axis=1), axis=0).div(
                    scores.std(axis=1).replace(0, np.nan), axis=0
                )
            stacked = scores.stack().rename(name)
            X_parts.append(stacked)

        X = pd.concat(X_parts, axis=1)
        y = forward_returns.stack().rename("forward_return")

        common_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if train_mask is not None:
            train_mask = train_mask.loc[common_idx] if hasattr(train_mask, "loc") else train_mask
            train_idx = common_idx[train_mask]
        else:
            train_idx = common_idx

        X_train = X.loc[train_idx].values
        y_train = y.loc[train_idx].values

        if len(X_train) < n_factors * 2:
            logger.warning(
                "Fewer training samples (%d) than 2× n_factors (%d). Results may be unstable.",
                len(X_train),
                n_factors,
            )

        self._model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_train)
        train_r2 = float(
            1 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
        )

        coefficients = dict(zip(self._factor_names, self._model.coef_))

        all_pred = self._model.predict(X.values)
        composite = pd.Series(all_pred, index=common_idx, name=composite_name)
        composite_df = composite.unstack()

        result = RidgeCombineResult(
            coefficients=coefficients,
            intercept=float(self._model.intercept_),
            alpha=self.alpha,
            n_factors=n_factors,
            n_train_obs=len(X_train),
            train_r2=train_r2,
            composite_scores=composite_df,
            composite_name=composite_name,
            diagnostics={
                "coef_norm": float(np.linalg.norm(self._model.coef_)),
                "coef_range": float(np.max(self._model.coef_) - np.min(self._model.coef_)),
            },
        )
        return result

    def predict(self, factor_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate composite scores using the fitted model.

        Args:
            factor_scores: Dict of {factor_name: DataFrame(n_dates, n_tickers)}.

        Returns:
            DataFrame of composite scores.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_parts: list[pd.DataFrame] = []
        for name in self._factor_names:
            scores = factor_scores[name].copy()
            if self.standardize:
                scores = scores.sub(scores.mean(axis=1), axis=0).div(
                    scores.std(axis=1).replace(0, np.nan), axis=0
                )
            X_parts.append(scores.stack().rename(name))

        X = pd.concat(X_parts, axis=1)
        common_idx = X.dropna().index

        preds = self._model.predict(X.loc[common_idx].values)
        composite = pd.Series(preds, index=common_idx, name="composite")
        return composite.unstack()


def from_dsl_recipes(
    recipes: dict[str, str],
    df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_start: str | None = None,
    train_end: str | None = None,
    alpha: float = 1.0,
) -> RidgeCombineResult:
    """End-to-end: parse DSL recipes, evaluate, and Ridge-combine.

    Args:
        recipes: Dict of {name: dsl_expression_string}.
        df: OHLCV DataFrame (MultiIndex or regular).
        forward_returns: Forward returns DataFrame.
        train_start: Start date for training period.
        train_end: End date for training period.
        alpha: Ridge regularization strength.

    Returns:
        RidgeCombineResult.
    """
    from src.patterns.dsl.executor import evaluate_string

    factor_scores: dict[str, pd.DataFrame] = {}
    for name, recipe in recipes.items():
        factor_scores[name] = evaluate_string(recipe, df)

    train_mask = None
    if train_start is not None or train_end is not None:
        idx = next(iter(factor_scores.values())).index
        mask = pd.Series(True, index=idx)
        if train_start:
            mask &= idx >= pd.Timestamp(train_start)
        if train_end:
            mask &= idx <= pd.Timestamp(train_end)
        stacked_mask = pd.DataFrame(
            {name: mask.values for name in factor_scores.keys()},
            index=idx,
        ).stack()
        train_mask = stacked_mask

    combiner = RidgeSignalCombiner(alpha=alpha)
    return combiner.fit(factor_scores, forward_returns, train_mask=train_mask)
