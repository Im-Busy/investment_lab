"""
Hidden Markov Model (HMM) Regime Detector

Uses Gaussian HMM to detect latent market regimes from technical features.
Implemented with hmmlearn library.

Features:
- Automatic regime discovery via HMM state transitions
- Probability-based regime assignment
- Transition matrix analysis
- Supports 3-5 hidden states

Example:
    >>> from src.ml.hmm_regime import HMMRegimeDetector
    >>> detector = HMMRegimeDetector(n_states=4)
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
    >>> probs = detector.predict_proba(features)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary, RegimeType


class HMMRegimeDetector(RegimeDetectorBase):
    """
    Hidden Markov Model regime detector.

    Uses Gaussian HMM to identify latent market regimes from technical features.
    The HMM assumes:
    - Market operates in K latent states (regimes)
    - Transitions between states follow a Markov process
    - Observations in each state follow a Gaussian distribution

    Args:
        n_states: Number of hidden regimes (3-5 recommended)
        covariance_type: HMM covariance type ("diag", "spherical", "tied", "full")
        max_iter: Maximum EM iterations for convergence
        random_state: Random seed for reproducibility
        n_init: Number of HMM initializations (not used in hmmlearn 0.3+)

    Attributes:
        model_: Trained GaussianHMM model
        transition_matrix_: Estimated state transition matrix
        means_: Mean of observations in each state
        covariances_: Covariance of observations in each state
    """

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "diag",
        max_iter: int = 200,
        random_state: int = 42,
        n_init: int = 3,
        min_samples: int = 50,
    ):
        super().__init__(name="HMMRegimeDetector")

        if n_states < 2 or n_states > 10:
            raise ValueError("n_states must be between 2 and 10")

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init
        self.min_samples = min_samples

        self.model_ = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.scaler_ = StandardScaler()
        self.regime_mapping_: Dict[int, str] = {}
        self.fitted_n_samples_ = 0

    def fit(self, data: pd.DataFrame) -> HMMRegimeDetector:
        """
        Fit HMM on input features.

        Args:
            data: DataFrame with technical features (rows=observations, columns=features)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If insufficient samples or features
        """
        from hmmlearn import hmm

        self._validate_data(data)

        # Scale features
        X = self.scaler_.fit_transform(data)

        # Fit HMM
        self.model_ = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.max_iter,  # Note: parameter is 'n_iter' not 'max_iter'
            random_state=self.random_state,
            verbose=False,
        )

        self.model_.fit(X)
        self.is_fitted = True
        self.fitted_n_samples_ = len(data)

        # Store model parameters
        self.transition_matrix_ = self.model_.transmat_
        self.means_ = self.model_.means_
        self.covariances_ = self.model_.covars_

        # Create interpretable regime mapping based on volatility
        self._create_regime_mapping(X)

        return self

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data before fitting."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(data)} < {self.min_samples} (minimum required)"
            )

        if data.isna().any().any():
            raise ValueError("Input data contains NaN values. Handle missing values first.")

        if len(data.columns) == 0:
            raise ValueError("Input data has no features")

    def _create_regime_mapping(self, X: np.ndarray) -> None:
        """
        Create interpretable regime labels based on state characteristics.

        Maps HMM states to meaningful regime names based on:
        - Mean return (bullish vs bearish)
        - Variance (high vs low volatility)
        """
        state_stats = []

        for i in range(self.n_states):
            mean_return = self.means_[i].mean()
            variance = (
                self.covariances_[i].mean()
                if hasattr(self.covariances_[i], "mean")
                else self.covariances_[i]
            )
            state_stats.append(
                {
                    "state": i,
                    "mean_return": mean_return,
                    "variance": variance,
                }
            )

        # Sort states by mean return
        sorted_states = sorted(state_stats, key=lambda x: x["mean_return"], reverse=True)

        # Assign regime names
        self.regime_mapping_ = {}
        n_high_vol = max(1, self.n_states // 3)
        n_low_vol = max(1, self.n_states // 3)

        for rank, state_info in enumerate(sorted_states):
            state_idx = state_info["state"]
            variance_rank = sorted(
                state_stats,
                key=lambda x: x["variance"],
                reverse=True,
            ).index(state_info)

            if variance_rank < n_high_vol:
                self.regime_mapping_[state_idx] = f"High_Vol_{rank}"
            elif variance_rank >= self.n_states - n_low_vol:
                self.regime_mapping_[state_idx] = f"Low_Vol_{rank}"
            else:
                self.regime_mapping_[state_idx] = f"Neutral_{rank}"

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for input data.

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            Series of regime labels (strings) with same index as input
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.scaler_.transform(data)
        hidden_states = self.model_.predict(X)

        # Map to interpretable labels
        labels = [self.regime_mapping_.get(s, f"State_{s}") for s in hidden_states]

        return pd.Series(labels, index=data.index, name="hmm_regime")

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities for input data.

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            DataFrame with probability for each regime (columns sum to 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.scaler_.transform(data)
        probs = self.model_.predict_proba(X)

        # Create column names from regime mapping
        columns = [self.regime_mapping_.get(i, f"State_{i}") for i in range(self.n_states)]

        return pd.DataFrame(
            probs,
            columns=columns,
            index=data.index,
        )

    def get_hidden_states(self, data: pd.DataFrame) -> pd.Series:
        """
        Get raw HMM hidden state indices (0 to n_states-1).

        Args:
            data: DataFrame with features

        Returns:
            Series of integer state labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.scaler_.transform(data)
        hidden_states = self.model_.predict(X)

        return pd.Series(hidden_states, index=data.index, name="hmm_state")

    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about detected regimes.

        Returns:
            RegimeSummary with label distribution and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        labels = list(self.regime_mapping_.values())
        unique_labels = list(set(labels))

        # Create a temporary distribution from mapping
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = sum(label_counts.values())
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=self.n_states,
            regime_labels=unique_labels,
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "transition_matrix": self.transition_matrix_.tolist()
                if self.transition_matrix_ is not None
                else None,
                "state_means": self.means_.tolist() if self.means_ is not None else None,
                "n_samples_fitted": self.fitted_n_samples_,
                "n_features": len(self.means_[0]) if self.means_ is not None else 0,
                "convergence": getattr(self.model_, "converged_", True),
                "n_iter": getattr(self.model_, "n_iter_", 0),
            },
        )

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the estimated state transition matrix.

        Returns:
            DataFrame with transition probabilities P(state_j | state_i)
        """
        if not self.is_fitted or self.transition_matrix_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        columns = [self.regime_mapping_.get(i, f"State_{i}") for i in range(self.n_states)]

        return pd.DataFrame(
            self.transition_matrix_,
            index=columns,
            columns=columns,
        )

    def get_expected_duration(self) -> pd.Series:
        """
        Calculate expected duration (in bars) for each regime.

        For HMM, expected duration in state i is: 1 / (1 - p_ii)
        where p_ii is the self-transition probability.

        Returns:
            Series with expected duration for each regime
        """
        if not self.is_fitted or self.transition_matrix_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        durations = {}
        for i in range(self.n_states):
            p_ii = self.transition_matrix_[i, i]
            expected_duration = 1 / (1 - p_ii) if p_ii < 1 else float("inf")
            label = self.regime_mapping_.get(i, f"State_{i}")
            durations[label] = expected_duration

        return pd.Series(durations, name="expected_duration_bars")

    def get_regime_volatility(self) -> pd.DataFrame:
        """
        Get volatility metrics for each regime.

        Returns:
            DataFrame with mean and variance for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        results = []
        for i in range(self.n_states):
            label = self.regime_mapping_.get(i, f"State_{i}")
            mean_ret = self.means_[i].mean()
            var = (
                self.covariances_[i].mean()
                if hasattr(self.covariances_[i], "mean")
                else self.covariances_[i]
            )
            results.append(
                {
                    "regime": label,
                    "mean_return": mean_ret,
                    "variance": var,
                    "volatility": np.sqrt(var),
                }
            )

        return pd.DataFrame(results).set_index("regime")
