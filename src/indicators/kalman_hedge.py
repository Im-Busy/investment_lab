"""
Kalman Filter Hedge Ratio for Pair Trading

Implements a Kalman filter to dynamically estimate the hedge ratio
between two cointegrated assets.

Reference:
- Chan, E. (2008). Quantitative Trading: How to Build Your Own Algorithmic Trading Business
- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pykalman import KalmanFilter


@dataclass
class KalmanHedgeResult:
    """Result from Kalman filter hedge ratio calculation."""

    hedge_ratios: np.ndarray
    spread: np.ndarray
    spread_std: np.ndarray
    state_covariance: np.ndarray


class KalmanHedgeRatio:
    """
    Kalman Filter for dynamic hedge ratio estimation.

    Models the relationship: price_A = β * price_B + ε
    where β evolves over time according to a random walk.

    Parameters:
        observation_std: Standard deviation of observation noise
        state_std: Standard deviation of state transition (hedge ratio drift)
        initial_hedge_ratio: Initial hedge ratio estimate
        initial_covariance: Initial state covariance
    """

    def __init__(
        self,
        observation_std: float = 1.0,
        state_std: float = 0.0001,
        initial_hedge_ratio: float = 1.0,
        initial_covariance: float = 1.0,
    ):
        self.observation_std = observation_std
        self.state_std = state_std
        self.initial_hedge_ratio = initial_hedge_ratio
        self.initial_covariance = initial_covariance

    def fit(self, price_a: np.ndarray, price_b: np.ndarray) -> KalmanHedgeResult:
        """
        Fit Kalman filter to price data.

        Args:
            price_a: Price series of asset A (dependent variable)
            price_b: Price series of asset B (independent variable)

        Returns:
            KalmanHedgeResult with hedge ratios, spread, and uncertainty
        """
        n = len(price_a)

        # Observation matrix (time-varying)
        observation_matrix = price_b.reshape(-1, 1)

        # Initialize Kalman Filter
        kf = KalmanFilter(
            observation_matrices=observation_matrix,
            observation_offsets=np.zeros(n),
            transition_matrices=np.eye(1),
            transition_offsets=np.zeros(1),
            observation_covariance=self.observation_std**2,
            transition_covariance=self.state_std**2 * np.eye(1),
            initial_state_mean=[[self.initial_hedge_ratio]],
            initial_state_covariance=[[self.initial_covariance]],
            em_vars=["transition_covariance", "observation_covariance"],
        )

        # Use EM algorithm to estimate noise parameters
        kf = kf.em(price_a, n_iter=5)

        # Smooth the estimates
        state_means, state_covariances = kf.smooth(price_a)

        # Extract hedge ratios
        hedge_ratios = state_means.flatten()

        # Calculate spread
        spread = price_a - hedge_ratios * price_b

        # Calculate spread standard deviation from state covariance
        spread_std = np.sqrt(state_covariances.flatten() * price_b**2 + self.observation_std**2)

        return KalmanHedgeResult(
            hedge_ratios=hedge_ratios,
            spread=spread,
            spread_std=spread_std,
            state_covariance=state_covariances,
        )

    def update(
        self,
        price_a: float,
        price_b: float,
        prev_hedge_ratio: float,
        prev_covariance: float,
    ) -> Tuple[float, float, float]:
        """
        Update hedge ratio with new observation (online mode).

        Args:
            price_a: Current price of asset A
            price_b: Current price of asset B
            prev_hedge_ratio: Previous hedge ratio estimate
            prev_covariance: Previous state covariance

        Returns:
            Tuple of (new_hedge_ratio, new_covariance, spread)
        """
        # State prediction
        pred_hedge = prev_hedge_ratio
        pred_cov = prev_covariance + self.state_std**2

        # Observation prediction
        pred_price = pred_hedge * price_b
        pred_obs_var = pred_cov * price_b**2 + self.observation_std**2

        # Kalman gain
        kalman_gain = pred_cov * price_b / pred_obs_var

        # State update
        innovation = price_a - pred_price
        new_hedge = pred_hedge + kalman_gain * innovation
        new_cov = (1 - kalman_gain * price_b) * pred_cov

        spread = price_a - new_hedge * price_b

        return new_hedge, new_cov, spread
