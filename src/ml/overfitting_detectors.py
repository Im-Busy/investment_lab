"""Overfitting detection from research papers (5520 + Backtest Overfitting papers)."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TrainingHistoryOverfitDetector:
    """Detect overfitting from training history curves (5520 paper, ICLR 2023).

    Trains a time-series classifier on validation loss curves to identify overfit models.
    Non-intrusive — uses validation loss which is a byproduct of training.

    Paper result: F1=0.91 on real-world DL training histories, 32% earlier stopping.
    """

    def __init__(
        self,
        divergence_threshold: float = 0.15,
        window_size: int = 20,
        min_epochs: int = 10,
    ):
        self.divergence_threshold = divergence_threshold
        self.window_size = window_size
        self.min_epochs = min_epochs

    def compute_divergence_score(
        self,
        train_losses: list[float],
        val_losses: list[float],
    ) -> float:
        """Compute overfitting divergence score from loss curves.

        Measures the gap between training and validation loss in the trailing window.
        Returns score in [0, 1] where > 0.7 indicates likely overfitting.

        Args:
            train_losses: Per-epoch training loss values
            val_losses: Per-epoch validation loss values
        """
        if len(train_losses) < self.min_epochs or len(val_losses) < self.min_epochs:
            return 0.0

        train_window = np.array(train_losses[-self.window_size :])
        val_window = np.array(val_losses[-self.window_size :])

        train_trend = np.polyfit(range(len(train_window)), train_window, 1)[0]
        val_trend = np.polyfit(range(len(val_window)), val_window, 1)[0]

        divergence = 0.0
        if train_trend < 0 and val_trend > 0:
            gap_ratio = abs(val_trend / (train_trend + 1e-8))
            divergence = min(gap_ratio / self.divergence_threshold, 1.0)

        final_gap = abs(val_window[-1] - train_window[-1]) / (abs(train_window[-1]) + 1e-8)
        divergence = max(divergence, min(final_gap / self.divergence_threshold, 1.0))

        return divergence

    def is_overfit(
        self, train_losses: list[float], val_losses: list[float]
    ) -> tuple[bool, float, Optional[int]]:
        """Detect if model is overfit and find optimal stopping epoch.

        Returns:
            (is_overfit, divergence_score, optimal_epoch)
        """
        score = self.compute_divergence_score(train_losses, val_losses)
        is_overfit = score > 0.7

        optimal_epoch = None
        if len(val_losses) >= self.min_epochs:
            optimal_epoch = int(np.argmin(val_losses))

        return is_overfit, score, optimal_epoch


class SyntheticOOSComparator:
    """Synthetic OOS comparison framework (Backtest Overfitting paper, SSRN 4686376).

    Generates synthetic OOS datasets using parametric market models
    (Heston, Merton Jump Diffusion, regime-switching).
    Compares real OOS performance against null distribution of synthetic runs.
    Computes a Generalization Score.

    Paper result: CPCV has lowest PBO among all CV methods.
    """

    DEFAULT_HESTON_PARAMS = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "v0": 0.04,
    }

    def __init__(
        self,
        n_synthetic_runs: int = 100,
        seed: int = 42,
    ):
        self.n_synthetic_runs = n_synthetic_runs
        self.rng = np.random.RandomState(seed)

    def generate_synthetic_returns(
        self,
        n_days: int,
        model: str = "heston",
    ) -> np.ndarray:
        """Generate synthetic daily returns from parametric market model.

        Args:
            n_days: Number of trading days to simulate
            model: "heston", "merton", "regime_switch", or "drift_burst"
        """
        if model == "heston":
            return self._heston_simulation(n_days)
        elif model == "merton":
            return self._merton_simulation(n_days)
        elif model == "regime_switch":
            return self._regime_switch_simulation(n_days)
        elif model == "drift_burst":
            return self._drift_burst_simulation(n_days)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _heston_simulation(self, n_days: int) -> np.ndarray:
        """Heston stochastic volatility model (Euler discretization)."""
        p = self.DEFAULT_HESTON_PARAMS
        dt = 1 / 252
        returns = np.zeros(n_days)
        v = p["v0"]

        for t in range(n_days):
            z1 = self.rng.randn()
            z2 = p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * self.rng.randn()
            v = max(
                v
                + p["kappa"] * (p["theta"] - v) * dt
                + p["sigma"] * np.sqrt(max(v, 0)) * np.sqrt(dt) * z2,
                1e-8,
            )
            returns[t] = np.sqrt(v) * np.sqrt(dt) * z1

        return returns

    def _merton_simulation(self, n_days: int) -> np.ndarray:
        """Merton Jump Diffusion model."""
        mu = 0.05 / 252
        sigma = 0.2 / np.sqrt(252)
        lambda_j = 0.1 / 252
        mu_j = -0.02
        sigma_j = 0.05

        diff = mu + sigma * self.rng.randn(n_days)
        n_jumps = self.rng.poisson(lambda_j, n_days)
        jumps = np.array([self.rng.normal(mu_j, sigma_j, int(nj)).sum() for nj in n_jumps])
        return diff + jumps

    def _regime_switch_simulation(self, n_days: int) -> np.ndarray:
        """Two-regime Markov switching model (bull/bear)."""
        mu = [0.0008, -0.0004]
        sigma = [0.012, 0.025]
        trans_prob = 0.02

        regime = 0 if self.rng.rand() < 0.7 else 1
        returns = np.zeros(n_days)

        for t in range(n_days):
            if self.rng.rand() < trans_prob:
                regime = 1 - regime
            returns[t] = self.rng.normal(mu[regime], sigma[regime])

        return returns

    def _drift_burst_simulation(self, n_days: int) -> np.ndarray:
        """Drift-Burst Hypothesis model with intermittent explosive drifts."""
        base_vol = 0.012
        burst_prob = 0.005
        burst_magnitude = 0.05

        returns = self.rng.normal(0, base_vol, n_days)
        bursts = self.rng.rand(n_days) < burst_prob
        returns[bursts] += self.rng.choice([-1, 1], size=bursts.sum()) * burst_magnitude

        return returns

    def compute_generalization_score(
        self,
        real_oos_sharpe: float,
        real_is_sharpe: float,
        n_oos_days: int,
        n_synthetic_runs: int | None = None,
    ) -> dict:
        """Compute generalization score by comparing real OOS to synthetic null.

        Returns dict with:
            - pbo: Probability of Backtest Overfitting (from paper)
            - generalization_score: 1.0 - PBO (higher = better generalization)
            - synthetic_sharpes: array of synthetic OOS Sharpes
            - is_better_than_random: bool if real OOS beats synthetic median
        """
        n_runs = n_synthetic_runs or self.n_synthetic_runs

        synthetic_sharpes = []
        for _ in range(n_runs):
            model_choice = self.rng.choice(["heston", "merton", "regime_switch"])
            syn_returns = self.generate_synthetic_returns(n_oos_days, model=model_choice)
            ann_return = syn_returns.mean() * 252
            ann_vol = syn_returns.std() * np.sqrt(252)
            syn_sharpe = ann_return / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
            synthetic_sharpes.append(syn_sharpe)

        synthetic_sharpes = np.array(synthetic_sharpes)

        pbo = (synthetic_sharpes >= real_oos_sharpe).mean()

        generalization_score = 1.0 - pbo

        is_oos_ratio = (
            real_is_sharpe / (real_oos_sharpe + 1e-8) if real_oos_sharpe > 0 else float("inf")
        )

        return {
            "pbo": float(pbo),
            "generalization_score": float(generalization_score),
            "is_oos_ratio": float(is_oos_ratio),
            "synthetic_median_sharpe": float(np.median(synthetic_sharpes)),
            "synthetic_std_sharpe": float(np.std(synthetic_sharpes)),
            "is_better_than_random": bool(real_oos_sharpe > np.median(synthetic_sharpes)),
            "n_synthetic_runs": n_runs,
        }
