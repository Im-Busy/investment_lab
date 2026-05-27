"""
DTW DeD-iMs convergence metric for GAN training on time series.

Combines Dynamic Time Warping (DTW) for temporal fidelity assessment
with Deep Dataset Dissimilarity Measure (DeD-iMs) for dataset-level
distribution comparison. Used to monitor TTS-GAN training progress
on financial time series where Wasserstein distance converges
prematurely before temporal patterns stabilize.

Reference: Podobinski & Chudziak (2024), "Financial time series augmentation
using transformer based GAN architecture" — Warsaw University of Technology.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two 1-D sequences.

    Uses squared Euclidean cost and O(N*M) dynamic programming.
    For sequences of length 90-120, this is ~8k-14k operations.

    Args:
        x: First time series, shape (T,).
        y: Second time series, shape (T',).

    Returns:
        DTW distance (lower = more similar).
    """
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(np.sqrt(dtw[n, m]))


def _dtw_distance_batch(real_sample: np.ndarray, synth_sample: np.ndarray) -> float:
    """DTW distance between two multi-variate samples.

    Averages DTW across feature dimensions.

    Args:
        real_sample: Real sample, shape (T, D).
        synth_sample: Synthetic sample, shape (T, D).

    Returns:
        Mean DTW distance across D dimensions.
    """
    if real_sample.ndim == 1:
        return _dtw_distance(real_sample, synth_sample)
    distances = [
        _dtw_distance(real_sample[:, d], synth_sample[:, d]) for d in range(real_sample.shape[1])
    ]
    return float(np.mean(distances))


def compute_dedims(
    set_a: np.ndarray,
    set_b: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> float:
    """Deep Dataset Dissimilarity Measure (DeD-iMs) using DTW.

    Compares datasets by measuring how intra-dataset distances differ from
    cross-dataset distances. Uses DTW as the base distance metric.

    Algorithm (Stolte et al., 2024):
    1. Draw n random samples from sets A and B.
    2. For each sample in A, find closest sample in B via DTW → D_cross.
    3. For each sample in A, find closest sample in A via DTW → D_ref.
    4. DeD-iMs = mean(|D_cross - D_ref|).

    Args:
        set_a: Real dataset, shape (N_A, T, D).
        set_b: Synthetic dataset, shape (N_B, T, D).
        n_samples: Number of samples to draw for comparison.
        seed: Random seed for reproducibility.

    Returns:
        DeD-iMs dissimilarity score (lower = more similar datasets).
    """
    rng = np.random.RandomState(seed)
    n_a = min(n_samples, len(set_a))
    n_b = min(n_samples, len(set_b))

    idx_a = rng.choice(len(set_a), size=n_a, replace=False)
    idx_b = rng.choice(len(set_b), size=n_b, replace=False)
    idx_a_ref = rng.choice(len(set_a), size=n_a, replace=False)

    cross_dists = np.zeros(n_a)
    ref_dists = np.zeros(n_a)

    for i, ai in enumerate(idx_a):
        cross_dists[i] = min(_dtw_distance_batch(set_a[ai], set_b[bi]) for bi in idx_b)

    for i, ai in enumerate(idx_a):
        ar = idx_a_ref[i]
        ref_dists[i] = min(
            _dtw_distance_batch(set_a[ai], set_a[aj]) for j, aj in enumerate(idx_a) if j != i
        )

    return float(np.mean(np.abs(cross_dists - ref_dists)))


def compute_dtw_dedims(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> float:
    """Combined DTW DeD-iMs convergence metric.

    This is the primary metric for monitoring TTS-GAN training.
    Lower values indicate better temporal fidelity and distributional similarity.

    Args:
        real_data: Real time series, shape (N, T, D).
        synth_data: Synthetic time series, shape (N, T, D).
        n_samples: Samples for DeD-iMs computation.
        seed: Random seed.

    Returns:
        DTW DeD-iMs score.
    """
    return compute_dedims(real_data, synth_data, n_samples=n_samples, seed=seed)


def compute_wasserstein_distance(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    n_bins: int = 50,
) -> float:
    """Approximate 1-D Wasserstein distance via histogram matching.

    Computes the Earth Mover's Distance between real and synthetic
    data distributions by flattening all values into 1-D and comparing
    histograms.

    Args:
        real_data: Real time series, shape (N, T, D).
        synth_data: Synthetic time series, shape (N, T, D).
        n_bins: Number of histogram bins.

    Returns:
        Wasserstein distance (lower = more similar distributions).
    """
    real_flat = real_data.ravel()
    synth_flat = synth_data.ravel()
    combined = np.concatenate([real_flat, synth_flat])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

    real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_flat, bins=bins, density=True)

    real_cdf = (
        np.cumsum(real_hist) / np.sum(real_hist) if np.sum(real_hist) > 0 else np.zeros(n_bins)
    )
    synth_cdf = (
        np.cumsum(synth_hist) / np.sum(synth_hist) if np.sum(synth_hist) > 0 else np.zeros(n_bins)
    )

    return float(np.sum(np.abs(real_cdf - synth_cdf)) * (bins[1] - bins[0]))


class GANConvergenceMonitor:
    """Tracks GAN training convergence using DTW DeD-iMs and Wasserstein.

    Records metrics per epoch for convergence analysis and early stopping.
    """

    def __init__(self) -> None:
        self.dtw_dedims_history: list[float] = []
        self.wasserstein_history: list[float] = []
        self.best_dtw_dedims = float("inf")
        self.best_epoch = 0
        self._converged_epoch: Optional[int] = None

    def record(
        self,
        real_data: np.ndarray,
        synth_data: np.ndarray,
        epoch: int,
        n_samples: int = 50,
    ) -> dict[str, float]:
        """Compute and record convergence metrics for this epoch.

        Args:
            real_data: Real validation samples, shape (N, T, D).
            synth_data: Synthetic samples, shape (N, T, D).
            epoch: Current training epoch.
            n_samples: Samples for DeD-iMs.

        Returns:
            Dict with dtw_dedims and wasserstein values.
        """
        dtw_dedims = compute_dtw_dedims(real_data, synth_data, n_samples=n_samples)
        wasserstein = compute_wasserstein_distance(real_data, synth_data)

        self.dtw_dedims_history.append(dtw_dedims)
        self.wasserstein_history.append(wasserstein)

        if dtw_dedims < self.best_dtw_dedims:
            self.best_dtw_dedims = dtw_dedims
            self.best_epoch = epoch

        return {"dtw_dedims": dtw_dedims, "wasserstein": wasserstein, "epoch": epoch}

    def has_converged(self, patience: int = 10, min_improvement: float = 1e-4) -> bool:
        """Check if DTW DeD-iMs has plateaued for `patience` epochs.

        Args:
            patience: Number of epochs without improvement to declare convergence.
            min_improvement: Minimum relative improvement to count as progress.

        Returns:
            True if converged.
        """
        if len(self.dtw_dedims_history) < patience:
            return False
        recent = self.dtw_dedims_history[-patience:]
        best_recent = min(recent)
        return (self.best_dtw_dedims - best_recent) / max(
            self.best_dtw_dedims, 1e-8
        ) < min_improvement
