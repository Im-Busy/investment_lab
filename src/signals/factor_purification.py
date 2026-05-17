"""
R2: Factor Purification Module.

Before evaluating pattern signal quality, regress out sector membership and
market cap. Compute IC/Sharpe/IR on the purified residual signal to determine
whether the pattern has independent alpha or is merely a sector/size proxy.

Source: 华泰多因子系列1 §1.3

Formula:
    y = β0 + β1·sector_dummies + β2·log_mcap + ε
    purified_signal = ε
    purity_ratio = var(ε) / var(y)

Usage:
    purifier = FactorPurifier()
    purified, report = purifier.purify(signal_series, sector_labels, log_mcap)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PurificationReport:
    """Report from factor purification regression."""

    purity_ratio: float
    """Fraction of signal variance NOT explained by sector/size (var(ε)/var(y))."""

    sector_r2: float
    """R-squared from sector dummies alone."""

    market_cap_r2: float
    """Incremental R-squared from log(market cap) after sector."""

    total_r2: float
    """Total R-squared from sector + market cap."""

    sector_coefs: dict[str, float]
    """Coefficients for sector dummies."""

    market_cap_coef: float
    """Coefficient for log(market cap)."""

    n_samples: int
    """Number of observations used."""

    is_contaminated: bool
    """True if purity_ratio < 0.5 (more than half of variance is sector/size)."""


class FactorPurifier:
    """Purify pattern signals by regressing out sector and size factors.

    Uses OLS via numpy (no external dependencies). Suitable for panel data
    where each observation is a (pattern_signal, sector, mcap) tuple.

    Parameters:
        min_purity: Threshold below which a signal is flagged as contaminated.
        epsilon: Numerical tolerance for near-zero divisions.
    """

    def __init__(
        self,
        min_purity: float = 0.5,
        epsilon: float = 1e-10,
    ) -> None:
        self._min_purity = min_purity
        self._epsilon = epsilon

    def purify(
        self,
        signal: np.ndarray,
        sectors: np.ndarray,
        log_mcap: np.ndarray | None = None,
    ) -> tuple[np.ndarray, PurificationReport]:
        """Regress out sector and market cap from pattern signal.

        Args:
            signal: (N,) array of pattern signal values or returns.
            sectors: (N,) array of string sector labels (e.g., 'XLK', 'XLV').
            log_mcap: (N,) array of log(market cap). If None, size purification
                is skipped.

        Returns:
            Tuple of (purified_signal, purification_report).
        """
        signal = np.asarray(signal, dtype=np.float64)
        sectors = np.asarray(sectors, dtype=str)
        n = len(signal)

        # Drop NaN entries
        mask = np.isfinite(signal)
        if log_mcap is not None:
            log_mcap = np.asarray(log_mcap, dtype=np.float64)
            mask &= np.isfinite(log_mcap)
        n_valid = int(mask.sum())
        if n_valid < 10:
            logger.warning("Too few valid samples (%d) for purification", n_valid)
            return signal, PurificationReport(
                purity_ratio=1.0 if n_valid > 0 else 0.0,
                sector_r2=0.0,
                market_cap_r2=0.0,
                total_r2=0.0,
                sector_coefs={},
                market_cap_coef=0.0,
                n_samples=n_valid,
                is_contaminated=False,
            )

        y = signal[mask]
        sector_arr = sectors[mask]
        unique_sectors = sorted(set(sector_arr))

        # Build design matrix: [intercept, sector_dummies, log_mcap]
        cols = 1 + len(unique_sectors)
        if log_mcap is not None:
            cols += 1
        X = np.ones((n_valid, cols), dtype=np.float64)

        # Sector dummy coding (one-hot, drop first for identifiability)
        self._sector_map: dict[str, int] = {}
        for j, sec in enumerate(unique_sectors):
            col_idx = 1 + j
            X[:, col_idx] = (sector_arr == sec).astype(np.float64)
            self._sector_map[sec] = col_idx

        size_col = 1 + len(unique_sectors)
        if log_mcap is not None:
            X[:, size_col] = log_mcap[mask]

        # ── Compute total R² ──
        y_centered = y - y.mean()
        total_r2, sector_r2, mcap_r2 = self._decompose_r2(y_centered, X, size_col, log_mcap)

        # ── OLS via numpy ──
        coeffs, residuals = self._ols(X, y)

        # ── Purified signal ──
        purified_full = np.full_like(signal, np.nan, dtype=np.float64)
        predicted_masked = X @ coeffs
        purified_full[mask] = y - predicted_masked

        # ── Purity ratio ──
        var_y = float(np.var(y, ddof=1)) if len(y) > 1 else self._epsilon
        var_e = float(np.var(residuals, ddof=1)) if len(residuals) > 1 else self._epsilon
        purity_ratio = var_e / max(var_y, self._epsilon)

        # ── Coefficient extraction ──
        sector_coefs: dict[str, float] = {}
        for sec, col_idx in self._sector_map.items():
            sector_coefs[sec] = float(coeffs[col_idx])
        mcap_coef = float(coeffs[size_col]) if log_mcap is not None else 0.0

        report = PurificationReport(
            purity_ratio=min(purity_ratio, 1.0),
            sector_r2=min(sector_r2, 1.0),
            market_cap_r2=min(mcap_r2, 1.0),
            total_r2=min(total_r2, 1.0),
            sector_coefs=sector_coefs,
            market_cap_coef=mcap_coef,
            n_samples=n_valid,
            is_contaminated=purity_ratio < self._min_purity,
        )

        if report.is_contaminated:
            logger.info(
                "Signal CONTAMINATED: purity=%.2f (%.1f%% of variance is sector/size). "
                "Don't trust IS performance alone.",
                purity_ratio,
                (1 - purity_ratio) * 100,
            )

        return purified_full, report

    @staticmethod
    def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """OLS via normal equations: β = (X'X)^(-1) X'y."""
        try:
            coeffs = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coeffs
        return coeffs, residuals

    @staticmethod
    def _decompose_r2(
        y_centered: np.ndarray,
        X: np.ndarray,
        size_col: int,
        log_mcap: np.ndarray | None,
    ) -> tuple[float, float, float]:
        """Decompose R² into sector-only and incremental market-cap contributions."""
        ss_total = float(y_centered @ y_centered)
        if ss_total < 1e-15:
            return 0.0, 0.0, 0.0

        # Full model
        _, res_full = FactorPurifier._ols(X, y_centered)
        ss_res_full = float(res_full @ res_full)
        total_r2 = 1.0 - ss_res_full / ss_total

        # Sector-only model (drop size column)
        if log_mcap is not None:
            X_sector = np.delete(X, size_col, axis=1)
            _, res_sector = FactorPurifier._ols(X_sector, y_centered)
            ss_res_sector = float(res_sector @ res_sector)
            sector_r2 = 1.0 - ss_res_sector / ss_total
            mcap_r2 = total_r2 - sector_r2
        else:
            sector_r2 = total_r2
            mcap_r2 = 0.0

        return total_r2, sector_r2, mcap_r2


def purify_pattern_signals(
    signals: dict[str, np.ndarray],
    sectors: np.ndarray,
    log_mcap: np.ndarray | None = None,
    min_purity: float = 0.5,
) -> dict[str, PurificationReport]:
    """Convenience: purify all pattern signals and return reports.

    Args:
        signals: dict of pattern_name -> signal array.
        sectors: sector labels per bar.
        log_mcap: log market cap per bar.
        min_purity: contamination threshold.

    Returns:
        Dict of pattern_name -> PurificationReport.
    """
    purifier = FactorPurifier(min_purity=min_purity)
    reports: dict[str, PurificationReport] = {}
    for name, sig in signals.items():
        _, report = purifier.purify(sig, sectors, log_mcap)
        reports[name] = report
    return reports
