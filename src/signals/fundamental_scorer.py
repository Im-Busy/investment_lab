"""
H3: Sector-Specific Multi-Factor Scoring.

Per-sector fundamental factor models based on empirical sector IC divergence
(2026-05-17 session). Each sector uses a curated subset of factors where that
sector's factor IC is highest.

Sector factor model assignments (from session data):
    Technology:    ps_ratio (IC 0.74), pe_ratio (IC 0.68), ev_ebitda
    Healthcare:    profit_margin (IC 0.90), revenue_growth
    Financials:    pb_ratio (IC 0.95), ps_ratio, pe_ratio
    Energy:        ev_ebitda, ps_ratio, debt_equity
    Consumer:      pe_ratio, profit_margin, revenue_growth
    Industrials:   ev_ebitda, pb_ratio, pe_ratio
    Utilities:     dividend_yield, pb_ratio, pe_ratio

Usage:
    from src.signals.fundamental_scorer import FundamentalScorer

    scorer = FundamentalScorer()
    score = scorer.compute_sector_score("Technology", pe_ratio=22, ps_ratio=5.1, ...)
    # Returns float in [-1, 1] where positive = attractive fundamentals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np


@dataclass
class SectorFactorModel:
    """Per-sector fundamental factor configuration."""

    sector: str
    factors: list[str]
    weights: list[float]
    # Z-score thresholds for signal normalization
    z_clip: float = 2.0


# ── Pre-built sector models (from 2026-05-17 session) ──
SECTOR_MODELS: dict[str, SectorFactorModel] = {
    "Technology": SectorFactorModel(
        sector="Technology",
        factors=["ps_ratio", "pe_ratio", "ev_ebitda"],
        weights=[0.40, 0.35, 0.25],
    ),
    "Financials": SectorFactorModel(
        sector="Financials",
        factors=["pb_ratio", "ps_ratio", "pe_ratio"],
        weights=[0.50, 0.25, 0.25],
    ),
    "Healthcare": SectorFactorModel(
        sector="Healthcare",
        factors=["profit_margin", "revenue_growth", "pe_ratio"],
        weights=[0.50, 0.30, 0.20],
    ),
    "Energy": SectorFactorModel(
        sector="Energy",
        factors=["ev_ebitda", "ps_ratio", "debt_equity"],
        weights=[0.40, 0.35, 0.25],
    ),
    "Consumer Cyclical": SectorFactorModel(
        sector="Consumer Cyclical",
        factors=["pe_ratio", "profit_margin", "revenue_growth"],
        weights=[0.35, 0.35, 0.30],
    ),
    "Consumer Defensive": SectorFactorModel(
        sector="Consumer Defensive",
        factors=["pe_ratio", "profit_margin", "roe"],
        weights=[0.30, 0.40, 0.30],
    ),
    "Industrials": SectorFactorModel(
        sector="Industrials",
        factors=["ev_ebitda", "pb_ratio", "pe_ratio"],
        weights=[0.40, 0.30, 0.30],
    ),
    "Utilities": SectorFactorModel(
        sector="Utilities",
        factors=["dividend_yield", "pb_ratio", "pe_ratio"],
        weights=[0.40, 0.30, 0.30],
    ),
    "Real Estate": SectorFactorModel(
        sector="Real Estate",
        factors=["pb_ratio", "dividend_yield", "debt_equity"],
        weights=[0.40, 0.35, 0.25],
    ),
    "Basic Materials": SectorFactorModel(
        sector="Basic Materials",
        factors=["ev_ebitda", "pb_ratio", "debt_equity"],
        weights=[0.35, 0.35, 0.30],
    ),
    "Communication Services": SectorFactorModel(
        sector="Communication Services",
        factors=["pe_ratio", "ps_ratio", "profit_margin"],
        weights=[0.35, 0.35, 0.30],
    ),
}

# ── Factor direction mapping ──
# True = higher is better (multiply z-score by +1)
# False = lower is better (multiply z-score by -1)
FACTOR_DIRECTION: dict[str, bool] = {
    "pe_ratio": False,
    "pb_ratio": False,
    "ps_ratio": False,
    "ev_ebitda": False,
    "debt_equity": False,
    "short_pct_float": False,
    "beta": False,
    "roe": True,
    "roa": True,
    "profit_margin": True,
    "revenue_growth": True,
    "earnings_growth": True,
    "dividend_yield": True,
    "market_cap": True,
}


@dataclass
class SectorFundamentalScore:
    """Result of a sector fundamental evaluation."""

    sector: str
    composite_score: float  # in [-1, 1]
    factor_scores: dict[str, float]  # per-factor z-scores
    n_factors: int


class FundamentalScorer:
    """Compute per-sector fundamental attractiveness scores.

    Takes raw factor values and produces normalized composite scores
    suitable as signal modifiers for RulesFirstStrategy.

    Parameters:
        cross_sectional_z: If True, compute z-scores against a reference
            cross-section of sectors. If False, use absolute z-score thresholds.
        clip: Maximum absolute z-score before normalization.
    """

    # Approximate cross-sectional reference values (S&P 500 long-term medians)
    _REFERENCE: ClassVar[dict[str, float]] = {
        "pe_ratio": 20.0,
        "pb_ratio": 3.0,
        "ps_ratio": 1.5,
        "ev_ebitda": 12.0,
        "debt_equity": 0.8,
        "roe": 0.15,
        "roa": 0.05,
        "profit_margin": 0.10,
        "revenue_growth": 0.08,
        "earnings_growth": 0.10,
        "dividend_yield": 0.02,
        "beta": 1.0,
        "market_cap": 100e9,  # 100B
    }

    _SIGMA: ClassVar[dict[str, float]] = {
        "pe_ratio": 15.0,
        "pb_ratio": 2.5,
        "ps_ratio": 1.0,
        "ev_ebitda": 8.0,
        "debt_equity": 0.8,
        "roe": 0.15,
        "roa": 0.08,
        "profit_margin": 0.10,
        "revenue_growth": 0.12,
        "earnings_growth": 0.15,
        "dividend_yield": 0.02,
        "beta": 0.3,
        "market_cap": 200e9,
    }

    def __init__(self, clip: float = 2.0) -> None:
        self._clip = clip

    def compute_sector_score(
        self,
        sector: str,
        **factor_values: float,
    ) -> SectorFundamentalScore:
        """Compute fundamental attractiveness score for a sector.

        Args:
            sector: Sector name (must match SECTOR_MODELS key).
            **factor_values: Factor name → value mappings.

        Returns:
            SectorFundamentalScore with composite score in [-1, 1].
        """
        model = SECTOR_MODELS.get(sector)
        if model is None:
            return SectorFundamentalScore(
                sector=sector,
                composite_score=0.0,
                factor_scores={},
                n_factors=0,
            )

        factor_scores: dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for factor, weight in zip(model.factors, model.weights):
            value = factor_values.get(factor)
            if value is None or not np.isfinite(value):
                continue

            z = self._compute_z(factor, value)
            z_clipped = np.clip(z, -model.z_clip, model.z_clip)
            factor_scores[factor] = z_clipped
            weighted_sum += weight * z_clipped
            weight_total += weight

        if weight_total > 0:
            composite = weighted_sum / weight_total
            composite = np.tanh(composite)  # Squash to [-1, 1]
        else:
            composite = 0.0

        return SectorFundamentalScore(
            sector=sector,
            composite_score=float(composite),
            factor_scores=factor_scores,
            n_factors=len(factor_scores),
        )

    def _compute_z(self, factor: str, value: float) -> float:
        """Compute z-score relative to reference median."""
        ref = self._REFERENCE.get(factor, 0)
        sigma = self._SIGMA.get(factor, 1)
        z = (value - ref) / sigma
        if not FACTOR_DIRECTION.get(factor, True):
            z = -z
        return z

    def compute_score_from_dict(
        self,
        sector: str,
        all_factors: dict[str, float],
    ) -> SectorFundamentalScore:
        """Compute from a dict of all factor values."""
        return self.compute_sector_score(sector, **all_factors)

    @staticmethod
    def get_sector_factors(sector: str) -> list[str]:
        """Get the list of factors used for a given sector."""
        model = SECTOR_MODELS.get(sector)
        if model is None:
            return []
        return list(model.factors)

    @staticmethod
    def get_supported_sectors() -> list[str]:
        return list(SECTOR_MODELS.keys())


# Backward-compatible utility function
def sector_fundamental_score(
    sector: str,
    **factor_values: float,
) -> float:
    """Quick sector fundamental score (returns composite only)."""
    scorer = FundamentalScorer()
    result = scorer.compute_sector_score(sector, **factor_values)
    return result.composite_score
