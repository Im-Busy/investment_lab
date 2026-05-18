"""
R14: Factor Engine Library Wrapper.

Thin try-install wrapper around the open-source Factor Engine Python library
(https://github.com/factor-engine). If installable, provides 11 validated
factor computations for free. If not, falls back gracefully — no lock-in.

Source: Beyond Fama-French §5

Usage:
    engine = FactorEngineWrapper()
    if engine.available:
        df = engine.compute_factor("momentum", prices, returns)
    else:
        # Use fallback factor implementations
        pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known factors from Factor Engine library (verified 2024-12)
KNOWN_FACTORS = [
    "momentum",
    "size",
    "value",
    "quality",
    "low_volatility",
    "profitability",
    "investment",
    "leverage",
    "dividend_yield",
    "earnings_yield",
    "accruals",
]


@dataclass
class FactorEngineStatus:
    """Status of the Factor Engine library availability."""

    available: bool = False
    installed: bool = False
    message: str = ""
    supported_factors: list[str] = field(default_factory=list)
    unavailable_factors: list[str] = field(default_factory=list)


class FactorEngineWrapper:
    """Thin wrapper around the Factor Engine Python library.

    Tries to import `factor_engine` on initialization. If unavailable,
    sets `available=False` and all methods become no-ops returning None.

    Architecture: src/ml/factor_engine.py — optional factor feature provider.
    """

    def __init__(self) -> None:
        self._engine: Any = None
        self._status = FactorEngineStatus()

        try:
            import factor_engine  # type: ignore[import-untyped]

            self._engine = factor_engine
            self._status.installed = True
            self._status.available = True
            self._status.message = "Factor Engine v{} loaded.".format(
                getattr(factor_engine, "__version__", "unknown")
            )
            self._status.supported_factors = self._detect_supported_factors()
            self._status.unavailable_factors = [
                f for f in KNOWN_FACTORS if f not in self._status.supported_factors
            ]
            logger.info(
                "Factor Engine loaded: %d factors available.", len(self._status.supported_factors)
            )

        except ImportError:
            self._status.message = "Factor Engine not installed. Install with: uv add factor-engine"
            logger.info("Factor Engine not available. Use fallback factor implementations.")

    @property
    def available(self) -> bool:
        """Whether Factor Engine is importable and functional."""
        return self._status.available

    @property
    def status(self) -> FactorEngineStatus:
        """Detailed availability status."""
        return self._status

    def _detect_supported_factors(self) -> list[str]:
        """Detect which factors are available in the loaded engine."""
        if self._engine is None:
            return []

        detected: list[str] = []
        # Factor Engine uses decorator-based registration
        # Try calling each known factor function
        for factor_name in KNOWN_FACTORS:
            try:
                fn = getattr(self._engine, f"compute_{factor_name}", None)
                if fn is not None:
                    detected.append(factor_name)
            except Exception:
                continue

        return detected

    def compute_factor(
        self,
        factor_name: str,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Compute a single factor if the engine is available.

        Args:
            factor_name: One of KNOWN_FACTORS (e.g., "momentum", "size").
            prices: (T, N) price dataframe.
            returns: (T, N) return dataframe.
            **kwargs: Additional parameters passed to the factor function.

        Returns:
            (T, N) factor values, or None if unavailable.
        """
        if not self.available or self._engine is None:
            logger.debug("Factor Engine not available for %s.", factor_name)
            return None

        fn = getattr(self._engine, f"compute_{factor_name}", None)
        if fn is None:
            logger.warning("Factor '%s' not found in Factor Engine.", factor_name)
            return None

        try:
            result = fn(prices, returns, **kwargs)
            if isinstance(result, np.ndarray):
                return pd.DataFrame(result, index=prices.index, columns=prices.columns)
            return result
        except Exception as exc:
            logger.warning("Factor Engine compute_%s failed: %s", factor_name, exc)
            return None

    def compute_all_factors(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        factors: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Optional[pd.DataFrame]]:
        """Compute multiple factors.

        Args:
            prices: (T, N) price dataframe.
            returns: (T, N) return dataframe.
            factors: List of factor names (default: all detected).
            **kwargs: Passed to each factor computation.

        Returns:
            {factor_name: dataframe_or_None}
        """
        if factors is None:
            factors = self._status.supported_factors

        results: dict[str, Optional[pd.DataFrame]] = {}
        for name in factors:
            results[name] = self.compute_factor(name, prices, returns, **kwargs)
        return results

    def add_factor_features(
        self,
        df: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        factors: Optional[list[str]] = None,
        prefix: str = "fe_",
    ) -> pd.DataFrame:
        """Add Factor Engine features to an existing feature dataframe.

        Args:
            df: (T, F) existing feature dataframe.
            prices: (T, N) price dataframe.
            returns: (T, N) return dataframe.
            factors: Factor names to add (default: all detected).
            prefix: Column prefix for factor engine features.

        Returns:
            (T, F + K) dataframe with factor columns added.
        """
        if not self.available:
            logger.debug("Factor Engine unavailable — no features added.")
            return df

        factor_dfs = self.compute_all_factors(prices, returns, factors=factors)
        result = df.copy()

        for name, factor_df in factor_dfs.items():
            if factor_df is None:
                continue
            if factor_df.ndim == 2 and factor_df.shape[1] > 1:
                for col in factor_df.columns:
                    result[f"{prefix}{name}_{col}"] = factor_df[col]
            elif factor_df.ndim == 1:
                result[f"{prefix}{name}"] = factor_df
            else:
                result[f"{prefix}{name}"] = factor_df

        return result


# ── Module-level convenience function ──


def get_factor_engine() -> FactorEngineWrapper:
    """Get or create the module-level Factor Engine wrapper instance.

    Returns:
        FactorEngineWrapper (singleton per import).
    """
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = FactorEngineWrapper()
    return _ENGINE_INSTANCE


_ENGINE_INSTANCE: Optional[FactorEngineWrapper] = None
