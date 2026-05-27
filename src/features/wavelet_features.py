"""
Wavelet-based feature preprocessor for financial time series.

Multi-level Daubechies-4 wavelet decomposition for localized
time-frequency features. Provides deterministic signal enhancement
for CatBoost, LSTM, and RulesFirst scoring.

Feature categories:
1. Per-level statistics (mean, std, entropy, energy, ZCR) × 6 levels = 30 features
2. Cross-level correlation (adjacent detail coefficients) = 4 features
3. Volatility decomposition (cA3 trend, cD1 micro, cD3 macro) = 3 features
4. Regime shift ratio (cD3/cA3 > 1.5 warning) = 1 feature

Total: ~38 features per instrument series.

Reference: Phase 27B — Wavelet Feature Preprocessor (WDformer, AWEMixer, DB2-TransF papers).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SCIPY_AVAILABLE = False
try:
    from scipy.stats import entropy as scipy_entropy

    SCIPY_AVAILABLE = True
except ImportError:
    pass


def _safe_entropy(values: np.ndarray) -> float:
    if not SCIPY_AVAILABLE or len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=10, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(scipy_entropy(hist))


def _zero_crossing_rate(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    signs = np.sign(values)
    zero_crossings = np.sum(np.abs(np.diff(signs)) > 1)
    return zero_crossings / (len(values) - 1)


def _energy(values: np.ndarray) -> float:
    return float(np.sum(values**2))


def _wavelet_decompose(
    series: np.ndarray,
    wavelet: str = "db4",
    levels: int = 5,
) -> list[np.ndarray]:
    """Decompose a 1D signal using discrete wavelet transform.

    Returns [cA{levels}, cD{levels}, cD{levels-1}, ..., cD1].
    """
    import pywt

    coeffs = pywt.wavedec(series, wavelet, mode="per", level=levels)
    return list(coeffs)


def _extract_level_features(coeffs: np.ndarray, level_name: str) -> dict[str, float]:
    """Extract statistical features from a single wavelet decomposition level."""
    return {
        f"wavelet_{level_name}_mean": float(np.mean(coeffs)),
        f"wavelet_{level_name}_std": float(np.std(coeffs)),
        f"wavelet_{level_name}_entropy": _safe_entropy(coeffs),
        f"wavelet_{level_name}_energy": _energy(coeffs),
        f"wavelet_{level_name}_zcr": _zero_crossing_rate(coeffs),
    }


def compute_wavelet_features(
    series: pd.Series,
    wavelet: str = "db4",
    levels: int = 5,
    window: int = 128,
) -> pd.DataFrame:
    """Compute wavelet decomposition features for a price/volume series.

    Args:
        series: Input time series (e.g., Close price, returns, or volume).
        wavelet: Wavelet family (default 'db4' — most common in financial lit).
        levels: Number of decomposition levels (default 5).
        window: Rolling window length for decomposition (default 128, ~6 months).

    Returns:
        DataFrame with wavelet feature columns, indexed same as input.
    """
    import pywt

    values = series.values.astype(np.float64)
    n = len(values)
    feature_names = (
        [f"wavelet_cA{levels}_{s}" for s in ["mean", "std", "entropy", "energy", "zcr"]]
        + [
            f"wavelet_cD{lvl}_{s}"
            for lvl in range(levels, 0, -1)
            for s in ["mean", "std", "entropy", "energy", "zcr"]
        ]
        + [f"wavelet_cross_corr_{i}_{i - 1}" for i in range(levels, 1, -1)]
    )
    result = pd.DataFrame(np.nan, index=series.index, columns=feature_names)

    min_required = max(window, 2**levels)
    for i in range(min_required, n + 1):
        segment = values[i - window : i]
        try:
            coeffs = _wavelet_decompose(segment, wavelet=wavelet, levels=levels)
        except (ValueError, RuntimeError):
            continue

        row: dict[str, float] = {}
        cA = coeffs[0]
        row.update(_extract_level_features(cA, f"cA{levels}"))

        for lvl_idx in range(1, len(coeffs)):
            d_level = levels - lvl_idx + 1
            row.update(_extract_level_features(coeffs[lvl_idx], f"cD{d_level}"))

        for lvl_idx in range(1, len(coeffs)):
            if lvl_idx < len(coeffs) - 1:
                d_high = levels - lvl_idx + 1
                d_low = levels - lvl_idx
                c1 = coeffs[lvl_idx]
                c2 = coeffs[lvl_idx + 1]
                min_len = min(len(c1), len(c2))
                if min_len > 1:
                    c1_t = c1[:min_len]
                    c2_t = c2[:min_len]
                    corr_mat = np.corrcoef(c1_t, c2_t)
                    corr = corr_mat[0, 1]
                    if not np.isnan(corr):
                        row[f"wavelet_cross_corr_{d_high}_{d_low}"] = float(corr)

        for k, v in row.items():
            if k in result.columns:
                result.iloc[i - 1, result.columns.get_loc(k)] = v

    return result


def compute_wavelet_volatility_features(
    series: pd.Series,
    wavelet: str = "db4",
    levels: int = 3,
    vol_window: int = 20,
    decomp_window: int = 252,
) -> pd.DataFrame:
    """Wavelet-based volatility decomposition (B1.3 from AWEMixer).

    Decomposes rolling volatility into structural (cA3), micro (cD1),
    and macro (cD3) components. Computes regime shift warning ratio.

    Args:
        series: Price series.
        wavelet: Wavelet family.
        levels: Decomposition levels for vol series (default 3).
        vol_window: Window for rolling volatility computation (default 20).
        decomp_window: Window for wavelet decomposition (default 252).

    Returns:
        DataFrame with columns: wavelet_vol_structural, wavelet_vol_micro,
        wavelet_vol_macro, wavelet_vol_rshift_ratio.
    """
    import pywt

    returns = series.pct_change().dropna()
    vol_series = returns.rolling(vol_window).std().dropna()
    values = vol_series.values.astype(np.float64)
    n = len(values)

    columns = [
        "wavelet_vol_structural",
        "wavelet_vol_micro",
        "wavelet_vol_macro",
        "wavelet_vol_rshift_ratio",
    ]
    result = pd.DataFrame(np.nan, index=vol_series.index, columns=columns)

    min_required = max(decomp_window, 2**levels)
    for i in range(min_required, n + 1):
        segment = values[i - decomp_window : i]
        try:
            coeffs = _wavelet_decompose(segment, wavelet=wavelet, levels=levels)
        except (ValueError, RuntimeError):
            continue

        cA3 = coeffs[0]
        cD3 = coeffs[1]
        cD1 = coeffs[-1]

        struct_val = float(np.mean(cA3))
        macro_val = float(np.std(cD3))
        micro_val = float(np.std(cD1))

        result.iloc[i - 1, 0] = struct_val
        result.iloc[i - 1, 1] = micro_val
        result.iloc[i - 1, 2] = macro_val

        if struct_val > 1e-10:
            rshift = macro_val / struct_val
            result.iloc[i - 1, 3] = float(rshift)

    return result


class WaveletFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wavelet feature preprocessor.

    Extracts multi-level wavelet decomposition features from OHLCV data
    for consumption by CatBoost, LSTM, or any downstream model.

    Args:
        wavelet: Wavelet family (default 'db4').
        levels: Decomposition levels (default 5).
        window: Rolling window for decomposition (default 60).
        series_columns: Which OHLCV columns to decompose (default Close only).
        output: Feature set — 'all' (38 features), 'reduced' (20 most important),
            or 'regime_only' (4 vol decomposition features).
    """

    def __init__(
        self,
        wavelet: str = "db4",
        levels: int = 5,
        window: int = 128,
        series_columns: tuple[str, ...] = ("Close",),
        output: str = "all",
    ):
        self.wavelet = wavelet
        self.levels = levels
        self.window = window
        self.series_columns = series_columns
        self.output = output

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "WaveletFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract wavelet features from OHLCV data.

        Args:
            X: DataFrame with OHLCV columns.

        Returns:
            DataFrame of wavelet features only (same index).
        """
        import pywt

        parts: list[pd.DataFrame] = []

        for col in self.series_columns:
            if col not in X.columns:
                continue
            series = X[col]
            prefix = col.lower()

            price_wavelet = compute_wavelet_features(
                series, wavelet=self.wavelet, levels=self.levels, window=self.window
            )
            price_wavelet = price_wavelet.rename(
                columns=lambda c: f"{prefix}_{c}" if not c.startswith(prefix) else c
            )
            parts.append(price_wavelet)

            vol_wavelet = compute_wavelet_volatility_features(
                series, wavelet=self.wavelet, levels=3
            )
            vol_wavelet = vol_wavelet.rename(
                columns=lambda c: f"{prefix}_{c}" if not c.startswith(prefix) else c
            )
            parts.append(vol_wavelet)

        if not parts:
            return pd.DataFrame(index=X.index)

        result = pd.concat(parts, axis=1)

        if self.output == "regime_only":
            rshift_cols = [c for c in result.columns if "rshift_ratio" in c]
            struct_cols = [c for c in result.columns if "vol_structural" in c]
            return result[rshift_cols + struct_cols]

        if self.output == "reduced":
            priority_cols = [
                c
                for c in result.columns
                if any(
                    p in c
                    for p in ["rshift_ratio", "vol_structural", "entropy", "zcr", "cross_corr"]
                )
            ]
            return result[priority_cols]

        return result


def compute_wavelet_price_volume_features(
    df: pd.DataFrame,
    wavelet: str = "db4",
    levels: int = 5,
    window: int = 128,
) -> pd.DataFrame:
    """Convenience function: extract wavelet features from OHLCV DataFrame.

    Decomposes Close, High, Low, and Volume series. Merges into a single
    feature DataFrame suitable for joining with existing features.

    Args:
        df: OHLCV DataFrame.
        wavelet: Wavelet family.
        levels: Decomposition levels.
        window: Rolling window.

    Returns:
        DataFrame of wavelet features (same index as df).
    """
    extractor = WaveletFeatureExtractor(
        wavelet=wavelet,
        levels=levels,
        window=window,
        series_columns=("Close", "High", "Low"),
        output="all",
    )
    return extractor.transform(df)
