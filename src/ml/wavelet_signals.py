"""
C10: Wavelet Denoising & FFT Signal Processing for Trading.

High-frequency signal processing techniques applied to financial time series.
Wavelets decompose price data into multiple frequency bands, isolating trend
from noise. FFT detects dominant cycles for regime timing.

Models:
  WaveletDenoiser: Multi-level wavelet thresholding for price signal extraction.
    Uses PyWavelets (pywt) for discrete wavelet transform (DWT).
    Hard/soft thresholding removes Gaussian noise from returns and prices.
    Produces clean signal for trend detection and regime classification.

  FFTCycleDetector: Fast Fourier Transform for cycle frequency detection.
    Identifies dominant periodic components in price data.
    Strongest frequency → cycle length → regime adaptation parameter.

  SignalDecomposer: Full wavelet decomposition into approximation + details.
    Approximation = trend, details = noise/cycles at different scales.
    Computes signal-to-noise ratio and trend strength metrics.

Usage:
    >>> wd = WaveletDenoiser(wavelet="db8", level=4)
    >>> clean = wd.denoise(price_series)
    >>> fft = FFTCycleDetector()
    >>> cycles = fft.detect_cycles(price_series, n_cycles=3)
    >>> sd = SignalDecomposer()
    >>> decomp = sd.decompose(price_series)
    >>> print(decomp.snr_db, decomp.trend_strength)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pywt

    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
    logger.warning("pywt not installed; wavelet denoising disabled. Install via: uv add pywavelets")


# ── Wavelet Denoising ───────────────────────────────────────────────────


@dataclass
class DenoiseResult:
    denoised: np.ndarray
    noise: np.ndarray
    snr_db: float
    wavelet: str
    level: int
    method: str

    def to_dict(self) -> Dict:
        return {
            "snr_db": round(self.snr_db, 2),
            "wavelet": self.wavelet,
            "level": self.level,
            "method": self.method,
            "noise_std": round(float(np.nanstd(self.noise)), 6),
        }


class WaveletDenoiser:
    """Wavelet threshold denoising for financial time series.

    Decomposes signal via Discrete Wavelet Transform (DWT), thresholds
    detail coefficients to remove noise, then reconstructs. Wavelet
    denoising outperforms moving averages for non-stationary financial
    data because it adapts to frequency content locally.

    Args:
        wavelet: Wavelet family (db4-db20, sym4-sym20, coif1-coif5).
            Higher number = more vanishing moments = smoother.
        level: Decomposition levels. Higher = more noise removal but
            more signal distortion. Auto if None.
        method: "soft" (shrink toward zero) or "hard" (keep or zero).
        threshold: Noise threshold multiplier for coefficients (None = auto).
    """

    _DEFAULT_WAVELETS = ["db8", "db10", "db12", "sym8", "sym10", "coif3"]

    def __init__(
        self,
        wavelet: str = "db8",
        level: Optional[int] = None,
        method: str = "soft",
        threshold: Optional[float] = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.method = method
        self.threshold = threshold

    def denoise(self, series: np.ndarray) -> DenoiseResult:
        """Apply wavelet denoising to a time series.

        Args:
            series: 1-D array to denoise.

        Returns:
            DenoiseResult with denoised signal, extracted noise, SNR.
        """
        if not _HAS_PYWT:
            return self._fallback_denoise(series)

        series = np.asarray(series, dtype=float)
        valid = ~np.isnan(series)
        clean_data = series[valid]
        n = len(clean_data)

        if n < 4:
            noise = series - series
            return DenoiseResult(
                denoised=series.copy(),
                noise=noise,
                snr_db=float("inf"),
                wavelet=self.wavelet,
                level=0,
                method=self.method,
            )

        max_level = pywt.dwt_max_level(n, pywt.Wavelet(self.wavelet).dec_len)
        level = min(self.level or max_level, max_level) if max_level > 0 else 0

        if level == 0:
            noise = np.zeros_like(series)
            noise[valid] = 0.0
            return DenoiseResult(
                denoised=series.copy(),
                noise=noise,
                snr_db=float("inf"),
                wavelet=self.wavelet,
                level=0,
                method=self.method,
            )

        coeffs = pywt.wavedec(clean_data, self.wavelet, level=level)

        if self.threshold is not None:
            sigma = self.threshold
        else:
            detail_coeffs = np.concatenate([c for c in coeffs[1:]])
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745 if len(detail_coeffs) > 0 else 1.0

        threshold_value = (
            sigma * np.sqrt(2 * np.log(n))
            if self.method == "soft"
            else sigma * np.sqrt(2 * np.log(n)) * 1.5
        )

        coeffs_thresh = [coeffs[0]]
        for c in coeffs[1:]:
            if self.method == "soft":
                c_thresh = np.sign(c) * np.maximum(np.abs(c) - threshold_value, 0)
            else:
                c_thresh = c * (np.abs(c) >= threshold_value)
            coeffs_thresh.append(c_thresh)

        denoised_clean = pywt.waverec(coeffs_thresh, self.wavelet)[:n]

        denoised = series.copy()
        denoised[valid] = denoised_clean
        noise = np.zeros_like(series)
        noise[valid] = clean_data - denoised_clean

        snr = _compute_snr(series[valid], noise[valid])
        return DenoiseResult(
            denoised=denoised,
            noise=noise,
            snr_db=snr,
            wavelet=self.wavelet,
            level=level,
            method=self.method,
        )

    def _fallback_denoise(self, series: np.ndarray) -> DenoiseResult:
        """Savitzky-Golay smoothing fallback when pywt unavailable."""
        from scipy.signal import savgol_filter

        series = np.asarray(series, dtype=float)
        valid = ~np.isnan(series)
        n_valid = valid.sum()

        if n_valid < 5:
            return DenoiseResult(
                denoised=series.copy(),
                noise=np.zeros_like(series),
                snr_db=float("inf"),
                wavelet="sg_filter",
                level=0,
                method="soft",
            )

        denoised = series.copy()
        denoised[valid] = savgol_filter(series[valid], min(21, n_valid - (n_valid % 2) - 1), 3)
        noise = series - denoised
        snr = _compute_snr(series[valid], noise[valid])

        return DenoiseResult(
            denoised=denoised,
            noise=noise,
            snr_db=snr,
            wavelet="sg_filter",
            level=0,
            method="soft",
        )

    def denoise_dataframe(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Denoise a price series and return as DataFrame.

        Args:
            prices: Price series with datetime index.

        Returns:
            DataFrame with raw, denoised, noise columns.
        """
        raw = prices.values
        result = self.denoise(raw)
        return pd.DataFrame(
            {
                "raw": raw,
                "denoised": result.denoised,
                "noise": result.noise,
            },
            index=prices.index,
        )


# ── FFT Cycle Detection ─────────────────────────────────────────────────


@dataclass
class CycleResult:
    frequencies: np.ndarray
    amplitudes: np.ndarray
    periods: np.ndarray
    dominant_period: float
    dominant_amplitude: float
    signal_power_ratio: float

    def top_cycles(self, n: int = 3) -> List[Dict]:
        """Get top N cycles sorted by amplitude.

        Returns:
            List of {"period": days, "amplitude": relative, "frequency": Hz}.
        """
        idx = np.argsort(self.amplitudes)[::-1][: min(n, len(self.amplitudes))]
        return [
            {
                "period": round(float(self.periods[i]), 1),
                "amplitude": round(float(self.amplitudes[i]), 4),
                "frequency": round(float(self.frequencies[i]), 6),
            }
            for i in idx
            if self.amplitudes[i] > 0
        ]

    def to_dict(self) -> Dict:
        return {
            "dominant_period_days": round(float(self.dominant_period), 1),
            "dominant_amplitude": round(float(self.dominant_amplitude), 4),
            "signal_power_ratio": round(float(self.signal_power_ratio), 4),
            "top_cycles": self.top_cycles(3),
        }


class FFTCycleDetector:
    """Fast Fourier Transform cycle detection for financial time series.

    Identifies dominant periodic components in price or returns data.
    Converts strongest frequency to cycle period for regime-adaptive
    parameter tuning (e.g., lookback windows, trailing stop distances).

    Args:
        detrend: Remove linear trend before FFT (avoids DC component dominance).
        max_period: Maximum cycle period in bars to consider.
        min_period: Minimum cycle period in bars to consider.
    """

    def __init__(
        self,
        detrend: bool = True,
        max_period: int = 252,
        min_period: int = 3,
    ):
        self.detrend = detrend
        self.max_period = max_period
        self.min_period = min_period

    def detect_cycles(
        self,
        series: np.ndarray,
        n_cycles: int = 3,
    ) -> CycleResult:
        """Detect dominant cycles via FFT.

        Args:
            series: 1-D time series.
            n_cycles: Number of top cycles to return.

        Returns:
            CycleResult with frequencies, amplitudes, periods.
        """
        series = np.asarray(series, dtype=float)
        valid = ~np.isnan(series)
        clean = series[valid]
        n = len(clean)

        if n < self.min_period * 2:
            return CycleResult(
                frequencies=np.array([]),
                amplitudes=np.array([]),
                periods=np.array([]),
                dominant_period=float("nan"),
                dominant_amplitude=0.0,
                signal_power_ratio=0.0,
            )

        if self.detrend and n > 2:
            t = np.arange(n)
            slope, intercept = np.polyfit(t, clean, 1)
            signal = clean - (slope * t + intercept)
        else:
            signal = clean

        fft_vals = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(n)

        amplitudes = np.abs(fft_vals)
        frequencies = freq[1:]
        amplitudes = amplitudes[1:]
        periods = np.where(frequencies > 0, 1.0 / frequencies, float("inf"))

        mask = (periods >= self.min_period) & (periods <= self.max_period)
        frequencies = frequencies[mask]
        amplitudes = amplitudes[mask]
        periods = periods[mask]

        if len(amplitudes) == 0:
            return CycleResult(
                frequencies=np.array([]),
                amplitudes=np.array([]),
                periods=np.array([]),
                dominant_period=float("nan"),
                dominant_amplitude=0.0,
                signal_power_ratio=0.0,
            )

        dominant_idx = int(np.argmax(amplitudes))
        total_power = float(np.sum(fft_vals.real**2 + fft_vals.imag**2))
        signal_power = (
            float(np.sum(amplitudes[: min(n_cycles * 3, len(amplitudes))] ** 2))
            if len(amplitudes) > 0
            else 0.0
        )

        return CycleResult(
            frequencies=frequencies,
            amplitudes=amplitudes,
            periods=periods,
            dominant_period=float(periods[dominant_idx]),
            dominant_amplitude=float(amplitudes[dominant_idx]),
            signal_power_ratio=signal_power / total_power if total_power > 1e-12 else 0.0,
        )

    def detect_cycle_signal(
        self,
        series: np.ndarray,
    ) -> float:
        """Single trading signal based on dominant cycle phase.

        1 = near trough (bullish), -1 = near peak (bearish), 0 = neutral.

        Args:
            series: Price or returns series.

        Returns:
            Signal in [-1, 1].
        """
        result = self.detect_cycles(series)
        if np.isnan(result.dominant_period):
            return 0.0

        n = len(series)
        period = result.dominant_period
        phase = (n % max(period, 1.0)) / period * 2 * np.pi
        return float(-np.cos(phase))

    def rolling_cycles(
        self,
        series: np.ndarray,
        window: int = 252,
    ) -> pd.DataFrame:
        """Rolling cycle detection over a window.

        Args:
            series: 1-D price or returns series.
            window: Analysis window size.

        Returns:
            DataFrame with dominant_period and signal_power_ratio per time step.
        """
        series = np.asarray(series, dtype=float)
        n = len(series)
        periods = np.full(n, np.nan)
        power_ratios = np.full(n, np.nan)

        for t in range(window, n):
            result = self.detect_cycles(series[t - window : t])
            periods[t] = result.dominant_period
            power_ratios[t] = result.signal_power_ratio

        return pd.DataFrame({"dominant_period": periods, "signal_power_ratio": power_ratios})

    def detect_cycles_dataframe(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Detect cycles and return results as DataFrame.

        Args:
            prices: Price series with datetime index.

        Returns:
            DataFrame with rolling cycle diagnostics.
        """
        return self.rolling_cycles(prices.values)


# ── Signal Decomposer ───────────────────────────────────────────────────


@dataclass
class DecompositionResult:
    approximation: np.ndarray
    details: List[np.ndarray]
    snr_db: float
    trend_strength: float
    wavelet: str
    n_levels: int

    @property
    def detail_energy_ratio(self) -> List[float]:
        """Energy ratio of each detail level vs total energy."""
        total = float(np.sum(self.approximation**2) + sum(np.sum(d**2) for d in self.details))
        if total < 1e-12:
            return [0.0] * len(self.details)
        return [float(np.sum(d**2) / total) for d in self.details]

    def to_dict(self) -> Dict:
        return {
            "snr_db": round(self.snr_db, 2),
            "trend_strength": round(self.trend_strength, 4),
            "wavelet": self.wavelet,
            "n_levels": self.n_levels,
            "detail_energy": [round(r, 4) for r in self.detail_energy_ratio],
        }


class SignalDecomposer:
    """Multi-level wavelet decomposition for signal analysis.

    Decomposes price data into approximation (trend) and detail (noise/cycle)
    coefficients at multiple frequency bands. Computes SNR and trend strength
    for regime classification and signal quality assessment.

    Args:
        wavelet: PyWavelets wavelet name.
        level: Decomposition levels (None = auto).
    """

    def __init__(self, wavelet: str = "db8", level: Optional[int] = None):
        self.wavelet = wavelet
        self.level = level

    def decompose(self, series: np.ndarray) -> DecompositionResult:
        """Full wavelet decomposition.

        Args:
            series: 1-D time series.

        Returns:
            DecompositionResult with approximation and detail coefficients.
        """
        series = np.asarray(series, dtype=float)
        valid = ~np.isnan(series)
        clean = series[valid]
        n = len(clean)

        if not _HAS_PYWT or n < 4:
            return DecompositionResult(
                approximation=series.copy(),
                details=[],
                snr_db=float("nan"),
                trend_strength=0.0,
                wavelet=self.wavelet,
                n_levels=0,
            )

        max_lvl = pywt.dwt_max_level(n, pywt.Wavelet(self.wavelet).dec_len)
        level = min(self.level or max_lvl, max_lvl) if max_lvl > 0 else 0

        if level == 0:
            return DecompositionResult(
                approximation=series.copy(),
                details=[],
                snr_db=float("inf"),
                trend_strength=1.0,
                wavelet=self.wavelet,
                n_levels=0,
            )

        coeffs = pywt.wavedec(clean, self.wavelet, level=level)

        approx = series.copy()
        approx[valid] = pywt.upcoef("a", coeffs[0], self.wavelet, level=level, take=len(clean))

        details = []
        for i in range(1, level + 1):
            coeff_list = [np.zeros_like(c) for c in coeffs]
            coeff_list[i] = coeffs[i]
            detail = pywt.waverec(coeff_list, self.wavelet)[:n]
            details.append(detail)

        noise_est = np.concatenate([c for c in coeffs[1:]]) if len(coeffs) > 1 else np.array([0.0])
        sigma = np.median(np.abs(noise_est)) / 0.6745 if len(noise_est) > 0 else 1.0
        signal_power = float(np.sum(clean**2))
        noise_power = sigma**2 * n
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-12 else float("inf")

        trend_strength = (
            float(np.sum(approx[valid] ** 2) / signal_power) if signal_power > 1e-12 else 1.0
        )

        return DecompositionResult(
            approximation=approx,
            details=details,
            snr_db=snr,
            trend_strength=trend_strength,
            wavelet=self.wavelet,
            n_levels=level,
        )

    def decompose_dataframe(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Decompose price series and return components as DataFrame.

        Args:
            prices: Price series with datetime index.

        Returns:
            DataFrame with approximation, detail_levels, snr, trend columns.
        """
        values = prices.values
        result = self.decompose(values)

        data = {"price": values, "approximation": result.approximation}

        n = len(values)
        for i, detail in enumerate(result.details):
            col = f"detail_{i + 1}"
            detail_padded = np.full(n, np.nan)
            valid = ~np.isnan(values)
            clean_n = valid.sum()
            detail_padded[valid] = (
                detail[:clean_n]
                if len(detail) >= clean_n
                else np.pad(detail, (0, clean_n - len(detail)), constant_values=np.nan)
            )
            data[col] = detail_padded

        return pd.DataFrame(data, index=prices.index)


# ── FFT-Based Signal Generator ──────────────────────────────────────────


class FFTFilter:
    """FFT-based band-pass filter for signal extraction.

    Removes low-frequency trend and high-frequency noise, preserving
    only frequencies within a specified band. Useful for extracting
    mean-reverting signals or cyclical patterns from price data.

    Args:
        low_cutoff: Low-frequency cutoff in cycles per bar.
        high_cutoff: High-frequency cutoff in cycles per bar.
    """

    def __init__(self, low_cutoff: float = 0.001, high_cutoff: float = 0.1):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def filter(self, series: np.ndarray) -> np.ndarray:
        """Apply band-pass filter via FFT.

        Args:
            series: 1-D time series.

        Returns:
            Band-pass filtered signal (mean-reverting).
        """
        series = np.asarray(series, dtype=float)
        valid = ~np.isnan(series)
        clean = series[valid]
        n = len(clean)

        if n < 4:
            return series.copy()

        fft_vals = np.fft.rfft(clean)
        freq = np.fft.rfftfreq(n)

        mask = (freq >= self.low_cutoff) & (freq <= self.high_cutoff)
        fft_vals[0] = 0.0
        fft_vals[~mask] = 0.0

        filtered_clean = np.fft.irfft(fft_vals, n=n)

        result = series.copy()
        result[valid] = filtered_clean
        return result

    def filter_dataframe(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Band-pass filter price series and return as DataFrame.

        Args:
            prices: Price series with datetime index.

        Returns:
            DataFrame with raw, filtered, signal columns.
        """
        raw = prices.values
        filtered = self.filter(raw)
        signal = np.where(filtered > 0, 1.0, -1.0)
        return pd.DataFrame(
            {"raw": raw, "filtered": filtered, "signal": signal},
            index=prices.index,
        )


# ── Helpers ─────────────────────────────────────────────────────────────


def _compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    s_power = float(np.sum(signal**2))
    n_power = float(np.sum(noise**2))
    if n_power < 1e-12:
        return float("inf")
    if s_power < 1e-12:
        return 0.0
    return float(10 * np.log10(s_power / n_power))
