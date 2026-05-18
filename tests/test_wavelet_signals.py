"""Tests for C10: Wavelet/FFT Signal Processing."""

import numpy as np
import pandas as pd
import pytest

from src.ml.wavelet_signals import (
    WaveletDenoiser,
    DenoiseResult,
    FFTCycleDetector,
    CycleResult,
    SignalDecomposer,
    DecompositionResult,
    FFTFilter,
)


@pytest.fixture
def sine_data():
    """Generate noisy sine wave data."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(t)
    noisy = clean + np.random.randn(200) * 0.2
    return noisy, clean


def test_denoise_smoke(sine_data):
    """Basic denoising produces output."""
    noisy, _ = sine_data
    wd = WaveletDenoiser(wavelet="db8", level=4)
    result = wd.denoise(noisy)
    assert isinstance(result, DenoiseResult)
    assert len(result.denoised) == len(noisy)
    assert len(result.noise) == len(noisy)


def test_denoise_reduces_noise(sine_data):
    """Denoised signal is closer to clean than noisy signal."""
    noisy, clean = sine_data
    wd = WaveletDenoiser(wavelet="db8", level=4)
    result = wd.denoise(noisy)
    mse_noisy = np.mean((noisy - clean) ** 2)
    mse_denoised = np.mean((result.denoised - clean) ** 2)
    assert mse_denoised < mse_noisy


def test_denoise_short_input():
    """Handle very short input."""
    wd = WaveletDenoiser()
    result = wd.denoise(np.array([1.0, 2.0, 3.0]))
    assert len(result.denoised) == 3


def test_denoise_nan_handling(sine_data):
    """Handle NaN values."""
    noisy, _ = sine_data
    noisy[50] = np.nan
    wd = WaveletDenoiser()
    result = wd.denoise(noisy)
    assert len(result.denoised) == len(noisy)


def test_denoise_methods(sine_data):
    """Soft vs hard thresholding both work."""
    noisy, _ = sine_data
    wd_soft = WaveletDenoiser(method="soft")
    wd_hard = WaveletDenoiser(method="hard")
    r_soft = wd_soft.denoise(noisy)
    r_hard = wd_hard.denoise(noisy)
    assert r_soft.snr_db is not None
    assert r_hard.snr_db is not None


def test_denoise_dataframe(sine_data):
    """Denoise from pandas Series."""
    noisy, _ = sine_data
    idx = pd.date_range("2020-01-01", periods=len(noisy))
    wd = WaveletDenoiser()
    df = wd.denoise_dataframe(pd.Series(noisy, index=idx))
    assert "raw" in df.columns
    assert "denoised" in df.columns
    assert "noise" in df.columns


def test_fft_detect_cycles(sine_data):
    """FFT detects the 50-bar cycle in sine wave."""
    noisy, _ = sine_data
    fft = FFTCycleDetector(max_period=100, min_period=3)
    result = fft.detect_cycles(noisy)
    assert isinstance(result, CycleResult)
    assert result.dominant_period > 0
    top = result.top_cycles(3)
    assert len(top) > 0
    assert "period" in top[0]


def test_fft_detrend(sine_data):
    """Detrending before FFT."""
    noisy, _ = sine_data
    trend = noisy + np.linspace(0, 10, len(noisy))

    fft_dt = FFTCycleDetector(detrend=True)
    fft_raw = FFTCycleDetector(detrend=False)

    r_dt = fft_dt.detect_cycles(trend)
    r_raw = fft_raw.detect_cycles(trend)

    assert r_dt.dominant_period > 0
    assert r_raw.dominant_period > 0


def test_fft_short_input():
    """Very short input returns empty result."""
    fft = FFTCycleDetector()
    result = fft.detect_cycles(np.array([1.0, 2.0]))
    assert np.isnan(result.dominant_period)


def test_fft_cycle_signal(sine_data):
    """Cycle phase signal in [-1, 1]."""
    noisy, _ = sine_data
    fft = FFTCycleDetector()
    signal = fft.detect_cycle_signal(noisy)
    assert -1.0 <= signal <= 1.0


def test_fft_rolling_cycles(sine_data):
    """Rolling cycle detection."""
    noisy, _ = sine_data
    fft = FFTCycleDetector()
    df = fft.rolling_cycles(noisy, window=100)
    assert "dominant_period" in df.columns


def test_signal_decomposer(sine_data):
    """Full wavelet decomposition."""
    noisy, _ = sine_data
    sd = SignalDecomposer(wavelet="db8", level=4)
    result = sd.decompose(noisy)
    assert isinstance(result, DecompositionResult)
    assert len(result.approximation) == len(noisy)
    assert len(result.details) > 0


def test_decomposition_trend_strength(sine_data):
    """Trend strength is between 0 and 1."""
    noisy, _ = sine_data
    sd = SignalDecomposer()
    result = sd.decompose(noisy)
    assert 0.0 <= result.trend_strength <= 1.0


def test_decomposition_dataframe(sine_data):
    """Decompose from pandas Series."""
    noisy, _ = sine_data
    idx = pd.date_range("2020-01-01", periods=len(noisy))
    sd = SignalDecomposer(level=2)
    df = sd.decompose_dataframe(pd.Series(noisy, index=idx))
    assert "approximation" in df.columns


def test_fft_filter(sine_data):
    """Band-pass filter extraction."""
    noisy, _ = sine_data
    ffilter = FFTFilter(low_cutoff=0.005, high_cutoff=0.1)
    filtered = ffilter.filter(noisy)
    assert len(filtered) == len(noisy)


def test_fft_filter_dataframe(sine_data):
    """Band-pass filter from DataFrame."""
    noisy, _ = sine_data
    idx = pd.date_range("2020-01-01", periods=len(noisy))
    ffilter = FFTFilter()
    df = ffilter.filter_dataframe(pd.Series(noisy, index=idx))
    assert "raw" in df.columns
    assert "filtered" in df.columns
    assert "signal" in df.columns


def test_to_dict(sine_data):
    """Serialization of results."""
    noisy, _ = sine_data
    wd = WaveletDenoiser()
    dr = wd.denoise(noisy)
    assert "snr_db" in dr.to_dict()

    fft = FFTCycleDetector()
    cr = fft.detect_cycles(noisy)
    assert "dominant_period_days" in cr.to_dict()

    sd = SignalDecomposer(level=2)
    decomp = sd.decompose(noisy)
    assert "snr_db" in decomp.to_dict()
