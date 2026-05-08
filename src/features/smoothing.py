import pandas as pd


def kalman_filter_smooth(
    prices: pd.Series,
    transition_covariance: float = 0.01,
    observation_covariance: float = 1.0,
    initial_state_mean: float = 0.0,
    initial_state_covariance: float = 1.0,
) -> pd.Series:
    """Apply Kalman Filter for time-series denoising.

    Uses a random-walk state-space model where the hidden state follows
    a Gaussian random walk and observations are a noisy measurement of that state.

    Args:
        prices: Single-asset price or return series.
        transition_covariance: Process noise (controls smoothness; smaller = smoother).
        observation_covariance: Measurement noise variance.
        initial_state_mean: Initial estimate of the hidden state.
        initial_state_covariance: Initial covariance of the state estimate.

    Returns:
        Filtered (smoothed) series with same index as input.
    """
    from pykalman import KalmanFilter

    kf = KalmanFilter(
        transition_matrices=[[1]],
        observation_matrices=[[1]],
        initial_state_mean=[initial_state_mean],
        initial_state_covariance=[[initial_state_covariance]],
        observation_covariance=[[observation_covariance]],
        transition_covariance=[[transition_covariance]],
    )
    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.ravel(), index=prices.index, name="kalman_filtered")


def wavelet_denoise(
    signal: pd.Series,
    wavelet: str = "db6",
    threshold_scale: float = 0.5,
    mode: str = "soft",
) -> pd.Series:
    """Denoise a time series using discrete wavelet transform.

    Decomposes the signal using the specified wavelet, applies soft thresholding
    to detail coefficients, and reconstructs the cleaned signal.

    Args:
        signal: Input time series (returns or prices).
        wavelet: Wavelet family and order (e.g., 'db6', 'sym5', 'coif3').
        threshold_scale: Scale factor for thresholding, multiplied by signal max.
            Smaller values remove more noise.  0.1 = mild, 0.5 = aggressive.
        mode: Thresholding mode ('soft' or 'hard').

    Returns:
        Reconstructed (denoised) signal with same index as input.
    """
    import pywt

    data = signal.values.copy()
    coefficients = pywt.wavedec(data, wavelet, mode="per")
    threshold = threshold_scale * abs(signal).max()
    coefficients[1:] = [pywt.threshold(i, value=threshold, mode=mode) for i in coefficients[1:]]
    reconstructed = pywt.waverec(coefficients, wavelet, mode="per")
    return pd.Series(reconstructed[: len(signal)], index=signal.index, name="wavelet_smoothed")


def list_wavelet_families() -> list:
    """Return all available wavelet families and their member names.

    Returns:
        List of family names (e.g., ['Haar', 'Daubechies', 'Symlets', ...]).
    """
    import pywt

    return pywt.families(short=False)
