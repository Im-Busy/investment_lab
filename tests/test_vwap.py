"""Unit tests for VWAP indicator."""

import numpy as np
import pandas as pd

from src.indicators.vwap import compute_vwap


def test_vwap_basic() -> None:
    """Test VWAP calculation with known values."""
    data = {
        "High": [100.0, 102.0, 101.0, 103.0, 104.0],
        "Low": [98.0, 99.0, 97.0, 100.0, 101.0],
        "Close": [99.0, 101.0, 98.0, 102.0, 103.0],
        "Volume": [1000.0, 1500.0, 2000.0, 1200.0, 800.0],
    }
    df = pd.DataFrame(data)
    result = compute_vwap(df, deviation_pct=1.0)

    # VWAP should be monotonically related to cumulative volume-weighted price
    assert "vwap" in result.columns
    assert "vwap_upper" in result.columns
    assert "vwap_lower" in result.columns

    # First bar VWAP = typical_price (since cumsum is just first value)
    typical_price_0 = (100 + 98 + 99) / 3.0
    assert np.isclose(result["vwap"].iloc[0], typical_price_0)

    # Upper/lower bands should be correct percentage
    assert np.isclose(result["vwap_upper"].iloc[0], typical_price_0 * 1.01)
    assert np.isclose(result["vwap_lower"].iloc[0], typical_price_0 * 0.99)

    print("test_vwap_basic PASSED")


def test_vwap_with_constant_volume() -> None:
    """Test VWAP with constant volume - should equal cumulative mean of typical price."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "High": np.random.uniform(100, 110, n),
            "Low": np.random.uniform(90, 100, n),
            "Close": np.random.uniform(95, 105, n),
            "Volume": np.ones(n) * 1000.0,
        }
    )
    result = compute_vwap(df, deviation_pct=0.5)

    # With constant volume, VWAP = cumsum(tp) / (n * vol) = cummean(tp)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    expected_vwap = tp.cumsum() / np.arange(1, n + 1)

    assert np.allclose(result["vwap"].values, expected_vwap.values)
    print("test_vwap_with_constant_volume PASSED")


if __name__ == "__main__":
    test_vwap_basic()
    test_vwap_with_constant_volume()
    print("All VWAP tests passed!")
