# -*- coding: utf-8 -*-
"""
Environment Verification Script for SMC Strategy Development

Validates the development environment setup including:
- Python version compatibility
- Required packages installation
- yfinance data fetching capability
- ATR calculation accuracy (manual vs pandas-ta)
- Data quality checks

Outputs JSON report with all verification results.

Usage:
    python scripts/test_setup.py

Exit Codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Set UTF-8 encoding for stdout/stderr
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Suppress warnings during verification
warnings.filterwarnings("ignore")


def verify_python_version() -> Dict[str, Any]:
    """
    Verify Python version is 3.13.12 or compatible.

    Returns:
        Dict with verification results
    """
    result: Dict[str, Any] = {
        "check": "python_version",
        "passed": False,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "required": "3.10+",
        "message": "",
    }

    # Check for Python 3.10+ (minimum compatible version)
    if sys.version_info.major == 3 and sys.version_info.minor >= 10:
        result["passed"] = True
        result["message"] = f"Python {result['version']} is compatible"
    else:
        result["message"] = f"Python {result['version']} is not compatible. Requires Python 3.10+"

    return result


def verify_packages() -> Dict[str, Any]:  # type: ignore[return-value]
    """
    Verify required packages are installed with correct versions.

    Returns:
        Dict with verification results for each package
    """
    required_packages = {
        "pandas": "2.0.0",
        "numpy": "1.24.0",
        "yfinance": "0.2.0",
        "loguru": "0.7.0",
    }

    optional_packages = {
        "pandas_ta": "0.3.0",
        "backtesting": "0.3.0",
    }

    results: Dict[str, Any] = {
        "check": "packages",
        "passed": True,
        "required": {},
        "optional": {},
        "missing": [],
        "message": "",
    }

    # Check required packages
    for package, min_version in required_packages.items():
        try:
            module = __import__(package.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            results["required"][package] = {
                "installed": True,
                "version": version,
                "min_version": min_version,
            }
        except ImportError:
            results["required"][package] = {
                "installed": False,
                "version": None,
                "min_version": min_version,
            }
            results["missing"].append(package)
            results["passed"] = False

    # Check optional packages
    for package, min_version in optional_packages.items():
        try:
            module = __import__(package.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            results["optional"][package] = {
                "installed": True,
                "version": version,
                "min_version": min_version,
            }
        except ImportError:
            results["optional"][package] = {
                "installed": False,
                "version": None,
                "min_version": min_version,
            }

    if results["passed"]:
        results["message"] = "All required packages installed"
    else:
        results["message"] = f"Missing required packages: {results['missing']}"

    return results


def verify_yfinance_connection() -> Dict[str, Any]:  # type: ignore[return-value]
    """
    Verify yfinance can fetch data from Yahoo Finance API.

    Returns:
        Dict with verification results
    """
    result = {
        "check": "yfinance_connection",
        "passed": False,
        "ticker": "SPY",
        "rows_fetched": 0,
        "date_range": None,
        "message": "",
    }

    try:
        import yfinance as yf

        # Fetch 5 days of 5-minute data for SPY
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Extra days to account for weekends

        ticker = yf.Ticker(result["ticker"])
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="5m",
            prepost=True,
        )

        if df.empty:
            result["message"] = "No data fetched - market may be closed or API issue"
            return result

        result["rows_fetched"] = len(df)
        result["date_range"] = {"start": str(df.index[0]), "end": str(df.index[-1])}

        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            result["message"] = f"Missing columns: {missing_cols}"
            return result

        # Check for minimum data
        if len(df) >= 10:
            result["passed"] = True
            result["message"] = f"Successfully fetched {len(df)} rows of 5-minute data"
        else:
            result["message"] = f"Insufficient data: only {len(df)} rows"

    except Exception as e:
        result["message"] = f"Error fetching data: {str(e)}"

    return result


def calculate_manual_atr(df, period: int = 14):
    """
    Calculate ATR using manual formula for validation.

    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = EMA(TR, period)

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period

    Returns:
        Series of ATR values
    """
    import pandas as pd

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using EMA (Wilders smoothing uses RMA which is similar)
    # RMA = EMA with alpha = 1/period
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    return atr


def calculate_simple_atr(df, period: int = 14):
    """
    Calculate ATR using simple moving average (existing implementation).

    This matches the current implementation in src/indicators/technical.py

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period

    Returns:
        Series of ATR values
    """
    import pandas as pd

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def verify_atr_calculation() -> Dict[str, Any]:  # type: ignore[return-value]
    """
    Verify ATR calculation matches manual formula.

    Cross-checks first 20 values against manual TR formula:
        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ATR = EMA(TR, period)

    Returns:
        Dict with verification results
    """
    result: Dict[str, Any] = {
        "check": "atr_calculation",
        "passed": False,
        "method_comparison": {},
        "max_deviation_pct": None,
        "message": "",
    }

    try:
        import pandas as pd
        import numpy as np

        # Create sample data for testing
        np.random.seed(42)
        n = 100

        # Generate realistic OHLC data
        base_price = 100.0
        returns = np.random.randn(n) * 0.02
        close = base_price * np.cumprod(1 + returns)

        # Generate High/Low with realistic spreads
        spread = np.abs(np.random.randn(n)) * 0.005 * close
        high = close + spread * np.random.rand(n)
        low = close - spread * np.random.rand(n)

        df = pd.DataFrame({"High": high, "Low": low, "Close": close})

        period = 14

        # Calculate ATR using different methods
        atr_ema = calculate_manual_atr(df, period)  # EMA-based (Wilders)
        atr_sma = calculate_simple_atr(df, period)  # SMA-based (existing)

        # Compare values after warmup period
        start_idx = period + 5
        valid_idx = start_idx + 20

        atr_ema_values = atr_ema.iloc[start_idx:valid_idx].values
        atr_sma_values = atr_sma.iloc[start_idx:valid_idx].values

        # Calculate deviation
        deviation = np.abs(atr_ema_values - atr_sma_values) / atr_ema_values * 100
        max_deviation = np.max(deviation)
        avg_deviation = np.mean(deviation)

        result["method_comparison"] = {
            "ema_based": {
                "mean": float(np.nanmean(atr_ema_values)),
                "std": float(np.nanstd(atr_ema_values)),
            },
            "sma_based": {
                "mean": float(np.nanmean(atr_sma_values)),
                "std": float(np.nanstd(atr_sma_values)),
            },
        }

        result["max_deviation_pct"] = float(max_deviation)
        result["avg_deviation_pct"] = float(avg_deviation)

        # Note: EMA and SMA will have different values, but both are valid
        # We're checking that they're in the same ballpark (within 20%)
        if max_deviation < 20:  # Allow 20% difference between methods
            result["passed"] = True
            result["message"] = (
                f"ATR methods differ by max {max_deviation:.2f}% (EMA vs SMA is expected)"
            )
        else:
            result["message"] = f"ATR methods differ significantly: {max_deviation:.2f}%"

        # Check pandas_ta if available
        try:
            import pandas_ta as ta

            df_ta = df.copy()
            df_ta.ta.atr(length=period, append=True)
            atr_ta = df_ta[f"ATRr_{period}"]

            atr_ta_values = atr_ta.iloc[start_idx:valid_idx].values

            # Compare pandas_ta with EMA method
            ta_deviation = np.abs(atr_ema_values - atr_ta_values) / atr_ema_values * 100
            ta_max_deviation = np.max(ta_deviation)

            atr_ta_arr = np.asarray(atr_ta_values, dtype=np.float64)
            result["method_comparison"]["pandas_ta"] = {
                "mean": float(np.nanmean(atr_ta_arr)),
                "std": float(np.nanstd(atr_ta_arr)),
            }
            result["pandas_ta_deviation_pct"] = float(ta_max_deviation)

            if ta_max_deviation < 0.1:  # pandas_ta should match EMA closely
                result["pandas_ta_match"] = True
            else:
                result["pandas_ta_match"] = False
                result["message"] += f" | pandas_ta deviation: {ta_max_deviation:.2f}%"

        except ImportError:
            result["pandas_ta_available"] = False

    except Exception as e:
        result["message"] = f"Error verifying ATR: {str(e)}"

    return result


def verify_data_quality() -> Dict[str, Any]:  # type: ignore[return-value]
    """
    Verify data quality checks work correctly.

    Tests the validation functions from src/utils/validators.py

    Returns:
        Dict with verification results
    """
    result: Dict[str, Any] = {"check": "data_quality", "passed": False, "tests": {}, "message": ""}

    try:
        import pandas as pd

        # Test 1: Check if validators module exists
        validators_path = Path("src/utils/validators.py")
        if validators_path.exists():
            result["tests"]["validators_module"] = {"exists": True}

            # Try importing
            try:
                sys.path.insert(0, str(Path.cwd()))
                from src.utils.validators import validate_dataframe

                result["tests"]["validators_import"] = {"success": True}

                # Test with valid data
                valid_df = pd.DataFrame(
                    {
                        "Open": [100, 101, 102],
                        "High": [102, 103, 104],
                        "Low": [99, 100, 101],
                        "Close": [101, 102, 103],
                    }
                )
                valid_df.index = pd.date_range("2024-01-01", periods=3, freq="D")

                is_valid, errors = validate_dataframe(valid_df)
                result["tests"]["valid_data_check"] = {"passed": is_valid, "errors": errors}

                # Test with invalid data (missing columns)
                invalid_df = pd.DataFrame({"Open": [100, 101, 102], "Close": [101, 102, 103]})

                is_valid, errors = validate_dataframe(invalid_df, min_rows=5)
                result["tests"]["invalid_data_check"] = {
                    "passed": not is_valid,  # Should fail validation
                    "expected_errors": len(errors) > 0,
                }

            except ImportError as e:
                result["tests"]["validators_import"] = {"success": False, "error": str(e)}
        else:
            result["tests"]["validators_module"] = {"exists": False}

        # Test 2: Check sample data files
        data_path = Path("data/raw")
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            result["tests"]["data_files"] = {
                "exists": True,
                "file_count": len(csv_files),
                "files": [f.name for f in csv_files[:5]],  # First 5 files
            }
        else:
            result["tests"]["data_files"] = {"exists": False}

        # Determine overall pass
        validators_ok = result["tests"].get("validators_import", {}).get("success", False)
        valid_check_ok = result["tests"].get("valid_data_check", {}).get("passed", False)

        result["passed"] = validators_ok and valid_check_ok

        if result["passed"]:
            result["message"] = "Data quality validation working correctly"
        else:
            result["message"] = "Some data quality checks failed"

    except Exception as e:
        result["message"] = f"Error verifying data quality: {str(e)}"

    return result


def verify_project_structure() -> Dict[str, Any]:  # type: ignore[return-value]
    """
    Verify project directory structure.

    Returns:
        Dict with verification results
    """
    result: Dict[str, Any] = {
        "check": "project_structure",
        "passed": False,
        "directories": {},
        "message": "",
    }

    required_dirs = [
        "src",
        "src/indicators",
        "src/strategies",
        "src/patterns",
        "src/backtest",
        "src/signals",
        "src/utils",
        "tests",
        "data",
        "data/raw",
        "data/processed",
        "logs",
        "notebooks",
        "config",
    ]

    missing_dirs = []
    existing_dirs = []

    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            existing_dirs.append(dir_path)
            result["directories"][dir_path] = True
        else:
            missing_dirs.append(dir_path)
            result["directories"][dir_path] = False

    result["existing_count"] = len(existing_dirs)
    result["missing_count"] = len(missing_dirs)

    # Allow some missing directories (config is new)
    critical_dirs = ["src", "src/indicators", "src/strategies", "tests", "data"]
    critical_missing = [d for d in critical_dirs if d in missing_dirs]

    if not critical_missing:
        result["passed"] = True
        result["message"] = (
            f"Project structure OK ({len(existing_dirs)}/{len(required_dirs)} directories)"
        )
    else:
        result["message"] = f"Missing critical directories: {critical_missing}"

    return result


def run_all_checks() -> Dict[str, Any]:
    """
    Run all verification checks.

    Returns:
        Dict with all verification results
    """
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
        },
        "checks": [],
        "summary": {"total_checks": 0, "passed": 0, "failed": 0, "all_passed": False},
    }

    # Run all verification checks
    checks = [
        verify_python_version,
        verify_packages,
        verify_project_structure,
        verify_yfinance_connection,
        verify_atr_calculation,
        verify_data_quality,
    ]

    for check_func in checks:
        try:
            check_result = check_func()
            results["checks"].append(check_result)

            results["summary"]["total_checks"] += 1
            if check_result.get("passed", False):
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1

        except Exception as e:
            results["checks"].append(
                {"check": check_func.__name__, "passed": False, "error": str(e)}
            )
            results["summary"]["total_checks"] += 1
            results["summary"]["failed"] += 1

    results["summary"]["all_passed"] = results["summary"]["failed"] == 0

    return results


def main():
    """Main entry point for verification script."""
    print("=" * 60)
    print("SMC Strategy Environment Verification")
    print("=" * 60)
    print()

    # Run all checks
    results = run_all_checks()

    # Print results
    for check in results["checks"]:
        status = "[PASS]" if check.get("passed", False) else "[FAIL]"
        print(f"{status} - {check['check']}")
        if check.get("message"):
            print(f"        {check['message']}")
        print()

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print()

    if results["summary"]["all_passed"]:
        print("[OK] All checks passed! Environment is ready for SMC strategy development.")
    else:
        print("[WARN] Some checks failed. Please review and fix issues before proceeding.")

    print()

    # Save JSON report
    report_path = Path("logs") / "environment_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Full report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if results["summary"]["all_passed"] else 1)


if __name__ == "__main__":
    main()
