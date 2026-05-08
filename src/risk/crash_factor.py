# -*- coding: utf-8 -*-
"""
Crash Factor Model - R17/H5 Implementation

Logistic regression model from P13 (Crash-Based Strategies) for pre-trade risk filtering.

From handover document:
- R17: Crash factor as pre-trade risk filter
- H5: Crash factor filter reduces tail loss (Δ99thLoss ≤ -25%)
- P13: Crash-Based Quantitative Strategies (Finance Research Letters, 2022)

Features (from P13 Appendix A):
1. 12-month return
2. Excess return (vs market)
3. Total volatility
4. Skewness
5. Size (log market cap)
6. Turnover change
7. Firm age
8. Tangibility (assets / market cap)
9. Sales growth
10. Dummy for negative earnings

Usage:
    uv run python src/risk/crash_factor.py --data data/symbols/*.csv --output reports/cash_risk
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

warnings.filterwarnings("ignore")


@dataclass
class CrashFactorConfig:
    """Configuration for crash factor model."""

    # Features to use (from P13 specification)
    features: List[str] = None

    # Probability threshold for crash warning
    crash_threshold: float = 0.10

    # Lookback periods
    return_window: int = 252
    volatility_window: int = 60
    turnover_window: int = 20

    # Data requirements
    min_observations: int = 60

    def __post_init__(self):
        if self.features is None:
            self.features = [
                "return_12m",
                "excess_return",
                "total_volatility",
                "skewness",
                "size",
                "turnover_change",
                "firm_age",
                "tangibility",
                "sales_growth",
                "negative_earnings",
            ]


@dataclass
class CrashFactorResult:
    """Result for one stock."""

    symbol: str
    crash_probability: float
    crash_warning: bool
    feature_values: Dict[str, float]
    feature_contributions: Dict[str, float]
    risk_level: str
    recommendation: str


class CrashFactorModel:
    """
    Crash factor logistic regression model from P13.

    Based on Jang & Kang (2019) specification with 10 features.
    Predicts probability of stock price crash.

    P13 Reference:
    - "Crash-Based Quantitative Strategies", Finance Research Letters, 2022
    - CMRS quintiles show monotonic risk pattern
    - Top 10% crash probability should be excluded
    """

    def __init__(self, config: Optional[CrashFactorConfig] = None):
        """
        Args:
            config: Model configuration
        """
        self.config = config or CrashFactorConfig()

        # Coefficients from P13 (Jang & Kang 2019)
        # Note: These are approximate coefficients from the paper
        # In production, should train on historical data
        self.coefficients: Dict[str, float] = {
            "intercept": -2.50,
            "return_12m": -0.15,
            "excess_return": -0.20,
            "total_volatility": 1.50,
            "skewness": -0.30,
            "size": -0.10,
            "turnover_change": 0.80,
            "firm_age": -0.05,
            "tangibility": 0.20,
            "sales_growth": -0.10,
            "negative_earnings": 0.50,
        }

        self.feature_names = self.config.features

    def calculate_features(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        market_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate all 10 crash factor features.

        Args:
            prices: OHLCV DataFrame with 'Close'/'close', 'Volume'/'volume' columns
            fundamentals: Optional fundamental data DataFrame
            market_returns: Optional market returns for excess return calc

        Returns:
            DataFrame with feature columns
        """
        # Normalize column names to lowercase for internal use
        _prices = prices.rename(
            columns={
                "Close": "close",
                "Volume": "volume",
                "Open": "open",
                "High": "high",
                "Low": "low",
            }
        )

        features = {}

        # 1. 12-month return
        returns = _prices["close"].pct_change(self.config.return_window)
        features["return_12m"] = returns

        # 2. Excess return (vs market)
        if market_returns is not None:
            stock_returns = _prices["close"].pct_change()
            excess = stock_returns - market_returns.reindex(stock_returns.index, method="ffill")
            features["excess_return"] = excess.rolling(self.config.return_window).sum()
        else:
            features["excess_return"] = returns

        # 3. Total volatility
        daily_returns = _prices["close"].pct_change()
        features["total_volatility"] = daily_returns.rolling(
            self.config.volatility_window
        ).std() * np.sqrt(252)

        # 4. Skewness
        features["skewness"] = daily_returns.rolling(self.config.return_window).skew()

        # 5. Size (log market cap) - use volume as proxy if no fundamentals
        if fundamentals is not None and "market_cap" in fundamentals.columns:
            features["size"] = np.log(fundamentals["market_cap"])
        else:
            features["size"] = np.log(_prices["volume"] * _prices["close"])

        # 6. Turnover change
        turnover = _prices["volume"] / (_prices["close"].replace(0, np.nan))
        avg_turnover = turnover.rolling(self.config.turnover_window).mean()
        prev_turnover = avg_turnover.shift(self.config.turnover_window)
        features["turnover_change"] = np.where(
            prev_turnover > 0, (avg_turnover - prev_turnover) / prev_turnover, 0.0
        )

        # 7. Firm age (days since first data)
        features["firm_age"] = np.arange(len(prices)) / 252.0

        # 8. Tangibility (use volume/price ratio as proxy)
        features["tangibility"] = _prices["volume"] / (_prices["close"] + 1e-8)

        # 9. Sales growth (use volume change as proxy)
        features["sales_growth"] = _prices["volume"].pct_change(252)

        # 10. Negative earnings dummy (use negative return as proxy)
        features["negative_earnings"] = (daily_returns < 0).astype(int)

        features_df = pd.DataFrame(features)
        features_df = features_df.dropna()

        return features_df

    def calculate_crash_probability(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate crash probability from features.

        Args:
            features: DataFrame with feature columns

        Returns:
            Series with crash probabilities
        """
        if "intercept" not in features.columns:
            features = features.copy()
            features["intercept"] = 1.0

        # Calculate log-odds (linear combination)
        logodds = pd.Series(0.0, index=features.index)

        for feature in self.feature_names:
            if feature in features.columns:
                coef = self.coefficients.get(feature, 0.0)
                logodds += coef * features[feature]

        logodds += self.coefficients.get("intercept", 0.0)

        # Convert to probability using logistic function
        crash_prob = pd.Series(expit(logodds), index=features.index)

        return crash_prob

    def get_feature_contributions(
        self,
        feature_values: pd.Series,
    ) -> Dict[str, float]:
        """
        Get individual feature contributions to crash probability.

        Args:
            feature_values: Single row of features

        Returns:
            Dict mapping feature → contribution
        """
        contributions = {}

        for feature in self.feature_names:
            if feature in feature_values.index:
                contributions[feature] = (
                    self.coefficients.get(feature, 0.0) * feature_values[feature]
                )

        return contributions

    def predict(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
    ) -> CrashFactorResult:
        """
        Predict crash probability for one stock.

        Args:
            prices: OHLCV data
            fundamentals: Optional fundamental data
            symbol: Stock symbol

        Returns:
            CrashFactorResult with prediction and analysis
        """
        features = self.calculate_features(prices, fundamentals)

        if len(features) == 0:
            return CrashFactorResult(
                symbol=symbol,
                crash_probability=0.0,
                crash_warning=False,
                feature_values={},
                feature_contributions={},
                risk_level="UNKNOWN",
                recommendation="Insufficient data",
            )

        latest_features = features.iloc[-1]
        crash_prob = self.calculate_crash_probability(features).iloc[-1]

        feature_contributions = self.get_feature_contributions(latest_features)

        crash_warning = crash_prob >= self.config.crash_threshold

        if crash_prob >= 0.20:
            risk_level = "CRITICAL"
            recommendation = "EXCLUDE: High crash probability"
        elif crash_prob >= 0.10:
            risk_level = "HIGH"
            recommendation = "REDUCE: Above threshold"
        elif crash_prob >= 0.05:
            risk_level = "MEDIUM"
            recommendation = "MONITOR: Elevated risk"
        else:
            risk_level = "LOW"
            recommendation = "ACCEPT: Normal risk"

        return CrashFactorResult(
            symbol=symbol,
            crash_probability=float(crash_prob),
            crash_warning=crash_warning,
            feature_values=latest_features.to_dict(),
            feature_contributions=feature_contributions,
            risk_level=risk_level,
            recommendation=recommendation,
        )


class CrashFactorFilter:
    """
    Pre-trade crash risk filter.

    Uses crash factor model to filter high-risk trades.
    Implements H5: "Crash factor filter reduces tail loss"
    """

    def __init__(
        self,
        model: Optional[CrashFactorModel] = None,
        crash_threshold: float = 0.10,
        exclude_top_percentile: bool = True,
    ):
        """
        Args:
            model: Crash factor model
            crash_threshold: Probability threshold for warnings
            exclude_top_percentile: Exclude top 10% crash probability
        """
        self.model = model or CrashFactorModel()
        self.crash_threshold = crash_threshold
        self.exclude_top_percentile = exclude_top_percentile

    def filter_trades(
        self,
        candidate_trades: pd.DataFrame,
        prices_dict: Dict[str, pd.DataFrame],
        fundamentals_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter candidate trades by crash risk.

        Args:
            candidate_trades: DataFrame with 'symbol' column
            prices_dict: Dict[symbol -> prices DataFrame]
            fundamentals_dict: Optional Dict[symbol -> fundamentals]

        Returns:
            Tuple of (filtered_trades, excluded_trades)
        """
        crash_probs = []

        for symbol in candidate_trades["symbol"].unique():
            if symbol not in prices_dict:
                continue

            prices = prices_dict[symbol]
            fundamentals = fundamentals_dict.get(symbol) if fundamentals_dict else None

            result = self.model.predict(prices, fundamentals, symbol)
            crash_probs.append(
                {
                    "symbol": symbol,
                    "crash_probability": result.crash_probability,
                    "risk_level": result.risk_level,
                }
            )

        probs_df = pd.DataFrame(crash_probs)

        if len(probs_df) == 0:
            return candidate_trades, pd.DataFrame()

        if self.exclude_top_percentile:
            threshold = probs_df["crash_probability"].quantile(0.90)
            excluded = probs_df[probs_df["crash_probability"] >= threshold]
        else:
            excluded = probs_df[probs_df["crash_probability"] >= self.crash_threshold]

        excluded_symbols = set(excluded["symbol"])
        filtered = candidate_trades[~candidate_trades["symbol"].isin(excluded_symbols)]

        print(f"Crash filter: {len(excluded)} of {len(candidate_trades)} trades excluded")
        print(f"  Excluded symbols: {list(excluded_symbols)}")

        return filtered, excluded

    def analyze_portfolio_risk(
        self,
        positions: pd.DataFrame,
        prices_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Analyze portfolio-level crash risk.

        Args:
            positions: DataFrame with 'symbol', 'size' columns
            prices_dict: Dict[symbol -> prices]

        Returns:
            Dictionary with portfolio risk metrics
        """
        results = []

        for _, row in positions.iterrows():
            symbol = row["symbol"]
            if symbol not in prices_dict:
                continue

            prices = prices_dict[symbol]
            result = self.model.predict(prices, symbol=symbol)
            results.append(result)

        if len(results) == 0:
            return {"error": "No positions analyzed"}

        probabilities = [r.crash_probability for r in results]
        high_risk_count = sum(1 for r in results if r.crash_warning)

        return {
            "portfolio_crash_risk": float(np.mean(probabilities)),
            "max_crash_risk": float(np.max(probabilities)),
            "positions_at_risk": high_risk_count,
            "total_positions": len(results),
            "risk_pct": high_risk_count / len(results),
        }


def run_batch_analysis(
    data_dir: str,
    output_dir: str,
) -> Dict:
    """
    Run crash factor analysis on all stocks.

    Args:
        data_dir: Directory with stock data CSVs
        output_dir: Output directory

    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    model = CrashFactorModel()
    results = []

    data_files = list(Path(data_dir).glob("*.csv"))
    print(f"Analyzing {len(data_files)} stocks...")

    for filepath in data_files:
        symbol = filepath.stem
        try:
            prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if "close" not in prices.columns and "Close" not in prices.columns:
                continue

            result = model.predict(prices, symbol=symbol)
            results.append(asdict(result))
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")

    results_df = pd.DataFrame(results)

    # Sort by crash probability
    results_df = results_df.sort_values("crash_probability", ascending=False)

    # Save results
    json_path = os.path.join(output_dir, "crash_factor_analysis.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "analysis_results": results,
                "summary": {
                    "total_stocks": len(results),
                    "high_risk_count": len(results_df[results_df["crash_probability"] >= 0.10]),
                    "mean_crash_prob": results_df["crash_probability"].mean(),
                    "max_crash_prob": results_df["crash_probability"].max(),
                },
            },
            f,
            indent=2,
            default=str,
        )

    csv_path = os.path.join(output_dir, "crash_factor_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\nAnalysis complete: {len(results)} stocks")
    print(f"  High risk (≥10%): {len(results_df[results_df['crash_probability'] >= 0.10])}")
    print(f"  Critical (≥20%): {len(results_df[results_df['crash_probability'] >= 0.20])}")

    return {
        "results": results,
        "summary": {
            "total_stocks": len(results),
            "high_risk_count": len(results_df[results_df["crash_probability"] >= 0.10]),
        },
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Crash Factor Model - Pre-trade Risk Filter (R17)")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with stock data CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/cash_risk",
        help="Output directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Crash probability threshold",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    results = run_batch_analysis(args.data_dir, args.output_dir)

    print("\n✅ R17 Crash factor analysis complete!")
    print(f"\nKey Finding: {results['summary']['high_risk_count']} stocks with elevated crash risk")


if __name__ == "__main__":
    main()
