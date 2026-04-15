"""
Walk-Forward Validator for Pattern Selection

Validates patterns across in-sample, out-of-sample, and forward-validation
periods to detect overfitting and ensure robustness.

Data Split:
- In-Sample (IS): 60% - Pattern selection and parameter optimization
- Out-of-Sample (OOS): 20% - Validation of selected patterns
- Forward Validation (FV): 20% - Final check on unseen data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class OverfitStatus(Enum):
    """Overfitting detection status."""

    PASS = "PASS"
    MEDIUM_OVERFIT = "MEDIUM_OVERFIT"
    HIGH_OVERFIT = "HIGH_OVERFIT"
    OOS_UNPROFITABLE = "OOS_UNPROFITABLE"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    # Data split percentages
    in_sample_pct: float = 0.60
    out_of_sample_pct: float = 0.20
    forward_pct: float = 0.20

    # Minimum Sharpe ratios per period
    min_is_sharpe: float = 0.5
    min_oos_sharpe: float = 0.3
    min_fv_sharpe: float = 0.0

    # Maximum degradation from IS to OOS
    min_degradation_ratio: float = 0.5

    # Minimum trades per period
    min_trades_per_period: int = 10


@dataclass
class PeriodResult:
    """Results for a single validation period."""

    period_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    passes_threshold: bool
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result for a pattern."""

    pattern_name: str
    is_result: PeriodResult
    oos_result: PeriodResult
    fv_result: Optional[PeriodResult]

    # Cross-period metrics
    degradation_absolute: float
    degradation_percentage: float
    overfit_status: OverfitStatus

    # Overall validation
    passes_validation: bool
    all_failure_reasons: List[str] = field(default_factory=list)

    # Period dates for reference
    is_dates: Tuple[pd.Timestamp, pd.Timestamp] = field(default_factory=lambda: (pd.NaT, pd.NaT))
    oos_dates: Tuple[pd.Timestamp, pd.Timestamp] = field(default_factory=lambda: (pd.NaT, pd.NaT))
    fv_dates: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None


def detect_overfitting(
    is_sharpe: float,
    oos_sharpe: float,
    degradation_ratio: float = 0.5,
) -> OverfitStatus:
    """
    Detect overfitting by comparing IS vs OOS performance.

    Args:
        is_sharpe: In-sample Sharpe ratio
        oos_sharpe: Out-of-sample Sharpe ratio
        degradation_ratio: Maximum allowed degradation (e.g., 0.5 = 50%)

    Returns:
        OverfitStatus indicating severity of overfitting
    """
    if is_sharpe <= 0:
        return OverfitStatus.OOS_UNPROFITABLE

    degradation = is_sharpe - oos_sharpe
    degradation_pct = degradation / is_sharpe

    if degradation_pct > 0.7:
        return OverfitStatus.HIGH_OVERFIT
    elif degradation_pct > degradation_ratio:
        return OverfitStatus.MEDIUM_OVERFIT
    elif oos_sharpe < 0:
        return OverfitStatus.OOS_UNPROFITABLE
    else:
        return OverfitStatus.PASS


class WalkForwardValidator:
    """
    Walk-forward validation for pattern selection.

    Splits data into IS/OOS/FV periods and validates patterns
    across all periods to detect overfitting.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def split_data(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into IS, OOS, and FV periods.

        Args:
            data: OHLCV DataFrame with DatetimeIndex

        Returns:
            Dict with keys 'is', 'oos', 'fv' (fv may be None if not enough data)
        """
        n = len(data)
        is_end = int(n * self.config.in_sample_pct)
        oos_end = int(n * (self.config.in_sample_pct + self.config.out_of_sample_pct))

        is_data = data.iloc[:is_end]
        oos_data = data.iloc[is_end:oos_end]
        fv_data = data.iloc[oos_end:] if oos_end < n else None

        return {
            "is": is_data,
            "oos": oos_data,
            "fv": fv_data,
        }

    def get_split_dates(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Get the date ranges for each split.

        Args:
            data: OHLCV DataFrame with DatetimeIndex

        Returns:
            Dict with date ranges for each period
        """
        splits = self.split_data(data)

        result = {}
        for key, split_data in splits.items():
            if split_data is not None and len(split_data) > 0:
                result[key] = (split_data.index[0], split_data.index[-1])

        return result

    def run_validation(
        self,
        pattern_name: str,
        performance_metrics: Dict[str, Dict[str, Any]],
        config: Optional[WalkForwardConfig] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation using pre-computed performance metrics.

        This method expects metrics for each period to be provided, as the
        actual backtesting is done externally.

        Args:
            pattern_name: Name of the pattern being validated
            performance_metrics: Dict with keys 'is', 'oos', 'fv' (optional)
                Each value should contain: sharpe_ratio, total_trades,
                win_rate, total_return_pct, max_drawdown_pct, profit_factor,
                start_date, end_date
            config: Optional override for validation config

        Returns:
            WalkForwardResult with validation status
        """
        cfg = config or self.config
        all_failures = []

        # Build IS result
        is_metrics = performance_metrics.get("is", {})
        is_result = self._build_period_result(
            "In-Sample",
            is_metrics,
            sharpe_threshold=cfg.min_is_sharpe,
            cfg=cfg,
        )
        if not is_result.passes_threshold:
            all_failures.extend([f"IS: {r}" for r in is_result.failure_reasons])

        # Build OOS result
        oos_metrics = performance_metrics.get("oos", {})
        oos_result = self._build_period_result(
            "Out-of-Sample",
            oos_metrics,
            sharpe_threshold=cfg.min_oos_sharpe,
            cfg=cfg,
        )
        if not oos_result.passes_threshold:
            all_failures.extend([f"OOS: {r}" for r in oos_result.failure_reasons])

        # Calculate degradation
        is_sharpe = is_metrics.get("sharpe_ratio", 0.0)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
        degradation = is_sharpe - oos_sharpe
        degradation_pct = degradation / is_sharpe if is_sharpe > 0 else float("inf")

        # Detect overfitting
        overfit_status = detect_overfitting(is_sharpe, oos_sharpe, cfg.min_degradation_ratio)

        if overfit_status == OverfitStatus.HIGH_OVERFIT:
            all_failures.append(f"HIGH_OVERFIT: {degradation_pct:.0%} degradation from IS to OOS")
        elif overfit_status == OverfitStatus.MEDIUM_OVERFIT:
            all_failures.append(f"MEDIUM_OVERFIT: {degradation_pct:.0%} degradation from IS to OOS")
        elif overfit_status == OverfitStatus.OOS_UNPROFITABLE:
            all_failures.append("OOS_UNPROFITABLE: OOS Sharpe is negative")

        # Build FV result if available
        fv_result = None
        fv_metrics = performance_metrics.get("fv")
        if fv_metrics:
            fv_result = self._build_period_result(
                "Forward-Validation",
                fv_metrics,
                sharpe_threshold=cfg.min_fv_sharpe,
                cfg=cfg,
            )
            if not fv_result.passes_threshold:
                all_failures.extend([f"FV: {r}" for r in fv_result.failure_reasons])

        # Overall validation passes if no failures
        passes_validation = len(all_failures) == 0

        return WalkForwardResult(
            pattern_name=pattern_name,
            is_result=is_result,
            oos_result=oos_result,
            fv_result=fv_result,
            degradation_absolute=degradation,
            degradation_percentage=degradation_pct,
            overfit_status=overfit_status,
            passes_validation=passes_validation,
            all_failure_reasons=all_failures,
            is_dates=(
                is_metrics.get("start_date", pd.NaT),
                is_metrics.get("end_date", pd.NaT),
            ),
            oos_dates=(
                oos_metrics.get("start_date", pd.NaT),
                oos_metrics.get("end_date", pd.NaT),
            ),
            fv_dates=(
                (fv_metrics.get("start_date"), fv_metrics.get("end_date")) if fv_metrics else None
            ),
        )

    def _build_period_result(
        self,
        period_name: str,
        metrics: Dict[str, Any],
        sharpe_threshold: float,
        cfg: WalkForwardConfig,
    ) -> PeriodResult:
        """Build PeriodResult from metrics dict and check thresholds."""
        failures = []

        sharpe = metrics.get("sharpe_ratio", 0.0)
        trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)
        max_dd = metrics.get("max_drawdown_pct", 0.0)

        # Check minimum trades
        if trades < cfg.min_trades_per_period:
            failures.append(f"Insufficient trades: {trades} < {cfg.min_trades_per_period}")

        # Check Sharpe threshold
        if sharpe < sharpe_threshold:
            failures.append(f"Sharpe {sharpe:.2f} < threshold {sharpe_threshold}")

        # Check profit factor
        if profit_factor < 1.0 and trades >= cfg.min_trades_per_period:
            failures.append(f"Profit factor {profit_factor:.2f} < 1.0")

        return PeriodResult(
            period_name=period_name,
            start_date=metrics.get("start_date", pd.NaT),
            end_date=metrics.get("end_date", pd.NaT),
            total_trades=trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            total_return_pct=metrics.get("total_return_pct", 0.0),
            max_drawdown_pct=max_dd,
            profit_factor=profit_factor,
            passes_threshold=len(failures) == 0,
            failure_reasons=failures,
        )

    def validate_multiple_patterns(
        self,
        pattern_metrics: Dict[str, Dict[str, Dict[str, Any]]],
        config: Optional[WalkForwardConfig] = None,
    ) -> pd.DataFrame:
        """
        Validate multiple patterns and return summary DataFrame.

        Args:
            pattern_metrics: Dict[pattern_name -> Dict[period -> metrics]]
            config: Optional validation config

        Returns:
            DataFrame with validation results for all patterns
        """
        results = []
        for pattern_name, metrics in pattern_metrics.items():
            result = self.run_validation(pattern_name, metrics, config)
            results.append(
                {
                    "pattern_name": pattern_name,
                    "is_sharpe": result.is_result.sharpe_ratio,
                    "is_trades": result.is_result.total_trades,
                    "is_passes": result.is_result.passes_threshold,
                    "oos_sharpe": result.oos_result.sharpe_ratio,
                    "oos_trades": result.oos_result.total_trades,
                    "oos_passes": result.oos_result.passes_threshold,
                    "fv_sharpe": result.fv_result.sharpe_ratio if result.fv_result else None,
                    "fv_trades": result.fv_result.total_trades if result.fv_result else None,
                    "fv_passes": result.fv_result.passes_threshold if result.fv_result else None,
                    "degradation_abs": result.degradation_absolute,
                    "degradation_pct": result.degradation_percentage,
                    "overfit_status": result.overfit_status.value,
                    "passes_validation": result.passes_validation,
                    "failure_reasons": "; ".join(result.all_failure_reasons),
                }
            )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("degradation_pct", ascending=True)

        return df
