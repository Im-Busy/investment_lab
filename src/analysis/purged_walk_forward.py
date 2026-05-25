"""Purged Walk-Forward Validator with Statistical Gates.

Extends the basic WalkForwardValidator with purging and pre-committed
decision gates from the literature:

- Purge gaps between IS and OOS to prevent information leakage from
  overlapping features (e.g., 20-day MA crossing the boundary).

- Walk-Forward Efficiency (WFE): OOS_return / IS_return — must be > 0.5.

- Majority-Pass: >50% of windows must have OOS Sharpe > 0.

- Catastrophic Veto: Any single window with OOS return < -30% is a hard fail.

- Purged OOS chain: All OOS segments concatenated into a single equity curve
  for the true out-of-sample performance estimate.

References:
    AlgoXpert Alpha Research Framework. arXiv:2603.09219
    Arian, Norouzi, Seco. "Backtest Overfitting in the ML Era." SSRN 4686376
    Lopez de Prado, M. "Advances in Financial Machine Learning." Wiley 2018.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.deflated_sharpe import _sharpe_ratio

logger = logging.getLogger(__name__)

DEFAULT_PURGE_DAYS = 21  # ≈1 month calendar, removes MA/indicator overlap
DEFAULT_WFE_THRESHOLD = 0.50
DEFAULT_MAJORITY_THRESHOLD = 0.50
DEFAULT_CATASTROPHIC_THRESHOLD = -0.30


@dataclass
class PurgedWindowResult:
    """Result for a single purged walk-forward window.

    Attributes:
        window_idx: Sequential window number.
        is_start, is_end: In-sample date range.
        oos_start, oos_end: Out-of-sample date range.
        purge_start, purge_end: Purge gap date range (may be empty).
        is_sharpe: In-sample annualized Sharpe.
        oos_sharpe: Out-of-sample annualized Sharpe.
        is_return: In-sample total return.
        oos_return: Out-of-sample total return.
        wfe: Walk-Forward Efficiency (oos_return / is_return).
        trades: Number of trades (if available).
        oos_returns: Array of OOS period returns (for equity curve chaining).
    """

    window_idx: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    purge_start: pd.Timestamp | None
    purge_end: pd.Timestamp | None
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    wfe: float
    trades: int
    oos_returns: np.ndarray
    is_returns: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class PurgedWFAReport:
    """Aggregated purged walk-forward analysis report.

    Attributes:
        windows: List of per-window results.
        n_windows: Number of windows evaluated.
        purge_days: Purge gap in trading days.
        chained_oos_sharpe: Sharpe of concatenated OOS segments.
        chained_oos_return: Total return of concatenated OOS segments.
        mean_wfe: Mean Walk-Forward Efficiency across windows.
        majority_pass_fraction: Fraction of windows with OOS Sharpe > 0.
        catastrophic_veto_triggered: True if any window had OOS return < -30%.
        wfe_pass: True if mean WFE > 0.50.
        majority_pass: True if >50% of windows have OOS Sharpe > 0.
        overall_pass: True if all three gates pass.
        gate_results: Dict of {gate_name: passed}.
    """

    windows: list[PurgedWindowResult]
    n_windows: int
    purge_days: int
    chained_oos_sharpe: float
    chained_oos_return: float
    mean_wfe: float
    majority_pass_fraction: float
    catastrophic_veto_triggered: bool
    wfe_pass: bool
    majority_pass: bool
    overall_pass: bool
    gate_results: dict[str, bool]


class PurgedWalkForwardValidator:
    """Walk-forward validator with purge gaps and pre-committed decision gates.

    Splits history into rolling IS/OOS window pairs separated by a purge gap.
    Reports WFE, majority-pass rate, catastrophic veto, and chained OOS equity
    curve metrics.

    Usage:
        validator = PurgedWalkForwardValidator(
            is_days=4*252,  # 4 years training
            oos_days=252,   # 1 year test
            purge_days=21,  # 1 month purge gap
            step_days=126,  # 6-month step
        )
        report = validator.validate(returns_series, dates)
    """

    def __init__(
        self,
        is_days: int = 4 * 252,
        oos_days: int = 252,
        purge_days: int = DEFAULT_PURGE_DAYS,
        step_days: int = 126,
        wfe_threshold: float = DEFAULT_WFE_THRESHOLD,
        majority_threshold: float = DEFAULT_MAJORITY_THRESHOLD,
        catastrophic_threshold: float = DEFAULT_CATASTROPHIC_THRESHOLD,
        periods_per_year: int = 252,
    ):
        self.is_days = is_days
        self.oos_days = oos_days
        self.purge_days = purge_days
        self.step_days = step_days
        self.wfe_threshold = wfe_threshold
        self.majority_threshold = majority_threshold
        self.catastrophic_threshold = catastrophic_threshold
        self.periods_per_year = periods_per_year

    def validate(
        self,
        returns: np.ndarray | pd.Series,
        dates: pd.DatetimeIndex | None = None,
        trades_per_window: list[int] | None = None,
    ) -> PurgedWFAReport:
        """Run purged walk-forward validation.

        Args:
            returns: Array of strategy period returns.
            dates: DatetimeIndex for timestamps. If None, uses range index.
            trades_per_window: Optional per-window trade counts.

        Returns:
            PurgedWFAReport with all window results and gate status.
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        n = len(returns)

        if dates is None:
            dates = pd.date_range("2000-01-01", periods=n, freq="B")
        elif len(dates) != n:
            raise ValueError(f"dates length ({len(dates)}) != returns length ({n})")

        window_span = self.is_days + self.purge_days + self.oos_days
        if n < window_span:
            logger.warning("Insufficient data: need %d bars, have %d", window_span, n)
            return PurgedWFAReport(
                windows=[],
                n_windows=0,
                purge_days=self.purge_days,
                chained_oos_sharpe=0.0,
                chained_oos_return=0.0,
                mean_wfe=0.0,
                majority_pass_fraction=0.0,
                catastrophic_veto_triggered=True,
                wfe_pass=False,
                majority_pass=False,
                overall_pass=False,
                gate_results={"wfe": False, "majority": False, "catastrophic": False},
            )

        windows: list[PurgedWindowResult] = []
        all_oos_returns: list[float] = []
        idx = 0

        while idx + window_span <= n:
            is_start = idx
            is_end = idx + self.is_days
            purge_start = is_end
            purge_end = min(purge_start + self.purge_days, n)
            oos_start = purge_end
            oos_end = min(oos_start + self.oos_days, n)

            if oos_end > n:
                break

            is_rets = returns[is_start:is_end]
            oos_rets = returns[oos_start:oos_end]

            is_sharpe = _sharpe_ratio(is_rets, self.periods_per_year)
            oos_sharpe = _sharpe_ratio(oos_rets, self.periods_per_year)
            is_return = float(np.prod(1.0 + is_rets) - 1.0)
            oos_return = float(np.prod(1.0 + oos_rets) - 1.0)
            wfe = oos_return / is_return if is_return != 0 else float("-inf")

            trades = (
                trades_per_window[len(windows)]
                if trades_per_window and len(windows) < len(trades_per_window)
                else 0
            )

            windows.append(
                PurgedWindowResult(
                    window_idx=len(windows),
                    is_start=dates[is_start],
                    is_end=dates[is_end - 1],
                    oos_start=dates[oos_start],
                    oos_end=dates[oos_end - 1],
                    purge_start=dates[purge_start] if purge_start < n else None,
                    purge_end=dates[purge_end - 1] if purge_end <= n else None,
                    is_sharpe=is_sharpe,
                    oos_sharpe=oos_sharpe,
                    is_return=is_return,
                    oos_return=oos_return,
                    wfe=wfe,
                    trades=trades,
                    oos_returns=oos_rets.copy(),
                    is_returns=is_rets.copy(),
                )
            )
            all_oos_returns.append(oos_rets)
            idx += self.step_days

        if not windows:
            return PurgedWFAReport(
                windows=[],
                n_windows=0,
                purge_days=self.purge_days,
                chained_oos_sharpe=0.0,
                chained_oos_return=0.0,
                mean_wfe=0.0,
                majority_pass_fraction=0.0,
                catastrophic_veto_triggered=True,
                wfe_pass=False,
                majority_pass=False,
                overall_pass=False,
                gate_results={"wfe": False, "majority": False, "catastrophic": False},
            )

        # Chain OOS segments
        chained = np.concatenate(all_oos_returns)
        chained_sharpe = _sharpe_ratio(chained, self.periods_per_year)
        chained_return = float(np.prod(1.0 + chained) - 1.0)

        # Compute gates
        mean_wfe = float(np.mean([w.wfe for w in windows]))
        oos_positive = [w for w in windows if w.oos_sharpe > 0]
        majority_frac = len(oos_positive) / len(windows) if windows else 0.0
        catastrophic = any(w.oos_return < self.catastrophic_threshold for w in windows)

        wfe_pass = mean_wfe > self.wfe_threshold
        majority_pass = majority_frac >= self.majority_threshold
        overall_pass = wfe_pass and majority_pass and (not catastrophic)

        return PurgedWFAReport(
            windows=windows,
            n_windows=len(windows),
            purge_days=self.purge_days,
            chained_oos_sharpe=chained_sharpe,
            chained_oos_return=chained_return,
            mean_wfe=mean_wfe,
            majority_pass_fraction=majority_frac,
            catastrophic_veto_triggered=catastrophic,
            wfe_pass=wfe_pass,
            majority_pass=majority_pass,
            overall_pass=overall_pass,
            gate_results={
                "wfe": wfe_pass,
                "majority": majority_pass,
                "catastrophic": not catastrophic,
            },
        )

    def format_report(self, report: PurgedWFAReport) -> str:
        """Generate a human-readable purged walk-forward report.

        Args:
            report: PurgedWFAReport from validate().

        Returns:
            Multi-line formatted string.
        """
        lines = [
            "=" * 70,
            "PURGED WALK-FORWARD ANALYSIS REPORT",
            "=" * 70,
            "",
            f"  Windows: {report.n_windows} (IS={self.is_days}d, OOS={self.oos_days}d, "
            f"purge={self.purge_days}d, step={self.step_days}d)",
            "",
            f"  Chained OOS Sharpe:     {report.chained_oos_sharpe:>10.3f}",
            f"  Chained OOS Return:     {report.chained_oos_return:>10.1%}",
            f"  Mean WFE:              {report.mean_wfe:>10.3f}",
            f"  Majority-Pass Fraction:  {report.majority_pass_fraction:>8.1%}",
            "",
            "  Decision Gates:",
        ]
        for gate_name in ("wfe", "majority", "catastrophic"):
            passed = report.gate_results[gate_name]
            status = "PASS" if passed else "FAIL"
            desc = {
                "wfe": f"WFE > {self.wfe_threshold} (mean={report.mean_wfe:.3f})",
                "majority": f"> {self.majority_threshold:.0%} windows positive Sharpe ({report.majority_pass_fraction:.0%})",
                "catastrophic": f"No window < {self.catastrophic_threshold:.0%} return",
            }[gate_name]
            lines.append(f"    [{status}] {desc}")

        lines.extend(["", "-" * 70, "PER-WINDOW DETAILS:", "-" * 70])

        for w in report.windows:
            veto_marker = " ⚠ CATASTROPHIC" if w.oos_return < self.catastrophic_threshold else ""
            lines.append(
                f"  Window {w.window_idx}: IS {w.is_start.date()} -> {w.is_end.date()} | "
                f"OOS {w.oos_start.date()} -> {w.oos_end.date()}{veto_marker}"
            )
            lines.append(
                f"    IS Sharpe={w.is_sharpe:+.3f} (Return {w.is_return:+.1%}) | "
                f"OOS Sharpe={w.oos_sharpe:+.3f} (Return {w.oos_return:+.1%}) | "
                f"WFE={w.wfe:.3f} | Trades={w.trades}"
            )

        lines.extend(["", "=" * 70])
        if report.overall_pass:
            lines.append("CONCLUSION: STRATEGY PASSES ALL WALK-FORWARD GATES")
        else:
            failed = [k for k, v in report.gate_results.items() if not v]
            lines.append(f"CONCLUSION: STRATEGY FAILS GATES: {', '.join(failed)}")
        lines.append("=" * 70)

        return "\n".join(lines)
