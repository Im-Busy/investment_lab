"""Regime Audit: Per-Regime Performance Analysis with DSR.

Computes strategy performance conditioned on market regime, using the
SVMRegimeClassifier (or a simpler VIX-based heuristic) to label each
trading day as one of: BULL, BEAR, LOW_VOL, NORMAL, HIGH_VOL, CRISIS.

Produces:
- Per-regime Sharpe, return, max drawdown, win rate, profit factor
- Per-regime DSR (Deflated Sharpe Ratio) for statistical significance
- Consistency score: fraction of regimes where strategy is profitable
- Gate: any regime with negative Sharpe → sizing problem (not signal failure)

References:
    Paiva et al. (2016) — SVM for regime detection (82% precision)
    Harvey, Liu, Zhu (2016) — t > 3.0 threshold for significance
    Bailey & Lopez de Prado (2014) — DSR for per-regime evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.deflated_sharpe import (
    PSRResult,
    _sharpe_ratio,
    compute_dsr_from_returns,
    compute_psr_from_returns,
)

logger = logging.getLogger(__name__)

DEFAULT_VIX_THRESHOLDS = {"LOW_VOL": 15.0, "NORMAL": 25.0, "HIGH_VOL": 35.0}


class MarketRegime(Enum):
    """Market regime categories."""

    BULL = "bull"
    BEAR = "bear"
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class RegimeStats:
    """Performance statistics for a single regime.

    Attributes:
        regime: Market regime label.
        n_days: Number of trading days in this regime.
        n_trades: Number of trades executed in this regime.
        sharpe: Annualized Sharpe ratio.
        total_return: Cumulative return (fraction).
        max_drawdown: Maximum drawdown (negative).
        win_rate: Fraction of positive-return periods.
        profit_factor: Gross profit / gross loss.
        avg_return: Average daily return.
        volatility: Annualized standard deviation.
        psr: Probabilistic Sharpe Ratio (vs benchmark=0).
        dsr: Deflated Sharpe Ratio (vs expected max).
        is_profitable: True if total_return > 0 and sharpe > 0.
    """

    regime: str
    n_days: int
    n_trades: int
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_return: float
    volatility: float
    psr: float
    dsr: float | None
    is_profitable: bool


@dataclass
class RegimeAuditReport:
    """Full regime audit report.

    Attributes:
        total_n_days: Total trading days across all regimes.
        total_n_trades: Total trades.
        overall_sharpe: Overall annualized Sharpe.
        overall_return: Overall total return.
        regimes: List of RegimeStats, one per regime present.
        consistency_score: Fraction of regimes where strategy is profitable.
        all_regimes_pass: True if every regime has positive Sharpe.
        worst_regime: Name of worst-performing regime.
        worst_regime_sharpe: Sharpe of worst-performing regime.
        regime_labels: Array of regime labels per bar.
        summary: Dict summary for JSON export.
    """

    total_n_days: int
    total_n_trades: int
    overall_sharpe: float
    overall_return: float
    regimes: list[RegimeStats]
    consistency_score: float
    all_regimes_pass: bool
    worst_regime: str
    worst_regime_sharpe: float
    regime_labels: np.ndarray
    summary: dict = field(default_factory=dict)


def _classify_regime_by_vix(
    vix_values: np.ndarray,
    spy_returns: np.ndarray | None = None,
    bear_threshold: float = -0.20,
    bull_threshold: float = 0.20,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Classify regimes using VIX levels + SPY drawdown.

    Hierarchical classification:
        1. SPY drawdown < -20% → BEAR
        2. SPY drawdown > +20% from trough → BULL
        3. VIX >= 35 → CRISIS
        4. VIX >= 25 → HIGH_VOL
        5. VIX < 15 → LOW_VOL
        6. Otherwise → NORMAL

    Args:
        vix_values: Array of VIX values per bar.
        spy_returns: Array of SPY returns per bar (for drawdown).
        bear_threshold: Drawdown threshold for BEAR.
        bull_threshold: Recovery threshold for BULL.
        thresholds: Dict of {"LOW_VOL": 15, "NORMAL": 25, "HIGH_VOL": 35}.

    Returns:
        Array of regime label strings.
    """
    if thresholds is None:
        thresholds = DEFAULT_VIX_THRESHOLDS

    n = len(vix_values)
    labels = np.full(n, "NORMAL", dtype=object)

    if spy_returns is not None:
        cum = np.cumprod(1.0 + spy_returns)
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        trough = np.minimum.accumulate(drawdown)
        recovered = _recovery_pct_from_trough(drawdown, trough)

        labels[drawdown <= bear_threshold] = "BEAR"
        labels[(drawdown > bear_threshold) & (recovered > bull_threshold)] = "BULL"

    vix = np.asarray(vix_values)
    labels[vix >= thresholds["HIGH_VOL"]] = "CRISIS"
    labels[(vix >= thresholds["NORMAL"]) & (vix < thresholds["HIGH_VOL"])] = "HIGH_VOL"
    labels[vix < thresholds["LOW_VOL"]] = "LOW_VOL"

    # CRISIS overrides BULL/BEAR — if VIX >= 35 it's a crisis regardless of trend
    crisis_mask = vix >= thresholds["HIGH_VOL"]
    labels[crisis_mask] = "CRISIS"

    return labels


def _recovery_pct_from_trough(drawdown: np.ndarray, trough: np.ndarray) -> np.ndarray:
    """Compute recovery percentage from the historical trough."""
    divisor = np.where(trough < -0.001, -trough, 0.001)
    safe_dd = np.where(drawdown < 0, drawdown, 0)
    return safe_dd / divisor


def classify_regimes_from_returns(
    returns: np.ndarray,
    vix: np.ndarray | None = None,
    spy_returns: np.ndarray | None = None,
    regime_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Classify each bar into a market regime.

    Precedence:
        1. If regime_labels provided (from SVMRegimeClassifier), use those.
        2. If VIX + SPY returns provided, use VIX heuristic.
        3. Otherwise, label all as "NORMAL".

    Args:
        returns: Strategy returns (used to determine number of bars).
        vix: VIX values array (same length as returns).
        spy_returns: SPY benchmark returns (same length as returns).
        regime_labels: Pre-computed regime labels (from SVM).

    Returns:
        Array of regime label strings.
    """
    n = len(returns)
    if regime_labels is not None and len(regime_labels) == n:
        return np.asarray(regime_labels, dtype=object)
    if vix is not None and len(vix) == n:
        return _classify_regime_by_vix(vix, spy_returns)
    return np.full(n, "NORMAL", dtype=object)


def compute_per_regime_metrics(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    trades_mask: np.ndarray | None = None,
    periods_per_year: int = 252,
    n_trials: int = 100,
) -> list[RegimeStats]:
    """Compute performance metrics for each market regime.

    Args:
        returns: Array of strategy period returns.
        regime_labels: Array of regime labels (same length as returns).
        trades_mask: Boolean array marking bars where trades occurred.
        periods_per_year: Annualization factor.
        n_trials: Number of trials for DSR correction.

    Returns:
        List of RegimeStats, one per present regime.
    """
    returns = np.asarray(returns)
    regime_labels = np.asarray(regime_labels)
    results: list[RegimeStats] = []

    unique_regimes = sorted(set(regime_labels))
    if "UNKNOWN" in unique_regimes:
        unique_regimes.remove("UNKNOWN")
        unique_regimes.append("UNKNOWN")

    for regime in unique_regimes:
        mask = regime_labels == regime
        regime_rets = returns[mask]
        n_days = len(regime_rets)

        if n_days < 10:
            continue

        n_trades = int(np.sum(trades_mask[mask])) if trades_mask is not None else 0

        sharpe = _sharpe_ratio(regime_rets, periods_per_year)
        total_return = float(np.prod(1.0 + regime_rets) - 1.0)

        cum = np.cumprod(1.0 + regime_rets)
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - peak) / np.where(peak > 0, peak, 1.0)))

        win_rate = float(np.mean(regime_rets > 0))
        gross_profit = float(np.sum(regime_rets[regime_rets > 0]))
        gross_loss = float(np.abs(np.sum(regime_rets[regime_rets < 0])))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_return = float(np.mean(regime_rets))
        volatility = float(np.std(regime_rets, ddof=1) * np.sqrt(periods_per_year))

        try:
            psr_result = compute_psr_from_returns(
                regime_rets, benchmark=0.0, periods_per_year=periods_per_year
            )
            psr = psr_result.psr
        except Exception:
            psr = 0.0

        try:
            dsr_result = compute_dsr_from_returns(
                regime_rets, n_trials=n_trials, periods_per_year=periods_per_year
            )
            dsr = dsr_result.psr
        except Exception:
            dsr = None

        results.append(
            RegimeStats(
                regime=regime,
                n_days=n_days,
                n_trades=n_trades,
                sharpe=sharpe,
                total_return=total_return,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_return=avg_return,
                volatility=volatility,
                psr=psr,
                dsr=dsr,
                is_profitable=total_return > 0 and sharpe > 0,
            )
        )

    return results


def audit_regimes(
    returns: np.ndarray,
    vix: np.ndarray | None = None,
    spy_returns: np.ndarray | None = None,
    regime_labels: np.ndarray | None = None,
    trades_mask: np.ndarray | None = None,
    periods_per_year: int = 252,
    n_trials: int = 100,
    dates: pd.DatetimeIndex | None = None,
    use_svm: bool = False,
    svm_model_path: str | None = None,
) -> RegimeAuditReport:
    """Run a full regime audit on strategy returns.

    Args:
        returns: Array of strategy period returns.
        vix: VIX values for regime classification.
        spy_returns: SPY benchmark returns for drawdown calculation.
        regime_labels: Optional pre-computed regime labels (overrides auto-classify).
        trades_mask: Boolean mask of bars where trades occurred.
        periods_per_year: Annualization factor.
        n_trials: Multiple-testing trials for DSR.
        dates: DatetimeIndex for labels.
        use_svm: If True, use SVMRegimeClassifier instead of VIX heuristic.
        svm_model_path: Path to saved SVM model.

    Returns:
        RegimeAuditReport with per-regime statistics.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if use_svm and dates is not None:
        try:
            from src.ml.svm_regime import SVMRegimeClassifier
            import pandas as pd

            if regime_labels is None:
                svm = SVMRegimeClassifier()
                if svm_model_path:
                    svm.load(svm_model_path)
                # Build minimal OHLCV DataFrame
                if spy_returns is not None:
                    spy_cum = np.cumprod(1.0 + spy_returns[-n:])
                else:
                    spy_cum = np.cumprod(1.0 + returns)
                spy_open = np.roll(spy_cum, 1)
                spy_open[0] = 1.0
                df = pd.DataFrame(
                    {
                        "Open": spy_open[:n],
                        "High": spy_cum[:n] * 1.01,
                        "Low": spy_cum[:n] * 0.99,
                        "Close": spy_cum[:n],
                    },
                    index=dates[-n:] if len(dates) >= n else dates,
                )

                if svm._is_fitted or svm_model_path:
                    regime_labels = []
                    for i in range(svm.window, n):
                        win = df.iloc[i - svm.window : i]
                        try:
                            result = svm.predict(win)
                            regime_labels.append(result.regime)
                        except Exception:
                            regime_labels.append("unknown")
                    # Pad early bars
                    regime_labels = ["unknown"] * svm.window + regime_labels
                    regime_labels = np.array(regime_labels, dtype=object)
        except ImportError:
            logger.warning("SVM regime classifier not available, using VIX heuristic")
        except Exception:
            logger.warning("SVM regime classification failed, using VIX heuristic", exc_info=True)

    if regime_labels is None:
        regime_labels = classify_regimes_from_returns(returns, vix, spy_returns)

    overall_sharpe = _sharpe_ratio(returns, periods_per_year)
    overall_return = float(np.prod(1.0 + returns) - 1.0)
    total_trades = int(np.sum(trades_mask)) if trades_mask is not None else 0

    regime_stats = compute_per_regime_metrics(
        returns, regime_labels, trades_mask, periods_per_year, n_trials
    )

    profitable_regimes = [r for r in regime_stats if r.is_profitable]
    consistency_score = len(profitable_regimes) / len(regime_stats) if regime_stats else 0.0
    all_pass = all(r.is_profitable for r in regime_stats)

    if regime_stats:
        worst = min(regime_stats, key=lambda r: r.sharpe)
        worst_name, worst_sharpe = worst.regime, worst.sharpe
    else:
        worst_name, worst_sharpe = "NONE", 0.0

    summary = {
        "overall_sharpe": overall_sharpe,
        "overall_return": overall_return,
        "total_days": n,
        "total_trades": total_trades,
        "n_regimes": len(regime_stats),
        "consistency_score": consistency_score,
        "all_regimes_pass": all_pass,
        "worst_regime": worst_name,
        "worst_sharpe": worst_sharpe,
    }

    return RegimeAuditReport(
        total_n_days=n,
        total_n_trades=total_trades,
        overall_sharpe=overall_sharpe,
        overall_return=overall_return,
        regimes=regime_stats,
        consistency_score=consistency_score,
        all_regimes_pass=all_pass,
        worst_regime=worst_name,
        worst_regime_sharpe=worst_sharpe,
        regime_labels=regime_labels,
        summary=summary,
    )


def format_regime_report(report: RegimeAuditReport) -> str:
    """Generate a human-readable regime audit report.

    Args:
        report: RegimeAuditReport from audit_regimes().

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "=" * 70,
        "REGIME AUDIT REPORT",
        "=" * 70,
        "",
        f"  Total Days:           {report.total_n_days}",
        f"  Total Trades:          {report.total_n_trades}",
        f"  Overall Sharpe:        {report.overall_sharpe:>10.3f}",
        f"  Overall Return:        {report.overall_return:>10.1%}",
        f"  Consistency Score:     {report.consistency_score:>10.0%}  "
        f"({sum(1 for r in report.regimes if r.is_profitable)}/{len(report.regimes)} regimes profitable)",
        f"  All Regimes Pass:      {'YES' if report.all_regimes_pass else 'NO'}",
        f"  Worst Regime:          {report.worst_regime} (Sharpe={report.worst_regime_sharpe:.3f})",
        "",
        "-" * 70,
        f"  {'Regime':<14} {'Days':>5} {'Trades':>7} {'Sharpe':>8} {'Return':>9} "
        f"{'MaxDD':>8} {'Win%':>6} {'PF':>6} {'PSR':>7} {'DSR':>7}",
        "-" * 70,
    ]

    for r in report.regimes:
        dsr_str = f"{r.dsr:.3f}" if r.dsr is not None else "  N/A  "
        status = "+" if r.is_profitable else "-"
        lines.append(
            f"  [{status}] {r.regime:<11} {r.n_days:>5} {r.n_trades:>7} "
            f"{r.sharpe:>+8.3f} {r.total_return:>9.1%} {r.max_drawdown:>8.1%} "
            f"{r.win_rate:>6.1%} {r.profit_factor:>6.1f} {r.psr:>7.3f} {dsr_str:>7}"
        )

    lines.extend(
        [
            "",
            "-" * 70,
        ]
    )

    if report.all_regimes_pass:
        lines.append("  CONCLUSION: Strategy is profitable across ALL regimes.")
    else:
        failed = [r.regime for r in report.regimes if not r.is_profitable]
        lines.append(f"  CONCLUSION: Strategy FAILS in: {', '.join(failed)}")
        lines.append("  This is a SIZING problem, not necessarily a signal problem.")
        lines.append("  Consider: regime-adaptive position sizing or regime gates.")

    lines.append("=" * 70)
    return "\n".join(lines)
