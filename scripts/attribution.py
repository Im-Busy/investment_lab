"""
R10: Performance Attribution Decomposition.

Decompose strategy returns via factor regression:
    r_P = α + β_market·r_market + β_sector·r_sector + Σ β_style·r_style + ε

Identifies genuine alpha (intercept) vs factor beta (market, sector, style).
Post-hoc CLI analysis — not real-time.

Source: 华泰多因子 §4, Beyond Fama-French §3

Usage:
    # Basic: strategy vs market
    uv run scripts/attribution.py --strategy outputs/strategy_returns.csv \\
        --market data/raw/SPY_daily.csv

    # With sector and style factors
    uv run scripts/attribution.py --strategy outputs/strategy_returns.csv \\
        --market data/raw/SPY_daily.csv --sector outputs/sector_returns.csv \\
        --style outputs/style_returns.csv

    # Save report
    uv run scripts/attribution.py --strategy outputs/strategy_returns.csv \\
        --market data/raw/SPY_daily.csv --json-output outputs/attribution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── Output formatting ──
_STAR_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, ".")]


def _stars(p: float) -> str:
    for threshold, symbol in _STAR_THRESHOLDS:
        if p < threshold:
            return symbol
    return ""


# ── Data classes ──


@dataclass
class FactorExposure:
    """Single factor exposure from OLS regression."""

    name: str
    beta: float
    beta_se: float
    tstat: float
    pvalue: float
    contribution: float  # beta * mean(factor_return)


@dataclass
class AttributionReport:
    """Full performance attribution decomposition.

    Attributes:
        strategy_name: Label for the strategy being analyzed.
        alpha: Annualized specific return (intercept * 252).
        alpha_daily: Daily alpha (OLS intercept).
        alpha_tstat: t-statistic for the intercept.
        alpha_pvalue: P-value for the intercept.
        r_squared: Model R² (fraction of variance explained by factors).
        adj_r_squared: Adjusted R².
        exposures: List of factor exposures sorted by |tstat| descending.
        total_factor_return: Sum of factor contributions (annualized).
        total_strategy_return: Annualized strategy return.
        unexplained_pct: Percent of strategy return not explained by factors.
    """

    strategy_name: str
    alpha: float
    alpha_daily: float
    alpha_tstat: float
    alpha_pvalue: float
    r_squared: float
    adj_r_squared: float
    exposures: list[FactorExposure] = field(default_factory=list)
    total_factor_return: float = 0.0
    total_strategy_return: float = 0.0
    unexplained_pct: float = 0.0


# ── Core logic ──


def _load_returns(csv_path: str, col: Optional[str] = None) -> pd.Series:
    """Load daily returns from CSV.

    If col is None, uses the first numeric column.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if col is not None:
        return df[col].astype(float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric columns in {csv_path}")
    return df[numeric_cols[0]].astype(float)


def _load_close_to_returns(csv_path: str) -> pd.Series:
    """Load price data and convert to daily returns."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "Close" in df.columns:
        return df["Close"].pct_change().dropna()
    if "Adj Close" in df.columns:
        return df["Adj Close"].pct_change().dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError(f"No Close or numeric columns in {csv_path}")
    return df[numeric_cols[0]].pct_change().dropna()


def decompose_strategy(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: Optional[pd.DataFrame] = None,
    style_returns: Optional[pd.DataFrame] = None,
    strategy_name: str = "Strategy",
) -> AttributionReport:
    """Decompose strategy returns via OLS factor regression.

    Args:
        strategy_returns: Daily strategy return series.
        market_returns: Daily market return series.
        sector_returns: DataFrame of sector factor returns (columns = sector names).
        style_returns: DataFrame of style factor returns (columns = style names).
        strategy_name: Label for the strategy.

    Returns:
        AttributionReport with alpha, betas, R², and factor contributions.
    """
    aligned = pd.DataFrame({"strategy": strategy_returns, "market": market_returns}).dropna()
    common_idx = aligned.index

    factor_list: list[pd.Series] = [aligned["market"]]
    factor_names: list[str] = ["Market"]

    if sector_returns is not None:
        sec_aligned = sector_returns.reindex(common_idx).dropna(axis=1, how="all")
        if not sec_aligned.empty:
            common_idx = sec_aligned.index.intersection(aligned.index)
            for col in sec_aligned.columns:
                factor_list.append(sec_aligned[col])
                factor_names.append(f"Sector:{col}")

    if style_returns is not None:
        sty_aligned = style_returns.reindex(common_idx).dropna(axis=1, how="all")
        if not sty_aligned.empty:
            common_idx = sty_aligned.index.intersection(common_idx)
            for col in sty_aligned.columns:
                factor_list.append(sty_aligned[col])
                factor_names.append(f"Style:{col}")

    aligned = aligned.loc[common_idx]
    if len(aligned) < 30:
        raise ValueError(f"Insufficient data after alignment: {len(aligned)} < 30 bars")

    x_data = pd.concat(factor_list, axis=1).loc[common_idx]
    x_data.columns = factor_names
    x_data = x_data.dropna()

    final_idx = aligned.index.intersection(x_data.index)
    y = aligned.loc[final_idx, "strategy"]
    x = x_data.loc[final_idx]

    x_sm = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, x_sm, missing="drop").fit()

    alpha_daily = float(model.params[0])
    alpha = alpha_daily * 252
    alpha_tstat = float(model.tvalues[0]) if len(model.tvalues) > 0 else 0.0
    alpha_pvalue = float(model.pvalues[0]) if len(model.pvalues) > 0 else 1.0

    exposures: list[FactorExposure] = []
    total_contrib = 0.0
    for i, name in enumerate(factor_names):
        beta = float(model.params[i + 1])
        beta_se = float(model.bse[i + 1])
        tstat = float(model.tvalues[i + 1])
        pvalue = float(model.pvalues[i + 1])
        factor_mean = float(x[name].mean())
        contribution = beta * factor_mean * 252
        total_contrib += contribution
        exposures.append(
            FactorExposure(
                name=name,
                beta=beta,
                beta_se=beta_se,
                tstat=tstat,
                pvalue=pvalue,
                contribution=contribution,
            )
        )

    exposures.sort(key=lambda e: abs(e.tstat), reverse=True)

    total_strategy_return = float(y.mean()) * 252
    unexplained_pct = (
        (total_strategy_return - total_contrib) / max(abs(total_strategy_return), 1e-10) * 100
    )

    return AttributionReport(
        strategy_name=strategy_name,
        alpha=alpha,
        alpha_daily=alpha_daily,
        alpha_tstat=alpha_tstat,
        alpha_pvalue=alpha_pvalue,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        exposures=exposures,
        total_factor_return=total_contrib,
        total_strategy_return=total_strategy_return,
        unexplained_pct=unexplained_pct,
    )


# ── Formatting ──


def format_report(report: AttributionReport) -> str:
    """Format an AttributionReport as a readable string."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"  PERFORMANCE ATTRIBUTION: {report.strategy_name}")
    lines.append("=" * 72)

    lines.append("\n  Regression: r_P = α + Σ β_i · r_factor_i")
    lines.append(f"  Observations: {len(report.exposures) + 1} parameters")

    lines.append(f"\n  {'─' * 68}")
    lines.append(f"  Alpha (annualized):     {report.alpha:>10.4f}  ({report.alpha * 100:+.2f}%)")
    lines.append(
        f"  Alpha t-stat:           {report.alpha_tstat:>10.4f}{_stars(report.alpha_pvalue)}"
    )
    lines.append(f"  Alpha p-value:          {report.alpha_pvalue:>10.4f}")
    lines.append(f"  R²:                     {report.r_squared:>10.4f}")
    lines.append(f"  Adj R²:                 {report.adj_r_squared:>10.4f}")
    lines.append(f"  Total Strategy Return:  {report.total_strategy_return * 100:>10.2f}%")
    lines.append(f"  Total Factor Return:    {report.total_factor_return * 100:>10.2f}%")
    lines.append(f"  Unexplained:            {report.unexplained_pct:>10.2f}%")

    if report.exposures:
        lines.append(f"\n  {'Factor':<24} {'Beta':>8} {'SE':>8} {'t':>8} {'p':>8} {'Contrib%':>10}")
        lines.append(f"  {'─' * 24} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")
        for exp in report.exposures:
            lines.append(
                f"  {exp.name:<24} {exp.beta:>8.4f} {exp.beta_se:>8.4f} "
                f"{exp.tstat:>8.2f} {exp.pvalue:>8.4f} {exp.contribution * 100:>10.2f}%"
            )

    interpretation = _interpret(report)
    if interpretation:
        lines.append("\n  Interpretation:")
        for line in interpretation:
            lines.append(f"    {line}")

    lines.append("")
    return "\n".join(lines)


def _interpret(report: AttributionReport) -> list[str]:
    """Generate interpretation lines for the report."""
    lines: list[str] = []

    if report.alpha_pvalue < 0.05 and report.alpha > 0:
        lines.append(
            f"✅ Significant positive alpha: {report.alpha * 100:.2f}%/yr (p={report.alpha_pvalue:.4f})"
        )
    elif report.alpha_pvalue < 0.05 and report.alpha < 0:
        lines.append(
            f"❌ Significant negative alpha: {report.alpha * 100:.2f}%/yr (p={report.alpha_pvalue:.4f})"
        )
    else:
        lines.append(f"⏸️  Alpha not statistically significant (p={report.alpha_pvalue:.4f})")

    if report.r_squared > 0.7:
        lines.append(
            f"📊 High factor dependence: R²={report.r_squared:.2f} — most return explained by factors"
        )
    elif report.r_squared > 0.3:
        lines.append(f"📊 Moderate factor dependence: R²={report.r_squared:.2f}")
    else:
        lines.append(
            f"📊 Low factor dependence: R²={report.r_squared:.2f} — returns largely idiosyncratic"
        )

    dominant = report.exposures[0] if report.exposures else None
    if dominant and abs(dominant.tstat) > 2:
        direction = "long" if dominant.beta > 0 else "short"
        lines.append(f"📈 Dominant exposure: {dominant.name} ({direction}, t={dominant.tstat:.1f})")

    return lines


def report_to_dict(report: AttributionReport) -> dict:
    """Convert AttributionReport to JSON-serializable dict."""
    return {
        "strategy_name": report.strategy_name,
        "alpha": report.alpha,
        "alpha_daily": report.alpha_daily,
        "alpha_tstat": report.alpha_tstat,
        "alpha_pvalue": report.alpha_pvalue,
        "r_squared": report.r_squared,
        "adj_r_squared": report.adj_r_squared,
        "total_strategy_return": report.total_strategy_return,
        "total_factor_return": report.total_factor_return,
        "unexplained_pct": report.unexplained_pct,
        "exposures": [
            {
                "name": e.name,
                "beta": e.beta,
                "beta_se": e.beta_se,
                "tstat": e.tstat,
                "pvalue": e.pvalue,
                "contribution": e.contribution,
            }
            for e in report.exposures
        ],
    }


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance Attribution: decompose strategy returns into factor betas + alpha"
    )
    parser.add_argument("--strategy", required=True, help="CSV with strategy daily returns")
    parser.add_argument(
        "--strategy-col", help="Column name in strategy CSV (default: first numeric)"
    )
    parser.add_argument("--market", required=True, help="CSV with market daily prices or returns")
    parser.add_argument(
        "--market-col", help="Column name in market CSV (default: Close or first numeric)"
    )
    parser.add_argument("--sector", help="CSV with sector factor returns (columns = sector names)")
    parser.add_argument("--style", help="CSV with style factor returns (columns = style names)")
    parser.add_argument("--json-output", help="Save JSON report to file")
    parser.add_argument("--name", default="Strategy", help="Label for the strategy")
    args = parser.parse_args()

    # Load strategy returns
    strategy = _load_returns(args.strategy, args.strategy_col)
    print(f"  Strategy: {len(strategy)} bars loaded")

    # Load market returns (handle prices vs returns)
    market_df = pd.read_csv(args.market, index_col=0, parse_dates=True)
    if args.market_col:
        market_ret = market_df[args.market_col].astype(float).pct_change().dropna()
    elif "Close" in market_df.columns:
        market_ret = market_df["Close"].astype(float).pct_change().dropna()
    elif "Adj Close" in market_df.columns:
        market_ret = market_df["Adj Close"].astype(float).pct_change().dropna()
    else:
        numeric = market_df.select_dtypes(include=[np.number]).columns[0]
        market_ret = market_df[numeric].astype(float).pct_change().dropna()
    print(f"  Market:  {len(market_ret)} bars loaded")

    # Load optional factors
    sector_returns: Optional[pd.DataFrame] = None
    style_returns: Optional[pd.DataFrame] = None
    if args.sector:
        sector_returns = pd.read_csv(args.sector, index_col=0, parse_dates=True)
        print(f"  Sector:  {len(sector_returns.columns)} factors, {len(sector_returns)} bars")
    if args.style:
        style_returns = pd.read_csv(args.style, index_col=0, parse_dates=True)
        print(f"  Style:   {len(style_returns.columns)} factors, {len(style_returns)} bars")

    # Decompose
    report = decompose_strategy(
        strategy_returns=strategy,
        market_returns=market_ret,
        sector_returns=sector_returns,
        style_returns=style_returns,
        strategy_name=args.name,
    )

    # Output
    print(format_report(report))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str))
        print(f"  Report saved to {args.json_output}")


if __name__ == "__main__":
    main()
