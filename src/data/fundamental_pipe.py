"""P28-7: Fundamental analysis pipeline — to_toolkit() pattern.

Bridges FinanceDatabase symbol filtering with FinanceToolkit (installed as
transitive dependency) to extract 60+ financial ratios (ROE, D/E, margins,
growth rates, valuation multiples) for feature engineering.

Pattern: filter tickers via FinanceDatabase → pipe into FinanceToolkit →
structured fundamental DataFrame ready for ML pipeline.

Source: FinanceDatabase + FinanceToolkit (JerBouma).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data.financedb_layer import get_universe

logger = logging.getLogger(__name__)

FRATIO_DEFAULTS = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "days_of_sales_outstanding",
    "days_of_inventory_outstanding",
    "operating_cycle",
    "days_of_payables_outstanding",
    "cash_conversion_cycle",
    "gross_profit_margin",
    "operating_profit_margin",
    "net_profit_margin",
    "effective_tax_rate",
    "return_on_assets",
    "return_on_equity",
    "return_on_capital_employed",
    "debt_to_assets",
    "debt_to_equity",
    "debt_to_capital",
    "total_debt_to_capitalization",
    "interest_coverage_ratio",
    "cash_flow_to_debt",
    "company_equity_multiplier",
    "receivables_turnover",
    "payables_turnover",
    "inventory_turnover",
    "fixed_asset_turnover",
    "asset_turnover",
    "operating_cash_flow_to_sales",
    "free_cash_flow_to_sales",
    "cash_flow_coverage",
    "short_term_coverage",
    "capital_expenditure_coverage",
    "dividend_paid_and_capex_coverage",
    "dividend_payout",
    "price_to_earnings",
    "price_to_book",
    "price_to_sales",
    "price_earnings_to_growth",
    "price_to_cash_flow",
    "enterprise_value_multiple",
    "price_to_free_cash_flow",
    "price_to_operating_cash_flow",
    "earnings_yield",
    "free_cash_flow_yield",
    "dividend_yield",
    "payout_ratio",
    "tangible_asset_value",
    "net_current_asset_value",
    "enterprise_value",
    "market_cap",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebit",
    "ebitda",
    "total_assets",
    "total_debt",
    "free_cash_flow",
    "capital_expenditures",
    "depreciation_and_amortization",
    "research_and_development",
    "weighted_average_shares_diluted",
    "revenue_growth",
    "net_income_growth",
    "ebitda_growth",
    "free_cash_flow_growth",
]


@dataclass
class FundamentalBatch:
    """Result of a batch fundamental analysis run."""

    success: dict[str, pd.DataFrame]
    """Ticker → ratio DataFrame (row per quarter)."""

    failed: dict[str, str]
    """Ticker → error message."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Batch metadata (sector, country, etc.)."""

    @property
    def coverage(self) -> float:
        total = len(self.success) + len(self.failed)
        return len(self.success) / max(total, 1) * 100


def _get_toolkit():
    """Lazy-import FinanceToolkit."""
    import financetoolkit as ft

    return ft


def to_toolkit(
    tickers: list[str],
    *,
    ratios: Optional[list[str]] = None,
    quarters: int = 20,
    progress_bar: bool = False,
) -> FundamentalBatch:
    """Pipe tickers into FinanceToolkit for deep fundamental analysis.

    Args:
        tickers: List of ticker symbols.
        ratios: Specific ratio codes to fetch (default: FRATIO_DEFAULTS).
        quarters: Number of quarterly periods to fetch.
        progress_bar: Show tqdm progress bar.

    Returns:
        FundamentalBatch with structured ratio DataFrames per ticker.

    Example:
        >>> tech_tickers = get_universe(
        ...     country="United States",
        ...     sector="Information Technology",
        ...     market_cap="Large Cap",
        ... )
        >>> batch = to_toolkit(tech_tickers[:10], quarters=8)
        >>> for ticker, df in batch.success.items():
        ...     roe = df.loc["Return on Equity"].iloc[:, -1]
        ...     print(f"{ticker} latest ROE: {roe:.2%}")
    """
    ft = _get_toolkit()
    ratio_list = ratios or FRATIO_DEFAULTS
    success: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}

    for ticker in tickers:
        try:
            company = ft.Toolkit(
                ticker,
                api_key="",
                start_date=None,
                quarterly=True,
                progress_bar=progress_bar,
            )
            ratio_df = company.ratios.collect_all_ratios(
                rounding=4,
                growth=True,
            )
            raw = ratio_df.loc["ln_cross_sectional_score"]
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                selected = raw.loc[raw.index.intersection(ratio_list)]
                if not selected.empty:
                    success[ticker] = selected
                    continue
            logger.debug("%s: no ratio data returned", ticker)
            failed[ticker] = "No ratio data available"
        except Exception as e:
            logger.debug("%s: %s", ticker, e)
            failed[ticker] = str(e)

    logger.info(
        "Fundamental pipe: %d/%d tickers (%d%% coverage)",
        len(success),
        len(tickers),
        int(FundamentalBatch(success, failed).coverage),
    )
    return FundamentalBatch(success=success, failed=failed)


def fundamental_features_for_ml(
    tickers: list[str],
    *,
    ratios: Optional[list[str]] = None,
    quarters: int = 12,
) -> pd.DataFrame:
    """Extract fundamental features in ML-ready wide format.

    Each ticker gets one row with columns like `ticker`, `roe_tm`,
    `debt_to_equity_tm`, etc. (tm = trailing mean over available quarters).

    Args:
        tickers: List of ticker symbols.
        ratios: Ratio codes to fetch (default: key FRATIO_DEFAULTS subset).
        quarters: Number of quarterly periods.

    Returns:
        DataFrame indexed by ticker with fundamental ratio columns.
    """
    ml_ratios = ratios or [
        "return_on_equity",
        "return_on_assets",
        "net_profit_margin",
        "debt_to_equity",
        "current_ratio",
        "interest_coverage_ratio",
        "revenue_growth",
        "free_cash_flow_yield",
        "price_to_earnings",
        "price_to_book",
        "earnings_yield",
        "enterprise_value_multiple",
        "operating_cash_flow_to_sales",
        "asset_turnover",
        "price_earnings_to_growth",
        "dividend_yield",
    ]
    batch = to_toolkit(tickers, ratios=ml_ratios, quarters=quarters)
    rows: list[dict[str, Any]] = []
    for ticker, df in batch.success.items():
        row: dict[str, Any] = {"ticker": ticker}
        for ratio_id in df.index:
            vals = df.loc[ratio_id].dropna().values
            if len(vals) > 0:
                row[f"{ratio_id}_mean"] = float(vals.mean())
                row[f"{ratio_id}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                row[f"{ratio_id}_last"] = float(vals[-1])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")


def pipe_sector_fundamentals(
    country: str = "United States",
    sector: str = "Information Technology",
    market_cap: str = "Large Cap",
    max_tickers: int = 200,
    **to_toolkit_kwargs,
) -> FundamentalBatch:
    """End-to-end: get tickers from FinanceDatabase → pipe into FinanceToolkit.

    Args:
        country: Country filter.
        sector: GICS sector filter.
        market_cap: Market cap class filter.
        max_tickers: Max tickers to analyze.
        **to_toolkit_kwargs: Passed to to_toolkit().

    Returns:
        FundamentalBatch with results.
    """
    symbols = get_universe(
        country=country,
        sector=sector,
        market_cap=market_cap,
    )[:max_tickers]
    logger.info("Piping %d tickers from %s/%s → FinanceToolkit", len(symbols), country, sector)
    batch = to_toolkit(symbols, **to_toolkit_kwargs)
    batch.meta = {"country": country, "sector": sector, "market_cap": market_cap}
    return batch
