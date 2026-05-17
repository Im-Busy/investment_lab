"""
DSL Executor — evaluates factor expression trees against OHLCV DataFrames.

Computes factor scores point-in-time: for each date t, the expression is
evaluated using only data available at or before t (no forward-looking).

The executor operates on a MultiIndex DataFrame with (date, ticker) index
or a regular DataFrame with ticker columns and date index.

Base variables (point-in-time columns required):
    open, high, low, close, volume, mcap, dvol

Derived variables (computed on-the-fly):
    ret_cc, ret_log, hl_range, rel_volume, realized_vol,
    price_to_ma_N, volume_pct_chg

Computational pipeline:
    1. Parser produces FactorExpression tree (grammar.py)
    2. Validator checks constraints (validator.py)
    3. Executor traverses tree bottom-up against the panel data
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.patterns.dsl.grammar import (
    FactorExpression,
    ConstantExpr,
    CrossSectionalExpr,
    LinearComboExpr,
    NonlinearExpr,
    TimeSeriesExpr,
    VariableExpr,
)

logger = logging.getLogger(__name__)

BASE_VARIABLES: set[str] = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mcap",
    "dvol",
}

DERIVED_VARIABLES: set[str] = {
    "ret_cc",
    "ret_log",
    "hl_range",
    "rel_volume",
    "realized_vol",
    "price_to_ma_20",
    "price_to_ma_50",
    "price_to_ma_200",
    "volume_pct_chg",
}

ALLOWED_VARIABLES: set[str] = BASE_VARIABLES | DERIVED_VARIABLES

ALLOWED_OPERATORS: dict[str, str] = {
    "rank": "cross_sectional",
    "zscore": "cross_sectional",
    "lag": "time_series",
    "ma": "time_series",
    "std": "time_series",
    "diff": "time_series",
    "pct_chg": "time_series",
    "log": "nonlinear",
    "abs": "nonlinear",
    "clip": "nonlinear",
    "sqrt": "nonlinear",
}


class DSLExecutionError(Exception):
    """Raised when a DSL expression cannot be executed against the data."""


class DSLValidationError(Exception):
    """Raised when a DSL expression fails structural validation."""


def _compute_derived(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute derived variables from base OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and date index.
            Shape: (n_dates, n_tickers) for each column or MultiIndex (date, ticker).

    Returns:
        Dictionary of {variable_name: DataFrame} for derived variables.
    """
    derived: dict[str, pd.DataFrame] = {}

    close = df["close"]
    derived["ret_cc"] = close.pct_change()
    derived["ret_log"] = np.log(close / close.shift(1))
    derived["hl_range"] = (df["high"] - df["low"]) / close

    if "volume" in df.columns:
        vol = df["volume"]
        derived["volume_pct_chg"] = vol.pct_change()
        derived["rel_volume"] = vol / vol.rolling(20, min_periods=1).mean()

    close_ma20 = close.rolling(20, min_periods=5).mean()
    close_ma50 = close.rolling(50, min_periods=10).mean()
    close_ma200 = close.rolling(200, min_periods=30).mean()
    derived["price_to_ma_20"] = close / close_ma20
    derived["price_to_ma_50"] = close / close_ma50
    derived["price_to_ma_200"] = close / close_ma200

    if "close" in df.columns and "volume" not in df.columns:
        derived["realized_vol"] = close.pct_change().rolling(20, min_periods=5).std()
    elif "close" in df.columns:
        derived["realized_vol"] = close.pct_change().rolling(20, min_periods=5).std()

    return derived


def _build_data_map(df: pd.DataFrame, variables_needed: set[str]) -> dict[str, pd.DataFrame]:
    """Build a map from variable name to DataFrame for evaluation.

    Handles both MultiIndex (date, ticker) and regular (date, columns=tickers) DataFrames.
    """
    data_map: dict[str, pd.DataFrame] = {}

    has_multiindex = isinstance(df.index, pd.MultiIndex)

    for var in variables_needed:
        if var in BASE_VARIABLES:
            if has_multiindex:
                data_map[var] = df[var].unstack()
            elif var in df.columns:
                data_map[var] = df[[var]]
            elif isinstance(df.columns, pd.MultiIndex):
                try:
                    data_map[var] = df.xs(var, level=0, axis=1)
                except KeyError:
                    pass
            else:
                pass  # Check derived

    if has_multiindex:
        unstacked_close = df["close"].unstack()
        unstacked_high = df["high"].unstack() if "high" in df.columns else None
        unstacked_low = df["low"].unstack() if "low" in df.columns else None
        unstacked_volume = df["volume"].unstack() if "volume" in df.columns else None
    else:
        unstacked_close = df[["close"]] if "close" in df.columns else None
        unstacked_high = df[["high"]] if "high" in df.columns else None
        unstacked_low = df[["low"]] if "low" in df.columns else None
        unstacked_volume = df[["volume"]] if "volume" in df.columns else None

    if unstacked_close is not None:
        panel: dict[str, pd.DataFrame] = {"close": unstacked_close}
        if unstacked_high is not None:
            panel["high"] = unstacked_high
        if unstacked_low is not None:
            panel["low"] = unstacked_low
        if unstacked_volume is not None:
            panel["volume"] = unstacked_volume

        derived = _compute_derived_from_panel(panel)
        for dv in variables_needed:
            if dv in derived:
                data_map[dv] = derived[dv]

    return data_map


def _compute_derived_from_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute derived variables from a panel dict of base DataFrames."""
    derived: dict[str, pd.DataFrame] = {}
    c = panel["close"]
    derived["ret_cc"] = c.pct_change()
    derived["ret_log"] = np.log(c / c.shift(1))
    if "high" in panel and "low" in panel:
        derived["hl_range"] = (panel["high"] - panel["low"]) / c
    if "volume" in panel:
        v = panel["volume"]
        derived["volume_pct_chg"] = v.pct_change()
        derived["rel_volume"] = v / v.rolling(20, min_periods=1).mean()
    derived["price_to_ma_20"] = c / c.rolling(20, min_periods=5).mean()
    derived["price_to_ma_50"] = c / c.rolling(50, min_periods=10).mean()
    derived["price_to_ma_200"] = c / c.rolling(200, min_periods=30).mean()
    derived["realized_vol"] = c.pct_change().rolling(20, min_periods=5).std()
    return derived


def _eval_node(
    expr: FactorExpression,
    data_map: dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
    ticker_index: pd.Index,
) -> pd.DataFrame:
    """Recursively evaluate an expression node against the data.

    All results share the same (date x ticker) shape as the input data.
    """
    if isinstance(expr, VariableExpr):
        name = expr.name
        if name in data_map:
            return data_map[name].reindex(index=date_index).copy()
        raise DSLExecutionError(
            f"Variable '{name}' not found in data. Available: {sorted(data_map.keys())}"
        )

    if isinstance(expr, ConstantExpr):
        return pd.DataFrame(expr.value, index=date_index, columns=ticker_index)

    if isinstance(expr, CrossSectionalExpr):
        arg_val = _eval_node(expr.arg, data_map, date_index, ticker_index)
        if expr.method == "rank":
            result = arg_val.rank(axis=1, pct=True, method="average")
        elif expr.method == "zscore":
            result = arg_val.sub(arg_val.mean(axis=1), axis=0).div(
                arg_val.std(axis=1).replace(0, np.nan), axis=0
            )
        else:
            raise DSLExecutionError(f"Unknown cross-sectional method: {expr.method}")
        result.columns = ticker_index
        return result

    if isinstance(expr, TimeSeriesExpr):
        arg_val = _eval_node(expr.arg, data_map, date_index, ticker_index)
        k = expr.param
        if expr.method == "lag":
            result = arg_val.shift(k)
        elif expr.method == "ma":
            result = arg_val.rolling(k, min_periods=max(1, k // 4)).mean()
        elif expr.method == "std":
            result = arg_val.rolling(k, min_periods=max(1, k // 4)).std()
        elif expr.method == "diff":
            result = arg_val.diff(k)
        elif expr.method == "pct_chg":
            result = arg_val.pct_change(k)
        else:
            raise DSLExecutionError(f"Unknown time-series method: {expr.method}")
        result.columns = ticker_index
        return result

    if isinstance(expr, NonlinearExpr):
        arg_val = _eval_node(expr.arg, data_map, date_index, ticker_index)
        if expr.method == "log":
            offset = expr.param1 if expr.param1 is not None else 1.0
            result = np.log(offset + arg_val.clip(lower=0))
        elif expr.method == "abs":
            result = arg_val.abs()
        elif expr.method == "clip":
            lo = expr.param1 if expr.param1 is not None else -np.inf
            hi = expr.param2 if expr.param2 is not None else np.inf
            result = arg_val.clip(lower=lo, upper=hi)
        elif expr.method == "sqrt":
            result = np.sqrt(arg_val.clip(lower=0))
        else:
            raise DSLExecutionError(f"Unknown nonlinear method: {expr.method}")
        result.columns = ticker_index
        return result

    if isinstance(expr, LinearComboExpr):
        total: pd.DataFrame | None = None
        for coeff, term in expr.terms:
            term_val = _eval_node(term, data_map, date_index, ticker_index)
            weighted = term_val * coeff
            if total is None:
                total = weighted
            else:
                total = total + weighted
        if total is None:
            return pd.DataFrame(0.0, index=date_index, columns=ticker_index)
        total.columns = ticker_index
        return total

    raise DSLExecutionError(f"Unknown expression type: {type(expr)}")


def _prepare_panel_data(df: pd.DataFrame, required_columns: set[str]) -> dict[str, pd.DataFrame]:
    """Prepare panel data from a DataFrame, handling MultiIndex and regular formats.

    Returns a dict of {variable_name: DataFrame} with date as index, tickers as columns,
    plus derived variables. All available base variables are loaded to support
    transitive dependencies of derived variables.
    """
    panel: dict[str, pd.DataFrame] = {}
    has_multiindex = isinstance(df.index, pd.MultiIndex)

    for col in BASE_VARIABLES:
        if has_multiindex and col in df.columns:
            panel[col] = df[col].unstack()
        elif not has_multiindex and col in df.columns:
            panel[col] = df[[col]]
        elif not has_multiindex and isinstance(df.columns, pd.MultiIndex):
            try:
                panel[col] = df.xs(col, level=0, axis=1)
            except KeyError:
                pass

    if "close" in panel:
        derived = _compute_derived_from_panel(panel)
        panel.update(derived)

    return panel


def evaluate(expr: FactorExpression, df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a factor expression tree against an OHLCV DataFrame.

    Args:
        expr: FactorExpression tree to evaluate.
        df: Input DataFrame. Must have columns matching the expression's variables.
            Supports MultiIndex (date, ticker) or regular (date index).

    Returns:
        DataFrame of factor scores with shape (n_dates, n_tickers).
        Values are NaN where insufficient data history exists.

    Raises:
        DSLExecutionError: If a required variable is missing from the data.
        DSLValidationError: If the expression fails structural validation.
    """
    from src.patterns.dsl.validator import validate_recipe

    val_result = validate_recipe(expr)
    if not val_result.valid:
        raise DSLValidationError(f"Expression validation failed: {'; '.join(val_result.errors)}")

    variables_needed = expr.variables()
    panel = _prepare_panel_data(df, variables_needed)

    if not panel:
        raise DSLExecutionError("No base variables found in data. Provide at minimum 'close'.")

    sample_df = next(iter(panel.values()))
    date_index = sample_df.index
    ticker_index = sample_df.columns

    result = _eval_node(expr, panel, date_index, ticker_index)
    return result


def evaluate_string(expr_str: str, df: pd.DataFrame) -> pd.DataFrame:
    """Parse a DSL string and evaluate it against an OHLCV DataFrame.

    Convenience function combining parse + validate + evaluate.

    Args:
        expr_str: Factor expression in DSL syntax.
        df: Input DataFrame.

    Returns:
        DataFrame of factor scores.
    """
    from src.patterns.dsl.grammar import parse_expr

    expr = parse_expr(expr_str)
    return evaluate(expr, df)


def evaluate_factor_panel(recipes: dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate multiple factor recipes against the same data.

    Args:
        recipes: Dict of {factor_name: dsl_expression_string}.
        df: Input DataFrame.

    Returns:
        DataFrame with MultiIndex columns (factor_name, ticker) and date index.
    """
    results: dict[str, pd.DataFrame] = {}
    for name, recipe_str in recipes.items():
        results[name] = evaluate_string(recipe_str, df)
    return pd.concat(results, axis=1, names=["factor", "ticker"])


def compute_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """Compute daily Information Coefficient between factor scores and forward returns.

    Args:
        scores: Factor scores DataFrame with shape (n_dates, n_tickers).
        forward_returns: Forward returns DataFrame with same shape.
        method: 'pearson' or 'spearman'.

    Returns:
        Series of daily IC values.

    Raises:
        ValueError: If shapes don't match or method is invalid.
    """
    if scores.shape != forward_returns.shape:
        raise ValueError(
            f"Shape mismatch: scores {scores.shape} vs forward_returns {forward_returns.shape}"
        )
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown IC method: {method}. Use 'pearson' or 'spearman'.")

    ic_values: list[float] = []
    for idx in range(len(scores)):
        s = scores.iloc[idx].values
        r = forward_returns.iloc[idx].values
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() >= 10:
            if method == "pearson":
                ic = np.corrcoef(s[mask], r[mask])[0, 1]
            else:
                from scipy.stats import spearmanr

                ic = spearmanr(s[mask], r[mask])[0]
            ic_values.append(ic)
        else:
            ic_values.append(np.nan)

    return pd.Series(ic_values, index=scores.index, name="IC")
