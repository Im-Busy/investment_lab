import pandas as pd
import numpy as np
from typing import Optional, Sequence


def compute_returns(
    prices: pd.DataFrame,
    lags: Sequence[int] = (1,),
    method: str = "pct_change",
) -> pd.DataFrame:
    """Compute multi-period returns from a price matrix.

    Args:
        prices: DataFrame with date index and ticker columns (wide format).
        lags: Look-back periods for return calculation.
        method: One of 'pct_change' or 'log'.

    Returns:
        DataFrame with same shape as prices, columns named ``return_{lag}``.
    """
    results = {}
    for lag in lags:
        if method == "pct_change":
            results[f"return_{lag}"] = prices.pct_change(lag).stack()
        elif method == "log":
            results[f"return_{lag}"] = np.log(prices / prices.shift(lag)).stack()
        else:
            raise ValueError(f"Unknown method: {method}")
    return pd.DataFrame(results)


def compute_momentum(
    returns_df: pd.DataFrame,
    reversal_horizon: int = 1,
    momentum_horizons: Sequence[int] = (2, 3, 6, 9, 12),
) -> pd.DataFrame:
    """Compute momentum factors from pre-calculated returns.

    Momentum at horizon H is return_Hm minus the reversal leg (return_1m).
    Classic 3-12 momentum: return_12m - return_3m.

    Args:
        returns_df: MultiIndex DataFrame (date, ticker) with columns
            like ``return_1``, ``return_2``, etc.
        reversal_horizon: Horizon for the short-term reversal leg.
        momentum_horizons: Horizons for momentum factors.

    Returns:
        DataFrame with momentum columns appended.
    """
    momentum = pd.DataFrame(index=returns_df.index)
    reversal_col = f"return_{reversal_horizon}"
    for h in momentum_horizons:
        ret_col = f"return_{h}"
        if ret_col in returns_df.columns and reversal_col in returns_df.columns:
            momentum[f"momentum_{h}"] = returns_df[ret_col] - returns_df[reversal_col]
    if "return_12" in returns_df.columns and "return_3" in returns_df.columns:
        momentum["momentum_3_12"] = returns_df["return_12"] - returns_df["return_3"]
    return momentum


def compute_lagged_features(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    max_lag: int = 6,
) -> pd.DataFrame:
    """Create lagged versions of specified columns for supervised learning.

    Appends columns like ``{col}_t-1``, ``{col}_t-2``, ..., ``{col}_t-{max_lag}``
    grouped by the 'ticker' level.

    Args:
        data: MultiIndex DataFrame indexed by (date, ticker).
        feature_cols: Columns to lag.
        max_lag: Maximum lag period.

    Returns:
        data with lagged feature columns appended (modifies input in place).
    """
    for t in range(1, max_lag + 1):
        for col in feature_cols:
            if col in data.columns:
                data[f"{col}_t-{t}"] = data.groupby(level="ticker")[col].shift(t)
    return data


def compute_forward_returns(
    data: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3, 6, 12),
    return_col_prefix: str = "return_",
) -> pd.DataFrame:
    """Create forward return targets by shifting returns backward.

    Appends columns ``target_{h}m`` which are shifted so the target value
    at row t represents the return from t to t+h.

    Args:
        data: MultiIndex DataFrame indexed by (date, ticker).
        horizons: Forward horizons.
        return_col_prefix: Prefix for the return column.

    Returns:
        data with target columns appended.
    """
    for h in horizons:
        ret_col = f"{return_col_prefix}{h}"
        if ret_col in data.columns:
            data[f"target_{h}m"] = data.groupby(level="ticker")[ret_col].shift(-h)
    return data


def winsorize_multiindex(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorize a MultiIndex series at given percentiles.

    Clips extreme values at cross-sectional quantile boundaries.

    Args:
        series: A pandas Series (typically from .stack() operations).
        lower: Lower quantile (e.g., 0.01 for 1% winsorization).
        upper: Upper quantile (e.g., 0.99 for 99% winsorization).

    Returns:
        Winsorized Series with same index.
    """
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))


def compute_rolling_factor_betas(
    data: pd.DataFrame,
    factor_data: pd.DataFrame,
    target_col: str = "return_1",
    window: int = 24,
) -> pd.DataFrame:
    """Compute rolling Fama-French style factor betas using RollingOLS.

    Args:
        data: MultiIndex DataFrame (date, ticker) with the target return column.
        factor_data: DataFrame with factor columns aligned to data's index.
        target_col: Column name containing stock returns.
        window: Rolling window length for OLS estimation.

    Returns:
        DataFrame of factor betas per (date, ticker).
    """
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS

    betas = data.groupby(level="ticker", group_keys=False).apply(
        lambda x: (
            RollingOLS(
                endog=x[target_col],
                exog=sm.add_constant(factor_data.loc[x.index].dropna(how="all")),
                window=min(window, x.shape[0] - 1),
            )
            .fit(params_only=True)
            .params.drop(columns="const", errors="ignore")
        )
    )
    return betas


def impute_factor_betas(
    data: pd.DataFrame,
    factor_cols: Sequence[str],
    method: str = "mean",
) -> pd.DataFrame:
    """Impute missing factor betas per ticker.

    Args:
        data: DataFrame with factor columns.
        factor_cols: Names of factor columns to impute.
        method: One of 'mean', 'ffill', 'median'.

    Returns:
        data with imputed columns (modifies input in place).
    """
    for col in factor_cols:
        if method == "mean":
            data[col] = data.groupby("ticker")[col].transform(lambda x: x.fillna(x.mean()))
        elif method == "ffill":
            data[col] = data.groupby("ticker")[col].transform(lambda x: x.ffill().bfill())
        elif method == "median":
            data[col] = data.groupby("ticker")[col].transform(lambda x: x.fillna(x.median()))
    return data


def spearman_correlation_matrix(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute Spearman rank correlation matrix for feature columns.

    Args:
        data: DataFrame of features.
        columns: Subset of columns to use. None means all numeric columns.

    Returns:
        Spearman correlation matrix as a DataFrame.
    """
    cols = columns if columns is not None else data.select_dtypes(include="number").columns
    return data[cols].corr(method="spearman")
