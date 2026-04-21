from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import pandas as pd


def load_hdf5_panel(
    store_path: Union[str, Path],
    key: str,
    columns: Optional[Sequence[str]] = None,
    date_range: Optional[tuple] = None,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a panel dataset from HDF5 store.

    Replicates ML4T pattern A1: HDF5 Store Pattern.

    Args:
        store_path: Path to the .h5 file.
        key: HDF5 key (e.g., 'quandl/wiki/prices').
        columns: Columns to select from the store.
        date_range: Tuple of (start_date, end_date) for slicing.
        tickers: List of ticker symbols to filter.

    Returns:
        DataFrame with MultiIndex (date, ticker).
    """
    idx = pd.IndexSlice
    with pd.HDFStore(str(store_path), mode="r") as store:
        df = store[key]

        if columns is not None:
            df = df[columns]

        if date_range is not None:
            start, end = date_range
            df = df.loc[idx[str(start) : str(end), :]]

        if tickers is not None:
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[idx[:, list(tickers)]]

        return df


def unstack_prices(
    panel_df: pd.DataFrame,
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """Convert a MultiIndex panel of prices to wide format (date x ticker).

    Replicates the IndexSlice + unstack pattern from ML4T.

    Args:
        panel_df: MultiIndex DataFrame with columns including the price column.
        price_column: Name of the column to unstack.

    Returns:
        Wide-format DataFrame with dates as index and tickers as columns.
    """
    if isinstance(panel_df.columns, pd.MultiIndex):
        return panel_df[price_column].unstack("ticker")
    return panel_df[[price_column]].unstack("ticker")


def load_fama_french_factors(
    factor_set: str = "F-F_Research_Data_5_Factors_2x3",
    start: str = "2000",
    end: Optional[str] = None,
    drop_rf: bool = True,
) -> pd.DataFrame:
    """Download Fama-French factor data from Kenneth French's website.

    Replicates ML4T pattern A3: Fama-French Factor Download.

    Args:
        factor_set: Factor dataset name. Common options:
            - 'F-F_Research_Data_Factors' (3 factors: Mkt-RF, SMB, HML)
            - 'F-F_Research_Data_5_Factors_2x3' (5 factors)
            - 'F-F_Momentum_Factor' (momentum)
            - 'F-F_Research_Data_FactorsWeekly'
        start: Start date string for pandas.
        end: End date string. None means latest available.
        drop_rf: Whether to drop the risk-free rate column.

    Returns:
        DataFrame of monthly factor returns with percentage values (divide by 100
        to get decimal returns).
    """
    import pandas_datareader.data as web

    data = web.DataReader(factor_set, "famafrench", start=start)
    if isinstance(data, dict):
        # Returns dict with key 0 for the main dataset
        df = data[0]
    else:
        df = data

    if drop_rf and "RF" in df.columns:
        df = df.drop("RF", axis=1)

    return df


def join_metadata_filter(
    prices: pd.DataFrame,
    metadata: Union[pd.DataFrame, pd.Series],
    metadata_col: Optional[str] = None,
) -> tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
    """Filter price universe to intersection with metadata index.

    Replicates the universe filtering pattern (A4) from ML4T.

    Args:
        prices: Wide-format price DataFrame (tickers as columns).
        metadata: DataFrame or Series with stock metadata.
        metadata_col: If metadata is a DataFrame, column to filter on.
            If provided, filters to rows where metadata_col is truthy.

    Returns:
        Tuple of (filtered_prices, filtered_metadata) restricted to common tickers.
    """
    if metadata_col is not None and isinstance(metadata, pd.DataFrame):
        metadata = metadata[metadata_col].dropna()

    shared = prices.columns.intersection(metadata.index)
    return prices.loc[:, shared], metadata.loc[shared]


def store_to_hdf5(
    data: Union[pd.DataFrame, pd.Series],
    store_path: Union[str, Path],
    key: str,
    mode: str = "a",
    format: str = "table",
    data_columns: Optional[List[str]] = None,
) -> None:
    """Persist DataFrame to HDF5 store.

    Args:
        data: DataFrame or Series to save.
        store_path: Path to the .h5 file.
        key: HDF5 key path (e.g., 'quandl/wiki/prices').
        mode: 'a' to append, 'w' to overwrite.
        format: 'table' for queryable, 'fixed' for faster.
        data_columns: Columns to index for queries (only with format='table').
    """
    with pd.HDFStore(str(store_path), mode=mode) as store:
        store.put(key, data, format=format, data_columns=data_columns)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS string.

    Replicates ML4T utils.py format_time utility.

    Args:
        seconds: Numeric time in seconds.

    Returns:
        Formatted string like '00:01:23'.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"
