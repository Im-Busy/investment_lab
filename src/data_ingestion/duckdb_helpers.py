"""DuckDB helpers for SQL-based feature engineering over Parquet stores.

Provides zero-copy SQL querying of parquet feature files via DuckDB.
Useful when pandas chaining exceeds 20+ lines or cross-ticker joins are needed.

Usage:
    from src.data_ingestion.duckdb_helpers import query_feature_store

    df = query_feature_store(
        "SELECT ticker, date, ma_cross_score FROM features WHERE ticker='SPY'"
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

FEATURES_DIR = Path("experiments") / "features"
FEATURE_PARQUET = FEATURES_DIR / "feature_store.parquet"
LABELS_PARQUET = FEATURES_DIR / "labels.parquet"


@contextmanager
def duckdb_parquet(parquet_paths: list[Path | str]) -> duckdb.DuckDBPyConnection:
    """Context manager for DuckDB connection with registered parquet views.

    Args:
        parquet_paths: One or more parquet file paths to register as views.

    Yields:
        A DuckDB connection with registered 'features' and optional 'labels' views.
    """
    con = duckdb.connect(":memory:")
    try:
        for path in parquet_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Parquet file not found: {path}")
            view_name = path.stem.replace(".", "_").removesuffix("_parquet")
            con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")
        yield con
    finally:
        con.close()


def query_feature_store(sql: str) -> Any:
    """Run a SQL query against the feature store parquet with column name sanitization.

    The feature store parquet contains pandas-generated column names that may
    include special characters. DuckDB handles quoted identifiers for these.

    Args:
        sql: SQL query string. References tables 'feature_store' and 'labels'.

    Returns:
        Query result — call .df() on the result for a pandas DataFrame.
    """
    con = duckdb.connect(":memory:")
    try:
        if FEATURE_PARQUET.exists():
            con.execute(
                f"CREATE VIEW feature_store AS SELECT * FROM read_parquet('{FEATURE_PARQUET}')"
            )
        if LABELS_PARQUET.exists():
            con.execute(f"CREATE VIEW labels AS SELECT * FROM read_parquet('{LABELS_PARQUET}')")
        return con.sql(sql)
    finally:
        con.close()


def query_feature_store_df(sql: str) -> "pd.DataFrame":
    """Run SQL and return a pandas DataFrame directly."""

    result = query_feature_store(sql)
    return result.df()


def cross_ticker_rank(
    metric: str,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 10,
) -> "pd.DataFrame":
    """Rank tickers by a feature metric within each date using SQL.

    Args:
        metric: Column name in feature_store to rank by.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).
        top_n: Number of top tickers per date.

    Returns:
        DataFrame with columns: ticker, date, {metric}, rank.
    """
    where_clauses = []
    if start_date:
        where_clauses.append(f"date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"date <= '{end_date}'")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
        SELECT ticker, date, "{metric}",
               ROW_NUMBER() OVER (PARTITION BY date ORDER BY "{metric}" DESC) AS rank
        FROM feature_store
        {where}
        QUALIFY rank <= {top_n}
        ORDER BY date, rank
    """
    return query_feature_store_df(sql)


def feature_summary_stats() -> "pd.DataFrame":
    """Return summary statistics for all numeric feature columns using SQL.

    Returns:
        DataFrame with columns: column_name, count, mean, std, min, max.
    """
    sql = """
        SELECT column_name, count, mean, std, min, max
        FROM (
            UNPIVOT (
                SELECT * FROM feature_store
            ) ON COLUMNS(* EXCLUDE (ticker, date))
            INTO NAME column_name VALUE value
        )
        GROUP BY column_name
    """
    return query_feature_store_df(sql)


def join_features_labels(
    ticker: str | None = None,
    horizon: int = 5,
) -> "pd.DataFrame":
    """Join features with labels for a specific ticker and horizon.

    Args:
        ticker: Optional ticker filter.
        horizon: Forward return horizon (1, 5, or 20).

    Returns:
        DataFrame with features joined to their corresponding forward returns.
    """
    where = f"WHERE f.ticker = '{ticker}' AND" if ticker else "WHERE"
    sql = f"""
        SELECT f.*, l.forward_return_{horizon}d AS label_{horizon}d
        FROM feature_store f
        LEFT JOIN labels l ON f.ticker = l.ticker AND f.date = l.date
        {where} l.forward_return_{horizon}d IS NOT NULL
    """
    return query_feature_store_df(sql)
