"""P28-11: Faiss-based similar chart pattern search engine.

Given a query OHLCV window, finds the 10 most similar historical windows
and returns what happened next — a visual pattern matching tool for
trader intuition and pattern validation.

Source: awesome-ai-in-finance Chart Library pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SEARCH_DIMS = 40
_N_CANDIDATES = 50


@dataclass
class SimilarPattern:
    """A single similar historical pattern match."""

    start_idx: int
    end_idx: int
    similarity: float
    date_start: str
    date_end: str
    forward_return: float
    forward_max_dd: float
    forward_direction: str


@dataclass
class SearchResult:
    """Results of a similarity search."""

    query: np.ndarray
    matches: list[SimilarPattern]
    mean_forward_return: float
    median_forward_return: float
    positive_pct: float
    n_matches: int


def _ohlcv_to_feature_window(
    df: pd.DataFrame,
    window: int = 20,
) -> np.ndarray:
    """Convert OHLCV data to normalized feature windows for similarity search.

    Each window is normalized: (price - first_close) / first_close
    to make shape comparison scale-invariant.
    """
    closes = df["Close"].values.astype(np.float64)
    vols = df["Volume"].values.astype(np.float64)

    n_windows = max(0, len(closes) - window)
    if n_windows == 0:
        return np.empty((0, _SEARCH_DIMS))

    features = np.zeros((n_windows, _SEARCH_DIMS), dtype=np.float32)

    for i in range(n_windows):
        seg = closes[i : i + window]
        base = seg[0]
        if base > 0:
            norm = (seg - base) / base
        else:
            norm = np.zeros(window)

        vol_seg = vols[i : i + window]
        vol_base = vol_seg[0] if vol_seg[0] > 0 else 1.0
        vol_norm = (vol_seg - vol_base) / vol_base

        start, mid, end = 0, window // 2, window - 1
        features[i, :window] = norm
        features[i, window:] = vol_norm

    return features


def _build_index(features: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index over features."""
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(_SEARCH_DIMS)
    index.add(features.astype(np.float32))
    return index


def search_similar_patterns(
    df: pd.DataFrame,
    query_end_idx: int,
    window: int = 20,
    forward_bars: int = 10,
    top_k: int = 10,
) -> SearchResult:
    """Search for the top_k most similar historical chart patterns.

    Args:
        df: OHLCV DataFrame with DateTimeIndex and Close/Volume columns.
        query_end_idx: Index of the last bar in the query window.
        window: Length of the pattern window in bars.
        forward_bars: Bars to look forward for "what happened next" analysis.
        top_k: Number of similar patterns to return.

    Returns:
        SearchResult with matches and forward statistics.
    """
    features = _ohlcv_to_feature_window(df, window)
    if len(features) < 2:
        empty: list[SimilarPattern] = []
        return SearchResult(
            query=np.zeros(window),
            matches=empty,
            mean_forward_return=0.0,
            median_forward_return=0.0,
            positive_pct=0.0,
            n_matches=0,
        )

    query_start = max(0, query_end_idx - window + 1)
    query_idx = query_start

    query_vec = features[query_idx : query_idx + 1].copy()

    is_future = np.arange(len(features)) >= query_idx
    train_features = features[~is_future]
    train_indices = np.arange(len(features))[~is_future]

    if len(train_features) == 0:
        empty: list[SimilarPattern] = []
        return SearchResult(
            query=query_vec[0],
            matches=empty,
            mean_forward_return=0.0,
            median_forward_return=0.0,
            positive_pct=0.0,
            n_matches=0,
        )

    index = _build_index(train_features)
    faiss.normalize_L2(query_vec)

    distances, ann_indices = index.search(
        query_vec.astype(np.float32),
        min(top_k * 3, len(train_features)),
    )

    closes = df["Close"].values.astype(np.float64)
    dates = df.index

    matches: list[SimilarPattern] = []
    seen_starts: set[int] = set()

    for dist, ann_i in zip(distances[0], ann_indices[0]):
        if dist <= 0:
            continue
        original_idx = int(train_indices[ann_i])
        if original_idx in seen_starts:
            continue
        if original_idx >= len(closes) - forward_bars:
            continue

        seen_starts.add(original_idx)
        fwd_start = original_idx + window
        fwd_end = min(fwd_start + forward_bars, len(closes) - 1)
        if fwd_end <= fwd_start:
            continue

        entry_price = closes[fwd_start]
        exit_price = closes[fwd_end]
        if entry_price <= 0:
            continue

        fwd_return = (exit_price - entry_price) / entry_price
        fwd_segment = closes[fwd_start : fwd_end + 1]
        fwd_peak = np.max(fwd_segment)
        fwd_dd = min(0.0, (fwd_segment.min() - fwd_peak) / fwd_peak) if fwd_peak > 0 else 0.0

        matches.append(
            SimilarPattern(
                start_idx=original_idx,
                end_idx=original_idx + window,
                similarity=float(dist),
                date_start=str(dates[original_idx])[:10],
                date_end=str(dates[original_idx + min(window - 1, len(dates) - 1 - original_idx)])[
                    :10
                ],
                forward_return=fwd_return,
                forward_max_dd=fwd_dd,
                forward_direction="bullish" if fwd_return > 0 else "bearish",
            )
        )

        if len(matches) >= top_k:
            break

    returns = [m.forward_return for m in matches]
    positive = sum(1 for r in returns if r > 0)

    return SearchResult(
        query=query_vec[0],
        matches=sorted(matches, key=lambda m: m.similarity, reverse=True),
        mean_forward_return=float(np.mean(returns)) if returns else 0.0,
        median_forward_return=float(np.median(returns)) if returns else 0.0,
        positive_pct=positive / len(matches) * 100 if matches else 0.0,
        n_matches=len(matches),
    )


def search_rolling(
    df: pd.DataFrame,
    window: int = 20,
    forward_bars: int = 10,
    top_k: int = 10,
    step: int = 5,
    start_idx: int = 100,
) -> list[SearchResult]:
    """Run similarity search at regular intervals (rolling window scan).

    Useful for backtest-style "what would the pattern search have predicted
    at each step?"
    """
    results: list[SearchResult] = []
    for i in range(start_idx, len(df) - forward_bars, step):
        result = search_similar_patterns(
            df,
            query_end_idx=i,
            window=window,
            forward_bars=forward_bars,
            top_k=top_k,
        )
        results.append(result)
    return results


def format_result_table(result: SearchResult) -> str:
    """Format a search result as a readable table."""
    lines = [
        f"Pattern Search: {result.n_matches} matches found",
        f"Mean forward return: {result.mean_forward_return:.2%}",
        f"Median forward return: {result.median_forward_return:.2%}",
        f"Positive %: {result.positive_pct:.1f}%",
        "-" * 70,
        f"{'Similarity':>10} {'Date':>12} {'Fwd Return':>12} {'Fwd DD':>10} {'Dir':>8}",
        "-" * 70,
    ]
    for m in result.matches:
        lines.append(
            f"{m.similarity:10.4f} {m.date_start:>12} "
            f"{m.forward_return:12.2%} {m.forward_max_dd:10.2%} "
            f"{m.forward_direction:>8}"
        )
    return "\n".join(lines)
