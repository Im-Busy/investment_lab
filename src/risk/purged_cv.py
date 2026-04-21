from typing import Optional, Generator, Tuple
import numpy as np
import pandas as pd


class PurgedTimeSeriesCV:
    """Walk-forward cross-validation for multi-asset time series with purging.

    Generates train/test split tuples (train_idx, test_idx) that respect
    temporal ordering and purge overlapping observations between train and test
    periods to eliminate label leakage.

    Adapted from ML4T utils.py MultipleTimeSeriesCV.

    Assumes the input DataFrame/Series has a MultiIndex with levels
    ('date', 'ticker') or similar.

    Args:
        n_splits: Number of walk-forward splits.
        train_period_length: Number of days in each training period.
        test_period_length: Number of days in each test period.
        lookahead: Number of days to purge after training (gap between
            train end and test start). Set to test_period_length for
            full purging.
        date_idx: Name of the date level in the MultiIndex.
        shuffle: If True, randomly shuffle dates within each train period.
    """

    def __init__(
        self,
        n_splits: int = 3,
        train_period_length: int = 126,
        test_period_length: int = 21,
        lookahead: Optional[int] = None,
        date_idx: str = "date",
        shuffle: bool = False,
    ) -> None:
        self.n_splits = n_splits
        self.train_length = train_period_length
        self.test_length = test_period_length
        self.lookahead = lookahead or test_period_length
        self.date_idx = date_idx
        self.shuffle = shuffle

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test index pairs.

        Args:
            X: Feature DataFrame with MultiIndex (date, ticker).
            y: Target Series (ignored, for sklearn compatibility).
            groups: Optional group labels (ignored).

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        n_days = len(days)

        split_idxs = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead
            train_start_idx = train_end_idx + self.train_length

            if train_start_idx >= n_days or test_end_idx >= n_days:
                break

            split_idxs.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))

        dates = X.reset_index()[[self.date_idx]]

        for train_start, train_end, test_start, test_end in split_idxs:
            train_mask = (dates[self.date_idx] > days[train_start]) & (
                dates[self.date_idx] <= days[train_end]
            )
            test_mask = (dates[self.date_idx] > days[test_start]) & (
                dates[self.date_idx] <= days[test_end]
            )

            train_idx = dates[train_mask].index.to_numpy()
            test_idx = dates[test_mask].index.to_numpy()

            if self.shuffle:
                np.random.shuffle(train_idx)

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits
