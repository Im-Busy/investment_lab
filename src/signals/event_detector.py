"""Event-driven trading signal detectors from research papers.

Based on:
- Building a Calendar of Events Database by Analyzing Financial Spikes
- Event-Based Trading: Building Superior Trading Strategies
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinancialSpikeDetector:
    """Detect abnormal price/volume spikes and map to events.

    Based on: Building a Calendar of Events Database by Analyzing Financial Spikes.
    """

    def __init__(
        self,
        price_z_threshold: float = 2.5,
        volume_z_threshold: float = 3.0,
        lookback_days: int = 60,
    ) -> None:
        self.price_z_threshold = price_z_threshold
        self.volume_z_threshold = volume_z_threshold
        self.lookback_days = lookback_days

    def detect_spikes(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """Detect price and volume spike days.

        Returns:
            DataFrame with columns: date, price_spike, volume_spike, spike_score, return.
        """
        if len(close) < self.lookback_days + 1:
            logger.warning("Insufficient data for spike detection: %d bars", len(close))
            return pd.DataFrame(
                {
                    "date": close.index,
                    "price_spike": False,
                    "volume_spike": False,
                    "spike_score": 0.0,
                    "return": np.nan,
                }
            )

        returns = close.pct_change()
        vol_changes = volume.pct_change()
        vol_changes.replace([np.inf, -np.inf], np.nan, inplace=True)

        rolling_mean = returns.rolling(self.lookback_days, min_periods=30).mean()
        rolling_std = returns.rolling(self.lookback_days, min_periods=30).std()
        ret_z = (returns - rolling_mean) / rolling_std

        vol_mean = vol_changes.rolling(self.lookback_days, min_periods=30).mean()
        vol_std = vol_changes.rolling(self.lookback_days, min_periods=30).std()
        vol_z = (vol_changes - vol_mean) / vol_std

        price_spike = ret_z.abs() > self.price_z_threshold
        volume_spike = vol_z > self.volume_z_threshold

        spike_score = (ret_z.abs() / self.price_z_threshold).clip(0, 2) + (
            vol_z / self.volume_z_threshold
        ).clip(0, 2)

        return pd.DataFrame(
            {
                "date": close.index,
                "price_spike": price_spike.fillna(False),
                "volume_spike": volume_spike.fillna(False),
                "spike_score": spike_score.fillna(0.0).values,
                "return": returns.values,
            }
        )


class EventPatternDetector:
    """Detect event-driven trading patterns: pre-event compression + post-event drift.

    Based on: Event-Based Trading — Building Superior Trading Strategies.
    """

    def __init__(
        self,
        pre_event_compression_days: int = 5,
        post_event_drift_days: int = 3,
    ) -> None:
        self.pre_event_compression_days = pre_event_compression_days
        self.post_event_drift_days = post_event_drift_days

    def detect_pre_event_compression(
        self,
        close: pd.Series,
        event_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Detect volatility compression before known events.

        Returns:
            Series of compression scores (0=normal, 1=extreme compression).
        """
        returns = close.pct_change()
        rolling_vol = returns.rolling(self.pre_event_compression_days, min_periods=2).std()
        all_vol_mean = rolling_vol.mean()

        compression = pd.Series(0.0, index=close.index)
        if all_vol_mean is None or all_vol_mean == 0:
            return compression

        for event_date in event_dates:
            pre_days = pd.date_range(
                event_date - pd.Timedelta(days=self.pre_event_compression_days + 1),
                event_date - pd.Timedelta(days=1),
            )
            pre_days = pre_days[pre_days.isin(close.index)]
            if len(pre_days) < 2:
                continue
            pre_vol = rolling_vol.loc[pre_days].mean()
            if pd.notna(pre_vol) and all_vol_mean > 0:
                compression.loc[pre_days] = max(0.0, 1.0 - pre_vol / all_vol_mean)

        return compression

    def detect_post_event_drift(
        self,
        close: pd.Series,
        event_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Detect directional drift after events.

        Returns:
            Series of drift signals (+1=up drift, -1=down drift, 0=no drift).
        """
        returns = close.pct_change()
        drift_signal = pd.Series(0.0, index=close.index)

        for event_date in event_dates:
            post_window = pd.date_range(
                event_date,
                event_date + pd.Timedelta(days=self.post_event_drift_days),
            )
            post_window = post_window[post_window.isin(close.index)]
            if len(post_window) == 0:
                continue
            drift_ret = returns.loc[post_window].mean()
            if pd.notna(drift_ret):
                drift_signal.loc[post_window] = np.sign(drift_ret)

        return drift_signal


class EventSignalSuppressor:
    """Suppress ML entry signals on high-impact event days.

    Usage:
        suppressor = EventSignalSuppressor(event_calendar_db)
        if suppressor.is_event_day(date_val):
            # skip entry, but allow exits
    """

    def __init__(
        self,
        db_path: str = "data/event_calendar.duckdb",
        min_impact: str = "HIGH",
    ) -> None:
        self._db_path = db_path
        self._min_impact = min_impact
        from src.data_ingestion.event_calendar import EventCalendarDB

        self._db = EventCalendarDB(db_path)

    def is_event_day(
        self,
        check_date: str | date | pd.Timestamp,
        symbol: str | None = None,
    ) -> bool:
        """Check if a date is a high-impact event day."""
        if isinstance(check_date, pd.Timestamp):
            check_date = check_date.date()
        return self._db.is_event_day(check_date, symbol=symbol, min_impact=self._min_impact)

    def get_impact(
        self,
        check_date: str | date | pd.Timestamp,
        symbol: str | None = None,
    ) -> str | None:
        """Get the highest impact level for events on the given date."""
        if isinstance(check_date, pd.Timestamp):
            check_date = check_date.date()
        return self._db.get_event_impact(check_date, symbol=symbol)

    def preload_event_dates(
        self,
        date_index: pd.DatetimeIndex,
        symbol: str | None = None,
    ) -> pd.Series:
        """Precompute boolean Series of event days for a given date index.

        Returns:
            Boolean Series indexed by date_index, True for event days.
        """
        events = self._db.get_events_in_range(
            str(date_index.min().date()),
            str(date_index.max().date()),
            min_impact=self._min_impact,
        )
        if events.empty:
            return pd.Series(False, index=date_index)

        event_dates = pd.to_datetime(events["event_date"]).dt.date
        is_event = pd.Series(False, index=date_index)
        is_event[date_index.map(lambda d: d.date() in event_dates.values)] = True
        return is_event

    def close(self) -> None:
        self._db.close()
