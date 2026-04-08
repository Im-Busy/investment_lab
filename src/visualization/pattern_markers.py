"""
Pattern Markers for Chart Visualization

Converts pattern detection results to chart markers for visualization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import pandas as pd


class MarkerType(Enum):
    """Types of chart markers."""

    ENTRY = "entry"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    EXIT = "exit"


@dataclass
class PatternMarker:
    """
    Marker for a detected pattern on a chart.

    Attributes:
        timestamp: Bar timestamp
        price: Marker price level
        pattern_name: Name of the pattern
        direction: 'LONG' or 'SHORT'
        confidence: Signal confidence (0.0 to 1.0)
        marker_type: Type of marker (entry, stop, target)
        category: Pattern category (basic, harmonic, complex, classic, smc)
    """

    timestamp: pd.Timestamp
    price: float
    pattern_name: str
    direction: str
    confidence: float
    marker_type: MarkerType
    category: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert marker to dictionary."""
        return {
            "timestamp": str(self.timestamp),
            "price": self.price,
            "pattern_name": self.pattern_name,
            "direction": self.direction,
            "confidence": self.confidence,
            "marker_type": self.marker_type.value,
            "category": self.category,
        }


class PatternMarkerGenerator:
    """
    Generate markers for pattern visualization on charts.

    Converts trading signals and pattern detection results into
    chart markers that can be plotted with mplfinance or other
    charting libraries.
    """

    # Marker styles by direction
    DIRECTION_STYLES = {
        "LONG": {"marker": "^", "color": "green", "offset": -1},
        "SHORT": {"marker": "v", "color": "red", "offset": 1},
    }

    # Marker styles by type
    TYPE_STYLES = {
        MarkerType.ENTRY: {"size": 120, "alpha": 1.0},
        MarkerType.STOP_LOSS: {"size": 80, "alpha": 0.7},
        MarkerType.TAKE_PROFIT_1: {"size": 60, "alpha": 0.7},
        MarkerType.TAKE_PROFIT_2: {"size": 60, "alpha": 0.7},
        MarkerType.TAKE_PROFIT_3: {"size": 60, "alpha": 0.7},
        MarkerType.EXIT: {"size": 100, "alpha": 0.8},
    }

    # Pattern category colors
    CATEGORY_COLORS = {
        "basic": "#1f77b4",  # Blue
        "harmonic": "#9467bd",  # Purple
        "complex": "#ff7f0e",  # Orange
        "classic": "#17becf",  # Cyan
        "smc": "#2ca02c",  # Green
        "breakout": "#d62728",  # Red
    }

    # Pattern name to category mapping
    PATTERN_CATEGORIES = {
        # Basic patterns
        "MSLSupport": "basic",
        "MarketStructureLow": "basic",
        "MatchingLows": "basic",
        "NR7InsideDay": "basic",
        "NBarDecline": "basic",
        "FloorPivotBreakout": "basic",
        # Harmonic patterns
        "GartleyPattern": "harmonic",
        "ABCPattern": "harmonic",
        "SymmetricTriangle": "harmonic",
        "BollingerBands": "harmonic",
        # Breakout patterns
        "Donchian Channel Breakout": "breakout",
        "DonchianChannelBreakout": "breakout",
        # Complex patterns
        "CupHandle": "complex",
        "HeadShoulders": "complex",
        "SpikeLedge": "complex",
        "ThreeHills": "complex",
        "ParabolicArc": "complex",
        # Classic patterns
        "DoubleTop": "classic",
        "DoubleBottom": "classic",
        "TraderVic2B": "classic",
        "TripleTop": "classic",
        "DeadCatBounce": "classic",
        # SMC patterns
        "SMCReversal": "smc",
        "LiquiditySweep": "smc",
        "IFVG": "smc",
        "MSS": "smc",
    }

    def __init__(self):
        """Initialize the pattern marker generator."""
        self.markers: List[PatternMarker] = []

    def generate_markers(
        self,
        signals: List[Dict[str, Any]],
        include_stops: bool = True,
        include_targets: bool = True,
    ) -> List[PatternMarker]:
        """
        Convert trading signals to chart markers.

        Args:
            signals: List of signal dictionaries
            include_stops: Whether to include stop loss markers
            include_targets: Whether to include take profit markers

        Returns:
            List of PatternMarker objects
        """
        self.markers = []

        for signal in signals:
            # Get signal details
            ts_val = signal.get("timestamp")
            if ts_val is None:
                continue
            timestamp = pd.to_datetime(ts_val)
            entry_price = signal.get("entry_price")
            stop_loss = signal.get("stop_loss")
            direction = signal.get("direction", "LONG")
            pattern_name = signal.get("pattern_name", "Unknown")
            confidence = signal.get("confidence", 0.5)
            category = self._get_category(pattern_name)

            # Entry marker
            if entry_price is not None:
                self.markers.append(
                    PatternMarker(
                        timestamp=timestamp,
                        price=entry_price,
                        pattern_name=pattern_name,
                        direction=direction,
                        confidence=confidence,
                        marker_type=MarkerType.ENTRY,
                        category=category,
                    )
                )

            # Stop loss marker
            if include_stops and stop_loss is not None:
                self.markers.append(
                    PatternMarker(
                        timestamp=timestamp,
                        price=stop_loss,
                        pattern_name=pattern_name,
                        direction=direction,
                        confidence=confidence,
                        marker_type=MarkerType.STOP_LOSS,
                        category=category,
                    )
                )

            # Take profit markers
            if include_targets:
                for i, tp_key in enumerate(["take_profit_1", "take_profit_2", "take_profit_3"], 1):
                    tp_price = signal.get(tp_key)
                    if tp_price is not None:
                        marker_type = getattr(MarkerType, f"TAKE_PROFIT_{i}")
                        self.markers.append(
                            PatternMarker(
                                timestamp=timestamp,
                                price=tp_price,
                                pattern_name=pattern_name,
                                direction=direction,
                                confidence=confidence,
                                marker_type=marker_type,
                                category=category,
                            )
                        )

        return self.markers

    def generate_trade_markers(self, trades: List[Dict[str, Any]]) -> List[PatternMarker]:
        """
        Convert completed trades to chart markers.

        Args:
            trades: List of trade dictionaries with entry/exit info

        Returns:
            List of PatternMarker objects
        """
        self.markers = []

        for trade in trades:
            pattern_name = trade.get("pattern", "Unknown")
            direction = trade.get("direction", "LONG")
            category = self._get_category(pattern_name)

            # Entry marker
            entry_ts = trade.get("entry_time")
            entry_time = pd.to_datetime(entry_ts) if entry_ts is not None else None
            entry_price = trade.get("entry_price")

            if entry_time is not None and entry_price is not None:
                self.markers.append(
                    PatternMarker(
                        timestamp=entry_time,
                        price=entry_price,
                        pattern_name=pattern_name,
                        direction=direction,
                        confidence=1.0,  # Executed trade
                        marker_type=MarkerType.ENTRY,
                        category=category,
                    )
                )

            # Exit marker
            exit_ts = trade.get("exit_time")
            exit_time = pd.to_datetime(exit_ts) if exit_ts is not None else None
            exit_price = trade.get("exit_price")

            if exit_time is not None and exit_price is not None:
                self.markers.append(
                    PatternMarker(
                        timestamp=exit_time,
                        price=exit_price,
                        pattern_name=pattern_name,
                        direction=direction,
                        confidence=1.0,  # Executed trade
                        marker_type=MarkerType.EXIT,
                        category=category,
                    )
                )

        return self.markers

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert markers to DataFrame for easier manipulation.

        Returns:
            DataFrame with marker information
        """
        if not self.markers:
            return pd.DataFrame()

        return pd.DataFrame([m.to_dict() for m in self.markers])

    def to_mplfinance_format(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Convert markers to mplfinance addplot format.

        Args:
            df: OHLCV DataFrame with datetime index

        Returns:
            Dictionary of marker type -> price Series
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        result = {
            "entry_long": pd.Series(index=df.index, dtype=float),
            "entry_short": pd.Series(index=df.index, dtype=float),
            "stop_loss": pd.Series(index=df.index, dtype=float),
            "take_profit": pd.Series(index=df.index, dtype=float),
            "exit": pd.Series(index=df.index, dtype=float),
        }

        for marker in self.markers:
            if marker.timestamp in df.index:
                if marker.marker_type == MarkerType.ENTRY:
                    if marker.direction == "LONG":
                        result["entry_long"].loc[marker.timestamp] = marker.price  # type: ignore[index]
                    else:
                        result["entry_short"].loc[marker.timestamp] = marker.price  # type: ignore[index]
                elif marker.marker_type == MarkerType.STOP_LOSS:
                    result["stop_loss"].loc[marker.timestamp] = marker.price  # type: ignore[index]
                elif marker.marker_type in [
                    MarkerType.TAKE_PROFIT_1,
                    MarkerType.TAKE_PROFIT_2,
                    MarkerType.TAKE_PROFIT_3,
                ]:
                    result["take_profit"].loc[marker.timestamp] = marker.price  # type: ignore[index]
                elif marker.marker_type == MarkerType.EXIT:
                    result["exit"].loc[marker.timestamp] = marker.price  # type: ignore[index]

        return result

    def get_marker_style(self, marker: PatternMarker) -> Dict[str, Any]:
        """
        Get plotting style for a marker.

        Args:
            marker: PatternMarker object

        Returns:
            Dictionary with marker, color, size, alpha
        """
        direction_style = self.DIRECTION_STYLES.get(marker.direction, self.DIRECTION_STYLES["LONG"])
        type_style = self.TYPE_STYLES.get(marker.marker_type, self.TYPE_STYLES[MarkerType.ENTRY])
        _ = self.CATEGORY_COLORS.get(marker.category, "#1f77b4")

        # Override color for non-entry markers
        color: str
        if marker.marker_type == MarkerType.STOP_LOSS:
            color = "red"
        elif marker.marker_type in [
            MarkerType.TAKE_PROFIT_1,
            MarkerType.TAKE_PROFIT_2,
            MarkerType.TAKE_PROFIT_3,
        ]:
            color = "blue"
        elif marker.marker_type == MarkerType.EXIT:
            color = "purple"
        else:
            color = direction_style.get("color", "#1f77b4")  # type: ignore[assignment]

        return {
            "marker": direction_style["marker"],
            "color": color,
            "size": type_style["size"],
            "alpha": type_style["alpha"],
        }

    def _get_category(self, pattern_name: str) -> str:
        """Get pattern category from pattern name."""
        # Check direct mapping
        if pattern_name in self.PATTERN_CATEGORIES:
            return self.PATTERN_CATEGORIES[pattern_name]

        # Check partial match
        pattern_lower = pattern_name.lower()
        for key, category in self.PATTERN_CATEGORIES.items():
            if key.lower() in pattern_lower or pattern_lower in key.lower():
                return category

        return "unknown"

    def filter_by_category(self, category: str) -> List[PatternMarker]:
        """
        Filter markers by pattern category.

        Args:
            category: Pattern category to filter

        Returns:
            List of markers in the specified category
        """
        return [m for m in self.markers if m.category == category]

    def filter_by_direction(self, direction: str) -> List[PatternMarker]:
        """
        Filter markers by signal direction.

        Args:
            direction: 'LONG' or 'SHORT'

        Returns:
            List of markers with the specified direction
        """
        return [m for m in self.markers if m.direction == direction]

    def filter_by_type(self, marker_type: MarkerType) -> List[PatternMarker]:
        """
        Filter markers by marker type.

        Args:
            marker_type: Type of marker to filter

        Returns:
            List of markers of the specified type
        """
        return [m for m in self.markers if m.marker_type == marker_type]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of markers.

        Returns:
            Dictionary with marker counts and statistics
        """
        if not self.markers:
            return {"total": 0}

        df = self.to_dataframe()

        return {
            "total": len(self.markers),
            "by_category": df["category"].value_counts().to_dict(),
            "by_direction": df["direction"].value_counts().to_dict(),
            "by_type": df["marker_type"].value_counts().to_dict(),
            "patterns": df["pattern_name"].unique().tolist(),
        }
