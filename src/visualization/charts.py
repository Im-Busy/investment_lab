"""
mplfinance Chart Integration

Generates candlestick charts with pattern markers and indicators.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False


@dataclass
class MarkerStyle:
    """Style configuration for chart markers."""
    marker: str
    color: str
    size: int
    offset: float  # Price offset for visibility


class ChartGenerator:
    """
    Generate candlestick charts with pattern overlays using mplfinance.
    """
    
    # Default marker styles for different signal types
    SIGNAL_STYLES = {
        'LONG': MarkerStyle(marker='^', color='green', size=100, offset=-0.002),
        'SHORT': MarkerStyle(marker='v', color='red', size=100, offset=0.002),
        'EXIT_LONG': MarkerStyle(marker='x', color='blue', size=80, offset=0.002),
        'EXIT_SHORT': MarkerStyle(marker='x', color='blue', size=80, offset=-0.002)
    }
    
    # Pattern category colors
    CATEGORY_COLORS = {
        'basic': '#1f77b4',      # Blue
        'harmonic': '#9467bd',    # Purple
        'complex': '#ff7f0e',     # Orange
        'classic': '#17becf',     # Cyan
        'smc': '#2ca02c'          # Green
    }
    
    def __init__(self, default_style: str = 'charles'):
        """
        Initialize chart generator.
        
        Args:
            default_style: Default mplfinance style ('charles', 'yahoo', 'default', etc.)
        """
        if not MPLFINANCE_AVAILABLE:
            raise ImportError(
                "mplfinance is not installed. "
                "Install with: pip install mplfinance"
            )
        
        self.default_style = default_style
    
    def plot_with_patterns(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        title: str = "Price Chart with Patterns",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (14, 8),
        volume: bool = True
    ) -> Optional[str]:
        """
        Plot candlestick chart with pattern markers.
        
        Args:
            df: OHLCV DataFrame with datetime index
            signals: List of signal dictionaries with 'timestamp', 'direction', 'entry_price'
            title: Chart title
            save_path: Path to save the chart (optional)
            show: Whether to display the chart
            figsize: Figure size (width, height)
            volume: Whether to show volume panel
            
        Returns:
            Path to saved file if save_path provided, None otherwise
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
        
        # Create signal markers
        addplots = []
        marker_series = self._create_signal_markers(df, signals)
        
        if not marker_series.empty:
            addplots.append(
                mpf.make_addplot(
                    marker_series,
                    type='scatter',
                    markersize=100,
                    marker='^',
                    color='green'
                )
            )
        
        # Create stop loss markers
        sl_series = self._create_stop_loss_markers(df, signals)
        if not sl_series.empty:
            addplots.append(
                mpf.make_addplot(
                    sl_series,
                    type='scatter',
                    markersize=50,
                    marker='x',
                    color='red'
                )
            )
        
        # Create take profit markers
        tp_series = self._create_take_profit_markers(df, signals)
        for name, series in tp_series.items():
            if not series.empty:
                addplots.append(
                    mpf.make_addplot(
                        series,
                        type='scatter',
                        markersize=50,
                        marker='o',
                        color='blue'
                    )
                )
        
        # Plot settings
        kwargs = {
            'type': 'candle',
            'style': self.default_style,
            'title': title,
            'figsize': figsize,
            'volume': volume,
            'returnfig': True
        }
        
        if addplots:
            kwargs['addplot'] = addplots
        
        if save_path:
            kwargs['savefig'] = save_path
        
        # Create the plot
        fig, axes = mpf.plot(df, **kwargs)
        
        # Add legend for patterns
        if signals:
            self._add_legend(fig, axes[0], signals)
        
        if show:
            plt.show()
        elif save_path:
            plt.close(fig)
        
        return save_path
    
    def plot_with_trades(
        self,
        df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        title: str = "Price Chart with Trades",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Optional[str]:
        """
        Plot candlestick chart with trade entry/exit markers.
        
        Args:
            df: OHLCV DataFrame with datetime index
            trades: List of trade dictionaries
            title: Chart title
            save_path: Path to save the chart (optional)
            show: Whether to display the chart
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved file if save_path provided, None otherwise
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
        
        # Create entry and exit markers
        entry_markers = pd.Series(index=df.index, dtype=float)
        exit_markers = pd.Series(index=df.index, dtype=float)
        
        for trade in trades:
            entry_ts = trade.get('entry_time')
            if entry_ts is None:
                continue
            entry_time = pd.to_datetime(entry_ts)
            exit_ts = trade.get('exit_time')
            exit_time = pd.to_datetime(exit_ts) if exit_ts is not None else None
            entry_price = trade.get('entry_price')
            exit_price = trade.get('exit_price')
            
            # Entry marker
            if entry_time in df.index:
                entry_markers[entry_time] = entry_price  # type: ignore[index]
            
            # Exit marker
            if exit_time is not None and exit_price and exit_time in df.index:
                exit_markers[exit_time] = exit_price  # type: ignore[index]
        
        # Create addplots
        addplots = []
        
        if not entry_markers.dropna().empty:
            addplots.append(
                mpf.make_addplot(
                    entry_markers,
                    type='scatter',
                    markersize=120,
                    marker='^',
                    color='lime'
                )
            )
        
        if not exit_markers.dropna().empty:
            addplots.append(
                mpf.make_addplot(
                    exit_markers,
                    type='scatter',
                    markersize=120,
                    marker='v',
                    color='red'
                )
            )
        
        # Plot settings
        kwargs = {
            'type': 'candle',
            'style': self.default_style,
            'title': title,
            'figsize': figsize,
            'volume': True,
            'returnfig': True
        }
        
        if addplots:
            kwargs['addplot'] = addplots
        
        if save_path:
            kwargs['savefig'] = save_path
        
        # Create the plot
        fig, axes = mpf.plot(df, **kwargs)
        
        if show:
            plt.show()
        elif save_path:
            plt.close(fig)
        
        return save_path
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (14, 6)
    ) -> Optional[str]:
        """
        Plot equity curve over time.
        
        Args:
            equity_curve: DataFrame with 'timestamp' and 'equity' columns
            benchmark: Optional benchmark equity series
            title: Chart title
            save_path: Path to save the chart (optional)
            show: Whether to display the chart
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved file if save_path provided, None otherwise
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        if isinstance(equity_curve, pd.DataFrame):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            equity = equity_curve['equity']
        else:
            equity = equity_curve
        
        ax.plot(equity.index, equity.values, label='Strategy', linewidth=2)  # type: ignore[arg-type]
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, label='Benchmark',  # type: ignore[arg-type]
                   linewidth=1.5, alpha=0.7, linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        elif save_path:
            plt.close(fig)
        
        return save_path
    
    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (14, 5)
    ) -> Optional[str]:
        """
        Plot drawdown over time.
        
        Args:
            equity_curve: DataFrame with 'timestamp' and 'equity' columns
            title: Chart title
            save_path: Path to save the chart (optional)
            show: Whether to display the chart
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved file if save_path provided, None otherwise
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate drawdown
        if isinstance(equity_curve, pd.DataFrame):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            equity = equity_curve['equity']
        else:
            equity = equity_curve
        
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0,  # type: ignore[arg-type]
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)  # type: ignore[arg-type]
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        elif save_path:
            plt.close(fig)
        
        return save_path
    
    def _create_signal_markers(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]]
    ) -> pd.Series:
        """Create marker series for signal entry points."""
        markers = pd.Series(index=df.index, dtype=float)
        
        for signal in signals:
            ts_val = signal.get('timestamp')
            if ts_val is None:
                continue
            timestamp = pd.to_datetime(ts_val)
            entry_price = signal.get('entry_price')
            
            if timestamp in df.index and entry_price is not None:
                markers[timestamp] = entry_price  # type: ignore[index]
        
        return markers
    
    def _create_stop_loss_markers(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]]
    ) -> pd.Series:
        """Create marker series for stop loss levels."""
        markers = pd.Series(index=df.index, dtype=float)
        
        for signal in signals:
            ts_val = signal.get('timestamp')
            if ts_val is None:
                continue
            timestamp = pd.to_datetime(ts_val)
            stop_loss = signal.get('stop_loss')
            
            if timestamp in df.index and stop_loss is not None:
                markers[timestamp] = stop_loss  # type: ignore[index]
        
        return markers
    
    def _create_take_profit_markers(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]]
    ) -> Dict[str, pd.Series]:
        """Create marker series for take profit levels."""
        tp1 = pd.Series(index=df.index, dtype=float)
        tp2 = pd.Series(index=df.index, dtype=float)
        tp3 = pd.Series(index=df.index, dtype=float)
        
        for signal in signals:
            ts_val = signal.get('timestamp')
            if ts_val is None:
                continue
            timestamp = pd.to_datetime(ts_val)
            
            if timestamp in df.index:
                if signal.get('take_profit_1'):
                    tp1[timestamp] = signal['take_profit_1']  # type: ignore[index]
                if signal.get('take_profit_2'):
                    tp2[timestamp] = signal['take_profit_2']  # type: ignore[index]
                if signal.get('take_profit_3'):
                    tp3[timestamp] = signal['take_profit_3']  # type: ignore[index]
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3
        }
    
    def _add_legend(
        self,
        fig: Any,
        ax: Any,
        signals: List[Dict[str, Any]]
    ) -> None:
        """Add legend with pattern counts."""
        # Count patterns by category
        pattern_counts: Dict[str, int] = {}  # noqa: F841
        for signal in signals:
            pattern = signal.get('pattern_name', 'Unknown')
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Create legend text
        legend_text = "Patterns: " + ", ".join(
            f"{name}({count})" for name, count in pattern_counts.items()
        )
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', bbox=props)
