"""
Backtest Adapter - Unified Interface for Backtesting Frameworks

Provides a common interface for both homemade BacktestEngine and backtesting.py.
Allows progressive migration from homemade engine to backtesting.py.

Usage:
    adapter = BacktestAdapter(engine="backtesting.py")
    result = adapter.run_backtest(
        data=df,
        patterns=patterns,
        config=config
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd

try:
    from backtesting import Backtest
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

from .engine import BacktestEngine, BacktestConfig as HomemadeConfig, BacktestResult as HomemadeResult
from ..strategies.backtest_py.runner import BacktestPyRunner
from ..strategies.backtest_py.multi_pattern_strategy_optimized import MultiPatternStrategyOptimized


class BacktestEngineType(Enum):
    """Available backtest engine types."""
    HOMEMADE = "homemade"
    BACKTESTING_PY = "backtesting.py"
    AUTO = "auto"  # Choose best available


@dataclass
class UnifiedBacktestConfig:
    """
    Unified configuration for all backtest engines.
    
    Maps configuration between homemade engine and backtesting.py.
    """
    # Core parameters
    initial_equity: float = 100000.0
    commission_per_trade: float = 0.0
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    position_sizing_method: str = "fixed_fractional"
    risk_per_trade: float = 0.02  # 2%
    max_open_positions: int = 5
    min_confidence: float = 0.5
    
    # Pattern selection
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    min_confluence_count: int = 2
    
    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Engine-specific overrides
    engine_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_homemade(self) -> HomemadeConfig:
        """Convert to homemade engine configuration."""
        return HomemadeConfig(
            initial_equity=self.initial_equity,
            commission_per_trade=self.commission_per_trade,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            position_sizing_method=self.position_sizing_method,
            risk_per_trade=self.risk_per_trade,
            max_open_positions=self.max_open_positions,
            min_confidence=self.min_confidence,
        )
    
    def to_backtesting_py_params(self) -> Dict[str, Any]:
        """Convert to backtesting.py strategy parameters."""
        params = {
            "initial_cash": self.initial_equity,
            "commission": self.commission_pct,
            "min_confidence": self.min_confidence,
            "min_confluence_count": self.min_confluence_count,
        }
        
        # Add pattern filters if specified
        if self.include_patterns:
            params["include_patterns_only"] = ",".join(self.include_patterns)
        if self.exclude_patterns:
            params["exclude_patterns"] = ",".join(self.exclude_patterns)
            
        # Merge engine-specific overrides
        params.update(self.engine_params)
        return params


@dataclass
class UnifiedBacktestResult:
    """
    Unified result format for all backtest engines.
    """
    # Core results
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, Any]
    
    # Additional data
    signals: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[UnifiedBacktestConfig] = None
    engine_type: BacktestEngineType = BacktestEngineType.AUTO
    
    # Raw engine results (for debugging)
    raw_results: Optional[Any] = None
    
    @classmethod
    def from_homemade(cls, result: HomemadeResult, config: UnifiedBacktestConfig) -> "UnifiedBacktestResult":
        """Create from homemade engine result."""
        return cls(
            equity_curve=result.equity_curve if result.equity_curve is not None else pd.DataFrame(),
            trades=result.trades if result.trades is not None else pd.DataFrame(),
            metrics=result.metrics if result.metrics is not None else {},
            signals=result.signals,
            config=config,
            engine_type=BacktestEngineType.HOMEMADE,
            raw_results=result,
        )
    
    @classmethod
    def from_backtesting_py(cls, results: Dict[str, Any], config: UnifiedBacktestConfig) -> "UnifiedBacktestResult":
        """Create from backtesting.py results."""
        # Extract equity curve
        equity_curve_raw = results.get("equity_curve")
        if equity_curve_raw is not None:
            if hasattr(equity_curve_raw, 'Equity'):
                equity_curve = pd.DataFrame({
                    'timestamp': equity_curve_raw.index,
                    'equity': equity_curve_raw['Equity'].values
                })
            else:
                equity_curve = pd.DataFrame(equity_curve_raw)
        else:
            equity_curve = pd.DataFrame()
        
        # Extract trades
        trades_raw = results.get("trades")
        if trades_raw is not None:
            trades = trades_raw if isinstance(trades_raw, pd.DataFrame) else pd.DataFrame(trades_raw)
        else:
            trades = pd.DataFrame()
        
        # Extract metrics
        stats = results.get("stats", {})
        metrics = {
            "total_return_pct": stats.get("Return [%]", 0),
            "sharpe_ratio": stats.get("Sharpe Ratio", 0),
            "max_drawdown_pct": stats.get("Max. Drawdown [%]", 0),
            "win_rate_pct": stats.get("Win Rate [%]", 0),
            "total_trades": stats.get("# Trades", 0),
            "profit_factor": stats.get("Profit Factor", 0),
        }
        
        return cls(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            config=config,
            engine_type=BacktestEngineType.BACKTESTING_PY,
            raw_results=results,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "equity_curve": self.equity_curve.to_dict("records") if len(self.equity_curve) > 0 else [],
            "trades": self.trades.to_dict("records") if len(self.trades) > 0 else [],
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else None,
            "engine_type": self.engine_type.value,
            "total_signals": len(self.signals),
        }


class BacktestAdapter:
    """
    Unified adapter for running backtests with any engine.
    
    Provides a consistent interface regardless of underlying engine.
    """
    
    def __init__(
        self,
        engine: Union[BacktestEngineType, str] = BacktestEngineType.AUTO,
        default_config: Optional[UnifiedBacktestConfig] = None,
    ):
        """
        Initialize backtest adapter.
        
        Args:
            engine: Engine type to use ('homemade', 'backtesting.py', or 'auto')
            default_config: Default configuration for all backtests
        """
        if isinstance(engine, str):
            engine = BacktestEngineType(engine.lower())
            
        self.engine_type = engine
        self.default_config = default_config or UnifiedBacktestConfig()
        
        # Determine actual engine if AUTO
        if self.engine_type == BacktestEngineType.AUTO:
            self.engine_type = self._select_best_engine()
            
        # Validate engine availability
        if self.engine_type == BacktestEngineType.BACKTESTING_PY and not BACKTESTING_AVAILABLE:
            raise ImportError(
                "backtesting.py is not installed. Install with: pip install backtesting"
            )
    
    def _select_best_engine(self) -> BacktestEngineType:
        """Select the best available engine."""
        if BACKTESTING_AVAILABLE:
            return BacktestEngineType.BACKTESTING_PY
        return BacktestEngineType.HOMEMADE
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        patterns: Optional[List[Any]] = None,
        config: Optional[UnifiedBacktestConfig] = None,
        **kwargs,
    ) -> UnifiedBacktestResult:
        """
        Run backtest with configured engine.
        
        Args:
            data: OHLCV DataFrame with datetime index
            patterns: List of pattern detectors (required for homemade engine)
            config: Backtest configuration
            **kwargs: Additional parameters passed to engine
            
        Returns:
            UnifiedBacktestResult with standardized format
        """
        config = config or self.default_config
        
        if self.engine_type == BacktestEngineType.HOMEMADE:
            return self._run_homemade(data, patterns, config, **kwargs)
        else:
            return self._run_backtesting_py(data, config, **kwargs)
    
    def _run_homemade(
        self,
        data: pd.DataFrame,
        patterns: Optional[List[Any]],
        config: UnifiedBacktestConfig,
        **kwargs,
    ) -> UnifiedBacktestResult:
        """Run backtest using homemade engine."""
        if patterns is None:
            raise ValueError("Homemade engine requires patterns list")
        
        # Create homemade engine
        homemade_config = config.to_homemade()
        engine = BacktestEngine(patterns=patterns, config=homemade_config)
        
        # Run backtest
        result = engine.run(
            df=data,
            start_date=config.start_date,
            end_date=config.end_date,
            **kwargs,
        )
        
        return UnifiedBacktestResult.from_homemade(result, config)
    
    def _run_backtesting_py(
        self,
        data: pd.DataFrame,
        config: UnifiedBacktestConfig,
        **kwargs,
    ) -> UnifiedBacktestResult:
        """Run backtest using backtesting.py."""
        # Create runner
        runner = BacktestPyRunner(
            data=data,
            cash=config.initial_equity,
            commission=config.commission_pct,
            output_dir="reports",
        )
        
        # Get strategy parameters
        strategy_params = config.to_backtesting_py_params()
        strategy_params.update(kwargs)
        
        # Run backtest
        results = runner.run(
            strategy_class=MultiPatternStrategyOptimized,
            start_date=config.start_date,
            end_date=config.end_date,
            **strategy_params,
        )
        
        return UnifiedBacktestResult.from_backtesting_py(results, config)
    
    def compare_engines(
        self,
        data: pd.DataFrame,
        patterns: Optional[List[Any]] = None,
        config: Optional[UnifiedBacktestConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest with both engines and compare results.
        
        Useful for validation during migration.
        """
        config = config or self.default_config
        results = {}
        
        # Try homemade engine if patterns provided
        if patterns is not None:
            try:
                homemade_result = self._run_homemade(data, patterns, config)
                results["homemade"] = homemade_result
            except Exception as e:
                results["homemade_error"] = str(e)
        
        # Try backtesting.py if available
        if BACKTESTING_AVAILABLE:
            try:
                backtesting_result = self._run_backtesting_py(data, config)
                results["backtesting.py"] = backtesting_result
            except Exception as e:
                results["backtesting.py_error"] = str(e)
        
        # Compare metrics if both succeeded
        if "homemade" in results and "backtesting.py" in results:
            comparison = self._compare_results(
                results["homemade"], 
                results["backtesting.py"]
            )
            results["comparison"] = comparison
            
        return results
    
    def _compare_results(
        self,
        result_a: UnifiedBacktestResult,
        result_b: UnifiedBacktestResult,
    ) -> Dict[str, Any]:
        """Compare two backtest results."""
        comparison = {}
        
        # Compare key metrics
        metrics_a = result_a.metrics
        metrics_b = result_b.metrics
        
        for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]:
            val_a = metrics_a.get(key, 0)
            val_b = metrics_b.get(key, 0)
            
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = abs(val_a - val_b)
                diff_pct = diff / max(abs(val_a), 1e-10) * 100
                
                comparison[key] = {
                    "engine_a": val_a,
                    "engine_b": val_b,
                    "absolute_diff": diff,
                    "percent_diff": diff_pct,
                    "within_tolerance": diff_pct < 5.0,  # 5% tolerance
                }
        
        # Compare trade counts
        trades_a = len(result_a.trades)
        trades_b = len(result_b.trades)
        comparison["trade_count"] = {
            "engine_a": trades_a,
            "engine_b": trades_b,
            "diff": abs(trades_a - trades_b),
        }
        
        return comparison


# Factory function for easy creation
def create_backtest_adapter(
    engine: Union[str, BacktestEngineType] = "auto",
    **config_kwargs,
) -> BacktestAdapter:
    """
    Create a BacktestAdapter with specified configuration.
    
    Args:
        engine: Engine type ('homemade', 'backtesting.py', or 'auto')
        **config_kwargs: Configuration parameters for UnifiedBacktestConfig
        
    Returns:
        BacktestAdapter instance
    """
    config = UnifiedBacktestConfig(**config_kwargs)
    return BacktestAdapter(engine=engine, default_config=config)