# Benchmark Comparison Metrics Implementation Plan

## Overview

Add benchmark comparison metrics to answer the question: "Is this the market growing or is my strategy being smart?"

## Current Metrics (Already Implemented)

From `src/backtest/metrics.py`:
- Trade statistics (total, winning, losing, win rate)
- Profit/Loss metrics (total PnL, avg win/loss, largest win/loss)
- Risk metrics (profit factor, expectancy, risk/reward ratio)
- Risk-adjusted metrics (Sharpe ratio, Sortino ratio, Calmar ratio)
- Drawdown metrics (max drawdown, avg drawdown, duration)
- Streak metrics (max win/loss streaks)
- Holding period metrics

## New Metrics to Implement

### 1. Benchmark Comparison Metrics

```python
class BenchmarkMetrics:
    """Compare strategy performance against a benchmark (e.g., SPY buy-and-hold)."""
    
    # Core metrics
    - alpha: float          # Jensen's alpha - excess return vs benchmark
    - beta: float           # Market sensitivity (volatility relative to benchmark)
    - information_ratio: float  # Risk-adjusted excess return
    - tracking_error: float     # StdDev of excess returns
    - r_squared: float          # Correlation with benchmark squared
    
    # Comparison metrics
    - strategy_return: float    # Total strategy return
    - benchmark_return: float   # Total benchmark return
    - excess_return: float      # Strategy - Benchmark
    - relative_strength: float  # Strategy performance / Benchmark performance
```

### 2. Additional Risk Metrics

```python
class RiskMetrics:
    """Advanced risk measurement metrics."""
    
    - var_95: float           # Value at Risk (95% confidence)
    - var_99: float           # Value at Risk (99% confidence)
    - cvar_95: float          # Conditional VaR (Expected Shortfall)
    - ulcer_index: float      # Drawdown severity and duration
    - recovery_factor: float  # Net profit / Max drawdown
    - pain_index: float       # Average drawdown over time
    - kelly_fraction: float   # Optimal position size
```

### 3. Statistical Significance Tests

```python
class SignificanceTests:
    """Statistical tests for strategy performance."""
    
    - alpha_t_stat: float     # T-statistic for alpha
    - alpha_p_value: float    # P-value for alpha significance
    - sharpe_confidence: tuple # 95% confidence interval for Sharpe
    - bootstrap_metrics: dict  # Bootstrap confidence intervals
```

## Implementation Steps

### Step 1: Create Benchmark Comparison Module

File: `src/backtest/benchmark.py`

```python
"""
Benchmark Comparison Module

Provides metrics to compare strategy performance against a benchmark.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkComparison:
    """
    Calculate benchmark comparison metrics.
    
    Compares strategy returns against a benchmark (e.g., SPY buy-and-hold)
    to determine if the strategy adds alpha or simply captures market beta.
    """
    
    @staticmethod
    def calculate(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            strategy_returns: Daily returns of the strategy
            benchmark_returns: Daily returns of the benchmark
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Dictionary with benchmark comparison metrics
        """
        # Align returns
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 30:
            return {'error': 'Insufficient data for benchmark comparison'}
        
        strat = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]
        
        metrics = {}
        
        # Basic comparison
        metrics['strategy_total_return'] = (1 + strat).prod() - 1
        metrics['benchmark_total_return'] = (1 + bench).prod() - 1
        metrics['excess_return'] = metrics['strategy_total_return'] - metrics['benchmark_total_return']
        
        # Beta: Cov(Strat, Bench) / Var(Bench)
        covariance = np.cov(strat, bench)[0, 1]
        benchmark_variance = np.var(bench, ddof=1)
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha: Strat_Return - (RF + Beta * (Bench_Return - RF))
        daily_rf = risk_free_rate / 252
        annualized_strat = np.mean(strat) * 252
        annualized_bench = np.mean(bench) * 252
        metrics['alpha'] = annualized_strat - (risk_free_rate + metrics['beta'] * (annualized_bench - risk_free_rate))
        
        # Tracking Error
        excess_returns = strat - bench
        metrics['tracking_error'] = np.std(excess_returns, ddof=1) * np.sqrt(252)
        
        # Information Ratio
        if metrics['tracking_error'] > 0:
            metrics['information_ratio'] = (metrics['excess_return'] * 252) / metrics['tracking_error']
        else:
            metrics['information_ratio'] = 0
        
        # R-squared
        correlation = np.corrcoef(strat, bench)[0, 1]
        metrics['r_squared'] = correlation ** 2
        metrics['correlation'] = correlation
        
        # M2 Modigliani
        strat_sharpe = (np.mean(strat) - daily_rf) / np.std(strat, ddof=1)
        bench_std = np.std(bench, ddof=1)
        metrics['m2_measure'] = strat_sharpe * bench_std * np.sqrt(252) + risk_free_rate
        
        # Treynor Ratio
        if metrics['beta'] > 0:
            metrics['treynor_ratio'] = (annualized_strat - risk_free_rate) / metrics['beta']
        else:
            metrics['treynor_ratio'] = 0
            
        return metrics
```

### Step 2: Create Risk Metrics Module

File: `src/backtest/risk_metrics.py`

```python
"""
Advanced Risk Metrics Module

Provides VaR, CVaR, Ulcer Index, and other risk measurements.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Calculate advanced risk metrics."""
    
    @staticmethod
    def calculate(
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Dict[str, Any]],
        initial_equity: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Calculate advanced risk metrics.
        
        Args:
            returns: Daily returns series
            equity_curve: Equity values over time
            trades: List of trade dictionaries
            initial_equity: Starting equity
            
        Returns:
            Dictionary with risk metrics
        """
        metrics = {}
        
        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Ulcer Index
        peak = equity_curve.expanding().max()
        drawdown_pct = (equity_curve - peak) / peak * 100
        metrics['ulcer_index'] = np.sqrt(np.mean(drawdown_pct ** 2))
        
        # Pain Index
        metrics['pain_index'] = np.mean(np.abs(drawdown_pct))
        
        # Recovery Factor
        total_profit = equity_curve.iloc[-1] - initial_equity
        max_drawdown = (peak - equity_curve).max()
        metrics['recovery_factor'] = total_profit / max_drawdown if max_drawdown > 0 else 0
        
        # Kelly Criterion
        wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in trades if t.get('pnl', 0) < 0]
        if wins and losses:
            win_rate = len(wins) / len(trades)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                metrics['kelly_fraction'] = win_rate - (1 - win_rate) / (avg_win / avg_loss)
            else:
                metrics['kelly_fraction'] = 0
        else:
            metrics['kelly_fraction'] = 0
            
        return metrics
```

### Step 3: Update PerformanceMetrics

Modify `src/backtest/metrics.py` to integrate new metrics:

```python
# Add to calculate() method
from .benchmark import BenchmarkComparison
from .risk_metrics import RiskMetrics

# After existing calculations
if benchmark_returns is not None:
    metrics.update(BenchmarkComparison.calculate(
        strategy_returns=equity_curve['returns'],
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    ))

metrics.update(RiskMetrics.calculate(
    returns=returns,
    equity_curve=equity_curve['equity'],
    trades=trades,
    initial_equity=initial_equity
))
```

### Step 4: Update Notebook

Add benchmark comparison section to `notebooks/05_spy_longterm_backtest.ipynb`:

```python
# Add benchmark comparison cell
# Load SPY buy-and-hold returns
benchmark_returns = df['Close'].pct_change().dropna()

# Calculate benchmark metrics
from src.backtest.benchmark import BenchmarkComparison

benchmark_metrics = BenchmarkComparison.calculate(
    strategy_returns=strategy_daily_returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02
)

# Display results
print("=" * 60)
print("BENCHMARK COMPARISON (vs SPY Buy & Hold)")
print("=" * 60)
print(f"Strategy Total Return: {benchmark_metrics['strategy_total_return']*100:.2f}%")
print(f"Benchmark Total Return: {benchmark_metrics['benchmark_total_return']*100:.2f}%")
print(f"Excess Return (Alpha): {benchmark_metrics['alpha']*100:.2f}%")
print(f"Beta: {benchmark_metrics['beta']:.3f}")
print(f"Information Ratio: {benchmark_metrics['information_ratio']:.3f}")
print(f"Tracking Error: {benchmark_metrics['tracking_error']*100:.2f}%")
print(f"R-Squared: {benchmark_metrics['r_squared']:.3f}")
print(f"Treynor Ratio: {benchmark_metrics['treynor_ratio']:.3f}")
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/backtest/benchmark.py` | Create | Benchmark comparison metrics |
| `src/backtest/risk_metrics.py` | Create | Advanced risk metrics |
| `src/backtest/metrics.py` | Modify | Integrate new metrics |
| `notebooks/05_spy_longterm_backtest.ipynb` | Modify | Add benchmark comparison section |

## Testing

1. Unit tests for each metric calculation
2. Integration test with existing backtest results
3. Validate against known benchmarks (e.g., SPY historical data)
