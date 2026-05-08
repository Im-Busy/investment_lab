# Notebook Refactoring Plan

## Executive Summary

This plan outlines a comprehensive refactoring strategy for all 7 Jupyter notebooks in the project to maximize modularity, improve maintainability, and facilitate rapid parameter tuning. The refactoring will abstract hardcoded variables into centralized configuration sections and encapsulate repetitive logic into reusable functions.

## Current State Analysis

### Notebooks Overview

| Notebook | Purpose | Lines | Hardcoded Values | Repetitive Patterns |
|----------|---------|-------|------------------|---------------------|
| `01_multi_pattern_backtest.ipynb` | Multi-pattern strategy backtest | ~300 | Data paths, backtest params | Data loading, report generation |
| `02_smc_backtest.ipynb` | SMC/ICT strategy backtest | ~427 | Session times, strategy params | Indicator detection, metrics display |
| `03_strategy_comparison.ipynb` | Strategy comparison analysis | ~389 | Date ranges, capital | Metrics calculation, plotting |
| `04_pattern_visualization.ipynb` | Pattern visualization | ~439 | Pattern list, data subset | Pattern detection loop, plotting |
| `05_spy_longterm_backtest.ipynb` | Long-term SPY backtest | ~1498 | Data path, strategy params | Data summary, annual returns |
| `06_pattern_contribution.ipynb` | Pattern contribution analysis | ~916 | Backtest params, thresholds | Signal analysis, visualization |
| `07_pattern_selection_framework.ipynb` | Pattern selection framework | ~3361 | All thresholds, paths | Solo backtests, visualization |

### Identified Issues

#### 1. Hardcoded Variables (Current State)

**Data Configuration:**
- Data paths: `'data/raw/SPY_daily.csv'`, `'data/raw/SPY_historical.csv'`
- Date ranges: `'2015-01-02'`, `'2024-12-31'`, `'2020-01-01'`
- Column mappings scattered throughout

**Backtest Parameters:**
- `initial_equity=100000`, `cash=100000`
- `commission=0.001`, `commission_pct=0.001`
- `risk_per_trade=0.01`, `risk_per_trade=0.02`
- `slippage_pct=0.0005`
- `max_open_positions=3`, `max_open_positions=5`

**Strategy Parameters:**
- `min_confidence=0.55`, `min_confidence=0.60`
- `min_confluence_count=2`
- SMC session times: `"00:00"`, `"08:00"`

**Selection Thresholds:**
- `min_trades=30`, `min_sharpe=0.5`
- `correlation_threshold=0.8`
- `positive_contribution_threshold=0.0`

**Visualization Settings:**
- Figure sizes: `figsize=(14, 8)`, `figsize=(16, 12)`
- DPI values: `dpi=150`
- Window periods: `window=252`

#### 2. Repetitive Code Patterns

1. **Path Setup Pattern** (all notebooks):
```python
project_root = Path('..').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

2. **Data Loading Pattern** (6 notebooks):
```python
data_path = project_root / 'data' / 'raw' / 'SPY_daily.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
```

3. **Data Summary Pattern** (4 notebooks):
```python
print(f"Date Range: {df.index.min()} to {df.index.max()}")
print(f"Total Trading Days: {len(df)}")
```

4. **Metrics Display Pattern** (3 notebooks):
```python
print("="*60)
print("STRATEGY PERFORMANCE SUMMARY")
print("="*60)
# ... repetitive metric printing
```

5. **Backtest Runner Pattern** (4 notebooks):
```python
runner = BacktestPyRunner(data=df, cash=..., commission=...)
results = runner.run(strategy_class=..., ...)
```

---

## Proposed Architecture

### 1. Centralized Configuration Schema

Create a standardized configuration cell at the top of each notebook:

```python
# ============================================================
# CONFIGURATION SECTION
# ============================================================
# Modify these parameters to customize the analysis
# ============================================================

CONFIG = {
    # Data Configuration
    'data': {
        'file': 'SPY_daily.csv',           # Data file name
        'directory': 'data/raw',           # Data directory
        'start_date': '2015-01-01',        # Start date filter (None = all)
        'end_date': '2024-12-31',          # End date filter (None = all)
        'columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
    },

    # Backtest Configuration
    'backtest': {
        'initial_equity': 100000,          # Starting capital
        'commission': 0.001,               # Commission rate (0.1%)
        'slippage': 0.0005,                # Slippage rate
        'exclusive_orders': True,          # Close positions before opening new
    },

    # Strategy Configuration
    'strategy': {
        'min_confidence': 0.55,            # Minimum signal confidence
        'risk_per_trade': 0.02,            # Risk per trade (2%)
        'max_open_positions': 3,           # Maximum concurrent positions
        'min_confluence_count': 2,         # Minimum patterns for signal
    },

    # Output Configuration
    'output': {
        'directory': 'reports',            # Output directory
        'save_plots': True,                # Save plots to disk
        'show_plots': True,                # Display plots in notebook
        'dpi': 150,                        # Plot resolution
    },
}
```

### 2. Reusable Utility Functions

Create a new module `src/utils/notebook_helpers.py`:

```python
"""
Notebook Helper Utilities

Centralized functions for common notebook operations.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import sys

# ============================================================
# Path and Environment Setup
# ============================================================

def setup_project_root() -> Path:
    """Add project root to sys.path and return Path object."""
    project_root = Path('..').resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

# ============================================================
# Data Loading and Validation
# ============================================================

def load_price_data(
    file_name: str,
    directory: str = 'data/raw',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: List[str] = None
) -> pd.DataFrame:
    """Load and prepare OHLCV data with optional filtering."""
    ...

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column naming (Title case)."""
    ...

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Validate DataFrame has required OHLCV columns."""
    ...

# ============================================================
# Data Summary and Display
# ============================================================

def print_data_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """Print standardized data summary."""
    ...

def print_backtest_summary(stats: Dict[str, Any], title: str = "Backtest Results"):
    """Print formatted backtest statistics."""
    ...

def print_metrics_table(
    metrics: Dict[str, float],
    title: str = "Performance Metrics",
    compare_to: Optional[Dict[str, float]] = None
):
    """Print metrics in formatted table with optional comparison."""
    ...

# ============================================================
# Backtest Runner Helpers
# ============================================================

def create_runner_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract backtest runner configuration from CONFIG dict."""
    ...

def run_backtest_with_logging(
    runner_class,
    df: pd.DataFrame,
    strategy_class,
    config: Dict[str, Any],
    enable_signal_log: bool = False
) -> Dict[str, Any]:
    """Run backtest with standardized logging and error handling."""
    ...

# ============================================================
# Visualization Helpers
# ============================================================

def create_equity_plot(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: tuple = (14, 6)
):
    """Create standardized equity curve plot."""
    ...

def create_drawdown_plot(
    equity_curve: pd.Series,
    figsize: tuple = (14, 4)
):
    """Create drawdown visualization."""
    ...

def create_returns_distribution(
    returns: pd.Series,
    figsize: tuple = (10, 6)
):
    """Create returns distribution histogram."""
    ...

# ============================================================
# Metrics Calculation
# ============================================================

def calculate_performance_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    ...

def calculate_trade_statistics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate trade-level statistics."""
    ...
```

### 3. Notebook-Specific Configuration Classes

For notebooks with extensive configuration (like `07_pattern_selection_framework.ipynb`), create dedicated config classes:

```python
# src/analysis/pattern_selector_config.py (enhancement)

@dataclass
class NotebookPatternSelectionConfig:
    """Extended configuration for pattern selection notebook."""

    # Inherit from PatternSelectionConfig
    # Add notebook-specific display options

    # Display Configuration
    show_progress: bool = True
    max_patterns_display: int = 20
    chart_width: int = 1000
    chart_height: int = 800

    # Quick Test Mode
    quick_test: bool = False
    quick_test_patterns: int = 5
```

---

## Implementation Plan

### Phase 1: Create Infrastructure (Priority: High)

#### Task 1.1: Create Notebook Helper Module
- Create `src/utils/notebook_helpers.py`
- Implement core utility functions
- Add comprehensive docstrings
- Create unit tests

#### Task 1.2: Define Configuration Schema
- Create configuration templates for each notebook type
- Document all configurable parameters
- Create validation functions

### Phase 2: Refactor Notebooks (Priority: High)

#### Task 2.1: Refactor `01_multi_pattern_backtest.ipynb`

**Before:**
```python
# Scattered throughout notebook
data_path = project_root / 'data' / 'raw' / 'SPY_daily.csv'
runner = BacktestPyRunner(data=df, cash=100000, commission=0.001)
results = runner.run(strategy_class=MultiPatternStrategySimple, min_confidence=0.55, risk_per_trade=0.02, max_open_positions=3)
```

**After:**
```python
# ============================================================
# CONFIGURATION SECTION
# ============================================================
CONFIG = {
    'data': {'file': 'SPY_daily.csv'},
    'backtest': {'initial_equity': 100000, 'commission': 0.001},
    'strategy': {'min_confidence': 0.55, 'risk_per_trade': 0.02, 'max_open_positions': 3},
    'output': {'directory': 'reports', 'save_plots': True},
}

# ============================================================
# SETUP
# ============================================================
from src.utils.notebook_helpers import setup_project_root, load_price_data, print_data_summary
project_root = setup_project_root()

# ============================================================
# DATA LOADING
# ============================================================
df = load_price_data(**CONFIG['data'])
print_data_summary(df)
```

#### Task 2.2: Refactor `02_smc_backtest.ipynb`

**Key Changes:**
- Abstract SMC strategy parameters into CONFIG
- Create `SMC_CONFIG` section for session times and indicator parameters
- Use helper functions for indicator detection display

#### Task 2.3: Refactor `03_strategy_comparison.ipynb`

**Key Changes:**
- Consolidate comparison parameters
- Create reusable comparison visualization functions
- Abstract benchmark calculation

#### Task 2.4: Refactor `04_pattern_visualization.ipynb`

**Key Changes:**
- Create pattern list configuration
- Abstract visualization loop into function
- Standardize marker generation

#### Task 2.5: Refactor `05_spy_longterm_backtest.ipynb`

**Key Changes:**
- Consolidate all strategy parameters
- Create annual return analysis function
- Standardize tearsheet generation

#### Task 2.6: Refactor `06_pattern_contribution.ipynb`

**Key Changes:**
- Create analysis configuration section
- Abstract signal log analysis
- Create reusable contribution visualization

#### Task 2.7: Refactor `07_pattern_selection_framework.ipynb`

**Key Changes:**
- This notebook already has good structure with `PatternSelectionConfig`
- Enhance with additional display options
- Create quick-test mode configuration
- Add progress display configuration

### Phase 3: Create Notebook Configuration Module (Priority: Medium)

Create `src/utils/notebook_config.py`:

```python
"""
Notebook Configuration Module

Provides standardized configuration classes for all notebooks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class DataConfig:
    """Data loading configuration."""
    file: str = 'SPY_daily.csv'
    directory: str = 'data/raw'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    columns: List[str] = field(default_factory=lambda: ['Open', 'High', 'Low', 'Close', 'Volume'])

@dataclass
class BacktestConfig:
    """Backtest runner configuration."""
    initial_equity: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    exclusive_orders: bool = True

@dataclass
class StrategyConfig:
    """Strategy parameters configuration."""
    min_confidence: float = 0.55
    risk_per_trade: float = 0.02
    max_open_positions: int = 3
    min_confluence_count: int = 2

@dataclass
class OutputConfig:
    """Output and display configuration."""
    directory: str = 'reports'
    save_plots: bool = True
    show_plots: bool = True
    dpi: int = 150

@dataclass
class NotebookConfig:
    """Master configuration for notebooks."""
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        return {
            'data': self.data.__dict__,
            'backtest': self.backtest.__dict__,
            'strategy': self.strategy.__dict__,
            'output': self.output.__dict__,
        }
```

---

## Validation Checklist

After refactoring, each notebook should:

- [ ] Have a clear CONFIG section at the top
- [ ] Import helper functions from `src.utils.notebook_helpers`
- [ ] Use `setup_project_root()` for path setup
- [ ] Use `load_price_data()` for data loading
- [ ] Use `print_data_summary()` for data display
- [ ] Use `print_backtest_summary()` for results display
- [ ] Have no hardcoded paths or parameters in implementation cells
- [ ] Run successfully from top to bottom
- [ ] Produce identical results to pre-refactoring version

---

## Benefits

1. **Modularity**: Clear separation between configuration and implementation
2. **Maintainability**: Single source of truth for parameters
3. **Rapid Iteration**: Change parameters in one place
4. **Reusability**: Shared functions across notebooks
5. **Documentation**: Self-documenting configuration sections
6. **Testing**: Easier to test with different parameter sets
7. **Collaboration**: Clear structure for team members

---

## File Structure After Refactoring

```
src/
├── utils/
│   ├── __init__.py
│   ├── helpers.py              # Existing helpers
│   ├── validators.py          # Existing validators
│   ├── notebook_helpers.py     # NEW: Notebook utilities
│   └── notebook_config.py      # NEW: Configuration classes
├── config.py                   # Existing config (enhanced)

notebooks/
├── 01_multi_pattern_backtest.ipynb      # Refactored
├── 02_smc_backtest.ipynb                # Refactored
├── 03_strategy_comparison.ipynb          # Refactored
├── 04_pattern_visualization.ipynb       # Refactored
├── 05_spy_longterm_backtest.ipynb        # Refactored
├── 06_pattern_contribution.ipynb         # Refactored
└── 07_pattern_selection_framework.ipynb  # Refactored
```

---

## Timeline

| Phase | Tasks | Priority |
|-------|-------|----------|
| Phase 1 | Create helper module and config schema | High |
| Phase 2 | Refactor all 7 notebooks | High |
| Phase 3 | Create configuration module | Medium |
| Phase 4 | Validate and test | High |

---

## Questions for Clarification

Before implementation, please confirm:

1. **Configuration Format**: Should we use Python dictionaries (as shown) or YAML/JSON files for configuration? Python dicts are more flexible for notebooks.

2. **Backward Compatibility**: Should the helper functions maintain backward compatibility with existing code patterns?

3. **Additional Notebooks**: Are there any planned notebooks that should follow this pattern?

4. **Testing Requirements**: Should we create automated tests for the helper functions?

5. **Documentation**: Should we create a separate documentation file explaining the notebook patterns?
