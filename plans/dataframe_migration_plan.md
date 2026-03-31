# DataFrame Migration Plan: Trades Data Standardization

## Overview

This plan addresses the incomplete migration from `List[Dict[str, Any]]` to `pd.DataFrame` for trades data throughout the codebase. The current state has inconsistent handling that causes `ValueError: The truth value of a DataFrame is ambiguous` errors.

## Problem Statement

The codebase has two different patterns for handling trades data:

### Pattern 1 (Old - List Format)
- **Files:** `src/backtest/engine.py`, `src/backtest/metrics.py`
- **Format:** `List[Dict[str, Any]]`
- **Checks:** `if not trades:` (problematic for DataFrames)
- **Status:** Currently in use by the custom backtest engine

### Pattern 2 (New - DataFrame Format)
- **Files:** `src/strategies/backtest_py/runner.py`, `src/analysis/trade_attributor.py`
- **Format:** `pd.DataFrame`
- **Checks:** `if trades is not None and len(trades) > 0:` (correct approach)
- **Status:** Used by backtesting.py integration

### Inconsistency Issues
1. `src/backtest/metrics.py` line 40: `if not trades:` fails when trades is a DataFrame
2. `src/visualization/report.py` line 373: `if trades:` fails when trades is a DataFrame
3. `src/backtest/risk_metrics.py` line 174: Already fixed with proper check
4. Type hints are inconsistent across files

## Current State Analysis

### Files Using List Format (Need Migration)
| File | Function | Current Type | Status |
|------|----------|--------------|--------|
| `src/backtest/engine.py` | `_compile_trades()` | Returns `List[Dict]` | ❌ Needs update |
| `src/backtest/engine.py` | `BacktestResult.trades` | `List[Dict[str, Any]]` | ❌ Needs update |
| `src/backtest/metrics.py` | `calculate()` | Expects `List[Dict]` | ❌ Needs update |
| `src/backtest/metrics.py` | Line 40 check | `if not trades:` | ❌ Needs fix |

### Files Using DataFrame Format (Correct)
| File | Function | Current Type | Status |
|------|----------|--------------|--------|
| `src/strategies/backtest_py/runner.py` | `get_trades()` | Returns `pd.DataFrame` | ✅ Correct |
| `src/analysis/trade_attributor.py` | `__init__()` | Expects `pd.DataFrame` | ✅ Correct |
| `src/backtest/risk_metrics.py` | `calculate()` | Expects `Optional[List[Dict]]` | ⚠️ Mixed |

### Files With Problematic Checks
| File | Line | Current Check | Fix Required |
|------|------|---------------|--------------|
| `src/backtest/metrics.py` | 40 | `if not trades:` | ✅ Yes |
| `src/visualization/report.py` | 373 | `if trades:` | ✅ Yes |
| `src/backtest/risk_metrics.py` | 174 | `if trades is not None and len(trades) > 0:` | ✅ Already fixed |

## Migration Strategy

### Option A: Standardize on DataFrame (Recommended)
**Rationale:** DataFrames provide better performance, consistency with backtesting.py integration, and richer functionality for analysis.

**Pros:**
- Better performance for large datasets
- Consistent with modern pandas best practices
- Enables vectorized operations
- Better integration with visualization libraries
- Consistent with backtesting.py output

**Cons:**
- Requires updating multiple files
- Need to ensure backward compatibility during transition

### Option B: Standardize on List
**Rationale:** Maintain current custom engine format and update backtesting.py integration to convert.

**Pros:**
- Less changes required
- Simpler data structure

**Cons:**
- Performance overhead for large datasets
- Inconsistent with backtesting.py integration
- Limited analytical capabilities

## Implementation Plan

### Phase 1: Immediate Fixes (Critical)
Fix the ambiguous truth value errors without full migration.

#### Task 1.1: Fix `src/backtest/metrics.py`
**File:** `src/backtest/metrics.py`
**Line:** 40
**Change:**
```python
# Before
if not trades:

# After
if trades is None or len(trades) == 0:
```

#### Task 1.2: Fix `src/visualization/report.py`
**File:** `src/visualization/report.py`
**Line:** 373
**Change:**
```python
# Before
if trades:

# After
if trades is not None and len(trades) > 0:
```

#### Task 1.3: Update Type Hints in `src/backtest/metrics.py`
**File:** `src/backtest/metrics.py`
**Line:** 23
**Change:**
```python
# Before
def calculate(
    trades: List[Dict[str, Any]],
    ...

# After
def calculate(
    trades: Union[List[Dict[str, Any]], pd.DataFrame],
    ...
```

### Phase 2: Full DataFrame Migration (Recommended)

#### Task 2.1: Update `src/backtest/engine.py`

**Step 2.1.1:** Update `_compile_trades()` to return DataFrame
```python
def _compile_trades(self) -> pd.DataFrame:
    """Compile trades as DataFrame."""
    trades = []
    
    for position in self.position_manager.positions.values():
        if position.status == PositionStatus.CLOSED:
            trade = {
                'id': position.id,
                'pattern': position.pattern_name,
                'direction': position.direction.value,
                'entry_time': position.entry_time,
                'entry_price': position.entry_price,
                'exit_time': position.exit_time,
                'exit_price': position.exit_price,
                'size': position.size,
                'pnl': position.pnl,
                'pnl_pct': position.pnl_pct,
                'exit_reason': position.metadata.get('exit_reason', 'Unknown'),
                'stop_loss': position.stop_loss,
                'take_profit_1': position.take_profit_1,
                'take_profit_2': position.take_profit_2,
                'take_profit_3': position.take_profit_3
            }
            trades.append(trade)
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()
```

**Step 2.1.2:** Update `BacktestResult` dataclass
```python
@dataclass
class BacktestResult:
    """Result of a backtest run."""
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[BacktestConfig] = None
    signals: List[Dict[str, Any]] = field(default_factory=list)
```

#### Task 2.2: Update `src/backtest/metrics.py`

**Step 2.2.1:** Update function signature
```python
@staticmethod
def calculate(
    trades: pd.DataFrame,
    equity_curve: Optional[pd.DataFrame] = None,
    initial_equity: float = 100000.0,
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
```

**Step 2.2.2:** Update empty check
```python
if trades is None or len(trades) == 0:
    return {
        'total_trades': 0,
        'initial_equity': initial_equity,
        'final_equity': initial_equity,
        'total_return': 0.0,
        'message': 'No trades to analyze'
    }
```

**Step 2.2.3:** Update trade processing to use DataFrame operations
```python
# Before
wins = [t for t in trades if t.get('pnl', 0) > 0]
losses = [t for t in trades if t.get('pnl', 0) < 0]

# After
wins = trades[trades['pnl'] > 0]
losses = trades[trades['pnl'] < 0]
breakeven = trades[trades['pnl'] == 0]
```

#### Task 2.3: Update `src/backtest/risk_metrics.py`

**Step 2.3.1:** Update function signature
```python
@staticmethod
def calculate(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    initial_equity: float = 100000.0,
    confidence_levels: List[float] = [0.95, 0.99],
) -> Dict[str, Any]:
```

**Step 2.3.2:** Update Kelly Criterion calculation
```python
if trades is not None and len(trades) > 0:
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] < 0]
    
    if len(wins) > 0 and len(losses) > 0:
        win_rate = len(wins) / len(trades)
        avg_win = wins['pnl'].mean()
        avg_loss = abs(losses['pnl'].mean())
        # ... rest of calculation
```

#### Task 2.4: Update `src/visualization/report.py`

**Step 2.4.1:** Update function signature
```python
def generate_summary(
    self,
    trades: pd.DataFrame,
    equity_curve: Optional[pd.DataFrame] = None,
    initial_equity: float = 100000.0
) -> Dict[str, Any]:
```

**Step 2.4.2:** Update trade processing
```python
if trades is not None and len(trades) > 0:
    summary['trades']['total_trades'] = len(trades)
    
    if 'pnl' in trades.columns:
        winning = trades[trades['pnl'] > 0]
        losing = trades[trades['pnl'] < 0]
        
        summary['trades']['winning_trades'] = len(winning)
        summary['trades']['losing_trades'] = len(losing)
        summary['trades']['win_rate'] = len(winning) / len(trades) if len(trades) > 0 else 0
        summary['trades']['avg_win'] = float(winning['pnl'].mean()) if len(winning) > 0 else 0
        summary['trades']['avg_loss'] = float(losing['pnl'].mean()) if len(losing) > 0 else 0
        summary['trades']['total_pnl'] = float(trades['pnl'].sum())
```

### Phase 3: Validation and Testing

#### Task 3.1: Update Tests
- Update all test files that create mock trades data
- Ensure tests use DataFrame format
- Add tests for empty DataFrame handling

#### Task 3.2: Update Documentation
- Update docstrings to reflect DataFrame usage
- Update type hints throughout codebase
- Add examples of DataFrame trade format

#### Task 3.3: Integration Testing
- Test custom backtest engine with DataFrame output
- Test backtesting.py integration
- Test visualization with both engines
- Test trade attribution with DataFrame input

## Migration Checklist

### Phase 1: Immediate Fixes
- [ ] Fix `src/backtest/metrics.py` line 40
- [ ] Fix `src/visualization/report.py` line 373
- [ ] Update type hints in `src/backtest/metrics.py`
- [ ] Test fixes with existing code

### Phase 2: Full Migration
- [ ] Update `src/backtest/engine.py` `_compile_trades()`
- [ ] Update `src/backtest/engine.py` `BacktestResult` dataclass
- [ ] Update `src/backtest/metrics.py` function signature
- [ ] Update `src/backtest/metrics.py` trade processing logic
- [ ] Update `src/backtest/risk_metrics.py` function signature
- [ ] Update `src/backtest/risk_metrics.py` Kelly calculation
- [ ] Update `src/visualization/report.py` function signature
- [ ] Update `src/visualization/report.py` trade processing logic

### Phase 3: Validation
- [ ] Update unit tests
- [ ] Update integration tests
- [ ] Update documentation
- [ ] Performance testing
- [ ] End-to-end testing

## Trade Data Format

### Standard DataFrame Schema
```python
trade_schema = {
    'id': 'int64',              # Unique trade identifier
    'pattern': 'str',           # Pattern name that triggered trade
    'direction': 'str',         # 'long' or 'short'
    'entry_time': 'datetime64[ns]',  # Entry timestamp
    'entry_price': 'float64',   # Entry price
    'exit_time': 'datetime64[ns]',   # Exit timestamp
    'exit_price': 'float64',    # Exit price
    'size': 'float64',          # Position size
    'pnl': 'float64',           # Profit/Loss in currency
    'pnl_pct': 'float64',       # Profit/Loss as percentage
    'exit_reason': 'str',       # Reason for exit
    'stop_loss': 'float64',     # Stop loss price
    'take_profit_1': 'float64', # First take profit level
    'take_profit_2': 'float64', # Second take profit level
    'take_profit_3': 'float64', # Third take profit level
}
```

### Example Trade DataFrame
```python
import pandas as pd

trades = pd.DataFrame([
    {
        'id': 1,
        'pattern': 'MSL',
        'direction': 'long',
        'entry_time': pd.Timestamp('2024-01-01 10:00:00'),
        'entry_price': 100.0,
        'exit_time': pd.Timestamp('2024-01-01 11:00:00'),
        'exit_price': 105.0,
        'size': 100.0,
        'pnl': 500.0,
        'pnl_pct': 5.0,
        'exit_reason': 'take_profit',
        'stop_loss': 98.0,
        'take_profit_1': 105.0,
        'take_profit_2': None,
        'take_profit_3': None
    }
])
```

## Risk Assessment

### High Risk
- Breaking changes to `BacktestResult` dataclass
- Potential issues with existing notebooks/scripts
- Performance regression if not optimized

### Medium Risk
- Type hint inconsistencies during transition
- Test failures requiring updates
- Documentation gaps

### Low Risk
- Visualization compatibility
- Helper function updates

## Rollback Plan

If issues arise during migration:

1. **Phase 1 Rollback:** Revert specific line changes in metrics.py and report.py
2. **Phase 2 Rollback:** Keep List format in engine.py, add conversion layer
3. **Emergency:** Add compatibility wrapper functions

## Success Criteria

1. ✅ No `ValueError: The truth value of a DataFrame is ambiguous` errors
2. ✅ All trade data consistently represented as DataFrames
3. ✅ All tests passing
4. ✅ Performance maintained or improved
5. ✅ Both custom engine and backtesting.py produce compatible output
6. ✅ Visualization works with both engines

## Timeline

- **Phase 1:** 1-2 hours (immediate fixes)
- **Phase 2:** 4-6 hours (full migration)
- **Phase 3:** 2-3 hours (validation)

**Total Estimated Time:** 7-11 hours

## Dependencies

- pandas >= 2.0.0
- No breaking changes to external APIs
- All existing tests must pass after migration
