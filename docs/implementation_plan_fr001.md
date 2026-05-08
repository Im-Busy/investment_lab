# Implementation Plan: FR-001 - London Breakout Strategy

**Date:** 2026-04-20
**Feature Request:** FR-001
**Priority:** HIGH VALUE (15-25% Sharpe improvement on FX pairs)
**Estimated Time:** 2-3 hours
**Status:** Planning Phase

---

## 1. Strategy Overview

### Core Concept
Use Tokyo trading hour (2:00-3:00 EST) price action to predict London open (3:00 EST) breakout. Trade the first 30 minutes of London session with risk-managed entries.

### Trading Logic Flow
1. **Tokyo Hour Collection (2:00-3:00 EST)**: Collect high/low prices
2. **London Open (3:00 EST)**: Set upper/lower thresholds based on Tokyo range
3. **London Trading Window (3:00-3:30 EST)**: Monitor for breakout signals
4. **Entry Execution**: Buy on upper breakout, Sell on lower breakout
5. **Risk Management**: Stop loss filtering, reverse position detection
6. **London Close (12:00 EST)**: Clear all intraday positions

### Key Parameters
- **Tokyo Hour Window**: 2:00-3:00 EST (1 hour of price data)
- **London Trading Window**: 3:00-3:30 EST (30 minute execution window)
- **London Session End**: 12:00 EST (position clear)
- **Stop Loss Threshold**: 1% of price (`risky_stop=0.01`)
- **Time Zone Handling**: EST, GMT, Tokyo time conversions

---

## 2. Technical Specifications

### 2.1 Data Requirements
```
Format: Minute-frequency OHLCV data
Ticker: FX pairs (GBP/USD, EUR/USD, USD/JPY recommended)
Columns:
  - datetime: UTC timestamp
  - Open, High, Low, Close: Price levels
  - Volume: Trading volume (optional)
```

### 2.2 Time Zone Mapping
```
Tokyo (JST): 2:00-3:00 EST = 15:00-16:00 JST
London (GMT): 3:00-12:00 EST = 8:00-17:00 GMT
New York (EST): 2:00-12:00 EST
```

### 2.3 Signal Generation Logic
```python
# Tokyo hour collection (2:00-3:00 EST)
tokyo_high = max(prices during 2:00-3:00 EST)
tokyo_low = min(prices during 2:00-3:00 EST)
tokyo_range = tokyo_high - tokyo_low

# London open thresholds (3:00 EST)
london_open = price at 3:00 EST
upper_threshold = london_open + tokyo_range
lower_threshold = london_open - tokyo_range

# Breakout signals (3:00-3:30 EST)
if current_price > upper_threshold:
    signal = BUY if not stop_loss_triggered
elif current_price < lower_threshold:
    signal = SELL if not stop_loss_triggered

# Stop loss filter
if abs(price - entry_price) / entry_price > risky_stop:
    reject_signal = True
```

### 2.4 Position Management Rules
```python
# Single position per London session
cumulative_signals = sum(all signals this session)
if cumulative_signals >= 1:
    no_new_entries = True

# Reverse position detection
if existing_position == SHORT and current_price > upper_threshold:
    close_short() and open_long()

# Intraday clearing
if current_time >= 12:00 EST:
    close_all_positions()
```

---

## 3. File Structure

### New Files to Create
```
src/strategies/
└── london_breakout.py          # Main strategy implementation

tests/strategies/
└── test_london_breakout.py     # Comprehensive test suite

docs/
└── london_breakout.md          # Strategy documentation (optional)
```

### Existing Files to Reference
```
src/backtest/engine.py          # Custom backtest engine integration
src/indicators/technical.py    # Existing indicator utilities
src/data_ingestion/data_utils.py # Data loading helpers
```

---

## 4. Implementation Steps

### Phase 1: Core Strategy Skeleton (30 minutes)

#### Step 1.1: Create Strategy Class
**File:** `src/strategies/london_breakout.py`

```python
"""
London Breakout Strategy

Uses Tokyo hour (2:00-3:00 EST) to predict London open breakout.
Trades first 30 minutes of London session with risk management.
"""

from backtesting import Strategy
import pandas as pd
from typing import Optional
from dataclasses import dataclass

@dataclass
class LondonBreakoutParams:
    """Strategy parameters."""
    tokyo_start_hour: int = 2      # 2:00 EST
    tokyo_end_hour: int = 3        # 3:00 EST
    london_trading_minutes: int = 30 # 3:00-3:30 EST
    london_close_hour: int = 12     # 12:00 EST
    risky_stop: float = 0.01        # 1% stop loss
    param: float = 0.5              # Range multiplier (from Dual Thrust)

class LondonBreakoutStrategy(Strategy):
    """London Breakout Strategy for backtesting.py."""

    params = LondonBreakoutParams()

    def init(self) -> None:
        """Initialize strategy state."""
        self.tokyo_highs: list[float] = []
        self.tokyo_lows: list[float] = []
        self.tokyo_range: float = 0.0
        self.upper_threshold: float = 0.0
        self.lower_threshold: float = 0.0
        self.london_open_price: float = 0.0
        self.entry_price: float = 0.0
        self.signals_this_session: int = 0
        self.in_london_session: bool = False

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        current_time = self.data.index[-1]
        current_hour = current_time.hour

        # Implementation continues...
```

#### Step 1.2: Add Time Zone Handling
```python
    def _is_tokyo_hour(self, dt: pd.Timestamp) -> bool:
        """Check if time is within Tokyo hour window."""
        return dt.hour == self.params.tokyo_start_hour

    def _is_london_session_start(self, dt: pd.Timestamp) -> bool:
        """Check if time is London open."""
        return dt.hour == self.params.london_trading_start and dt.minute == 0

    def _is_london_trading_window(self, dt: pd.Timestamp) -> bool:
        """Check if within London trading window."""
        return (self.params.tokyo_end_hour <= dt.hour <
                self.params.tokyo_end_hour and
                dt.minute < self.params.london_trading_minutes)

    def _is_london_close(self, dt: pd.Timestamp) -> bool:
        """Check if London session ends."""
        return dt.hour >= self.params.london_close_hour
```

#### Step 1.3: Implement Tokyo Hour Collection
```python
    def _collect_tokyo_data(self) -> None:
        """Collect high/low during Tokyo hour."""
        self.tokyo_highs.append(self.data.High[-1])
        self.tokyo_lows.append(self.data.Low[-1])

    def _calculate_tokyo_range(self) -> float:
        """Calculate Tokyo hour price range."""
        if not self.tokyo_highs:
            return 0.0

        tokyo_high = max(self.tokyo_highs)
        tokyo_low = min(self.tokyo_lows)
        return tokyo_high - tokyo_low
```

#### Step 1.4: Set London Open Thresholds
```python
    def _set_london_thresholds(self) -> None:
        """Set upper/lower thresholds at London open."""
        self.tokyo_range = self._calculate_tokyo_range()
        self.london_open_price = self.data.Close[-1]

        self.upper_threshold = (
            self.london_open_price +
            self.params.param * self.tokyo_range
        )
        self.lower_threshold = (
            self.london_open_price -
            self.params.param * self.tokyo_range
        )
```

#### Step 1.5: Generate Breakout Signals
```python
    def _generate_signals(self) -> Optional[int]:
        """Generate buy/sell signals based on breakout."""
        current_price = self.data.Close[-1]

        # Upper breakout
        if current_price > self.upper_threshold:
            if self._check_stop_loss(current_price):
                return None
            return 1  # BUY

        # Lower breakout
        if current_price < self.lower_threshold:
            if self._check_stop_loss(current_price):
                return None
            return -1  # SELL

        return None

    def _check_stop_loss(self, price: float) -> bool:
        """Check if signal should be rejected by stop loss."""
        if self.entry_price == 0:
            return False
        return abs(price - self.entry_price) / self.entry_price > self.params.risky_stop
```

#### Step 1.6: Position Management
```python
    def _manage_positions(self, signal: int) -> None:
        """Manage positions based on signals."""
        current_price = self.data.Close[-1]

        # Single position per session
        if self.signals_this_session >= 1 and signal != 0:
            return

        # Reverse position detection
        if self.position and self.position.is_short and signal == 1:
            self.position.close()
            self.buy()
            self.entry_price = current_price
            self.signals_this_session += 1
            return

        if self.position and self.position.is_long and signal == -1:
            self.position.close()
            self.sell()
            self.entry_price = current_price
            self.signals_this_session += 1
            return

        # Regular entry
        if signal == 1 and not self.position:
            self.buy()
            self.entry_price = current_price
            self.signals_this_session += 1

        if signal == -1 and not self.position:
            self.sell()
            self.entry_price = current_price
            self. signals_this_session += 1

    def _clear_positions_at_close(self) -> None:
        """Clear all positions at London close."""
        if self.position:
            self.position.close()
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        """Reset session state."""
        self.tokyo_highs.clear()
        self.tokyo_lows.clear()
        self.tokyo_range = 0.0
        self.upper_threshold = 0.0
        self.lower_threshold = 0.0
        self.london_open_price = 0.0
        self.entry_price = 0.0
        self.signals_this_session = 0
        self.in_london_session = False
```

### Phase 2: Main Strategy Loop (30 minutes)

#### Step 2.1: Complete next() Method
```python
    def next(self) -> None:
        """Execute strategy logic on each bar."""
        current_time = self.data.index[-1]
        current_price = self.data.Close[-1]

        # Tokyo hour: Collect price data
        if self._is_tokyo_hour(current_time):
            self._collect_tokyo_data()
            return

        # London open: Set thresholds
        if self._is_london_session_start(current_time):
            self._set_london_thresholds()
            self.in_london_session = True
            return

        # London trading window: Generate signals
        if self._is_london_trading_window(current_time) and self.in_london_session:
            signal = self._generate_signals()
            if signal:
                self._manage_positions(signal)
            return

        # London close: Clear positions
        if self._is_london_close(current_time):
            self._clear_positions_at_close()
```

### Phase 3: Backtesting Integration (20 minutes)

#### Step 3.1: Create Backtest Runner
**File:** `tests/strategies/test_london_breakout.py`

```python
"""
Tests for London Breakout Strategy.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.strategies.london_breakout import (
    LondonBreakoutStrategy,
    LondonBreakoutParams
)
from src.backtest.engine import BacktestEngine


@pytest.fixture
def sample_fx_data():
    """Generate sample FX minute data for testing."""
    np.random.seed(42)
    n_points = 10000

    dates = pd.date_range(
        start='2023-01-01 00:00:00',
        periods=n_points,
        freq='1min'
    )

    base_price = 1.2500  # GBP/USD
    volatility = 0.0002

    prices = np.cumsum(
        np.random.randn(n_points) * volatility
    ) + base_price

    data = pd.DataFrame({
        'Open': prices + np.random.randn(n_points) * 0.0001,
        'High': prices + np.abs(np.random.randn(n_points)) * 0.0003,
        'Low': prices - np.abs(np.random.randn(n_points)) * 0.0003,
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n_points)
    }, index=dates)

    return data


@pytest.fixture
def strategy_params():
    """Default strategy parameters."""
    return LondonBreakoutParams(
        tokyo_start_hour=2,
        tokyo_end_hour=3,
        london_trading_minutes=30,
        london_close_hour=12,
        risky_stop=0.01,
        param=0.5
    )
```

#### Step 3.2: Unit Tests
```python
class TestLondonBreakoutStrategy:
    """Test suite for London Breakout Strategy."""

    def test_time_zone_detection(self, sample_fx_data, strategy_params):
        """Test Tokyo hour and London session detection."""
        strategy = LondonBreakoutStrategy()
        strategy.params = strategy_params

        # Test Tokyo hour
        tokyo_time = pd.Timestamp('2023-01-01 02:30:00')
        assert strategy._is_tokyo_hour(tokyo_time) == True

        # Test non-Tokyo hour
        non_tokyo_time = pd.Timestamp('2023-01-01 05:30:00')
        assert strategy._is_tokyo_hour(non_tokyo_time) == False

        # Test London open
        london_open = pd.Timestamp('2023-01-01 03:00:00')
        assert strategy._is_london_session_start(london_open) == True

    def test_tokyo_range_calculation(self, strategy_params):
        """Test Tokyo hour price range calculation."""
        strategy = LondonBreakoutStrategy()
        strategy.params = strategy_params

        # Simulate Tokyo hour data
        strategy.tokyo_highs = [1.2510, 1.2515, 1.2520]
        strategy.tokyo_lows = [1.2490, 1.2485, 1.2480]

        tokyo_range = strategy._calculate_tokyo_range()
        assert tokyo_range == pytest.approx(0.0040, rel=1e-4)

    def test_threshold_setting(self, strategy_params):
        """Test London open threshold calculation."""
        strategy = LondonBreakoutStrategy()
        strategy.params = strategy_params

        # Set Tokyo range
        strategy.tokyo_range = 0.0040
        strategy.london_open_price = 1.2500

        strategy = self._set_london_thresholds()

        expected_upper = 1.2500 + 0.5 * 0.0040
        expected_lower = 1.2500 - 0.5 * 0.0040

        assert strategy.upper_threshold == pytest.approx(expected_upper, rel=1e-4)
        assert strategy.lower_threshold == pytest.approx(expected_lower, rel=1e-4)

    def test_stop_loss_filtering(self, strategy_params):
        """Test stop loss signal rejection."""
        strategy = LondonBreakoutStrategy()
        strategy.params = strategy_params

        strategy.entry_price = 1.2500

        # Within stop loss
        assert strategy._check_stop_loss(1.2525) == False

        # Beyond stop loss (1%)
        assert strategy._check_stop_loss(1.2650) == True
        assert strategy._check_stop_loss(1.2350) == True

    def test_position_clearing(self, strategy_params):
        """Test London close position clearing."""
        strategy = LondonBreakoutStrategy()
        strategy.params = strategy_params

        # Simulate active session
        strategy.tokyo_highs = [1.2510]
        strategy.tokyo_lows = [1.2490]
        strategy.signals_this_session = 1

        strategy._clear_positions_at_close()

        assert len(strategy.tokyo_highs) == 0
        assert len(strategy.tokyo_lows) == 0
        assert strategy.signals_this_session == 0
```

#### Step 3.3: Integration Tests
```python
    def test_backtest_execution(self, sample_fx_data, strategy_params):
        """Test full backtest execution."""
        engine = BacktestEngine(
            strategy=LondonBreakoutStrategy,
            strategy_params=strategy_params,
            data=sample_fx_data,
            initial_cash=10000
        )

        results = engine.run()

        # Validate results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert len(results['trades']) > 0

    def test_signal_timing(self, sample_fx_data, strategy_params):
        """Test signals generate only during London trading window."""
        engine = BacktestEngine(
            strategy=LondonBreakoutStrategy,
            strategy_params=strategy_params,
            data=sample_fx_data,
            initial_cash=10000
        )

        results = engine.run()
        signals = results['signals']

        # Check signal timing
        for signal in signals:
            signal_time = signal['timestamp']
            hour = signal_time.hour

            # Signals should only be during 3:00-3:30 EST
            assert hour == 3
            assert signal_time.minute < 30
```

#### Step 3.4: Performance Tests
```python
    def test_performance_metrics(self, sample_fx_data, strategy_params):
        """Test strategy meets minimum performance criteria."""
        engine = BacktestEngine(
            strategy=LondonBreakoutStrategy,
            strategy_params=strategy_params,
            data=sample_fx_data,
            initial_cash=10000
        )

        results = engine.run()

        # Minimum viable thresholds
        assert len(results['trades']) >= 30  # Minimum trades
        assert results['sharpe_ratio'] > 0.0   # Positive risk-adjusted return
        assert results['max_drawdown'] > -1.0  # Drawdown not catastophic
```

### Phase 4: Data Validation (20 minutes)

#### Step 4.1: Create Data Validation Test
```python
class TestDataRequirements:
    """Test data requirements for London Breakout."""

    def test_minute_frequency_data(self):
        """Verify minute-frequency OHLCV data."""
        # Load validation dataset (GBP/USD 2022-2023)
        data_path = Path('data/validation/gbp_usd_2022_2023.csv')

        if data_path.exists():
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)

            # Check frequency
            time_diffs = data.index.to_series().diff()
            avg_diff = time_diffs.mean()

            assert avg_diff == pd.Timedelta(minutes=1)

    def test_timezone_coverage(self):
        """Verify data covers all required time zones."""
        data_path = Path('data/validation/gbp_usd_2022_2023.csv')

        if data_path.exists():
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)

            # Check Tokyo hour coverage (2:00-3:00 EST)
            tokyo_hours = data.index[data.index.hour == 2]
            assert len(tokyo_hours) > 0

            # Check London session coverage (3:00-12:00 EST)
            london_hours = data.index[
                (data.index.hour >= 3) &
                (data.index.hour < 12)
            ]
            assert len(london_hours) > 0
```

### Phase 5: Backtest Validation (30 minutes)

#### Step 5.1: Create Validation Runner
**File:** `validate_london_breakout.py`

```python
"""
Validation script for London Breakout Strategy.
"""

import pandas as pd
from pathlib import Path
from src.strategies.london_breakout import LondonBreakoutStrategy
from src.backtest.engine import BacktestEngine
import matplotlib.pyplot as plt


def run_validation():
    """Run full validation suite."""
    print("=== London Breakout Strategy Validation ===\n")

    # Load data
    data_path = Path('data/validation/gbp_usd_2022_2023.csv')

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please download GBP/USD minute data for 2022-2023")
        return

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} data points\n")

    # Run backtest
    engine = BacktestEngine(
        strategy=LondonBreakoutStrategy,
        initial_cash=10000,
        commission=0.0001  # 1 pip spread
    )

    results = engine.run(data)

    # Print metrics
    print("=== Backtest Results ===")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {len(results['trades'])}\n")

    # Validate metrics
    validate_results(results)

    # Plot equity curve
    plot_equity_curve(results)

    # Save results
    save_validation_results(results)


def validate_results(results: dict) -> None:
    """Validate results meet minimum criteria."""
    print("=== Validation ===")

    # Trade count
    trades = len(results['trades'])
    if trades < 50:
        print(f"⚠️  Warning: Only {trades} trades (expected 50+)")
    else:
        print(f"✅ Trade count: {trades} (meets 50+ minimum)")

    # Sharpe ratio
    sharpe = results['sharpe_ratio']
    if sharpe < 0.5:
        print(f"⚠️  Warning: Sharpe {sharpe:.2f} (expected > 0.5)")
    else:
        print(f"✅ Sharpe ratio: {sharpe:.2f} (exceeds 0.5 threshold)")

    # Max drawdown
    mdd = results['max_drawdown']
    if mdd > -0.3:
        print(f"✅ Max drawdown: {mdd:.2%} (within -30% limit)")
    else:
        print(f"⚠️  Warning: Max drawdown {mdd:.2%} (exceeds -30%)")


def plot_equity_curve(results: dict) -> None:
    """Plot equity curve."""
    equity = results['equity_curve']

    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity.values, label='London Breakout')
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial Capital')
    plt.title('London Breakout Strategy - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/london_breakout_equity.png', dpi=150)
    plt.close()


def save_validation_results(results: dict) -> None:
    """Save validation results to file."""
    results_path = Path('docs/london_breakout_results.json')

    results_data = {
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'win_rate': results['win_rate'],
        'total_trades': len(results['trades']),
        'profit_factor': results['profit_factor']
    }

    import json
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    run_validation()
```

### Phase 6: Documentation (15 minutes)

#### Step 6.1: Create Strategy Documentation
**File:** `docs/london_breakout.md`

```markdown
# London Breakout Strategy

## Overview

The London Breakout Strategy is a time-of-day momentum strategy that uses Tokyo trading hour price action to predict London session breakouts. It targets the first 30 minutes of the London session (3:00-3:30 EST) for execution.

## Strategy Logic

### 1. Tokyo Hour Collection (2:00-3:00 EST)
- Collect high and low prices during Tokyo hour
- Calculate price range: `tokyo_high - tokyo_low`

### 2. London Open Setup (3:00 EST)
- Set upper threshold: `london_open + 0.5 * tokyo_range`
- Set lower threshold: `london_open - 0.5 * tokyo_range`

### 3. Trading Window (3:00-3:30 EST)
- Monitor for price breakouts above upper or below lower thresholds
- Execute buy on upper breakout, sell on lower breakout
- Apply 1% stop loss filter
- Allow only 1 position per session
- Support reverse position detection

### 4. London Close (12:00 EST)
- Clear all intraday positions
- Reset session state

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokeyo_start_hour` | 2 | Tokyo hour start (EST) |
| `tokyo_end_hour` | 3 | Tokyo hour end (EST) |
| `london_trading_minutes` | 30 | London trading window duration |
| `london_close_hour` | 12 | London session end (EST) |
| `risky_stop` | 0.01 | Stop loss threshold (1%) |
| `param` | 0.5 | Range multiplier for thresholds |

## Data Requirements

- **Frequency:** Minute-level OHLCV data
- **Pairs:** FX pairs (GBP/USD recommended)
- **Time Period:** Minimum 2 years for robust validation
- **Coverage:** Must include Tokyo hour and London session times

## Expected Performance

- **Sharpe Ratio:** 0.5-1.5
- **Max Drawdown:** -10% to -25%
- **Win Rate:** 45-55%
- **Trade Frequency:** 50-100 trades per year on 2+ year data

## Validation Results

Run `python validate_london_breakout.py` to generate validation metrics and plots.

## Usage Example

```python
from src.strategies.london_breakout import LondonBreakoutStrategy
from src.backtest.engine import BacktestEngine

engine = BacktestEngine(
    strategy=LondonBreakoutStrategy,
    initial_cash=10000,
    commission=0.0001
)

results = engine.run(data)
```
```

---

## 5. Validation Criteria

### 5.1 Functional Validation
- ✅ Tokyo hour price data collection works correctly
- ✅ London open thresholds set accurately
- ✅ Breakout signals generate during 3:00-3:30 EST only
- ✅ Stop loss filter prevents weak entries
- ✅ Single position per session enforced
- ✅ Reverse position detection works
- ✅ London close clears all positions

### 5.2 Data Validation
- ✅ Minute-frequency data required
- ✅ Data covers Tokyo hour (2:00-3:00 EST)
- ✅ Data covers London session (3:00-12:00 EST)
- ✅ OHLCV columns present and valid

### 5.3 Performance Validation
- ✅ Backtest produces 50+ trades on 2+ year data
- ✅ Sharpe ratio > 0.5
- ✅ Max drawdown < -30%
- ✅ Win rate > 40%

### 5.4 Code Quality Validation
- ✅ All functions have type hints
- ✅ Public functions have docstrings
- ✅ Tests cover all code paths
- ✅ No hardcoded paths or values
- ✅ Exception handling implemented

---

## 6. Risk Considerations

### 6.1 Market Risks
- **Low Volatility Days:** Tokyo range too small → weak signals
- **False Breakouts:** Price moves beyond but reverses quickly
- **London Session Volatility:** High volatility during news events

### 6.2 Implementation Risks
- **Time Zone Errors:** Incorrect time conversion due to DST changes
- **Data Gaps:** Missing data during critical windows
- **Overfitting:** Parameters optimized on single pair/period

### 6.3 Mitigation Strategies
- Parameter sensitivity analysis across multiple FX pairs
- Include multiple validation periods (bull, bear, flat markets)
- Implement time zone aware datetime handling
- Add data quality checks for gaps

---

## 7. Success Metrics

### 7.1 Implementation Success
- All unit tests pass (`uv run pytest tests/strategies/test_london_breakout.py`)
- Integration tests pass with sample data
- Validation script runs without errors

### 7.2 Performance Success
- Sharpe ratio > 0.5 on validation data
- Win rate > 40% on validation data
- Trade frequency > 50 trades on 2+ year data

### 7.3 Code Quality Success
- Type hints on all function signatures
- Docstrings on all public functions
- Test coverage > 80%
- No linting errors (`uv run ruff src/strategies/london_breakout.py`)

---

## 8. Next Steps After FR-001

1. **Run Validation:** Execute `python validate_london_breakout.py`
2. **Review Results:** Check metrics against expected performance
3. **Parameter Tuning:** Adjust `param` and `risky_stop` if needed
4. **Cross-Validation:** Test on additional FX pairs (EUR/USD, USD/JPY)
5. **Documentation:** Update README.md with strategy summary
6. **Proceed to FR-002:** Implement Dual Thrust Strategy

---

## 9. Contingency Plans

### If Data Not Available
- Use synthetic minute data for initial testing
- Download free FX data from Kaggle or Alpha Vantage
- Start with lower-frequency data (5-minute) for development

### If Performance Poor
- Analyze signal generation timing
- Check time zone handling
- Review Tokyo range calculation
- Adjust thresholds and stop loss parameters

### If Tests Fail
- Review traceback for specific errors
- Check data format and index
- Verify time zone logic
- Add debug logging to next() method

---

**Implementation Start:** Ready to begin
**Estimated Completion:** 2-3 hours
**Next Review:** Backtest results and validation metrics
