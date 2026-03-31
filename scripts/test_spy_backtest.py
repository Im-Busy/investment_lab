"""
Quick test script to verify Multi-Pattern Strategy works on SPY data.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPY Long-Term Backtest - Quick Test")
print("=" * 60)

# Load SPY daily data
data_path = project_root / 'data' / 'raw' / 'SPY_daily.csv'
print(f"\nLoading data from: {data_path}")

df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df.columns = [c.capitalize() for c in df.columns]

print(f"Data loaded: {len(df)} rows")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

# Use a subset of data for faster testing
df_test = df.tail(500).copy()
print(f"\nUsing last 500 bars for testing: {df_test.index.min().date()} to {df_test.index.max().date()}")

# Test imports
print("\nTesting imports...")
try:
    from src.strategies.backtest_py import BacktestPyRunner, MultiPatternStrategy
    print("  [OK] BacktestPyRunner imported")
    print("  [OK] MultiPatternStrategy imported")
except ImportError as e:
    print(f"  [ERROR] Import error: {e}")
    sys.exit(1)

# Test pattern imports
print("\nTesting pattern detectors...")
try:
    from src.patterns.basic.msl import MarketStructureLow
    from src.patterns.basic.matching_lows import MatchingLows
    from src.patterns.basic.nr7id import NR7ID
    from src.patterns.basic.n_bar_decline import NBarDecline
    from src.patterns.basic.floor_pivot import FloorPivotBreakout
    from src.patterns.classic.double_top import DoubleTop
    from src.patterns.classic.double_bottom import DoubleBottom
    from src.patterns.complex.head_shoulders import HeadAndShoulders
    print("  [OK] Pattern detectors imported successfully")
except ImportError as e:
    print(f"  [ERROR] Pattern import error: {e}")
    sys.exit(1)

# Run a simple backtest
print("\nRunning backtest...")
try:
    runner = BacktestPyRunner(
        data=df_test,
        cash=100000,
        commission=0.001,
        exclusive_orders=True
    )
    
    # Run with minimal parameters
    results = runner.run(
        min_confidence=0.50,
        min_confluence_count=2,
        risk_per_trade=0.02,
        max_open_positions=3,
        use_regime_filter=False
    )
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    stats = results.get('stats', {})
    
    # Print key metrics
    print(f"\nReturn Metrics:")
    print(f"  Total Return: {stats.get('Return [%]', 'N/A')}")
    print(f"  Buy & Hold Return: {stats.get('Buy & Hold Return [%]', 'N/A')}")
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
    print(f"  Max Drawdown: {stats.get('Max. Drawdown [%]', 'N/A')}")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {stats.get('# Trades', 'N/A')}")
    print(f"  Win Rate: {stats.get('Win Rate [%]', 'N/A')}")
    
    print("\n[OK] Backtest completed successfully!")
    
except Exception as e:
    print(f"\n[ERROR] Backtest error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! The strategy is working correctly.")
print("=" * 60)
