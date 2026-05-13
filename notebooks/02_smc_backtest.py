# %% [markdown]
# # SMC/ICT Strategy Backtest Analysis
#
# This notebook demonstrates how to run the SMC/ICT reversal strategy
# using the custom backtest engine and generate visualization reports.
#
# **Note:** The SMC strategy requires 5-minute or finer intraday data.
#
# ---
#
# ## Quick Configuration Guide
#
# Modify the `CONFIG` dictionary in Section 1 to customize:
# - Data source and date range
# - SMC strategy parameters (session times, ATR settings)
# - Risk management parameters
# - Output settings

# %% [markdown]
# ---
#
# ## 1. Configuration Section
#
# **Modify parameters below to customize the SMC backtest.**

# %%
# ============================================================
# CONFIGURATION - Modify these parameters to customize analysis
# ============================================================

CONFIG = {
    # ----------------------------------------------------------
    # Data Configuration
    # ----------------------------------------------------------
    "data": {
        "file": "SPY_5min.csv",  # Primary: 5-minute data
        "fallback_file": "SPY_daily.csv",  # Fallback: daily data
        "directory": "data/raw",
        "start_date": None,
        "end_date": None,
        "columns": ["Open", "High", "Low", "Close", "Volume"],
    },
    # ----------------------------------------------------------
    # SMC Strategy Configuration
    # ----------------------------------------------------------
    "smc": {
        # Session times (UTC)
        "session_start": "00:00",  # Asian session start
        "session_end": "08:00",  # Asian session end
        # ATR settings
        "atr_period": 14,
        "atr_buffer_mult": 0.5,
        "ifvg_atr_mult": 1.2,
        "ifvg_proximity_mult": 1.5,
        # Risk management
        "risk_per_trade": 0.01,  # 1% risk per trade
        "slippage_buffer": 0.1,
        # Targets
        "target_1r": 1.0,  # First target at 1R (breakeven)
        "target_2r": 2.0,  # Second target at 2R
        "target_final": 2.5,  # Final target at 2.5R
        # Daily limits
        "daily_loss_limit": 0.03,  # 3% daily loss limit
        "max_trades_per_day": 3,
        # Confirmations
        "require_volume_confirmation": True,
        "require_mss_confirmation": True,
    },
    # ----------------------------------------------------------
    # Backtest Configuration
    # ----------------------------------------------------------
    "backtest": {
        "initial_equity": 100000,
        "commission_pct": 0.001,  # 0.1% commission
        "slippage_pct": 0.0005,  # 0.05% slippage
        "risk_per_trade": 0.01,
        "max_open_positions": 10,  # SMC typically trades one position
        "min_confidence": 0.5,
    },
    # ----------------------------------------------------------
    # Output Configuration
    # ----------------------------------------------------------
    "output": {
        "directory": "reports",
        "save_plots": True,
        "show_plots": True,
        "dpi": 150,
    },
}

# %% [markdown]
# ---
#
# ## 2. Setup and Imports

# %%
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# Setup project root FIRST (before any src imports)
candidates = [
    Path(".").resolve(),
    Path(".").resolve(),
]
project_root = None
for root in candidates:
    if (root / "src").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        project_root = root
        break
if project_root is None:
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Import notebook helpers
from src.utils.notebook_helpers import (
    load_price_data,
    print_data_summary,
)

# Standard imports

# Import SMC strategy
from src.strategies import SMCReversalStrategy, SMCConfig
from src.indicators.asian_range import detect_asian_range
from src.indicators.ifvg import detect_ifvg
from src.indicators.mss import detect_mss
from src.indicators.technical import atr as atr_indicator

print("✅ Imports successful!")
print(f"Project root: {project_root}")

# %% [markdown]
# ---
#
# ## 3. Load and Prepare Data
#
# The SMC strategy requires intraday data (5-minute bars recommended).

# %%
from pathlib import Path

# Force loading daily data since 5min isn't available
data_path = project_root / CONFIG["data"]["directory"] / "SPY_daily.csv"
assert data_path.exists(), "Need data file for SMC notebook"

print("WARNING: 5-minute data not found. Using daily data for demonstration.")
print("For proper SMC backtesting, please provide 5-minute OHLCV data.")

fallback_config = CONFIG.copy()
fallback_config["data"] = CONFIG["data"].copy()
fallback_config["data"]["file"] = "SPY_daily.csv"
fallback_config["data"]["start_date"] = "2024-01-01"
fallback_config["data"]["end_date"] = "2024-12-31"
df = load_price_data(fallback_config, project_root)
data_freq = "daily"

# Display data summary
print_data_summary(df, title=f"SMC Backtest Data ({data_freq})")

# Check data frequency
median_diff = df.index.to_series().diff().median()
print(f"Data frequency: {median_diff}")

# %% [markdown]
# ---
#
# ## 4. SMC Strategy Components
#
# Let's examine the SMC indicators individually.

# %%
# Detect Asian Range (for 5-minute data)
# The Asian session is typically 00:00-08:00 UTC

if data_freq == "5-minute":
    print("Detecting Asian Range...")
    asian_range = detect_asian_range(df)
    if asian_range:
        print(f"Asian Range High: {asian_range.high}")
        print(f"Asian Range Low: {asian_range.low}")
        print(f"Range Size: {asian_range.range_size}")
        print(f"Is Low Volatility: {asian_range.is_low_vol}")
else:
    print("Daily data detected - Asian Range detection requires intraday data")

# %%
# Detect IFVG (Inverse Fair Value Gaps)
print("Detecting IFVGs...")
atr_series = atr_indicator(df, period=14)
ifvg_list = detect_ifvg(df, atr=atr_series)

print(f"Found {len(ifvg_list)} IFVGs")
if ifvg_list:
    print("\nRecent IFVGs:")
    for ifvg in ifvg_list[:5]:
        print(f"  Direction: {ifvg.direction}, Range: [{ifvg.low:.2f}, {ifvg.high:.2f}]")
        print(f"    Filled: {ifvg.filled}, Gap Size: {ifvg.gap_size:.2f}")

# %%
# Detect Market Structure Shifts (limited sample for daily data)
print("Detecting Market Structure Shifts (sampling every 10th day)")
mss_list = []

# For daily data, sample every 10th bar to keep it fast
step = 10 if data_freq == "daily" else 1
for i in range(50, len(df), step):
    mss = detect_mss(df, i)
    if mss.detected and getattr(mss, "is_valid", False):
        mss_list.append(
            {
                "index": i,
                "timestamp": df.index[i],
                "direction": mss.direction,
                "break_price": mss.break_price,
            }
        )

print(f"Found {len(mss_list)} valid MSS signals")
if mss_list:
    print("\nRecent MSS signals:")
    for mss in mss_list[-5:]:
        print(f"  {mss['timestamp']}: {mss['direction']} at {mss['break_price']:.2f}")

# %% [markdown]
# ---
#
# ## 5. Configure SMC Strategy

# %%
# Create SMC configuration from CONFIG dict
smc_params = CONFIG["smc"]

smc_config = SMCConfig(
    session_start=smc_params["session_start"],
    session_end=smc_params["session_end"],
    atr_period=smc_params["atr_period"],
    atr_buffer_mult=smc_params["atr_buffer_mult"],
    ifvg_atr_mult=smc_params["ifvg_atr_mult"],
    ifvg_proximity_mult=smc_params["ifvg_proximity_mult"],
    risk_per_trade=smc_params["risk_per_trade"],
    slippage_buffer=smc_params["slippage_buffer"],
    target_1r=smc_params["target_1r"],
    target_2r=smc_params["target_2r"],
    target_final=smc_params["target_final"],
    daily_loss_limit=smc_params["daily_loss_limit"],
    max_trades_per_day=smc_params["max_trades_per_day"],
    require_volume_confirmation=smc_params["require_volume_confirmation"],
    require_mss_confirmation=smc_params["require_mss_confirmation"],
)

print("SMC Strategy Configuration:")
print(f"  Session: {smc_config.session_start} - {smc_config.session_end} UTC")
print(f"  Risk per trade: {smc_config.risk_per_trade * 100}%")
print(f"  Daily loss limit: {smc_config.daily_loss_limit * 100}%")
print(f"  Max trades per day: {smc_config.max_trades_per_day}")

# %% [markdown]
# ---
#
# ## 6. Run Backtest with Custom Engine

# %%
# Configure backtest
backtest_params = CONFIG["backtest"]

print("Backtest Configuration:")
print(f"  Initial equity: ${backtest_params['initial_equity']:,.0f}")
print(f"  Commission: {backtest_params['commission_pct'] * 100:.2f}%")
print(f"  Slippage: {backtest_params['slippage_pct'] * 100:.3f}%")

# %%
# Initialize strategy
smc_strategy = SMCReversalStrategy(config=smc_config)

print("✅ Strategy initialized")

# %%
# Run backtest
print("Running SMC backtest...")

# Note: SMC strategy works best on 5-minute data
# Results on daily data will be limited

signals = smc_strategy.run(df)

print("\nBacktest complete!")
print(f"Total signals: {len(signals)}")

# %% [markdown]
# ---
#
# ## 7. Analyze Results

# %%
# Display metrics
if signals:
    print("=" * 60)
    print("SMC STRATEGY PERFORMANCE")
    print("=" * 60)

    print(f"Total signals: {len(signals)}")
    long_signals = sum(1 for s in signals if s.direction == "long")
    short_signals = sum(1 for s in signals if s.direction == "short")
    print(f"Long signals: {long_signals}")
    print(f"Short signals: {short_signals}")
    print("=" * 60)

# %%
# Display signals
if signals:
    print("\n📊 Recent Signals:")
    for signal in signals[:10]:
        print(f"  {signal.timestamp}: {signal.direction} @ {signal.entry_price:.4f}")
else:
    print("\n⚠️ No signals generated. This is expected with daily data.")
    print("   SMC strategy requires 5-minute or finer intraday data.")

# %%
# Display signal details
if signals:
    print("\n📊 Signal Details:")
    for signal in signals[:5]:
        print(f"  {signal.timestamp}: {signal.direction}")
        print(f"    Entry: {signal.entry_price:.4f}, Stop: {signal.stop_loss:.4f}")
        print(
            f"    Targets: {signal.target_1:.4f}, {signal.target_2:.4f}, {signal.target_final:.4f}"
        )

# %% [markdown]
# ---
#
# ## 8. Summary

# %%
print("\n" + "=" * 60)
print("📊 SMC BACKTEST SUMMARY")
print("=" * 60)
print(f"\nData frequency: {data_freq}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total bars: {len(df):,}")

if data_freq == "daily":
    print("\n⚠️ NOTE: SMC strategy is designed for intraday data.")
    print("   For meaningful results, please provide 5-minute OHLCV data.")
    print("   The strategy looks for:")
    print("   - Asian Range liquidity sweeps")
    print("   - Inverse Fair Value Gaps (IFVG)")
    print("   - Market Structure Shifts (MSS)")
else:
    print(f"\nTotal signals: {len(signals)}")

    if signals:
        print("\nKey Metrics:")
        long_signals = sum(1 for s in signals if s.direction == "long")
        short_signals = sum(1 for s in signals if s.direction == "short")
        print(f"  Long signals: {long_signals}")
        print(f"  Short signals: {short_signals}")
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        print(f"  Avg confidence: {avg_confidence:.2f}")

print("\n" + "=" * 60)
