# %% [markdown]
# # Pattern Visualization
#
# This notebook demonstrates how to visualize detected patterns on price charts
# using mplfinance and the pattern marker system.

# %% [markdown]
# ## Configuration
#
# Centralized configuration for data, patterns, and visualization settings.

# %%
CONFIG = {
    # Data settings
    "data": {
        "filename": "SPY_daily.csv",
        "directory": "data/raw",
        "bars_to_load": 500,  # Focus on recent data for visualization
        "columns": ["Open", "High", "Low", "Close", "Volume"],
    },
    # Pattern detection settings
    "patterns": {
        "enabled": [
            "Market Structure Low",
            "Matching Lows",
            "NR7 Inside Day",
            "Double Top",
            "Double Bottom",
            "Head and Shoulders",
        ],
        "min_confidence": 0.0,  # Minimum confidence threshold
    },
    # Visualization settings
    "visualization": {
        "chart_style": "charles",
        "figure_size": (14, 8),
        "dpi": 150,
        "zoom_bars_before": 20,  # Bars before signal for zoom view
        "zoom_bars_after": 20,  # Bars after signal for zoom view
        "include_stops": True,
        "include_targets": True,
    },
    # Output settings
    "output": {
        "directory": "reports",
        "main_chart": "pattern_visualization.png",
        "confidence_chart": "pattern_confidence.png",
        "direction_chart": "pattern_direction.png",
    },
}

# %% [markdown]
# ## 1. Setup

# %%
import sys
from pathlib import Path

# Setup project root
project_root = Path(".").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import pattern detectors
from src.patterns.basic import MarketStructureLow, MatchingLows, NR7ID
from src.patterns.classic import DoubleTop, DoubleBottom
from src.patterns.complex import HeadAndShoulders

# Import visualization
from src.visualization import ChartGenerator, PatternMarkerGenerator

# Import helpers
from src.utils.notebook_helpers import (
    get_data_path,
    get_output_path,
    print_config,
)

# Print configuration
print_config(CONFIG, "Pattern Visualization Configuration")

# %% [markdown]
# ## 2. Load Data

# %%
# Construct data path
data_path = get_data_path(CONFIG, project_root)

# Load historical data
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df[CONFIG["data"]["columns"]]

# Focus on recent data for visualization
bars_to_load = CONFIG["data"]["bars_to_load"]
if bars_to_load and len(df) > bars_to_load:
    df = df.tail(bars_to_load)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# %% [markdown]
# ## 3. Detect Patterns

# %%
# Pattern name to class mapping
PATTERN_CLASSES = {
    "Market Structure Low": MarketStructureLow,
    "Matching Lows": MatchingLows,
    "NR7 Inside Day": NR7ID,
    "Double Top": DoubleTop,
    "Double Bottom": DoubleBottom,
    "Head and Shoulders": HeadAndShoulders,
}

# Initialize pattern detectors from config
patterns = {}
for name in CONFIG["patterns"]["enabled"]:
    if name in PATTERN_CLASSES:
        patterns[name] = PATTERN_CLASSES[name]()
    else:
        print(f"Warning: Unknown pattern '{name}'")

# Detect patterns
all_signals = []
min_confidence = CONFIG["patterns"]["min_confidence"]

for name, detector in patterns.items():
    print(f"Detecting {name}...")

    for i in range(detector.min_bars_required, len(df)):
        try:
            result = detector.detect(df, i)

            if result.detected and result.signal:
                # Apply confidence filter
                if result.signal.confidence < min_confidence:
                    continue

                signal = {
                    "timestamp": df.index[i],
                    "pattern_name": name,
                    "direction": result.signal.direction.value,
                    "entry_price": result.signal.entry_price,
                    "stop_loss": result.signal.stop_loss,
                    "take_profit_1": result.signal.take_profit_1,
                    "take_profit_2": result.signal.take_profit_2,
                    "take_profit_3": result.signal.take_profit_3,
                    "confidence": result.signal.confidence,
                }
                all_signals.append(signal)
        except Exception:
            pass  # Skip errors

print(f"\nTotal signals detected: {len(all_signals)}")

# Count by pattern
from collections import Counter

pattern_counts = Counter(s["pattern_name"] for s in all_signals)
print("\nSignals by pattern:")
for pattern, count in pattern_counts.items():
    print(f"  {pattern}: {count}")

# %% [markdown]
# ## 4. Generate Pattern Markers

# %%
# Generate markers for visualization
marker_gen = PatternMarkerGenerator()
markers = marker_gen.generate_markers(
    all_signals,
    include_stops=CONFIG["visualization"]["include_stops"],
    include_targets=CONFIG["visualization"]["include_targets"],
)

print(f"Total markers generated: {len(markers)}")

# Get marker summary
summary = marker_gen.get_summary()
print("\nMarker summary:")
for key, value in summary.items():
    if key != "patterns":
        print(f"  {key}: {value}")

# %% [markdown]
# ## 5. Visualize Patterns on Chart

# %%
# Create chart generator
chart_gen = ChartGenerator()

# Get output path
output_path = get_output_path(CONFIG, project_root, CONFIG["output"]["main_chart"])

# Plot with patterns
chart_gen.plot_with_patterns(
    df=df,
    signals=all_signals,
    title="SPY - Pattern Detection",
    save_path=str(output_path),
    show=True,
)


# %% [markdown]
# ## 6. Visualize Specific Pattern Types

# %%
def plot_pattern_type(pattern_name, df, signals, project_root, chart_gen, output_dir):
    """Plot chart with specific pattern type."""
    filtered = [s for s in signals if s["pattern_name"] == pattern_name]

    if not filtered:
        print(f"No {pattern_name} signals found")
        return

    # Generate output filename
    filename = f"{pattern_name.lower().replace(' ', '_')}.png"
    save_path = project_root / output_dir / filename

    try:
        chart_gen.plot_with_patterns(
            df=df,
            signals=filtered,
            title=f"SPY - {pattern_name} Signals ({len(filtered)} detected)",
            save_path=str(save_path),
            show=True,
        )
    except Exception as e:
        print(f"    Error plotting {pattern_name}: {e}")


# Plot each pattern type
for pattern_name in patterns.keys():
    print(f"Plotting {pattern_name}...")
    plot_pattern_type(
        pattern_name, df, all_signals, project_root, chart_gen, CONFIG["output"]["directory"]
    )

# %% [markdown]
# ## 7. Visualize Entry/Stop/Target Levels

# %%
# Create detailed visualization for a specific signal
if all_signals:
    import mplfinance as mpf

    viz_config = CONFIG["visualization"]

    # Get first signal
    signal = all_signals[0]
    signal_date = signal["timestamp"]

    # Get surrounding data using config settings
    start_idx = df.index.get_loc(signal_date) - viz_config["zoom_bars_before"]
    end_idx = df.index.get_loc(signal_date) + viz_config["zoom_bars_after"]

    if start_idx >= 0 and end_idx < len(df):
        df_zoom = df.iloc[start_idx:end_idx]

        # Create markers
        entry_markers = pd.Series(index=df_zoom.index, dtype=float)
        stop_markers = pd.Series(index=df_zoom.index, dtype=float)
        tp1_markers = pd.Series(index=df_zoom.index, dtype=float)

        if signal_date in df_zoom.index:
            entry_markers[signal_date] = signal["entry_price"]
            stop_markers[signal_date] = signal["stop_loss"]
            if signal.get("take_profit_1"):
                tp1_markers[signal_date] = signal["take_profit_1"]

        # Create addplots
        addplots = [
            mpf.make_addplot(
                entry_markers, type="scatter", markersize=200, marker="^", color="green"
            ),
            mpf.make_addplot(stop_markers, type="scatter", markersize=150, marker="x", color="red"),
        ]

        if signal.get("take_profit_1"):
            addplots.append(
                mpf.make_addplot(
                    tp1_markers, type="scatter", markersize=150, marker="o", color="blue"
                )
            )

        # Plot
        print("\nSignal Details:")
        print(f"  Pattern: {signal['pattern_name']}")
        print(f"  Direction: {signal['direction']}")
        print(f"  Entry: ${signal['entry_price']:.2f}")
        print(f"  Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"  Take Profit 1: ${signal.get('take_profit_1', 'N/A')}")
        print(f"  Confidence: {signal['confidence']:.2f}")

        mpf.plot(
            df_zoom,
            type="candle",
            style=viz_config["chart_style"],
            title=f"{signal['pattern_name']} - {signal['direction']}",
            addplot=addplots,
            figsize=viz_config["figure_size"],
            volume=True,
        )

# %% [markdown]
# ## 8. Pattern Confidence Distribution

# %%
# Analyze confidence distribution
if all_signals:
    confidences = [s["confidence"] for s in all_signals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(confidences, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_title("Signal Confidence Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].axvline(
        x=CONFIG["patterns"]["min_confidence"], color="red", linestyle="--", label="Min Threshold"
    )
    axes[0].legend()

    # By pattern
    pattern_conf = {}
    for s in all_signals:
        if s["pattern_name"] not in pattern_conf:
            pattern_conf[s["pattern_name"]] = []
        pattern_conf[s["pattern_name"]].append(s["confidence"])

    axes[1].boxplot(
        [pattern_conf[p] for p in pattern_conf.keys()], labels=[p[:15] for p in pattern_conf.keys()]
    )
    axes[1].set_title("Confidence by Pattern Type")
    axes[1].set_xlabel("Pattern")
    axes[1].set_ylabel("Confidence")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save using config
    output_path = get_output_path(CONFIG, project_root, CONFIG["output"]["confidence_chart"])
    plt.savefig(output_path, dpi=CONFIG["visualization"]["dpi"])
    plt.show()

# %% [markdown]
# ## 9. Pattern Direction Analysis

# %%
# Analyze direction distribution
if all_signals:
    directions = [s["direction"] for s in all_signals]
    long_count = directions.count("Long")
    short_count = directions.count("Short")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    axes[0].pie(
        [long_count, short_count],
        labels=["Long", "Short"],
        autopct="%1.1f%%",
        colors=["green", "red"],
        explode=(0.05, 0.05),
    )
    axes[0].set_title("Signal Direction Distribution")

    # Direction by pattern
    pattern_dir = {}
    for s in all_signals:
        if s["pattern_name"] not in pattern_dir:
            pattern_dir[s["pattern_name"]] = {"Long": 0, "Short": 0}
        pattern_dir[s["pattern_name"]][s["direction"]] += 1

    patterns_list = list(pattern_dir.keys())
    longs = [pattern_dir[p]["Long"] for p in patterns_list]
    shorts = [pattern_dir[p]["Short"] for p in patterns_list]

    x = np.arange(len(patterns_list))
    width = 0.35

    axes[1].bar(x - width / 2, longs, width, label="Long", color="green", alpha=0.7)
    axes[1].bar(x + width / 2, shorts, width, label="Short", color="red", alpha=0.7)
    axes[1].set_title("Direction by Pattern Type")
    axes[1].set_xlabel("Pattern")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p[:15] for p in patterns_list], rotation=45, ha="right")
    axes[1].legend()

    plt.tight_layout()

    # Save using config
    output_path = get_output_path(CONFIG, project_root, CONFIG["output"]["direction_chart"])
    plt.savefig(output_path, dpi=CONFIG["visualization"]["dpi"])
    plt.show()

# %% [markdown]
# ## 10. Summary

# %%
# Print summary
print("\n" + "=" * 60)
print("PATTERN DETECTION SUMMARY")
print("=" * 60)
print(f"\nData Period: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total Bars: {len(df)}")
print(f"\nTotal Signals Detected: {len(all_signals)}")
print("\nBy Pattern:")
for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {pattern}: {count}")

if all_signals:
    print("\nBy Direction:")
    print(f"  Long: {long_count} ({long_count / len(all_signals) * 100:.1f}%)")
    print(f"  Short: {short_count} ({short_count / len(all_signals) * 100:.1f}%)")
    print(f"\nAverage Confidence: {np.mean(confidences):.2f}")
    print(f"Min Confidence: {np.min(confidences):.2f}")
    print(f"Max Confidence: {np.max(confidences):.2f}")
print("=" * 60)
