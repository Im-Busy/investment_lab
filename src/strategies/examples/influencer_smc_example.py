"""
Influencer Framework + SMC Integration Example

Demonstrates how to use the modular influencer framework
and integrate it with existing SMC strategies.

Usage:
    uv run src/strategies/examples/influencer_smc_example.py
"""

import pandas as pd

from src.strategies.strategy_registry import (
    ConfluenceAggregatorPlugin,
    InfluencerMTFPlugin,
    PluginConfig,
    SMCReversalPlugin,
    StrategyRegistry,
)


def example_basic_usage():
    """
    Basic usage: Run influencer framework standalone.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Influencer Framework")
    print("=" * 60)

    # Fetch data (replace with your actual data source)
    # For demo, we'll create sample data
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    df_30min = pd.DataFrame(
        {
            "Open": 100 + (range(100)),
            "High": 102 + (range(100)),
            "Low": 98 + (range(100)),
            "Close": 101 + (range(100)),
            "Volume": [1000] * 100,
        },
        index=dates,
    )

    # For MTF, you need 4H data (aggregate from 30min)
    df_4h = df_30min.resample("4H").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    from src.strategies.influencer_confluence import (
        InfluencerConfig,
        InfluencerConfluenceScorer,
    )

    config = InfluencerConfig(
        htf_timeframe="4H",
        ltf_timeframe="30min",
        min_confluence_score=4.0,  # Minimum 4/5 factors must align
        volume_multiplier=2.0,  # Volume must be 2x average
    )

    scorer = InfluencerConfluenceScorer(config)

    # Scan for signals
    signals = scorer.scan(df_4h, df_30min)

    print(f"\nFound {len(signals)} confluence signals")

    for signal in signals:
        summary = scorer.get_summary(signal)
        print(f"\nSignal at {summary['timestamp']}")
        print(f"  Direction: {summary['direction']}")
        print(f"  Entry: {summary['entry']}")
        print(f"  Stop: {summary['stop_loss']}")
        print(f"  Target: {summary['take_profit']}")
        print(f"  Confluence: {summary['confluence_score']} ({summary['level']})")
        print("  Factors:")
        for factor, status in summary["factors"].items():
            print(f"    - {factor}: {status}")


def example_plugin_registry():
    """
    Plugin registry: Combine SMC + Influencer strategies.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Strategy Plugin Registry")
    print("=" * 60)

    # Create registry
    registry = StrategyRegistry()

    # Register SMC reversal plugin
    smc_plugin = SMCReversalPlugin(
        PluginConfig(
            name="SMC_Reversal",
            weight=1.0,
            priority=1,  # Evaluate first
        )
    )
    registry.register(smc_plugin)

    # Register influencer MTF plugin
    influencer_plugin = InfluencerMTFPlugin(
        PluginConfig(
            name="Influencer_MTF",
            weight=1.0,
            priority=2,
        )
    )
    # Note: You'll need to set htf_dataframe via set_htf_dataframe()
    # when you have actual 4H data
    registry.register(influencer_plugin)

    # Register confluence aggregator (combines signals)
    aggregator = ConfluenceAggregatorPlugin(
        PluginConfig(
            name="Confluence_Aggregator",
            weight=1.0,
            priority=100,  # Evaluate last
        )
    )
    aggregator.register_plugin(smc_plugin)
    aggregator.register_plugin(influencer_plugin)
    registry.register(aggregator)

    print(f"\nRegistered plugins: {registry.get_plugin_names()}")
    print(f"Plugin summary: {registry.get_plugin_summary()}")

    # Evaluate all plugins
    # signals = registry.evaluate(df_30min, htf_df=df_4h)

    # Or evaluate for confluence only (2+ plugins must agree)
    # confluence_signals = registry.evaluate_confluence(df_30min, min_plugins=2, htf_df=df_4h)


def example_selective_evaluation():
    """
    Selective evaluation: Only run specific plugins.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Selective Plugin Evaluation")
    print("=" * 60)

    registry = StrategyRegistry()

    # Register multiple plugins
    registry.register(SMCReversalPlugin())
    registry.register(InfluencerMTFPlugin())
    registry.register(ConfluenceAggregatorPlugin())

    # Enable/disable specific plugins
    registry.enable("SMC_Reversal")
    registry.disable("Influencer_MTF")  # Skip influencer for this run

    # Evaluate only specific plugins
    # signals = registry.evaluate(
    #     df_30min,
    #     plugin_names=["SMC_Reversal"],  # Only run SMC
    # )

    print("Plugins can be enabled/disabled dynamically")
    print("Use registry.enable('PluginName') and registry.disable('PluginName')")


def example_custom_weighting():
    """
    Custom weighting: Adjust plugin importance.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Plugin Weighting")
    print("=" * 60)

    registry = StrategyRegistry()

    # Register plugins with different weights
    smc = SMCReversalPlugin()
    smc.set_weight(1.2)  # Higher weight = more importance
    registry.register(smc)

    influencer = InfluencerMTFPlugin()
    influencer.set_weight(0.8)  # Lower weight
    registry.register(influencer)

    print("SMC weight: 1.2 (emphasized)")
    print("Influencer weight: 0.8 (de-emphasized)")
    print("\nAdjust weights based on your backtest results")


def example_filtering_by_quality():
    """
    Filter signals by quality/confidence.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Signal Quality Filtering")
    print("=" * 60)

    registry = StrategyRegistry()
    registry.register(SMCReversalPlugin())
    registry.register(InfluencerMTFPlugin())

    # signals = registry.evaluate(df_30min)

    # Filter by quality
    # high_quality = [s for s in signals if s.quality == SignalQuality.HIGH]
    # medium_quality = [s for s in signals if s.quality == SignalQuality.MEDIUM]

    # Filter by confidence
    # high_confidence = [s for s in signals if s.confidence >= 0.7]

    # Filter by plugin count (confluence)
    # multi_plugin = [s for s in signals if len(s.factors) >= 2]

    print("Signal quality levels: LOW, MEDIUM, HIGH")
    print("Filter signals based on your risk tolerance:")
    print("  - Conservative: Only HIGH quality, confidence >= 0.7")
    print("  - Moderate: MEDIUM+ quality, confidence >= 0.5")
    print("  - Aggressive: All signals, confidence >= 0.3")


def example_backtest_integration():
    """
    Integration with backtesting.py or custom backtest engine.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Backtest Integration")
    print("=" * 60)

    print("""
# For backtesting.py integration:

from backtesting import Backtest, Strategy
from src.strategies import StrategyRegistry, SMCReversalPlugin

class CustomBacktestStrategy(Strategy):
    params = (
        ('smc_weight', 1.0),
        ('influencer_weight', 1.0),
        ('min_confluence', 0.6),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = StrategyRegistry()
        self.registry.register(SMCReversalPlugin())
        self.registry.register(InfluencerMTFPlugin())

    def next(self):
        # Get signals from registry
        signals = self.registry.evaluate(self.data.df)

        # Filter by quality
        high_quality = [s for s in signals if s.confidence >= 0.7]

        # Execute trades
        for signal in high_quality:
            if signal.direction == 'long' and not self.position:
                self.buy()
            elif signal.direction == 'short' and not self.position:
                self.sell()

# Run backtest
bt = Backtest(df, CustomBacktestStrategy, cash=10000)
stats = bt.run()
print(stats)
""")


def example_with_real_data():
    """
    Full example with real data fetching.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Real Data Workflow")
    print("=" * 60)

    # Step 1: Fetch data at multiple timeframes
    print("\nStep 1: Fetch data")
    print("  - 4H DataFrame for higher timeframe bias")
    print("  - 30min DataFrame for entry timing")

    # df_4h = fetch_data("AAPL", timeframe="4h", days=90)
    # df_30min = fetch_data("AAPL", timeframe="30m", days=30)

    # Step 2: Initialize registry
    print("\nStep 2: Initialize strategy registry")
    registry = StrategyRegistry()

    # Step 3: Register plugins
    registry.register(SMCReversalPlugin(PluginConfig(name="SMC", weight=1.0, priority=1)))
    registry.register(InfluencerMTFPlugin(PluginConfig(name="Influencer", weight=1.0, priority=2)))

    # Step 4: Set HTF dataframe for MTF plugin
    # influencer_plugin = registry._plugins.get("Influencer_MTF")
    # if influencer_plugin:
    #     influencer_plugin.set_htf_dataframe(df_4h)

    # Step 5: Evaluate signals
    print("\nStep 5: Evaluate signals")
    # signals = registry.evaluate(df_30min)

    # Step 6: Filter and execute
    # high_quality_signals = [
    #     s for s in signals
    #     if s.quality == SignalQuality.HIGH and s.confidence >= 0.7
    # ]

    print("\nFull workflow:")
    print("  1. Fetch 4H and 30min data")
    print("  2. Initialize registry")
    print("  3. Register SMC + Influencer plugins")
    print("  4. Set HTF dataframe for MTF analysis")
    print("  5. Evaluate signals on 30min data")
    print("  6. Filter by quality/confidence")
    print("  7. Execute trades")


if __name__ == "__main__":
    print("Influencer Framework + SMC Integration Examples")
    print("=" * 60)

    # Run examples
    example_basic_usage()
    example_plugin_registry()
    example_selective_evaluation()
    example_custom_weighting()
    example_filtering_by_quality()
    example_backtest_integration()
    example_with_real_data()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Influencer framework is modular and composable")
    print("2. SMC strategies integrate via plugin registry")
    print("3. Signals can be filtered by quality/confidence")
    print("4. Plugin weights allow customization")
    print("5. Use confluence (2+ plugins agreeing) for high-probability setups")
