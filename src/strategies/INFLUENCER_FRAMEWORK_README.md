# Influencer Framework - Modular Trading System

## Overview

A modular trading system implementing the influencer's framework:
- **Multi-timeframe bias detection** (4H + 30min)
- **Volume confirmation** (the only non-price-derived indicator)
- **VWAP + SMA confluence** (institutional benchmark + self-fulfilling prophecy)
- **Fibonacci pullback levels** (works because everyone watches it)
- **Plugin architecture** for SMC strategy integration

## Core Principles

| Indicator | Role | Why It Works |
|-----------|------|--------------|
| **MACD** | Avoid for entries | Lags ~11 days in bear markets |
| **SMA 50/200** | Trend filter | Self-fulfilling prophecy (everyone uses it) |
| **Stochastics** | Use with caution | More noise than MACD, overreacts |
| **Fibonacci** | Pullback levels | Works because traders believe it works |
| **VWAP** | Institutional benchmark | "The King" - institutions measured against it |
| **Volume** | Confirmation | Only indicator not derived from price |

## Framework Strategy

```
4H Timeframe → Directional Bias
    ↓
30min Timeframe → Pullback Entry
    ↓
Volume Spike → Confirmation
    ↓
VWAP Reclaim → Entry Trigger
    ↓
SMA 50/200 → Trend Alignment
```

## Confluence Scoring (5 factors, 1 point each)

| Factor | Criteria | Points |
|--------|----------|--------|
| 4H Trend Alignment | HTF direction matches trade | +1 |
| Fibonacci Pullback | 38.2/50/61.8% retracement | +1 |
| Volume Spike | 2x+ average volume | +1 |
| VWAP Reclaim | Price reclaimed VWAP | +1 |
| SMA Alignment | Price > SMA50 > SMA200 (long) | +1 |

**Minimum score to trade:** 4/5 (or 5/5 in strict mode)

## Architecture

```
src/strategies/
├── multi_timeframe_bias.py    # 4H + 30min bias detector
├── volume_confirmation.py     # Volume analysis module
├── vwap_sma_confluence.py     # VWAP + SMA positioning
├── influencer_confluence.py   # Complete framework scorer
├── strategy_registry.py       # Plugin system for SMC integration
└── examples/
    └── influencer_smc_example.py  # Usage examples
```

## Quick Start

### Basic Usage

```python
from src.strategies import (
    InfluencerConfig,
    InfluencerConfluenceScorer,
)

# Configure
config = InfluencerConfig(
    htf_timeframe="4H",
    ltf_timeframe="30min",
    min_confluence_score=4.0,  # Minimum 4/5 factors
    volume_multiplier=2.0,      # Volume spike threshold
)

# Initialize scorer
scorer = InfluencerConfluenceScorer(config)

# Scan for signals
signals = scorer.scan(df_4h, df_30min)

# Review signals
for signal in signals:
    print(f"{signal.direction}: {signal.entry_price}")
    print(f"Confluence: {signal.confluence_score}/5.0")
    print(f"Factors: {signal.breakdown}")
```

### Plugin Registry (SMC + Influencer)

```python
from src.strategies import (
    StrategyRegistry,
    SMCReversalPlugin,
    InfluencerMTFPlugin,
    ConfluenceAggregatorPlugin,
    PluginConfig,
)

# Create registry
registry = StrategyRegistry()

# Register SMC strategy
smc = SMCReversalPlugin(PluginConfig(weight=1.0, priority=1))
registry.register(smc)

# Register Influencer framework
influencer = InfluencerMTFPlugin(PluginConfig(weight=1.0, priority=2))
registry.register(influencer)

# Register confluence aggregator
aggregator = ConfluenceAggregatorPlugin()
aggregator.register_plugin(smc)
aggregator.register_plugin(influencer)
registry.register(aggregator)

# Evaluate all plugins
signals = registry.evaluate(df_30min, htf_df=df_4h)

# Get confluence signals (2+ plugins agreeing)
confluence_signals = registry.evaluate_confluence(
    df_30min,
    min_plugins=2,
    htf_df=df_4h,
)
```

### Signal Quality Filtering

```python
from src.strategies import SignalQuality

# Get all signals
signals = registry.evaluate(df)

# Filter by quality
high_quality = [s for s in signals if s.quality == SignalQuality.HIGH]

# Filter by confidence
high_confidence = [s for s in signals if s.confidence >= 0.7]

# Filter by plugin count (confluence)
multi_plugin = [s for s in signals if len(s.factors) >= 2]
```

## Component Details

### Multi-Timeframe Bias Detector

Detects directional bias across timeframes:

```python
from src.strategies import (
    MultiTimeframeBiasDetector,
    MTFConfig,
    BiasState,
)

config = MTFConfig(
    htf_timeframe="4H",
    ltf_timeframe="30min",
    sma_short=50,
    sma_long=200,
)

detector = MultiTimeframeBiasDetector(config)
signals = detector.detect(df_4h, df_30min)

# Bias states:
# - BiasState.BULLISH_ALIGNED (4H up, 30min up)
# - BiasState.BULLISH_PULLBACK (4H up, 30min down) ← Entry zone
# - BiasState.BEARISH_ALIGNED (4H down, 30min down)
# - BiasState.BEARISH_RALLY (4H down, 30min up) ← Entry zone
```

### Volume Confirmation

```python
from src.strategies import VolumeConfirmation, VolumeConfig

config = VolumeConfig(
    volume_period=20,
    spike_threshold=2.0,      # 2x average = spike
    dry_up_threshold=0.5,     # 50% of average = dry up
)

volume = VolumeConfirmation(config)

# Confirm a move
confirmed, strength = volume.confirm_move(df, direction="long")

# Check for volume spike
is_spike = volume.is_volume_spike(df)

# Detect accumulation/distribution
is_accumulating = volume.detect_accumulation(df, min_bars=3)
is_distributing = volume.detect_distribution(df, min_bars=3)

# Detect divergence
divergence = volume.detect_volume_divergence(df)  # 'bullish' or 'bearish'
```

### VWAP + SMA Confluence

```python
from src.strategies import VWAPConfluenceAnalyzer, VWAPConfig

config = VWAPConfig(
    vwap_deviation_pct=0.5,   # Deviation bands
    sma_short=50,
    sma_long=200,
)

analyzer = VWAPConfluenceAnalyzer(config)

# Get current confluence
confluence = analyzer.get_current_confluence(df)

print(f"Price: ${confluence.price}")
print(f"VWAP: ${confluence.vwap}")
print(f"SMA50: ${confluence.sma_50}")
print(f"SMA200: ${confluence.sma_200}")
print(f"Bias: {confluence.bias}")  # 'bullish', 'bearish', or 'neutral'
print(f"Score: {confluence.confluence_score:.2f}")
```

### Strategy Plugins

Available plugins:

| Plugin | Type | Description |
|--------|------|-------------|
| `SMCReversalPlugin` | SMC | SMC/ICT reversal patterns |
| `InfluencerMTFPlugin` | Influencer | Multi-timeframe framework |
| `ConfluenceAggregatorPlugin` | Hybrid | Combines multiple plugins |

Plugin configuration:

```python
plugin = SMCReversalPlugin(
    PluginConfig(
        name="SMC_Reversal",
        enabled=True,
        weight=1.0,           # Importance in confluence
        min_confidence=0.5,   # Minimum confidence to generate signals
        priority=1,           # Execution order (lower = first)
    )
)
```

## Integration with Existing SMC Systems

### Option 1: Use as Plugin

```python
from src.strategies import StrategyRegistry, SMCReversalPlugin, InfluencerMTFPlugin

registry = StrategyRegistry()

# Register both SMC and Influencer
registry.register(SMCReversalPlugin())
registry.register(InfluencerMTFPlugin())

# Only take trades where both agree
confluence_signals = registry.evaluate_confluence(df, min_plugins=2)
```

### Option 2: Use as Filter

```python
from src.strategies import InfluencerConfluenceScorer

scorer = InfluencerConfluenceScorer()
mtf_signals = scorer.scan(df_4h, df_30min)

# Only take SMC signals that align with influencer framework
for smc_signal in smc_signals:
    matching_mtf = [
        s for s in mtf_signals
        if s.direction == smc_signal.direction
        and s.confluence_score >= 4.0
    ]
    if matching_mtf:
        # High-probability trade
        execute_trade(smc_signal)
```

### Option 3: Layered Approach

```
SMC Patterns → Initial Signal
    ↓
Influencer Framework → Confirmation Filter
    ↓
Volume Confirmation → Final Check
    ↓
Execute Trade
```

## Backtest Integration

### With backtesting.py

```python
from backtesting import Strategy, Backtest
from src.strategies import StrategyRegistry, SMCReversalPlugin, InfluencerMTFPlugin

class HybridStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = StrategyRegistry()
        self.registry.register(SMCReversalPlugin())
        self.registry.register(InfluencerMTFPlugin())
    
    def next(self):
        signals = self.registry.evaluate(self.data.df)
        
        # Filter for high-quality signals
        high_quality = [
            s for s in signals
            if s.quality == SignalQuality.HIGH
            and s.confidence >= 0.7
        ]
        
        for signal in high_quality:
            if signal.direction == 'long' and not self.position:
                self.buy()
            elif signal.direction == 'short' and not self.position:
                self.sell()

bt = Backtest(df, HybridStrategy, cash=10000)
stats = bt.run()
```

## Configuration Reference

### InfluencerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `htf_timeframe` | "4H" | Higher timeframe |
| `ltf_timeframe` | "30min" | Lower timeframe |
| `min_confluence_score` | 4.0 | Minimum score to trade (0-5) |
| `volume_multiplier` | 2.0 | Volume spike threshold |
| `fib_levels` | [0.382, 0.5, 0.618] | Fibonacci levels to watch |
| `atr_period` | 14 | ATR for stop calculation |
| `risk_per_trade` | 0.01 | Risk per trade |
| `require_all_factors` | False | Require all 5 factors (strict) |

### PluginConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Class name | Plugin identifier |
| `enabled` | True | Enable/disable plugin |
| `weight` | 1.0 | Weight in confluence scoring |
| `min_confidence` | 0.5 | Minimum confidence for signals |
| `priority` | 10 | Execution priority |

## Best Practices

1. **Always use volume confirmation** - It's the only non-price-derived indicator
2. **Wait for 4H pullback** - Don't chase entries against the HTF bias
3. **VWAP reclaim is key** - Institutions use VWAP as benchmark
4. **Multiple plugins = higher confidence** - SMC + Influencer agreeing = stronger signal
5. **Filter by quality** - Only trade HIGH quality signals initially
6. **Backtest your configuration** - Adjust weights based on historical performance

## Troubleshooting

### No signals generated

- Check that both HTF and LTF DataFrames have sufficient bars
- Verify volume multiplier isn't too high (try 1.5 instead of 2.0)
- Lower min_confluence_score temporarily for testing

### Too many false signals

- Increase min_confluence_score to 4.5 or 5.0
- Require volume confirmation on all trades
- Use confluence aggregator (2+ plugins must agree)

### Plugin conflicts

- Check plugin priorities (lower = executes first)
- Adjust weights to emphasize preferred strategy
- Use selective evaluation to test plugins independently
