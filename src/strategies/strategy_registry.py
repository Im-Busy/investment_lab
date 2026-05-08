"""
Strategy Plugin Registry

Plugin architecture for integrating influencer framework with existing SMC strategies.

Architecture:
1. Core strategies (SMC) run independently
2. Influencer framework acts as a filter/confluence layer
3. Plugin registry manages strategy registration and execution
4. Strategies can be combined for higher confluence

Example:
    >>> registry = StrategyRegistry()
    >>> registry.register(SMCReversalPlugin())
    >>> registry.register(InfluencerMTFPlugin())
    >>> signals = registry.evaluate(df)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class StrategyType(Enum):
    """Strategy type classification."""

    SMC = "smc"  # Smart Money Concepts
    INFLUENCER = "influencer"  # Influencer framework
    TECHNICAL = "technical"  # Traditional technical analysis
    HYBRID = "hybrid"  # Combination


class SignalQuality(Enum):
    """Signal quality rating."""

    LOW = "low"  # Single factor
    MEDIUM = "medium"  # 2-3 factors
    HIGH = "high"  # 4+ factors


@dataclass
class StrategySignal:
    """
    Unified strategy signal.

    Attributes:
        timestamp: Signal timestamp
        strategy_name: Name of generating strategy
        strategy_type: Type of strategy
        direction: Trade direction
        entry_price: Suggested entry
        stop_loss: Stop loss price
        take_profit: Take profit target
        confidence: Confidence score (0.0 to 1.0)
        quality: Signal quality rating
        factors: Contributing factors
        metadata: Additional information
    """

    timestamp: pd.Timestamp
    strategy_name: str
    strategy_type: StrategyType
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    quality: SignalQuality
    factors: List[str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class PluginConfig:
    """
    Plugin configuration.

    Attributes:
        name: Plugin name
        enabled: Whether plugin is active
        weight: Weight in final confluence (0.0 to 1.0)
        min_confidence: Minimum confidence to generate signals
        priority: Execution priority (lower = first)
    """

    name: str
    enabled: bool = True
    weight: float = 1.0
    min_confidence: float = 0.5
    priority: int = 10


class StrategyPlugin(ABC):
    """
    Abstract base class for strategy plugins.

    All strategy plugins must implement:
    - evaluate(): Generate signals from DataFrame
    - get_name(): Return plugin name
    - get_type(): Return strategy type
    """

    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize strategy plugin.

        Args:
            config: Plugin configuration
        """
        self.config = config or PluginConfig(name=self.__class__.__name__)
        self._enabled = True

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, **kwargs) -> List[StrategySignal]:
        """
        Evaluate strategy on DataFrame.

        Args:
            df: OHLCV DataFrame
            **kwargs: Additional arguments (e.g., htf_df for MTF strategies)

        Returns:
            List of StrategySignal objects
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return plugin name."""
        pass

    @abstractmethod
    def get_type(self) -> StrategyType:
        """Return strategy type."""
        pass

    def enable(self) -> None:
        """Enable plugin."""
        self._enabled = True

    def disable(self) -> None:
        """Disable plugin."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled and self.config.enabled

    def set_weight(self, weight: float) -> None:
        """Set plugin weight."""
        if 0.0 <= weight <= 1.0:
            self.config.weight = weight
        else:
            raise ValueError("Weight must be between 0.0 and 1.0")


class SMCReversalPlugin(StrategyPlugin):
    """
    SMC Reversal Strategy Plugin

    Integrates SMC reversal patterns with the plugin system.
    Wraps the existing SMCReversalStrategy for plugin compatibility.
    """

    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize SMC Reversal Plugin."""
        super().__init__(config or PluginConfig(name="SMC_Reversal", weight=1.0, priority=1))
        self._smc_strategy = None

    def evaluate(self, df: pd.DataFrame, **kwargs) -> List[StrategySignal]:
        """Evaluate SMC reversal patterns."""
        if not self.is_enabled():
            return []

        try:
            from .smc_reversal import SMCConfig, SMCReversalStrategy

            # Initialize if not already done
            if self._smc_strategy is None:
                self._smc_strategy = SMCReversalStrategy(
                    SMCConfig(
                        session_start="00:00",
                        session_end="08:00",
                        risk_per_trade=0.01,
                    )
                )

            # Run SMC strategy
            smc_signals = self._smc_strategy.run(df)

            # Convert to StrategySignal format
            strategy_signals = []
            for signal in smc_signals:
                strategy_signal = StrategySignal(
                    timestamp=signal.timestamp,
                    strategy_name="SMC_Reversal",
                    strategy_type=StrategyType.SMC,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.target_final,
                    confidence=signal.confidence,
                    quality=SignalQuality.HIGH
                    if signal.confidence >= 0.7
                    else SignalQuality.MEDIUM,
                    factors=["liquidity_sweep", "mss", "ifvg"],
                    metadata=signal.to_dict(),
                )
                strategy_signals.append(strategy_signal)

            return strategy_signals

        except ImportError:
            # SMC strategy not available
            return []
        except Exception as e:
            print(f"SMC Reversal Plugin error: {e}")
            return []

    def get_name(self) -> str:
        """Return plugin name."""
        return "SMC_Reversal"

    def get_type(self) -> StrategyType:
        """Return strategy type."""
        return StrategyType.SMC


class InfluencerMTFPlugin(StrategyPlugin):
    """
    Influencer Multi-Timeframe Plugin

    Integrates the influencer framework (4H bias + 30min entry + volume).
    Wraps InfluencerConfluenceScorer for plugin compatibility.
    """

    def __init__(
        self,
        config: Optional[PluginConfig] = None,
        htf_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize Influencer MTF Plugin.

        Args:
            config: Plugin configuration
            htf_df: Higher timeframe DataFrame (for MTF analysis)
        """
        super().__init__(config or PluginConfig(name="Influencer_MTF", weight=1.0, priority=2))
        self._htf_df = htf_df
        self._scorer = None

    def set_htf_dataframe(self, df: pd.DataFrame) -> None:
        """Set higher timeframe DataFrame for MTF analysis."""
        self._htf_df = df

    def evaluate(self, df: pd.DataFrame, **kwargs) -> List[StrategySignal]:
        """Evaluate influencer MTF framework."""
        if not self.is_enabled():
            return []

        if self._htf_df is None:
            return []

        try:
            from .influencer_confluence import InfluencerConfig, InfluencerConfluenceScorer

            # Initialize scorer if not already done
            if self._scorer is None:
                self._scorer = InfluencerConfluenceScorer(
                    InfluencerConfig(
                        htf_timeframe="4H",
                        ltf_timeframe="30min",
                        min_confluence_score=4.0,
                    )
                )

            # Get htf_df from kwargs or use stored
            htf_df = kwargs.get("htf_df", self._htf_df)

            # Run confluence scorer
            confluence_signals = self._scorer.scan(htf_df, df)

            # Convert to StrategySignal format
            strategy_signals = []
            for signal in confluence_signals:
                if signal.direction == "none":
                    continue

                strategy_signal = StrategySignal(
                    timestamp=signal.timestamp,
                    strategy_name="Influencer_MTF",
                    strategy_type=StrategyType.INFLUENCER,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence=signal.confluence_score / 5.0,  # Normalize to 0-1
                    quality=self._get_quality_from_score(signal.confluence_score),
                    factors=list(signal.breakdown.keys()),
                    metadata={
                        "htf_bias": signal.htf_bias.direction.value,
                        "ltf_bias": signal.ltf_bias.direction.value if signal.ltf_bias else "N/A",
                        "fib_level": signal.fib_level,
                        "vwap_status": signal.vwap_status,
                    },
                )
                strategy_signals.append(strategy_signal)

            return strategy_signals

        except Exception as e:
            print(f"Influencer MTF Plugin error: {e}")
            return []

    def _get_quality_from_score(self, score: float) -> SignalQuality:
        """Convert confluence score to quality rating."""
        if score >= 4.5:
            return SignalQuality.HIGH
        elif score >= 3.0:
            return SignalQuality.MEDIUM
        else:
            return SignalQuality.LOW

    def get_name(self) -> str:
        """Return plugin name."""
        return "Influencer_MTF"

    def get_type(self) -> StrategyType:
        """Return strategy type."""
        return StrategyType.INFLUENCER


class ConfluenceAggregatorPlugin(StrategyPlugin):
    """
    Confluence Aggregator Plugin

    Combines signals from multiple plugins into unified confluence score.
    Acts as a meta-plugin that evaluates agreement between strategies.
    """

    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize Confluence Aggregator."""
        super().__init__(
            config or PluginConfig(name="Confluence_Aggregator", weight=1.0, priority=100)
        )
        self._registered_plugins: List[StrategyPlugin] = []

    def register_plugin(self, plugin: StrategyPlugin) -> None:
        """Register a plugin for confluence evaluation."""
        if plugin not in self._registered_plugins:
            self._registered_plugins.append(plugin)

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin by name."""
        self._registered_plugins = [
            p for p in self._registered_plugins if p.get_name() != plugin_name
        ]

    def evaluate(self, df: pd.DataFrame, **kwargs) -> List[StrategySignal]:
        """Evaluate confluence across all registered plugins."""
        if not self.is_enabled() or not self._registered_plugins:
            return []

        # Collect signals from all plugins
        all_signals: Dict[str, List[StrategySignal]] = {}

        for plugin in self._registered_plugins:
            if plugin.is_enabled():
                signals = plugin.evaluate(df, **kwargs)
                if signals:
                    all_signals[plugin.get_name()] = signals

        if not all_signals:
            return []

        # Find confluence (same direction signals from multiple plugins)
        confluence_signals = self._aggregate_confluence(all_signals, df)

        return confluence_signals

    def _aggregate_confluence(
        self,
        all_signals: Dict[str, List[StrategySignal]],
        df: pd.DataFrame,
    ) -> List[StrategySignal]:
        """Aggregate signals into confluence signals."""
        if not all_signals:
            return []

        # Group by direction and approximate timestamp
        long_plugins = set()
        short_plugins = set()
        long_signals = []
        short_signals = []

        for plugin_name, signals in all_signals.items():
            for signal in signals:
                if signal.direction == "long":
                    long_plugins.add(plugin_name)
                    long_signals.append(signal)
                elif signal.direction == "short":
                    short_plugins.add(plugin_name)
                    short_signals.append(signal)

        confluence_results = []

        # Create confluence signal if 2+ plugins agree
        if len(long_plugins) >= 2 and long_signals:
            confluence_results.append(
                self._create_confluence_signal(
                    signals=long_signals,
                    direction="long",
                    plugins=long_plugins,
                    df=df,
                )
            )

        if len(short_plugins) >= 2 and short_signals:
            confluence_results.append(
                self._create_confluence_signal(
                    signals=short_signals,
                    direction="short",
                    plugins=short_plugins,
                    df=df,
                )
            )

        return confluence_results

    def _create_confluence_signal(
        self,
        signals: List[StrategySignal],
        direction: str,
        plugins: set,
        df: pd.DataFrame,
    ) -> StrategySignal:
        """Create aggregated confluence signal."""
        # Average entry/stop/target
        avg_entry = sum(s.entry_price for s in signals) / len(signals)
        avg_stop = sum(s.stop_loss for s in signals) / len(signals)
        avg_target = sum(s.take_profit for s in signals) / len(signals)

        # Weighted confidence
        total_weight = sum(s.confidence * self._get_plugin_weight(s.strategy_name) for s in signals)
        avg_confidence = total_weight / len(signals) if signals else 0

        # Quality based on plugin count
        if len(plugins) >= 3:
            quality = SignalQuality.HIGH
        elif len(plugins) >= 2:
            quality = SignalQuality.MEDIUM
        else:
            quality = SignalQuality.LOW

        timestamp = df.index[-1] if len(df) > 0 else pd.Timestamp.now()

        return StrategySignal(
            timestamp=timestamp,
            strategy_name="Confluence_Aggregator",
            strategy_type=StrategyType.HYBRID,
            direction=direction,
            entry_price=avg_entry,
            stop_loss=avg_stop,
            take_profit=avg_target,
            confidence=min(1.0, avg_confidence + len(plugins) * 0.1),  # Bonus for multiple plugins
            quality=quality,
            factors=list(plugins),
            metadata={
                "plugin_count": len(plugins),
                "plugins": list(plugins),
                "individual_confidences": [s.confidence for s in signals],
            },
        )

    def _get_plugin_weight(self, plugin_name: str) -> float:
        """Get weight for a plugin."""
        # Could be customized per plugin
        return 1.0

    def get_name(self) -> str:
        """Return plugin name."""
        return "Confluence_Aggregator"

    def get_type(self) -> StrategyType:
        """Return strategy type."""
        return StrategyType.HYBRID


class StrategyRegistry:
    """
    Strategy Plugin Registry

    Central registry for all strategy plugins.
    Manages plugin lifecycle, execution, and signal aggregation.

    Example:
        >>> registry = StrategyRegistry()
        >>> registry.register(SMCReversalPlugin())
        >>> registry.register(InfluencerMTFPlugin())
        >>> signals = registry.evaluate(df, htf_df=htf_df)
    """

    def __init__(self):
        """Initialize strategy registry."""
        self._plugins: Dict[str, StrategyPlugin] = {}
        self._execution_order: List[str] = []

    def register(self, plugin: StrategyPlugin, priority: Optional[int] = None) -> None:
        """
        Register a strategy plugin.

        Args:
            plugin: Strategy plugin instance
            priority: Execution priority (lower = first)
        """
        name = plugin.get_name()
        self._plugins[name] = plugin

        if priority is not None:
            plugin.config.priority = priority

        # Sort by priority
        self._execution_order = sorted(
            self._plugins.keys(),
            key=lambda n: self._plugins[n].config.priority,
        )

    def unregister(self, plugin_name: str) -> None:
        """Unregister a plugin by name."""
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            self._execution_order.remove(plugin_name)

    def enable(self, plugin_name: str) -> None:
        """Enable a plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].enable()

    def disable(self, plugin_name: str) -> None:
        """Disable a plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].disable()

    def set_weight(self, plugin_name: str, weight: float) -> None:
        """Set plugin weight."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].set_weight(weight)

    def evaluate(
        self,
        df: pd.DataFrame,
        plugin_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[StrategySignal]:
        """
        Evaluate all registered plugins.

        Args:
            df: OHLCV DataFrame
            plugin_names: Specific plugins to evaluate (default: all enabled)
            **kwargs: Additional arguments passed to plugins

        Returns:
            List of StrategySignal objects
        """
        signals = []

        # Determine which plugins to evaluate
        if plugin_names:
            plugins_to_eval = [self._plugins[n] for n in plugin_names if n in self._plugins]
        else:
            plugins_to_eval = [
                self._plugins[name]
                for name in self._execution_order
                if self._plugins[name].is_enabled()
            ]

        # Evaluate each plugin
        for plugin in plugins_to_eval:
            try:
                plugin_signals = plugin.evaluate(df, **kwargs)
                signals.extend(plugin_signals)
            except Exception as e:
                print(f"Plugin {plugin.get_name()} error: {e}")

        return signals

    def evaluate_confluence(
        self,
        df: pd.DataFrame,
        min_plugins: int = 2,
        **kwargs,
    ) -> List[StrategySignal]:
        """
        Evaluate confluence across multiple plugins.

        Only returns signals where 2+ plugins agree on direction.

        Args:
            df: OHLCV DataFrame
            min_plugins: Minimum plugins that must agree
            **kwargs: Additional arguments

        Returns:
            List of confluence signals
        """
        # Get all signals
        all_signals = self.evaluate(df, **kwargs)

        if not all_signals:
            return []

        # Group by direction and timestamp (same bar)
        long_signals = [s for s in all_signals if s.direction == "long"]
        short_signals = [s for s in all_signals if s.direction == "short"]

        confluence_signals = []

        # Check for long confluence
        if len(long_signals) >= min_plugins:
            confluence_signals.append(self._create_confluence_signal(long_signals, "long", df))

        # Check for short confluence
        if len(short_signals) >= min_plugins:
            confluence_signals.append(self._create_confluence_signal(short_signals, "short", df))

        return confluence_signals

    def _create_confluence_signal(
        self,
        signals: List[StrategySignal],
        direction: str,
        df: pd.DataFrame,
    ) -> StrategySignal:
        """Create aggregated confluence signal."""
        plugins = set(s.strategy_name for s in signals)

        # Average metrics
        avg_entry = sum(s.entry_price for s in signals) / len(signals)
        avg_stop = sum(s.stop_loss for s in signals) / len(signals)
        avg_target = sum(s.take_profit for s in signals) / len(signals)
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        # Bonus for multiple plugins
        plugin_bonus = min(0.2, len(plugins) * 0.05)
        final_confidence = min(1.0, avg_confidence + plugin_bonus)

        return StrategySignal(
            timestamp=df.index[-1] if len(df) > 0 else pd.Timestamp.now(),
            strategy_name="Confluence",
            strategy_type=StrategyType.HYBRID,
            direction=direction,
            entry_price=avg_entry,
            stop_loss=avg_stop,
            take_profit=avg_target,
            confidence=final_confidence,
            quality=SignalQuality.HIGH if len(plugins) >= 3 else SignalQuality.MEDIUM,
            factors=list(plugins),
            metadata={
                "plugin_count": len(plugins),
                "plugins": list(plugins),
                "confluence_type": "multi_plugin",
            },
        )

    def get_plugin_names(self) -> List[str]:
        """Get list of registered plugin names."""
        return list(self._plugins.keys())

    def get_plugin_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all registered plugins."""
        return {
            name: {
                "type": plugin.get_type().value,
                "enabled": plugin.is_enabled(),
                "weight": plugin.config.weight,
                "priority": plugin.config.priority,
                "min_confidence": plugin.config.min_confidence,
            }
            for name, plugin in self._plugins.items()
        }
