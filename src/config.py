"""Trading System Configuration

Centralized configuration management for the trading pattern detection system.
Uses Pydantic BaseSettings for env-var overrides, field validation, and .env support.

Environment variable format: BT_<SECTION>_<FIELD>
Example: BT_BACKTEST_COMMISSION_PCT=0.002
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PatternConfig(BaseModel):
    """Configuration for individual pattern detection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    lookback: int = 21
    min_bars: int = 20
    tolerance: float = 0.05
    entry_offset: float = 0.01
    stop_offset: float = 0.01
    volume_filter: bool = False
    confidence_base: float = 0.50


class RiskConfig(BaseModel):
    """Risk management configuration."""

    model_config = ConfigDict(extra="forbid")

    initial_equity: float = 100000.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.20
    max_open_positions: int = 5
    max_daily_loss: float = 0.03
    max_weekly_loss: float = 0.06
    max_monthly_loss: float = 0.10
    max_portfolio_heat: float = 0.06
    position_sizing_method: str = "fixed_fractional"
    atr_sizing_multiplier: float = 2.0


class SignalConfig(BaseModel):
    """Signal generation configuration."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = 0.50
    min_confluence_patterns: int = 2
    signal_decay_rate: float = 0.05
    signal_validity_bars: dict[str, int] = Field(
        default_factory=lambda: {"basic": 5, "harmonic": 10, "complex": 20, "classic": 15}
    )
    confluence_bonuses: dict[int, float] = Field(
        default_factory=lambda: {1: 0.0, 2: 0.10, 3: 0.20, 4: 0.30}
    )
    trend_alignment_bonus: float = 0.05


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    model_config = ConfigDict(extra="forbid")

    commission_per_trade: float = 0.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    use_take_profit_1: bool = True
    use_take_profit_2: bool = False
    use_take_profit_3: bool = False


class RegimeConfig(BaseModel):
    """Market regime detection configuration."""

    model_config = ConfigDict(extra="forbid")

    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_strong_threshold: float = 30.0
    atr_period: int = 14
    atr_lookback: int = 50
    atr_high_mult: float = 1.5
    atr_low_mult: float = 0.75
    sma_period: int = 50
    sma_fast_period: int = 20
    regime_adaptation: bool = True


_EMPTY_PATTERN_DICT: dict[str, PatternConfig] = {}


class TradingConfig(BaseSettings):
    """Main trading configuration with env-var override support.

    Environment variables use prefix BT_ and double-underscore for nesting.
    Example: BT_RISK__RISK_PER_TRADE=0.03 overrides risk.risk_per_trade.
    """

    model_config = SettingsConfigDict(
        env_prefix="BT_",
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    basic_patterns: dict[str, PatternConfig] = Field(default_factory=lambda: DEFAULT_BASIC_PATTERNS)
    harmonic_patterns: dict[str, PatternConfig] = Field(
        default_factory=lambda: DEFAULT_HARMONIC_PATTERNS
    )
    complex_patterns: dict[str, PatternConfig] = Field(
        default_factory=lambda: DEFAULT_COMPLEX_PATTERNS
    )
    classic_patterns: dict[str, PatternConfig] = Field(
        default_factory=lambda: DEFAULT_CLASSIC_PATTERNS
    )

    risk: RiskConfig = Field(default_factory=RiskConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)

    default_timeframe: str = "daily"
    htf_timeframe: str = "weekly"
    ltf_timeframe: str = "hourly"

    data_directory: str = "data"
    output_directory: str = "output"

    def get_pattern_config(self, pattern_name: str) -> PatternConfig | None:
        """Get configuration for a specific pattern."""
        pattern_name_lower = pattern_name.lower().replace(" ", "_").replace("-", "_")

        for patterns_dict in [
            self.basic_patterns,
            self.harmonic_patterns,
            self.complex_patterns,
            self.classic_patterns,
        ]:
            for key, config in patterns_dict.items():
                if key.lower() in pattern_name_lower or pattern_name_lower in key.lower():
                    return config

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary (backward-compatible)."""

        def _patterns_to_dict(d: dict[str, PatternConfig]) -> dict[str, dict[str, Any]]:
            return {k: v.model_dump() for k, v in d.items()}

        return {
            "basic_patterns": _patterns_to_dict(self.basic_patterns),
            "harmonic_patterns": _patterns_to_dict(self.harmonic_patterns),
            "complex_patterns": _patterns_to_dict(self.complex_patterns),
            "classic_patterns": _patterns_to_dict(self.classic_patterns),
            "risk": self.risk.model_dump(),
            "signal": self.signal.model_dump(),
            "backtest": self.backtest.model_dump(),
            "regime": self.regime.model_dump(),
            "default_timeframe": self.default_timeframe,
            "htf_timeframe": self.htf_timeframe,
            "ltf_timeframe": self.ltf_timeframe,
            "data_directory": self.data_directory,
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> TradingConfig:
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        if path.suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            import json

            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> TradingConfig:
        """Load configuration from file."""
        path = Path(filepath)

        if not path.exists():
            return cls()

        if path.suffix in (".yaml", ".yml"):
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        else:
            import json

            with open(path) as f:
                config_dict = json.load(f)

        return cls.from_dict(config_dict)


# Default pattern configurations
DEFAULT_BASIC_PATTERNS: dict[str, PatternConfig] = {
    "msl": PatternConfig(enabled=True, lookback=3, min_bars=5, confidence_base=0.55),
    "matching_lows": PatternConfig(
        enabled=True, lookback=5, min_bars=10, tolerance=0.05, confidence_base=0.50
    ),
    "nr7id": PatternConfig(enabled=True, lookback=7, min_bars=10, confidence_base=0.45),
    "n_bar_decline": PatternConfig(enabled=True, lookback=21, min_bars=25, confidence_base=0.50),
    "floor_pivot": PatternConfig(enabled=True, lookback=1, min_bars=5, confidence_base=0.45),
}

DEFAULT_HARMONIC_PATTERNS: dict[str, PatternConfig] = {
    "gartley": PatternConfig(
        enabled=True, lookback=50, min_bars=30, tolerance=0.05, confidence_base=0.60
    ),
    "abc": PatternConfig(
        enabled=True, lookback=30, min_bars=15, tolerance=0.05, confidence_base=0.55
    ),
    "symmetric_triangle": PatternConfig(
        enabled=True, lookback=30, min_bars=20, confidence_base=0.50
    ),
    "donchian": PatternConfig(enabled=True, lookback=20, min_bars=25, confidence_base=0.50),
    "bollinger": PatternConfig(enabled=True, lookback=20, min_bars=25, confidence_base=0.45),
}

DEFAULT_COMPLEX_PATTERNS: dict[str, PatternConfig] = {
    "cup_handle": PatternConfig(
        enabled=True, lookback=100, min_bars=50, tolerance=0.10, confidence_base=0.65
    ),
    "head_shoulders": PatternConfig(
        enabled=True, lookback=100, min_bars=40, tolerance=0.10, confidence_base=0.65
    ),
    "spike_ledge": PatternConfig(
        enabled=True, lookback=20, min_bars=15, tolerance=0.02, confidence_base=0.55
    ),
    "three_hills": PatternConfig(
        enabled=True, lookback=100, min_bars=50, tolerance=0.05, confidence_base=0.60
    ),
    "parabolic_arc": PatternConfig(enabled=True, lookback=50, min_bars=30, confidence_base=0.55),
}

DEFAULT_CLASSIC_PATTERNS: dict[str, PatternConfig] = {
    "double_top": PatternConfig(
        enabled=True, lookback=60, min_bars=20, tolerance=0.05, confidence_base=0.60
    ),
    "double_bottom": PatternConfig(
        enabled=True, lookback=60, min_bars=20, tolerance=0.05, confidence_base=0.60
    ),
    "trader_vic_2b": PatternConfig(enabled=True, lookback=21, min_bars=30, confidence_base=0.55),
    "triple_top": PatternConfig(
        enabled=True, lookback=100, min_bars=50, tolerance=0.03, confidence_base=0.65
    ),
    "dead_cat_bounce": PatternConfig(
        enabled=True, lookback=20, min_bars=25, tolerance=0.15, confidence_base=0.50
    ),
}

DEFAULT_CONFIG = TradingConfig()
