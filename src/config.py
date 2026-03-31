"""
Trading System Configuration

Centralized configuration management for the trading pattern detection system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml


@dataclass
class PatternConfig:
    """Configuration for individual pattern detection."""
    enabled: bool = True
    lookback: int = 21
    min_bars: int = 20
    tolerance: float = 0.05
    entry_offset: float = 0.01
    stop_offset: float = 0.01
    volume_filter: bool = False
    confidence_base: float = 0.50


@dataclass
class RiskConfig:
    """Risk management configuration."""
    initial_equity: float = 100000.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.20
    max_open_positions: int = 5
    max_daily_loss: float = 0.03
    max_weekly_loss: float = 0.06
    max_monthly_loss: float = 0.10
    max_portfolio_heat: float = 0.06
    position_sizing_method: str = 'fixed_fractional'
    atr_sizing_multiplier: float = 2.0


@dataclass
class SignalConfig:
    """Signal generation configuration."""
    min_confidence: float = 0.50
    min_confluence_patterns: int = 2
    signal_decay_rate: float = 0.05
    signal_validity_bars: Dict[str, int] = field(default_factory=lambda: {
        'basic': 5,
        'harmonic': 10,
        'complex': 20,
        'classic': 15
    })
    confluence_bonuses: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0,
        2: 0.10,
        3: 0.20,
        4: 0.30
    })
    trend_alignment_bonus: float = 0.05


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    commission_per_trade: float = 0.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    use_take_profit_1: bool = True
    use_take_profit_2: bool = False
    use_take_profit_3: bool = False


@dataclass
class RegimeConfig:
    """Market regime detection configuration."""
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


@dataclass
class TradingConfig:
    """
    Main trading configuration container.
    
    Contains all configuration settings for the trading system.
    """
    # Pattern configurations by category
    basic_patterns: Dict[str, PatternConfig] = field(default_factory=dict)
    harmonic_patterns: Dict[str, PatternConfig] = field(default_factory=dict)
    complex_patterns: Dict[str, PatternConfig] = field(default_factory=dict)
    classic_patterns: Dict[str, PatternConfig] = field(default_factory=dict)
    
    # System configurations
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    
    # Timeframe settings
    default_timeframe: str = 'daily'
    htf_timeframe: str = 'weekly'
    ltf_timeframe: str = 'hourly'
    
    # Data settings
    data_directory: str = 'data'
    output_directory: str = 'output'
    
    def __post_init__(self):
        """Initialize default pattern configurations."""
        if not self.basic_patterns:
            self.basic_patterns = self._default_basic_patterns()
        if not self.harmonic_patterns:
            self.harmonic_patterns = self._default_harmonic_patterns()
        if not self.complex_patterns:
            self.complex_patterns = self._default_complex_patterns()
        if not self.classic_patterns:
            self.classic_patterns = self._default_classic_patterns()
    
    def _default_basic_patterns(self) -> Dict[str, PatternConfig]:
        """Default configurations for basic patterns."""
        return {
            'msl': PatternConfig(
                enabled=True,
                lookback=3,
                min_bars=5,
                confidence_base=0.55
            ),
            'matching_lows': PatternConfig(
                enabled=True,
                lookback=5,
                min_bars=10,
                tolerance=0.05,
                confidence_base=0.50
            ),
            'nr7id': PatternConfig(
                enabled=True,
                lookback=7,
                min_bars=10,
                confidence_base=0.45
            ),
            'n_bar_decline': PatternConfig(
                enabled=True,
                lookback=21,
                min_bars=25,
                confidence_base=0.50
            ),
            'floor_pivot': PatternConfig(
                enabled=True,
                lookback=1,
                min_bars=5,
                confidence_base=0.45
            )
        }
    
    def _default_harmonic_patterns(self) -> Dict[str, PatternConfig]:
        """Default configurations for harmonic patterns."""
        return {
            'gartley': PatternConfig(
                enabled=True,
                lookback=50,
                min_bars=30,
                tolerance=0.05,
                confidence_base=0.60
            ),
            'abc': PatternConfig(
                enabled=True,
                lookback=30,
                min_bars=15,
                tolerance=0.05,
                confidence_base=0.55
            ),
            'symmetric_triangle': PatternConfig(
                enabled=True,
                lookback=30,
                min_bars=20,
                confidence_base=0.50
            ),
            'donchian': PatternConfig(
                enabled=True,
                lookback=20,
                min_bars=25,
                confidence_base=0.50
            ),
            'bollinger': PatternConfig(
                enabled=True,
                lookback=20,
                min_bars=25,
                confidence_base=0.45
            )
        }
    
    def _default_complex_patterns(self) -> Dict[str, PatternConfig]:
        """Default configurations for complex patterns."""
        return {
            'cup_handle': PatternConfig(
                enabled=True,
                lookback=100,
                min_bars=50,
                tolerance=0.10,
                confidence_base=0.65
            ),
            'head_shoulders': PatternConfig(
                enabled=True,
                lookback=100,
                min_bars=40,
                tolerance=0.10,
                confidence_base=0.65
            ),
            'spike_ledge': PatternConfig(
                enabled=True,
                lookback=20,
                min_bars=15,
                tolerance=0.02,
                confidence_base=0.55
            ),
            'three_hills': PatternConfig(
                enabled=True,
                lookback=100,
                min_bars=50,
                tolerance=0.05,
                confidence_base=0.60
            ),
            'parabolic_arc': PatternConfig(
                enabled=True,
                lookback=50,
                min_bars=30,
                confidence_base=0.55
            )
        }
    
    def _default_classic_patterns(self) -> Dict[str, PatternConfig]:
        """Default configurations for classic patterns."""
        return {
            'double_top': PatternConfig(
                enabled=True,
                lookback=60,
                min_bars=20,
                tolerance=0.05,
                confidence_base=0.60
            ),
            'double_bottom': PatternConfig(
                enabled=True,
                lookback=60,
                min_bars=20,
                tolerance=0.05,
                confidence_base=0.60
            ),
            'trader_vic_2b': PatternConfig(
                enabled=True,
                lookback=21,
                min_bars=30,
                confidence_base=0.55
            ),
            'triple_top': PatternConfig(
                enabled=True,
                lookback=100,
                min_bars=50,
                tolerance=0.03,
                confidence_base=0.65
            ),
            'dead_cat_bounce': PatternConfig(
                enabled=True,
                lookback=20,
                min_bars=25,
                tolerance=0.15,
                confidence_base=0.50
            )
        }
    
    def get_pattern_config(self, pattern_name: str) -> Optional[PatternConfig]:
        """Get configuration for a specific pattern."""
        pattern_name_lower = pattern_name.lower().replace(' ', '_').replace('-', '_')
        
        for patterns_dict in [
            self.basic_patterns,
            self.harmonic_patterns,
            self.complex_patterns,
            self.classic_patterns
        ]:
            for key, config in patterns_dict.items():
                if key.lower() in pattern_name_lower or pattern_name_lower in key.lower():
                    return config
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def pattern_dict_to_dict(d: Dict[str, PatternConfig]) -> Dict[str, Dict]:
            return {k: vars(v) for k, v in d.items()}
        
        return {
            'basic_patterns': pattern_dict_to_dict(self.basic_patterns),
            'harmonic_patterns': pattern_dict_to_dict(self.harmonic_patterns),
            'complex_patterns': pattern_dict_to_dict(self.complex_patterns),
            'classic_patterns': pattern_dict_to_dict(self.classic_patterns),
            'risk': vars(self.risk),
            'signal': {
                'min_confidence': self.signal.min_confidence,
                'min_confluence_patterns': self.signal.min_confluence_patterns,
                'signal_decay_rate': self.signal.signal_decay_rate,
                'signal_validity_bars': self.signal.signal_validity_bars,
                'confluence_bonuses': self.signal.confluence_bonuses,
                'trend_alignment_bonus': self.signal.trend_alignment_bonus,
            },
            'backtest': vars(self.backtest),
            'regime': vars(self.regime),
            'default_timeframe': self.default_timeframe,
            'htf_timeframe': self.htf_timeframe,
            'ltf_timeframe': self.ltf_timeframe,
            'data_directory': self.data_directory,
            'output_directory': self.output_directory,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """Create configuration from dictionary."""
        def dict_to_pattern_dict(d: Dict[str, Dict]) -> Dict[str, PatternConfig]:
            return {k: PatternConfig(**v) for k, v in d.items()}
        
        config = cls()
        
        if 'basic_patterns' in config_dict:
            config.basic_patterns = dict_to_pattern_dict(config_dict['basic_patterns'])
        if 'harmonic_patterns' in config_dict:
            config.harmonic_patterns = dict_to_pattern_dict(config_dict['harmonic_patterns'])
        if 'complex_patterns' in config_dict:
            config.complex_patterns = dict_to_pattern_dict(config_dict['complex_patterns'])
        if 'classic_patterns' in config_dict:
            config.classic_patterns = dict_to_pattern_dict(config_dict['classic_patterns'])
        
        if 'risk' in config_dict:
            config.risk = RiskConfig(**config_dict['risk'])
        if 'signal' in config_dict:
            signal_dict = config_dict['signal']
            config.signal = SignalConfig(
                min_confidence=signal_dict.get('min_confidence', 0.50),
                min_confluence_patterns=signal_dict.get('min_confluence_patterns', 2),
                signal_decay_rate=signal_dict.get('signal_decay_rate', 0.05),
                signal_validity_bars=signal_dict.get('signal_validity_bars', config.signal.signal_validity_bars),
                confluence_bonuses=signal_dict.get('confluence_bonuses', config.signal.confluence_bonuses),
                trend_alignment_bonus=signal_dict.get('trend_alignment_bonus', 0.05),
            )
        if 'backtest' in config_dict:
            config.backtest = BacktestConfig(**config_dict['backtest'])
        if 'regime' in config_dict:
            config.regime = RegimeConfig(**config_dict['regime'])
        
        config.default_timeframe = config_dict.get('default_timeframe', 'daily')
        config.htf_timeframe = config_dict.get('htf_timeframe', 'weekly')
        config.ltf_timeframe = config_dict.get('ltf_timeframe', 'hourly')
        config.data_directory = config_dict.get('data_directory', 'data')
        config.output_directory = config_dict.get('output_directory', 'output')
        
        return config
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TradingConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return cls()
        
        with open(filepath, 'r') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Default configuration instance
DEFAULT_CONFIG = TradingConfig()
