"""
Contribution Analysis System

This module provides tools for analyzing pattern contributions in multi-pattern trading strategies.
It includes four layers of analysis:

1. Signal Event Log (Layer 1) - Record all pattern detection events
2. Trade Attributor (Layer 2) - Match trades to contributing patterns
3. Ablation Engine (Layer 3) - Leave-one-out contribution analysis
4. Synergy Analyzer (Layer 4) - Pairwise pattern interactions

Usage:
    from src.analysis import SignalEventLog, TradeAttributor, AblationEngine, SynergyAnalyzer
    from src.analysis import ContributionReport, create_contribution_charts
"""

from .signal_event_log import SignalEvent, SignalEventLog
from .trade_attributor import AttributedTrade, TradeAttributor
from .ablation_engine import AblationResult, AblationEngine
from .synergy_analyzer import SynergyResult, SynergyAnalyzer
from .contribution_report import ContributionReport
from .contribution_charts import create_contribution_charts

__all__ = [
    # Layer 1
    "SignalEvent",
    "SignalEventLog",
    
    # Layer 2
    "AttributedTrade",
    "TradeAttributor",
    
    # Layer 3
    "AblationResult",
    "AblationEngine",
    
    # Layer 4
    "SynergyResult",
    "SynergyAnalyzer",
    
    # Aggregation & Reporting
    "ContributionReport",
    "create_contribution_charts",
]