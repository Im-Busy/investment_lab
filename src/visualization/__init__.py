"""
Visualization Module

Provides chart generation and report generation capabilities using
quantstats and mplfinance for both backtesting.py and custom engine results.
"""

from .tearsheet import TearsheetGenerator
from .charts import ChartGenerator
from .pattern_markers import PatternMarkerGenerator
from .report import ReportGenerator

__all__ = [
    'TearsheetGenerator',
    'ChartGenerator',
    'PatternMarkerGenerator',
    'ReportGenerator'
]
