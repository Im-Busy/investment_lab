"""Pattern reliability weights from authoritative sources.

Sources:
- Kirkpatrick/Fidelity (NCFE quantitative data)
- Duddella 2007 Trade Chart Patterns Guide
- Harmonic Pattern Trading Guides
- NCFE Technical Analysis Price Patterns

Used to weight confluence scoring by pattern reliability.
"""

PATTERN_RELIABILITY: dict[str, float] = {
    # NCFE-quantified
    "head_and_shoulders": 0.87,
    "inverse_head_shoulders": 0.80,
    "symmetrical_triangle": 0.77,
    "ascending_triangle": 0.78,
    "descending_triangle": 0.78,
    # Fidelity/Kirkpatrick
    "pipe_bottom": 0.85,
    "pipe_top": 0.85,
    "island_reversal_top": 0.88,
    "island_reversal_bottom": 0.82,
    "flag": 0.82,
    "pennant": 0.78,
    # Duddella 2007
    "dragon": 0.85,
    "inverse_dragon": 0.85,
    "three_hills": 0.85,
    "three_valleys": 0.85,
    "adam_eve": 0.83,
    "trader_vic_123": 0.80,
    "msl": 0.70,
    "msh": 0.70,
    "nr4": 0.80,
    "inside_bar": 0.65,
    "crown": 0.68,
    # Harmonic patterns
    "cypher": 0.88,
    "gartley": 0.82,
    "bat": 0.78,
    "butterfly": 0.76,
    "crab": 0.74,
    "shark": 0.72,
    "abc": 0.75,
    # Standard TA
    "double_top": 0.70,
    "double_bottom": 0.70,
    "triple_top": 0.70,
    "triple_bottom": 0.70,
    "wedge": 0.62,
    "rectangle": 0.72,
    "cup_handle": 0.73,
    # SMC components
    "sweep_reversal": 0.75,
    "breaker_block": 0.70,
    "mitigation_block": 0.65,
    "fvg_proximity": 0.55,
    "rejection_block": 0.60,
    "order_block": 0.65,
    # Candlestick
    "doji": 0.40,
    "harami": 0.45,
    "hammer": 0.55,
    "engulfing": 0.60,
    "dark_cloud": 0.58,
    "shooting_star": 0.55,
    "inverted_hammer": 0.50,
    # Technical
    "ichimoku": 0.65,
    "keltner_channel": 0.60,
    "williams_r": 0.50,
    "cci": 0.55,
    "bollinger_bands": 0.60,
    # Volatility
    "donchian_channel": 0.60,
    "key_reversal": 0.70,
    "round_top": 0.40,
    "round_bottom": 0.45,
    "v_top": 0.60,
    "v_bottom": 0.60,
    "scallop_ascending": 0.55,
    "scallop_descending": 0.55,
    "bump_and_run": 0.65,
    "quasimodo": 0.75,
}

MIN_PATTERN_RELIABILITY: float = 0.40
WEAK_PATTERN_THRESHOLD: float = 0.50


def get_pattern_reliability(pattern_name: str) -> float:
    """Get reliability weight for a pattern, defaulting to 0.50."""
    return PATTERN_RELIABILITY.get(pattern_name, 0.50)
