"""
Fibonacci Calculations

Fibonacci retracement and extension calculations for harmonic patterns.
"""

from typing import Dict, Literal
import numpy as np


# Standard Fibonacci ratios
FIBONACCI_LEVELS = {
    'retracement': {
        0.236: 0.236,
        0.382: 0.382,
        0.5: 0.5,
        0.618: 0.618,
        0.786: 0.786,
        0.886: 0.886,
    },
    'extension': {
        1.0: 1.0,
        1.13: 1.13,
        1.27: 1.27,
        1.414: 1.414,
        1.618: 1.618,
        2.0: 2.0,
        2.618: 2.618,
    }
}


def fibonacci_retracement(
    high: float,
    low: float,
    direction: Literal['up', 'down'] = 'up'
) -> Dict[float, float]:
    """
    Calculate Fibonacci retracement levels.
    
    For uptrend (direction='up'): levels are below the high
    For downtrend (direction='down'): levels are above the low
    
    Args:
        high: Swing high price
        low: Swing low price
        direction: 'up' for uptrend retracement, 'down' for downtrend retracement
        
    Returns:
        Dictionary mapping Fibonacci ratio to price level
    """
    if high <= low:
        raise ValueError("High must be greater than low")
    
    diff = high - low
    levels = {}
    
    if direction == 'up':
        # Retracement from high going down
        for ratio in FIBONACCI_LEVELS['retracement'].keys():
            levels[ratio] = high - (diff * ratio)
    else:
        # Retracement from low going up
        for ratio in FIBONACCI_LEVELS['retracement'].keys():
            levels[ratio] = low + (diff * ratio)
    
    return levels


def fibonacci_extension(
    high: float,
    low: float,
    direction: Literal['up', 'down'] = 'up'
) -> Dict[float, float]:
    """
    Calculate Fibonacci extension levels.
    
    For uptrend (direction='up'): levels are above the high
    For downtrend (direction='down'): levels are below the low
    
    Args:
        high: Swing high price
        low: Swing low price
        direction: 'up' for uptrend extension, 'down' for downtrend extension
        
    Returns:
        Dictionary mapping Fibonacci ratio to price level
    """
    if high <= low:
        raise ValueError("High must be greater than low")
    
    diff = high - low
    levels = {}
    
    if direction == 'up':
        # Extension from high going up
        for ratio in FIBONACCI_LEVELS['extension'].keys():
            levels[ratio] = high + (diff * (ratio - 1.0))
    else:
        # Extension from low going down
        for ratio in FIBONACCI_LEVELS['extension'].keys():
            levels[ratio] = low - (diff * (ratio - 1.0))
    
    return levels


def is_fib_ratio_match(
    actual_ratio: float,
    target_ratio: float,
    tolerance: float = 0.05
) -> bool:
    """
    Check if actual ratio matches target Fibonacci ratio within tolerance.
    
    Args:
        actual_ratio: The calculated ratio
        target_ratio: The target Fibonacci ratio
        tolerance: Acceptable deviation (default 5%)
        
    Returns:
        True if ratio matches within tolerance
    """
    if target_ratio == 0:
        return abs(actual_ratio) < tolerance
    
    deviation = abs(actual_ratio - target_ratio) / target_ratio
    return deviation <= tolerance


def calculate_ab_retracement(
    point_x: float,
    point_a: float,
    point_b: float
) -> float:
    """
    Calculate AB retracement ratio of XA leg.
    
    Args:
        point_x: X pivot price
        point_a: A pivot price
        point_b: B pivot price
        
    Returns:
        Retracement ratio (0.0 to 1.0+)
    """
    xa = abs(point_a - point_x)
    ab = abs(point_b - point_a)
    
    if xa == 0:
        return 0.0
    
    return ab / xa


def calculate_bc_retracement(
    point_a: float,
    point_b: float,
    point_c: float
) -> float:
    """
    Calculate BC retracement ratio of AB leg.
    
    Args:
        point_a: A pivot price
        point_b: B pivot price
        point_c: C pivot price
        
    Returns:
        Retracement ratio
    """
    ab = abs(point_b - point_a)
    bc = abs(point_c - point_b)
    
    if ab == 0:
        return 0.0
    
    return bc / ab


def calculate_cd_extension(
    point_b: float,
    point_c: float,
    point_d: float
) -> float:
    """
    Calculate CD extension ratio of BC leg.
    
    Args:
        point_b: B pivot price
        point_c: C pivot price
        point_d: D pivot price
        
    Returns:
        Extension ratio
    """
    bc = abs(point_c - point_b)
    cd = abs(point_d - point_c)
    
    if bc == 0:
        return 0.0
    
    return cd / bc


def calculate_xd_retracement(
    point_x: float,
    point_a: float,
    point_d: float
) -> float:
    """
    Calculate XD retracement ratio of XA leg.
    
    Args:
        point_x: X pivot price
        point_a: A pivot price
        point_d: D pivot price
        
    Returns:
        Retracement ratio
    """
    xa = abs(point_a - point_x)
    xd = abs(point_d - point_x)
    
    if xa == 0:
        return 0.0
    
    return xd / xa


def validate_gartley_ratios(
    point_x: float,
    point_a: float,
    point_b: float,
    point_c: float,
    point_d: float,
    tolerance: float = 0.05
) -> Dict[str, bool]:
    """
    Validate Fibonacci ratios for Gartley pattern.
    
    Gartley Requirements:
    - AB = 0.618 XA (±tolerance)
    - BC = 0.382-0.886 AB
    - CD = 1.13-1.618 BC
    - AD = 0.786 XA (±tolerance)
    
    Args:
        point_x, point_a, point_b, point_c, point_d: Pivot prices
        tolerance: Acceptable deviation from target ratios
        
    Returns:
        Dictionary with validation results for each ratio
    """
    ab_ratio = calculate_ab_retracement(point_x, point_a, point_b)
    bc_ratio = calculate_bc_retracement(point_a, point_b, point_c)
    cd_ratio = calculate_cd_extension(point_b, point_c, point_d)
    xd_ratio = calculate_xd_retracement(point_x, point_a, point_d)
    
    return {
        'ab_valid': is_fib_ratio_match(ab_ratio, 0.618, tolerance),
        'bc_valid': 0.382 - tolerance <= bc_ratio <= 0.886 + tolerance,
        'cd_valid': 1.13 - tolerance <= cd_ratio <= 1.618 + tolerance,
        'xd_valid': is_fib_ratio_match(xd_ratio, 0.786, tolerance),
        'ab_ratio': ab_ratio,
        'bc_ratio': bc_ratio,
        'cd_ratio': cd_ratio,
        'xd_ratio': xd_ratio,
        'all_valid': (
            is_fib_ratio_match(ab_ratio, 0.618, tolerance) and
            0.382 - tolerance <= bc_ratio <= 0.886 + tolerance and
            1.13 - tolerance <= cd_ratio <= 1.618 + tolerance and
            is_fib_ratio_match(xd_ratio, 0.786, tolerance)
        )
    }


def validate_abc_ratios(
    point_a: float,
    point_b: float,
    point_c: float,
    tolerance: float = 0.05
) -> Dict[str, bool]:
    """
    Validate Fibonacci ratios for ABC pattern.
    
    ABC Requirements:
    - BC = 0.382-0.618 AB
    
    Args:
        point_a, point_b, point_c: Pivot prices
        tolerance: Acceptable deviation from target ratios
        
    Returns:
        Dictionary with validation results
    """
    bc_ratio = calculate_bc_retracement(point_a, point_b, point_c)
    
    return {
        'bc_valid': 0.382 - tolerance <= bc_ratio <= 0.618 + tolerance,
        'bc_ratio': bc_ratio,
        'all_valid': 0.382 - tolerance <= bc_ratio <= 0.618 + tolerance
    }


def calculate_pattern_depth(
    high: float,
    low: float
) -> float:
    """
    Calculate pattern depth (range).
    
    Args:
        high: Pattern high
        low: Pattern low
        
    Returns:
        Pattern depth (high - low)
    """
    return abs(high - low)


def calculate_target_levels(
    entry: float,
    pattern_depth: float,
    direction: Literal['long', 'short'],
    multipliers: list = [0.62, 1.0, 1.27, 1.62]
) -> Dict[float, float]:
    """
    Calculate target levels based on pattern depth.
    
    Args:
        entry: Entry price
        pattern_depth: Pattern depth (high - low)
        direction: 'long' or 'short'
        multipliers: List of target multipliers
        
    Returns:
        Dictionary mapping multiplier to target price
    """
    targets = {}
    
    for mult in multipliers:
        if direction == 'long':
            targets[mult] = entry + (pattern_depth * mult)
        else:
            targets[mult] = entry - (pattern_depth * mult)
    
    return targets


def is_price_near_level(
    price: float,
    level: float,
    tolerance_pct: float = 0.02
) -> bool:
    """
    Check if price is near a Fibonacci level.
    
    Args:
        price: Current price
        level: Target level
        tolerance_pct: Tolerance as percentage (default 2%)
        
    Returns:
        True if price is within tolerance of level
    """
    if level == 0:
        return False
    
    deviation = abs(price - level) / level
    return deviation <= tolerance_pct
