"""Volume confirmation rules per pattern type.

All rules from: Kirkpatrick/Fidelity, NCFE, Duddella 2007, Warrior Trading.
"""


def get_volume_rule(pattern_type: str) -> dict:
    """Return volume confirmation rule for a given pattern type."""
    rules: dict[str, dict] = {
        "head_and_shoulders": {
            "high_volume_at_head": True,
            "declining_volume_at_shoulder": True,
            "min_volume_mult": 1.2,
        },
        "triangle": {
            "declining_volume_during_formation": True,
            "volume_surge_on_breakout": True,
            "min_volume_mult": 1.5,
            "breakout_bar_volume": "must_exceed_20_bar_avg",
        },
        "double_top_bottom": {
            "first_peak_volume_heavier": True,
            "breakout_volume_surge": True,
            "min_volume_mult": 1.3,
        },
        "cup_handle": {
            "volume_drying_during_handle": True,
            "breakout_volume_surge": True,
            "min_volume_mult": 1.4,
        },
        "v_top_bottom": {
            "high_volume_on_spike": True,
            "higher_volume_on_reversal": True,
            "min_volume_mult": 1.5,
        },
        "round_top_bottom": {
            "low_volume_during_formation": True,
            "volume_spike_on_breakout": True,
            "min_volume_mult": 1.5,
        },
        "gap": {
            "gap_size_2_5x_10day_atr_max": True,
            "volume_surge_on_gap": True,
            "min_volume_mult": 1.8,
        },
        "breakout": {
            "volume_must_exceed_20_bar_avg": True,
            "min_volume_mult": 1.3,
        },
        "abcd_flag": {
            "high_volume_on_A": True,
            "lower_volume_on_BC": True,
            "volume_surge_on_D": True,
        },
    }
    return rules.get(pattern_type, {"min_volume_mult": 1.0})
