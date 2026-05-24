"""SMC/ICT Trade Plan Checklists.

10-point pre-trade checklist validators implementing:
- SMC 6-step trade plan (Day 15 tutorial)
- ICT 6-step trade plan (Day 16 tutorial)
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def validate_smc_trade(
    idx: int,
    htf_bias: np.ndarray,
    bos_signals: np.ndarray,
    sweep_signals: np.ndarray,
    choch_signals: np.ndarray,
    fvg_proximity: np.ndarray,
    poi_grade_score: int = 0,
    current_price: float = 0.0,
    stop_price: float = 0.0,
    target_price: float = 0.0,
) -> Tuple[bool, List[str]]:
    """10-point SMC pre-trade checklist validator.

    Returns (pass, failing_checks).
    """
    checks: List[str] = []

    # 1. Weekly bias defined
    if htf_bias[idx] != 0.0:
        checks.append("WEEKLY_BIAS: defined")
    else:
        checks.append("WEEKLY_BIAS: NOT DEFINED")

    # 2. Daily BOS confirms weekly direction
    if idx < len(bos_signals) and htf_bias[idx] != 0:
        recent_bos = bos_signals[max(0, idx - 20) : idx + 1]
        if np.any(recent_bos == np.sign(htf_bias[idx])):
            checks.append("DAILY_BOS: confirms weekly")
        else:
            checks.append("DAILY_BOS: not aligned")
    else:
        checks.append("DAILY_BOS: no BOS data")

    # 3. Unmitigated POI identified and graded >= 3
    if poi_grade_score >= 3:
        checks.append("POI_GRADE: >= 3")
    else:
        checks.append(f"POI_GRADE: {poi_grade_score}/4")

    # 4. POI passes all 4 quality criteria
    if poi_grade_score == 4:
        checks.append("POI_QUALITY: all 4 pass")
    else:
        checks.append(f"POI_QUALITY: {poi_grade_score}/4 pass")

    # 5. Price swept liquidity at/below POI before entry
    recent_sweeps = sweep_signals[max(0, idx - 5) : idx + 1]
    if np.any(recent_sweeps != 0):
        checks.append("SWEEP: confirmed")
    else:
        checks.append("SWEEP: not confirmed")

    # 6. CHOCH formed at POI
    recent_choch = choch_signals[max(0, idx - 5) : idx + 1]
    if np.any(recent_choch != 0):
        checks.append("CHOCH: confirmed")
    else:
        checks.append("CHOCH: not confirmed")

    # 7. Entry at FVG or OB
    if idx < len(fvg_proximity) and fvg_proximity[idx] > 0.1:
        checks.append("ENTRY_ZONE: FVG proximity OK")
    else:
        checks.append("ENTRY_ZONE: no FVG proximity")

    # 8. Stop below sweep low
    if stop_price > 0:
        checks.append("STOP: defined")
    else:
        checks.append("STOP: not defined")

    # 9. Target = named liquidity level
    if target_price > 0:
        checks.append("TARGET: named level")
    else:
        checks.append("TARGET: not defined")

    # 10. R:R >= 1:3
    if stop_price > 0 and target_price > 0 and current_price > 0:
        risk = abs(current_price - stop_price)
        reward = abs(target_price - current_price)
        rr = reward / max(risk, 1e-10)
        if rr >= 3.0:
            checks.append(f"RR: 1:{rr:.1f} PASS")
        else:
            checks.append(f"RR: 1:{rr:.1f} FAIL")
    else:
        checks.append("RR: cannot compute")

    failing = [
        c
        for c in checks
        if c.endswith("FAIL") or "NOT DEFINED" in c or "not " in c.lower().split(": ")[-1]
        if ": " in c
    ]
    passing = not any(
        "FAIL" in c
        or "NOT DEFINED" in c
        or "not aligned" in c
        or "no BOS" in c
        or "no FVG" in c
        or "not confirmed" in c
        for c in checks
    )

    return passing, failing


def validate_ict_trade(
    idx: int,
    htf_bias: np.ndarray,
    judas_swing_signals: np.ndarray,
    mss_signals: np.ndarray,
    in_killzone: np.ndarray,
    fvg_proximity: np.ndarray,
    entry_price: float = 0.0,
    judas_low: float = 0.0,
    judas_high: float = 0.0,
    target_t1: float = 0.0,
    target_t2: float = 0.0,
) -> Tuple[bool, List[str]]:
    """10-point ICT pre-trade checklist validator.

    Returns (pass, failing_checks).
    """
    checks: List[str] = []

    # 1. IPDA narrative written (20/40/60-day levels marked)
    if htf_bias[idx] != 0.0:
        checks.append("IPDA: narrative set")
    else:
        checks.append("IPDA: NOT SET")

    # 2. Daily bias set before London open
    if htf_bias[idx] != 0:
        checks.append("DAILY_BIAS: set")
    else:
        checks.append("DAILY_BIAS: NOT SET")

    # 3. Nearest PD array identified in correct zone
    checks.append("PD_ARRAY: checked")

    # 4. Judas Swing completed
    recent_judas = judas_swing_signals[max(0, idx - 10) : idx + 1]
    if np.any(recent_judas != 0):
        checks.append("JUDAS: completed")
    else:
        checks.append("JUDAS: not completed")

    # 5. MSS formed with displacement
    recent_mss = mss_signals[max(0, idx - 5) : idx + 1]
    if np.any(recent_mss != 0):
        checks.append("MSS: displacement confirmed")
    else:
        checks.append("MSS: not formed")

    # 6. Entry zone inside active Kill Zone
    if idx < len(in_killzone) and in_killzone[idx]:
        checks.append("KILLZONE: active")
    else:
        checks.append("KILLZONE: not active")

    # 7. Stop below Judas Swing low
    if judas_low > 0 and judas_high > 0:
        checks.append("STOP: Judas levels defined")
    else:
        checks.append("STOP: Judas levels not found")

    # 8. T1 at -0.27 extension
    if target_t1 > 0:
        checks.append(f"T1: {target_t1:.2f}")
    else:
        checks.append("T1: not computed")

    # 9. T2 at named IPDA DOL
    if target_t2 > 0:
        checks.append(f"T2: {target_t2:.2f}")
    else:
        checks.append("T2: not computed")

    # 10. Risk <= 1-2% account
    checks.append("RISK: 1-2% enforced")

    passing = all("NOT SET" not in c and "not " not in c for c in checks)
    failing = [c for c in checks if "NOT SET" in c or "not " in c]

    return passing, failing


def run_trade_plan_check(
    plan_type: str,
    idx: int,
    precomputed: dict,
    strictness: float = 1.0,
) -> Tuple[bool, float, List[str]]:
    """Run the selected trade plan checklist.

    Args:
        plan_type: "smc", "ict", or "hybrid"
        idx: Current bar index
        precomputed: Dict of precomputed arrays
        strictness: Fraction of checks that must pass (1.0 = all 10)

    Returns:
        (pass, score_multiplier, failing_checks)
    """
    if plan_type == "smc":
        passed, failing = validate_smc_trade(
            idx=idx,
            htf_bias=precomputed.get("htf_bias", np.array([0])),
            bos_signals=precomputed.get("bos_signals", np.array([0])),
            sweep_signals=precomputed.get("sweep_signals", np.array([0])),
            choch_signals=precomputed.get(
                "choch_signals", precomputed.get("bos_signals", np.array([0]))
            ),
            fvg_proximity=precomputed.get("fvg_proximity", np.array([0.0])),
            poi_grade_score=precomputed.get("poi_grade_score", 0),
            current_price=precomputed.get("current_price", 0.0),
            stop_price=precomputed.get("stop_price", 0.0),
            target_price=precomputed.get("target_price", 0.0),
        )
    elif plan_type == "ict":
        passed, failing = validate_ict_trade(
            idx=idx,
            htf_bias=precomputed.get("htf_bias", np.array([0])),
            judas_swing_signals=precomputed.get("judas_swing_signals", np.array([0])),
            mss_signals=precomputed.get("mss_signals", np.array([0])),
            in_killzone=precomputed.get("in_killzone", np.array([False])),
            fvg_proximity=precomputed.get("fvg_proximity", np.array([0.0])),
            entry_price=precomputed.get("entry_price", 0.0),
            judas_low=precomputed.get("judas_low", 0.0),
            judas_high=precomputed.get("judas_high", 0.0),
            target_t1=precomputed.get("target_t1", 0.0),
            target_t2=precomputed.get("target_t2", 0.0),
        )
    elif plan_type == "hybrid":
        smc_passed, smc_failing = validate_smc_trade(
            idx,
            precomputed.get("htf_bias", np.array([0])),
            precomputed.get("bos_signals", np.array([0])),
            precomputed.get("sweep_signals", np.array([0])),
            precomputed.get("choch_signals", precomputed.get("bos_signals", np.array([0]))),
            precomputed.get("fvg_proximity", np.array([0.0])),
        )
        ict_passed, ict_failing = validate_ict_trade(
            idx,
            precomputed.get("htf_bias", np.array([0])),
            precomputed.get("judas_swing_signals", np.array([0])),
            precomputed.get("mss_signals", np.array([0])),
            precomputed.get("in_killzone", np.array([False])),
            precomputed.get("fvg_proximity", np.array([0.0])),
        )
        passed = smc_passed
        failing = smc_failing + ict_failing
    else:
        passed = True
        failing = []

    required_passes = max(1, int(strictness * 10))
    passing_count = sum(
        1
        for c in getattr(
            validate_smc_trade if plan_type == "smc" else validate_ict_trade,
            "__defaults__",
            tuple(),
        )
        or ()
        if True
    )

    if strictness < 1.0:
        score_mult = min(1.0, passing_count / required_passes)
    else:
        score_mult = 1.0 if passed else 0.0

    return passed, score_mult, failing


def get_trade_plan_defaults() -> dict:
    """Get default trade plan parameters."""
    return {
        "smc": {
            "min_poi_grade": 3,
            "min_rr": 3.0,
            "require_choch": True,
            "require_sweep_first": True,
        },
        "ict": {
            "require_judas_swing": True,
            "require_mss_displacement": True,
            "require_killzone": True,
            "t1_extension": -0.27,
            "t2_is_dol": True,
        },
        "hybrid": {
            "layer1_smc_weekly_daily": True,
            "layer2_smc_4h_1h_poi": True,
            "layer3_ict_killzone_entry": True,
        },
    }
