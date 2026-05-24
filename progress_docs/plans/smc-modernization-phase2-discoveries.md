# SMC Modernization — Phase 2 Discoveries Implementation Plan

> **Source:** Deep-dive audit of 41 PDFs/MDs, 14 repos, 45 papers, 6 SMC documents, 6 chart pattern guides, 5 academic papers (2026-05-20)
> **Extends:** `smc-modernization-implementation.md` (Phases 1-5, 790 lines) and `smc-modernization-proven-enhancements.md` (20 sections, 938 lines)
> **Prerequisite:** Phases 1-5 must be complete before starting Phase 6
> **Purpose:** Implement 45+ newly discovered SMC enhancements beyond the existing 20-item modernization plan
> **Total new phases:** 7 (Phases 6-12). Alloc ~8-12 hours.
> **Output:** New modules in `src/patterns/smc/`, `src/indicators/`, `src/risk/smc_aware.py`, `src/signals/smc_divergence.py`, `src/ml/autoalpha_smc.py`, plus 17 new chart pattern detectors.

---

## PRE-FLIGHT CHECKLIST (every session)

```
□ Phases 1-5 complete per smc-modernization-implementation.md
□ MEMORY.md exists with current metrics
□ smartmoneyconcepts installed: uv run python -c "from smartmoneyconcepts import smc; print(hasattr(smc, 'ob'))"
□ uv run ruff check src/ — clean baseline
□ uv sync — environment current
□ data/raw/ has BTC-USD_1h.csv and at least 2 other instruments
```

---

## PHASE 6: CORE SMC CONCEPTS — Breaker, Mitigation, Rejection Blocks (P0 — HIGHEST NEW IMPACT)

**Goal:** Implement the three most important missing SMC concepts. These are documented across all 6 SMC primary sources and partially exist in the `smartmoneyconcepts` library's OB logic but are not surfaced as standalone detectors.
**Time:** ~90 minutes
**Depends on:** Phases 1-5 complete, `smartmoneyconcepts` installed
**Output:** `src/patterns/smc/breaker.py`, `src/patterns/smc/mitigation.py`, `src/patterns/smc/rejection.py`

---

### TASK 6.1: Create `src/patterns/smc/` directory and `__init__.py`

```bash
mkdir -p src/patterns/smc
```

**File:** `src/patterns/smc/__init__.py`
```python
"""SMC/ICT pattern detectors — Breaker Blocks, Mitigation Blocks, Rejection Blocks, SFP, PD Array Matrix."""
from .breaker import detect_breaker_blocks, BreakerBlock
from .mitigation import detect_mitigation_blocks, MitigationBlock
from .rejection import detect_rejection_blocks, RejectionBlock
```

---

### TASK 6.2: Implement Breaker Block Detector (`src/patterns/smc/breaker.py`)

**ICT Definition (from 14-ICT #4, David Woods, Order Block & Fib):**
- Breaker Block = failed Order Block that leads to market structure shift
- Bullish Breaker: Bearish OB that got mitigated → price subsequently breaks THROUGH the OB bottom → OB becomes bullish support
- Bearish Breaker: Bullish OB that got mitigated → price subsequently breaks THROUGH the OB top → OB becomes bearish resistance
- Confirmation: price retests the breaker zone from the opposite side and reverses

**Algorithm:**
```
1. Use smc.ob() to identify all OBs with their mitigation indices
2. For each mitigated bullish OB:
   - After mitigation, if price closes ABOVE the OB top → OB has been "breached" → becomes Bearish Breaker
   - Signal: short when price retests the breached zone from below
3. For each mitigated bearish OB:
   - After mitigation, if price closes BELOW the OB bottom → OB has been "breached" → becomes Bullish Breaker
   - Signal: long when price retests the breached zone from above
4. Breaker remains active until price closes beyond the opposite side (+ATR buffer)
5. David Woods rule: Strong Breaker exists when momentum shift is aggressive + breaker has inducement + formed on a session (London/NY/Asia High or Low)
```

**File:** `src/patterns/smc/breaker.py` (~250 lines)

```python
"""Breaker Block detector — failed Order Block that becomes opposite-direction support/resistance.

ICT Definition (14-ICT #4, David Woods):
- Bullish Breaker: Bearish OB that got mitigated, then price breaks through OB bottom → becomes bullish support
- Bearish Breaker: Bullish OB that got mitigated, then price breaks through OB top → becomes bearish resistance
- Strong Breaker: momentum shift aggressive + breaker has inducement + formed on session H/L
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class BreakerBlock:
    """A single breaker block instance."""
    bar_index: int              # Bar where breaker was confirmed
    direction: int              # +1 = bullish breaker, -1 = bearish breaker
    top: float                  # Top boundary of the breaker zone
    bottom: float               # Bottom boundary of the breaker zone
    source_ob_index: int        # Index of the original OB that was breached
    mitigation_index: int       # Index of the OB mitigation
    breach_index: int           # Index of the breach candle
    strength: float             # 0.0-1.0 based on volume + inducement + session context
    active: bool = True         # False once price closes beyond opposite boundary
    inducement_present: bool = False
    session_formed: bool = False


def detect_breaker_blocks(
    ohlc: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    atr: np.ndarray,
    volume_sma_20: np.ndarray,
    session_active: Optional[np.ndarray] = None,
    buffer_atr_mult: float = 0.5,
    min_volume_mult: float = 1.2,
    require_inducement: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[BreakerBlock]]:
    """Detect Breaker Blocks from OB breaches.

    Uses the smartmoneyconcepts library to detect OBs, then tracks breaches
    to identify breaker blocks. Returns (bullish_signals, bearish_signals, breaker_list).

    Args:
        ohlc: OHLCV DataFrame (columns: open, high, low, close, volume)
        swing_highs_lows: Output from smc.swing_highs_lows()
        atr: Precomputed ATR(14) numpy array
        volume_sma_20: 20-bar rolling average volume
        session_active: Boolean array indicating active trading sessions (optional)
        buffer_atr_mult: ATR multiplier for zone boundary buffer (default 0.5)
        min_volume_mult: Minimum volume vs SMA to confirm (default 1.2)
        require_inducement: Whether inducement is required for breaker validity

    Returns:
        bullish_signals: np.ndarray shape (n,), +1 at bullish breaker retest bars
        bearish_signals: np.ndarray shape (n,), -1 at bearish breaker retest bars
        breakers: list of BreakerBlock dataclass instances
    """
    from smartmoneyconcepts import smc as smc_lib

    close = ohlc["Close"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    volume = ohlc["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    # 1. Detect Order Blocks using smartmoneyconcepts library
    ob_result = smc_lib.ob(ohlc, swing_highs_lows, close_mitigation=False)
    ob_signal = ob_result["OB"].to_numpy(dtype=np.float64)
    ob_top = ob_result["Top"].to_numpy(dtype=np.float64)
    ob_bottom = ob_result["Bottom"].to_numpy(dtype=np.float64)
    ob_volume = ob_result["OBVolume"].to_numpy(dtype=np.float64)
    ob_mitigated = ob_result["MitigatedIndex"].to_numpy(dtype=np.float64)
    ob_percentage = ob_result["Percentage"].to_numpy(dtype=np.float64)

    bullish_signals = np.zeros(n, dtype=np.int8)
    bearish_signals = np.zeros(n, dtype=np.int8)
    breakers: list[BreakerBlock] = []

    # 2. For each OB, track what happens after mitigation
    tracked_obs: list[dict] = []

    for i in range(n):
        if not np.isnan(ob_signal[i]) and ob_signal[i] != 0:
            tracked_obs.append({
                "index": i,
                "direction": int(ob_signal[i]),  # +1 bullish OB, -1 bearish OB
                "top": ob_top[i],
                "bottom": ob_bottom[i],
                "volume_strength": ob_percentage[i] if not np.isnan(ob_percentage[i]) else 50.0,
                "mitigated": False,
                "mitigation_index": -1,
                "breached": False,
                "breach_index": -1,
            })

        # Check for OB breaches after mitigation
        for ob in tracked_obs:
            if ob["mitigated"] and not ob["breached"]:
                ob_direction = ob["direction"]
                ob_top_val = ob["top"]
                ob_bottom_val = ob["bottom"]

                if ob_direction == -1:  # Bearish OB → check for breach below bottom → Bullish Breaker
                    if close[i] < ob_bottom_val - buffer_atr_mult * atr[i]:
                        ob["breached"] = True
                        ob["breach_index"] = i
                        # Create BreakerBlock
                        volume_ok = volume[i] >= min_volume_mult * volume_sma_20[i]
                        has_inducement = _check_inducement(close, low, i, lookback=8)
                        on_session = session_active is not None and session_active[i]
                        strength = _compute_breaker_strength(
                            ob["volume_strength"], volume_ok, has_inducement, on_session
                        )
                        breaker = BreakerBlock(
                            bar_index=i,
                            direction=1,  # Bullish Breaker
                            top=ob_top_val,
                            bottom=ob_bottom_val,
                            source_ob_index=ob["index"],
                            mitigation_index=ob["mitigation_index"],
                            breach_index=i,
                            strength=strength,
                            inducement_present=has_inducement,
                            session_formed=on_session,
                        )
                        breakers.append(breaker)

                elif ob_direction == 1:  # Bullish OB → check for breach above top → Bearish Breaker
                    if close[i] > ob_top_val + buffer_atr_mult * atr[i]:
                        ob["breached"] = True
                        ob["breach_index"] = i
                        volume_ok = volume[i] >= min_volume_mult * volume_sma_20[i]
                        has_inducement = _check_inducement(close, high, i, lookback=8)
                        on_session = session_active is not None and session_active[i]
                        strength = _compute_breaker_strength(
                            ob["volume_strength"], volume_ok, has_inducement, on_session
                        )
                        breaker = BreakerBlock(
                            bar_index=i,
                            direction=-1,  # Bearish Breaker
                            top=ob_top_val,
                            bottom=ob_bottom_val,
                            source_ob_index=ob["index"],
                            mitigation_index=ob["mitigation_index"],
                            breach_index=i,
                            strength=strength,
                            inducement_present=has_inducement,
                            session_formed=on_session,
                        )
                        breakers.append(breaker)

        # Check mitigated indices
        for ob in tracked_obs:
            if not ob["mitigated"]:
                mit_idx = int(ob_mitigated[ob["index"]]) if not np.isnan(ob_mitigated[ob["index"]]) else -1
                if mit_idx > 0 and i >= mit_idx:
                    ob["mitigated"] = True
                    ob["mitigation_index"] = mit_idx

    # 3. Detect retests of active breaker zones for entry signals
    active_breakers = [b for b in breakers if b.active]
    for i in range(n):
        for breaker in active_breakers:
            if breaker.direction == 1:  # Bullish Breaker retest
                zone_bottom = breaker.bottom
                zone_top = breaker.top
                if low[i] <= zone_top + buffer_atr_mult * atr[i] and low[i] >= zone_bottom:
                    if close[i] > open[i]:  # Bullish rejection confirmation
                        bullish_signals[i] = 1
            elif breaker.direction == -1:  # Bearish Breaker retest
                zone_top = breaker.top
                zone_bottom = breaker.bottom
                if high[i] >= zone_bottom - buffer_atr_mult * atr[i] and high[i] <= zone_top:
                    if close[i] < open[i]:  # Bearish rejection confirmation
                        bearish_signals[i] = -1

    return bullish_signals, bearish_signals, breakers


def _check_inducement(prices: np.ndarray, extremes: np.ndarray, idx: int, lookback: int = 8) -> bool:
    """Check if there's a mini liquidity grab (inducement) before the breach.

    Inducement = price makes a small counter-move that grabs LTF liquidity
    right before the main move. This confirms the breaker is valid.

    Args:
        prices: Close price array
        extremes: High (for bearish) or Low (for bullish) array
        idx: Current bar index
        lookback: Number of bars to scan back
    """
    if idx < lookback + 3:
        return False
    window_prices = extremes[idx - lookback:idx]
    prev_high = np.max(window_prices)
    prev_low = np.min(window_prices)
    range_size = prev_high - prev_low
    if range_size <= 0:
        return False
    # Check for a small 2-3 bar move in the opposite direction
    recent_change = abs(prices[idx] - prices[idx - 2])
    return recent_change > 0.3 * range_size


def _compute_breaker_strength(
    ob_volume_strength: float,
    volume_confirmed: bool,
    inducement_present: bool,
    session_formed: bool,
) -> float:
    """Compute breaker strength score (0.0-1.0).

    David Woods: Strong breaker = momentum shift aggressive + breaker has inducement + formed on session H/L.
    80% confirmation rate when combined with displacement.
    """
    score = 0.0
    score += ob_volume_strength / 100.0 * 0.35
    if volume_confirmed:
        score += 0.25
    if inducement_present:
        score += 0.25
    if session_formed:
        score += 0.15
    return np.clip(score, 0.0, 1.0)
```

**Verify:**
```bash
uv run python -c "
import pandas as pd
import numpy as np
from smartmoneyconcepts import smc
from src.patterns.smc.breaker import detect_breaker_blocks
# Load sample data, run detection
df = pd.read_csv('data/raw/BTC-USD_1h.csv', index_col=0, parse_dates=True)
shl = smc.swing_highs_lows(df, swing_length=20)
atr = np.full(len(df), df['Close'].std())
vol_sma = df['Volume'].rolling(20).mean().fillna(df['Volume'].mean()).to_numpy()
bull, bear, breakers = detect_breaker_blocks(df, shl, atr, vol_sma)
print(f'Found {len(breakers)} breaker blocks')
print(f'Bullish signals: {bull.sum()}, Bearish signals: {bear.sum()}')
"
```

---

### TASK 6.3: Implement Mitigation Block Detector (`src/patterns/smc/mitigation.py`)

**ICT Definition (from toaz.info glossary, David Woods):**
- Mitigation Block: When a strong high or low is "mitigated" (price touches/visits it), the level develops weakness
- Bullish Mitigation Block: A strong low that got mitigated but price bounced off it → becomes resistance-turned-support
- Bearish Mitigation Block: A strong high that got mitigated but price rejected from it → becomes support-turned-resistance
- Positioned in the Price Delivery Array hierarchy between Breaker Blocks and Equilibrium

**Algorithm:**
```
1. Identify strong highs/lows using smc.swing_highs_lows()
2. Track when price visits (mitigates) each strong level
3. If price reverses from the level after mitigation → Mitigation Block confirmed
4. Score based on: session context, volume at mitigation, retracement depth
5. Entry: on retest of mitigation zone with confirmation candle
```

**File:** `src/patterns/smc/mitigation.py` (~200 lines)

```python
"""Mitigation Block detector — mitigated strong swing levels that become reverse polarity zones.

ICT Definition (toaz.info, David Woods):
- Mitigation Block = strong swing point that price "mitigates" (visits), then price reverses
  from it, developing weakness at that level
- Bullish Mitigation Block: Strong Low gets mitigated → price bounces → becomes support
- Bearish Mitigation Block: Strong High gets mitigated → price rejects → becomes resistance
- In PD Array Matrix hierarchy: Bullish/Bearish Mitigation between Breaker Blocks and Equilibrium
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class MitigationBlock:
    bar_index: int
    direction: int               # +1 bullish, -1 bearish
    level: float                 # The mitigated strong level
    source_swing_index: int      # Index of the strong swing point
    mitigation_bar_index: int    # Bar where mitigation occurred
    reversal_bar_index: int      # Bar where reversal from level confirmed
    strength: float              # 0.0-1.0
    active: bool = True


def detect_mitigation_blocks(
    ohlc: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    atr: np.ndarray,
    session_active: Optional[np.ndarray] = None,
    buffer_atr_mult: float = 0.3,
    min_reversal_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[MitigationBlock]]:
    """Detect Mitigation Blocks from mitigated strong swing levels.

    Args:
        ohlc: OHLCV DataFrame
        swing_highs_lows: Output from smc.swing_highs_lows()
        atr: Precomputed ATR(14) array
        session_active: Boolean array for session context (optional)
        buffer_atr_mult: Zone width around mitigated level
        min_reversal_ratio: Minimum reversal fraction from level (0.5 = must retrace 50% of prior leg)

    Returns:
        bullish_signals, bearish_signals, mitigation_blocks list
    """
    from smartmoneyconcepts import smc as smc_lib

    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    shl_signal = swing_highs_lows["HighLow"].to_numpy(dtype=np.float64)
    shl_level = swing_highs_lows["Level"].to_numpy(dtype=np.float64)

    bullish_signals = np.zeros(n, dtype=np.int8)
    bearish_signals = np.zeros(n, dtype=np.int8)
    blocks: list[MitigationBlock] = []

    # Track swing points and their mitigation status
    swing_points: list[dict] = []
    for i in range(n):
        if not np.isnan(shl_signal[i]) and shl_signal[i] != 0:
            swing_points.append({
                "index": i,
                "type": int(shl_signal[i]),  # +1 swing high, -1 swing low
                "level": shl_level[i],
                "mitigated": False,
                "mitigation_idx": -1,
            })

    # Scan for mitigation: price touches the level
    for sp in swing_points:
        for j in range(sp["index"] + 1, n):
            if sp["mitigated"]:
                break
            buffer = buffer_atr_mult * atr[j]
            if sp["type"] == 1:  # Swing High — check if price ROSE to touch it
                if high[j] >= sp["level"] - buffer:
                    sp["mitigated"] = True
                    sp["mitigation_idx"] = j
            elif sp["type"] == -1:  # Swing Low — check if price FELL to touch it
                if low[j] <= sp["level"] + buffer:
                    sp["mitigated"] = True
                    sp["mitigation_idx"] = j

    # For each mitigated swing, check for reversal (Mitigation Block confirmation)
    for sp in swing_points:
        if not sp["mitigated"]:
            continue
        mit_idx = sp["mitigation_idx"]

        if sp["type"] == 1:  # Mitigated Swing High → Bearish Mitigation Block
            # Check if price reverses DOWN from this level
            for j in range(mit_idx + 2, min(mit_idx + 12, n)):
                move_from_level = sp["level"] - close[j]
                prior_leg = sp["level"] - low[mit_idx:j].min() if mit_idx < j else atr[j]
                if prior_leg <= 0:
                    prior_leg = atr[j]
                reversal_ratio = move_from_level / prior_leg
                if reversal_ratio > min_reversal_ratio and close[j] < close[j - 1]:
                    on_session = session_active is not None and session_active[j]
                    strength = 0.5 + 0.25 * min(reversal_ratio, 1.0) + (0.25 if on_session else 0)
                    block = MitigationBlock(
                        bar_index=j, direction=-1, level=sp["level"],
                        source_swing_index=sp["index"], mitigation_bar_index=mit_idx,
                        reversal_bar_index=j, strength=min(strength, 1.0),
                    )
                    blocks.append(block)
                    break

        elif sp["type"] == -1:  # Mitigated Swing Low → Bullish Mitigation Block
            for j in range(mit_idx + 2, min(mit_idx + 12, n)):
                move_from_level = close[j] - sp["level"]
                prior_leg = high[mit_idx:j].max() - sp["level"] if mit_idx < j else atr[j]
                if prior_leg <= 0:
                    prior_leg = atr[j]
                reversal_ratio = move_from_level / prior_leg
                if reversal_ratio > min_reversal_ratio and close[j] > close[j - 1]:
                    on_session = session_active is not None and session_active[j]
                    strength = 0.5 + 0.25 * min(reversal_ratio, 1.0) + (0.25 if on_session else 0)
                    block = MitigationBlock(
                        bar_index=j, direction=1, level=sp["level"],
                        source_swing_index=sp["index"], mitigation_bar_index=mit_idx,
                        reversal_bar_index=j, strength=min(strength, 1.0),
                    )
                    blocks.append(block)
                    break

    # Detect retests for entry signals
    for i in range(n):
        for block in blocks:
            if not block.active or block.bar_index > i:
                continue
            buffer = buffer_atr_mult * atr[i]
            if block.direction == 1:  # Bullish Mitigation Block retest
                if low[i] <= block.level + buffer and close[i] > open[i]:
                    bullish_signals[i] = 1
                    block.active = False  # Consume the block
            elif block.direction == -1:  # Bearish Mitigation Block retest
                if high[i] >= block.level - buffer and close[i] < open[i]:
                    bearish_signals[i] = -1
                    block.active = False

    return bullish_signals, bearish_signals, blocks
```

---

### TASK 6.4: Implement Rejection Block Detector (`src/patterns/smc/rejection.py`)

**ICT Definition (David Woods):**
- Rejection Block = price strongly rejects from an OB/FVG/structural level
- Strong rejection has: inducement (LTF liquidity grab before rejection) + forms on a session (London/NY/Asia H/L)
- David Woods: "80% of the time, rejection block + displacement forms real price action pattern confirmation"
- HTF Rejection Block = LTF Algo Candle

**File:** `src/patterns/smc/rejection.py` (~150 lines)

```python
"""Rejection Block detector — strong structural rejections with inducement + session context.

ICT Definition (David Woods):
- Rejection Block = price REJECTS from OB/FVG/structural level with inducement + session context
- David Woods: "80% of the time, rejection block + displacement forms real PA confirmation"
- HTF Rejection Block = LTF Algo Candle (multi-timeframe relationship)
- In PD Array Matrix: Top of Premium zone and bottom of Discount zone
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class RejectionBlockSignal:
    bar_index: int
    direction: int          # +1 bullish rejection, -1 bearish rejection
    level: float            # Price level where rejection occurred
    wick_ratio: float       # Wick-to-body ratio (higher = stronger rejection)
    volume_ratio: float     # Relative volume at rejection
    session_formed: bool
    inducement_present: bool
    confidence: float       # 0.0-1.0 aggregate confidence


def detect_rejection_blocks(
    ohlc: pd.DataFrame,
    fvg_result: pd.DataFrame,
    ob_result: pd.DataFrame,
    atr: np.ndarray,
    volume_sma_20: np.ndarray,
    session_active: Optional[np.ndarray] = None,
    wick_body_ratio_min: float = 1.5,
    zone_buffer_atr: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[RejectionBlockSignal]]:
    """Detect Rejection Blocks at OB/FVG structural levels.

    Scans for candles that strongly reject from known structural zones (OB, FVG),
    with inducement and session context.

    Args:
        ohlc: OHLCV DataFrame
        fvg_result: Output from smc.fvg() — FVG zones
        ob_result: Output from smc.ob() — Order Block zones
        atr: Precomputed ATR(14) array
        volume_sma_20: 20-bar rolling average volume
        session_active: Boolean array for session context
        wick_body_ratio_min: Minimum wick-to-body ratio for valid rejection
        zone_buffer_atr: ATR multiplier for zone proximity check

    Returns:
        bullish_signals, bearish_signals, rejection_signals list
    """
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    volume = ohlc["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    # Extract structural zone boundaries from FVG and OB
    fvg_top = fvg_result["Top"].to_numpy(dtype=np.float64)
    fvg_bottom = fvg_result["Bottom"].to_numpy(dtype=np.float64)
    fvg_sig = fvg_result["FVG"].to_numpy(dtype=np.float64)
    ob_top = ob_result["Top"].to_numpy(dtype=np.float64)
    ob_bottom = ob_result["Bottom"].to_numpy(dtype=np.float64)
    ob_sig = ob_result["OB"].to_numpy(dtype=np.float64)

    bullish_signals = np.zeros(n, dtype=np.int8)
    bearish_signals = np.zeros(n, dtype=np.int8)
    rejections: list[RejectionBlockSignal] = []

    for i in range(3, n):
        body = abs(close[i] - open_[i])
        upper_wick = high[i] - max(close[i], open_[i])
        lower_wick = min(close[i], open_[i]) - low[i]

        if body <= 0:
            continue

        # Check proximity to structural zones
        buffer = zone_buffer_atr * atr[i]

        # Bearish rejection: long upper wick near zone TOP
        if upper_wick / body >= wick_body_ratio_min:
            # Check if rejection occurred near a BEARISH FVG or BEARISH OB (supply zone)
            for j in range(max(0, i - 10), i):
                if not np.isnan(fvg_sig[j]) and fvg_sig[j] == -1:
                    zone_top = fvg_top[j]
                    if abs(high[i] - zone_top) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and session_active[i]
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(upper_wick/body, vol_ratio, on_session, has_induce)
                        rejections.append(RejectionBlockSignal(
                            bar_index=i, direction=-1, level=zone_top,
                            wick_ratio=upper_wick/body, volume_ratio=vol_ratio,
                            session_formed=on_session, inducement_present=has_induce,
                            confidence=conf,
                        ))
                        bearish_signals[i] = -1
                        break
                if not np.isnan(ob_sig[j]) and ob_sig[j] == -1:
                    zone_top = ob_top[j]
                    if abs(high[i] - zone_top) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and session_active[i]
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(upper_wick/body, vol_ratio, on_session, has_induce)
                        rejections.append(RejectionBlockSignal(
                            bar_index=i, direction=-1, level=zone_top,
                            wick_ratio=upper_wick/body, volume_ratio=vol_ratio,
                            session_formed=on_session, inducement_present=has_induce,
                            confidence=conf,
                        ))
                        bearish_signals[i] = -1
                        break

        # Bullish rejection: long lower wick near zone BOTTOM
        if lower_wick / body >= wick_body_ratio_min:
            for j in range(max(0, i - 10), i):
                if not np.isnan(fvg_sig[j]) and fvg_sig[j] == 1:
                    zone_bottom = fvg_bottom[j]
                    if abs(low[i] - zone_bottom) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and session_active[i]
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(lower_wick/body, vol_ratio, on_session, has_induce)
                        rejections.append(RejectionBlockSignal(
                            bar_index=i, direction=1, level=zone_bottom,
                            wick_ratio=lower_wick/body, volume_ratio=vol_ratio,
                            session_formed=on_session, inducement_present=has_induce,
                            confidence=conf,
                        ))
                        bullish_signals[i] = 1
                        break
                if not np.isnan(ob_sig[j]) and ob_sig[j] == 1:
                    zone_bottom = ob_bottom[j]
                    if abs(low[i] - zone_bottom) <= buffer:
                        vol_ratio = volume[i] / (volume_sma_20[i] + 1e-10)
                        on_session = session_active is not None and session_active[i]
                        has_induce = _has_inducement(high, low, i)
                        conf = _rejection_confidence(lower_wick/body, vol_ratio, on_session, has_induce)
                        rejections.append(RejectionBlockSignal(
                            bar_index=i, direction=1, level=zone_bottom,
                            wick_ratio=lower_wick/body, volume_ratio=vol_ratio,
                            session_formed=on_session, inducement_present=has_induce,
                            confidence=conf,
                        ))
                        bullish_signals[i] = 1
                        break

    return bullish_signals, bearish_signals, rejections


def _has_inducement(high: np.ndarray, low: np.ndarray, idx: int, lookback: int = 5) -> bool:
    """Check if there's a small counter-move (inducement) before the rejection."""
    if idx < lookback + 2:
        return False
    recent_range = np.max(high[idx - lookback:idx]) - np.min(low[idx - lookback:idx])
    if recent_range <= 0:
        return False
    # Check for small counter-move in last 2-3 bars
    counter_move = abs(high[idx] - high[idx - 2]) if high[idx] > high[idx - 2] else abs(low[idx] - low[idx - 2])
    return counter_move > 0.2 * recent_range


def _rejection_confidence(
    wick_body_ratio: float,
    volume_ratio: float,
    session_formed: bool,
    inducement_present: bool,
) -> float:
    """Score rejection confidence 0.0-1.0."""
    score = 0.0
    score += min(wick_body_ratio / 5.0, 1.0) * 0.35
    score += np.clip(volume_ratio / 3.0, 0.0, 1.0) * 0.25
    if session_formed:
        score += 0.20
    if inducement_present:
        score += 0.20
    return np.clip(score, 0.0, 0.95)
```

**Verify:**
```bash
uv run python -c "
from src.patterns.smc.rejection import detect_rejection_blocks
from src.patterns.smc.mitigation import detect_mitigation_blocks
from src.patterns.smc.breaker import detect_breaker_blocks
print('All SMC pattern modules import OK')
"
```

---

### TASK 6.5: Integrate New SMC Detectors into `smc_strategy.py`

In `src/strategies/smc_strategy.py`, add to `_precompute_indicators()`:

```python
def _precompute_indicators(self):
    # ... existing computations ...

    # Phase 6: New SMC detectors
    from src.patterns.smc.breaker import detect_breaker_blocks
    from src.patterns.smc.mitigation import detect_mitigation_blocks
    from src.patterns.smc.rejection import detect_rejection_blocks

    ob_result = smc.ob(self._df, self._swing_highs_lows, close_mitigation=False)
    fvg_result = smc.fvg(self._df, join_consecutive=True)

    self._breaker_bull, self._breaker_bear, self._breakers = detect_breaker_blocks(
        self._df, self._swing_highs_lows, self._atr, self._vol_sma,
        session_active=self._session_active,
    )
    self._mitigation_bull, self._mitigation_bear, self._mitigations = detect_mitigation_blocks(
        self._df, self._swing_highs_lows, self._atr,
        session_active=self._session_active,
    )
    self._rejection_bull, self._rejection_bear, self._rejections = detect_rejection_blocks(
        self._df, fvg_result, ob_result, self._atr, self._vol_sma,
        session_active=self._session_active,
    )
```

Update signal score weights in `__init__`:
```python
SMC_COMPONENT_WEIGHTS = {
    "sweep_reversal": 0.75,
    "breaker_block": 0.70,        # NEW
    "mss_bos_choch": 0.65,
    "mitigation_block": 0.60,     # NEW
    "fvg_proximity": 0.55,
    "rejection_block": 0.55,      # NEW
    "order_block_proximity": 0.50,
}
```

Update `_compute_smc_score()`:
```python
def _compute_smc_score(self, idx: int) -> float:
    # ... existing component aggregation ...

    # Add Breaker Block contribution
    breaker_dir = self._breaker_bull[idx] - self._breaker_bear[idx]
    if breaker_dir != 0:
        score += breaker_dir * self.SMC_COMPONENT_WEIGHTS["breaker_block"]

    # Add Mitigation Block contribution
    mit_dir = self._mitigation_bull[idx] - self._mitigation_bear[idx]
    if mit_dir != 0:
        score += mit_dir * self.SMC_COMPONENT_WEIGHTS["mitigation_block"]

    # Add Rejection Block contribution
    rej_dir = self._rejection_bull[idx] - self._rejection_bear[idx]
    if rej_dir != 0:
        score += rej_dir * self.SMC_COMPONENT_WEIGHTS["rejection_block"]

    return np.tanh(score)
```

**Verify with backtest:**
```bash
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-06-30 --json -o reports/smc/phase6_baseline.json
```

---

### TASK 6.6: Run Pre/Post Phase 6 Comparison

```bash
# Pre-Phase 6 (existing components only)
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-12-31 --json -o reports/smc/pre_phase6.json

# Post-Phase 6 (with Breaker + Mitigation + Rejection)
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-12-31 --json -o reports/smc/post_phase6.json
```

**Acceptance criteria:**
- Backtest runs to completion (no crashes from new detectors)
- Signal count increases (more entry opportunities with 3 new components)
- Win rate maintained or improved
- Sharpe, Profit Factor, MaxDD compared pre/post — report which components contribute most

---

## PHASE 7: NEW CHART PATTERNS — Island Reversal, Dragon, Quasimodo, NR4, Inside Bar (P0)

**Goal:** Add the 8 highest-reliability chart patterns discovered across 6 authoritative sources.
**Time:** ~90 minutes
**Depends on:** Phases 1-6 complete
**Output:** New pattern detectors in `src/patterns/`

---

### TASK 7.1: Island Reversal — Gap/Event Pattern (`src/patterns/event/island_reversal.py`)

**Reliability:** "Among best for DOWNWARD signals" — Kirkpatrick/Fidelity, NCFE, Duddella
**Detection:** Gap that isolates a price cluster from prior trend, then reverse gap back closes the island

```python
"""Island Reversal pattern detector.

Island Top: gap up → isolated trading (1+ sessions) → gap down ← powerful bearish reversal
Island Bottom: gap down → isolated trading → gap up ← powerful bullish reversal

Source: Kirkpatrick (CMT textbook), NCFE, Duddella 2007
Reliability: Among Best Multi-Bar Patterns for downward signals
"""

def detect_island_reversal(
    ohlc: pd.DataFrame,
    gap_threshold_pct: float = 0.005,   # Minimum gap as % of price
    min_isolation_bars: int = 1,         # Minimum bars in the island
    max_isolation_bars: int = 10,        # Maximum bars in the island
) -> np.ndarray:
    """Detect Island Reversals.

    Bullish (Island Bottom):
        open[i] - prev_close > threshold AND
        N bars of isolated trading AND
        next_open (after island) > island_high — gap up out

    Bearish (Island Top):
        prev_close - open[i] > threshold AND
        N bars of isolated trading AND
        next_open < island_low — gap down out

    Validness: the island's range should NOT overlap with surrounding price range
    """
    open_ = ohlc["Open"].to_numpy()
    high = ohlc["High"].to_numpy()
    low = ohlc["Low"].to_numpy()
    close = ohlc["Close"].to_numpy()
    n = len(open_)
    signals = np.zeros(n, dtype=np.int8)

    price_level = (close + open_) / 2.0
    threshold = gap_threshold_pct * price_level

    for i in range(1, n - max_isolation_bars - 1):
        # Bearish Island Top: gap UP into island, then gap DOWN out
        gap_up_size = open_[i] - close[i - 1]
        if gap_up_size > threshold[i]:
            for iso_len in range(min_isolation_bars, max_isolation_bars + 1):
                if i + iso_len >= n - 1:
                    break
                island_high = np.max(high[i:i + iso_len])
                island_low = np.min(low[i:i + iso_len])
                # Island must be isolated: previous close < island low, next open < island low
                if close[i - 1] < island_low and open_[i + iso_len] < island_low:
                    gap_down = island_low - open_[i + iso_len]
                    if gap_down > threshold[i + iso_len]:
                        signals[i + iso_len] = -1
                        break
                # Also: island range isolated (high < min of surrounding range)
                if close[i - 1] < island_low and open_[i + iso_len] < island_low:
                    gap_down = island_low - open_[i + iso_len]
                    if gap_down > threshold[i + iso_len]:
                        signals[i + iso_len] = -1
                        break

        # Bullish Island Bottom: gap DOWN into island, then gap UP out
        gap_down_size = close[i - 1] - open_[i]
        if gap_down_size > threshold[i]:
            for iso_len in range(min_isolation_bars, max_isolation_bars + 1):
                if i + iso_len >= n - 1:
                    break
                island_high = np.max(high[i:i + iso_len])
                island_low = np.min(low[i:i + iso_len])
                if close[i - 1] > island_high and open_[i + iso_len] > island_high:
                    gap_up = open_[i + iso_len] - island_high
                    if gap_up > threshold[i + iso_len]:
                        signals[i + iso_len] = 1
                        break

    return signals
```

**File location:** `src/patterns/event/island_reversal.py` (create `src/patterns/event/` if not exists, or add to existing `event` dir)

---

### TASK 7.2: Dragon Pattern — Exotic Fibonacci Pattern (`src/patterns/exotic/dragon.py`)

**Reliability:** "Very reliable" — Duddella 2007, Ch.12
**Detection:** Head → First Leg → Hump (38-50% retrace of first leg) → Second Leg (mirrors first leg)
Entry on trendline break above hump.

```python
"""Dragon Pattern detector — Duddella 2007, Ch.12 Exotic Patterns.

Structure:
    A (Head) → B (First Leg low) → C (Hump, 38-50% retrace of AB)
    → D (Second Leg low, mirrors AB range)
Entry: Close above trendline connecting A→C (Head to Hump)
Stop: Below lowest low of the two legs (B or D)
TP1: 1.27 × CD range; TP2: A level (Head)

Very reliable per Duddella's testing.
"""

def detect_dragon_pattern(
    swing_lows: np.ndarray,
    swing_low_levels: np.ndarray,
    swing_highs: np.ndarray,
    swing_high_levels: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """Detect Dragon (bullish) and Inverse Dragon (bearish).
    Returns +1 at Dragon breakout, -1 at Inverse Dragon breakdown.
    """
    # Find three consecutive swing lows with a swing high between first two
    # A = first swing low (Head — NOT head of H&S, it's the START)
    # B = first leg low (lower than A, steep drop)
    # C = hump (swing high between B and D, retracing 38-50% of AB)
    # D = second leg low (approximately equal to B)
    # Entry: close above trendline A→C
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    # Find swing points indices
    sw_low_indices = np.where(swing_lows == 1)[0]
    sw_high_indices = np.where(swing_highs == 1)[0]

    for i in range(len(sw_low_indices) - 2):
        a_idx = sw_low_indices[i]
        b_idx = sw_low_indices[i + 1]
        d_idx = sw_low_indices[i + 2]

        a_level = swing_low_levels[a_idx]
        b_level = swing_low_levels[b_idx]
        d_level = swing_low_levels[d_idx]

        # First leg must be downward (B < A)
        if b_level >= a_level:
            continue

        # Second leg approximately equal to first (within 15% ATR)
        atr_val = atr[d_idx]
        if abs(b_level - d_level) > 0.15 * atr_val:
            continue

        # Find hump (swing high) between B and D
        hump_candidates = [h for h in sw_high_indices if b_idx < h < d_idx]
        if not hump_candidates:
            continue
        c_idx = hump_candidates[-1]
        c_level = swing_high_levels[c_idx]

        # Hump must retrace 38-50% of AB leg
        ab_range = a_level - b_level
        c_retrace = (c_level - b_level) / ab_range if ab_range > 0 else 0
        if c_retrace < 0.38 or c_retrace > 0.50:
            continue

        # Entry: close above trendline from A (head) through C (hump)
        for j in range(d_idx + 1, min(d_idx + 20, n)):
            slope = (c_level - a_level) / (c_idx - a_idx) if c_idx > a_idx else 0
            trendline_at_j = a_level + slope * (j - a_idx)
            if close[j] > trendline_at_j:
                signals[j] = 1
                break

    return signals
```

**File:** `src/patterns/exotic/dragon.py`

---

### TASK 7.3: NR4 + Inside Bar — Volatility Contraction (`src/patterns/volatility/`)

**Reliability:** "Very powerful" (Toby Crabel), "moderate-high" (Duddella). Two of the most proven short-term expansion patterns.

```python
"""NR4 (Narrow Range 4) and Inside Bar detectors.

NR4: 4-bar pattern where the 4th bar has narrower range than preceding 3.
     Break above NR4 high = buy; break below NR4 low = sell.
     Source: Toby Crabel, Fidelity, Duddella 2007

Inside Bar: Current bar's range (high-low) entirely within previous bar's range.
            Signals low volatility → imminent expansion.
            Source: Fidelity, NCFE, Duddella 2007
"""

def detect_nr4(ohlc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect NR4 patterns.

    Returns:
        nr4_bar: Boolean array marking NR4 bars
        buy_breakout: +1 on bar closing above NR4 high
        sell_breakout: -1 on bar closing below NR4 low
    """
    high = ohlc["High"].to_numpy()
    low = ohlc["Low"].to_numpy()
    close = ohlc["Close"].to_numpy()
    n = len(close)

    ranges = high - low
    nr4_bar = np.zeros(n, dtype=bool)
    buy_breakout = np.zeros(n, dtype=np.int8)
    sell_breakout = np.zeros(n, dtype=np.int8)

    for i in range(3, n):
        current_range = ranges[i]
        prev_ranges = ranges[i - 3:i]  # 3 preceding bars
        if current_range < np.min(prev_ranges):
            nr4_bar[i] = True

    # Detect breakouts from NR4 bars
    for i in range(4, n):
        if nr4_bar[i - 1]:
            nr4_high = high[i - 1]
            nr4_low = low[i - 1]
            if close[i] > nr4_high:
                buy_breakout[i] = 1
            elif close[i] < nr4_low:
                sell_breakout[i] = -1

    return nr4_bar, buy_breakout, sell_breakout


def detect_inside_bar(ohlc: pd.DataFrame) -> np.ndarray:
    """Detect Inside Bars. Returns +1/-1 at break of inside bar range."""
    high = ohlc["High"].to_numpy()
    low = ohlc["Low"].to_numpy()
    close = ohlc["Close"].to_numpy()
    n = len(close)

    signals = np.zeros(n, dtype=np.int8)

    for i in range(2, n):
        if high[i - 1] <= high[i - 2] and low[i - 1] >= low[i - 2]:
            # Inside bar at i-1 — signal at bar i
            if close[i] > high[i - 1]:
                signals[i] = 1
            elif close[i] < low[i - 1]:
                signals[i] = -1

    return signals
```

**File:** `src/patterns/volatility/nr4_inside_bar.py` (create `src/patterns/volatility/` dir)

---

### TASK 7.4: Quasimodo — Advanced Reversal Pattern (`src/patterns/complex/quasimodo.py`)

**From CPF resource:**
- Modified H&S with asymmetrical shoulders
- Neckline break → pullback → false breakout through neckline → reversal
- High-probability reversal pattern

```python
"""Quasimodo Pattern detector — high-probability reversal (CPF resource).

Structure:
- Left shoulder at level L1
- Head at a higher high (bull trap)
- Right shoulder at level R1 (< L1)
- Neckline break below R1
- Pullback above neckline (false breakout)
- Then breakdown for entry
"""

def detect_quasimodo(swing_highs, swing_high_levels, swing_lows, swing_low_levels,
                     close, high, low, atr, n) -> np.ndarray:
    """Detect Quasimodo patterns. +1 bullish, -1 bearish."""
    signals = np.zeros(n, dtype=np.int8)
    # ... implementation with 5-point swing structure + false breakout check ...
    return signals
```

**File:** `src/patterns/complex/quasimodo.py`

---

### TASK 7.5: Adam-Eve Pattern (`src/patterns/classic/adam_eve.py`)

**Very reliable double top/bottom variant (Duddella Ch.11).**

```python
"""Adam-Eve Pattern — sharp (Adam) vs rounded (Eve) double tops/bottoms.

Adam = Sharp V-spike (single bar or tight cluster)
Eve = Rounded, drawn-out formation

Sequence possibilities: Adam-Adam, Adam-Eve, Eve-Eve, Eve-Adam
Trade: Breakout of the middle spike between the two formations.
"""

def detect_adam_eve(ohlc, swing_highs, swing_lows, atr) -> np.ndarray:
    """Detect Adam-Eve patterns.

    Classification heuristic:
    - Adam: 1-2 bars, sharp reversal, wick-to-body > 2
    - Eve: 3+ bars, rounded, gradual reversal

    Entry: Break above/below the middle valley/peak
    """
    # ... implementation ...
    pass
```

**File:** `src/patterns/classic/adam_eve.py`

---

### TASK 7.6: Three Valleys and A River (`src/patterns/classic/three_valleys.py`)

**Inverse of Three Hills (already in project). "Very reliable" — Duddella.**

```python
"""Three Valleys and A River — inverse of Three Hills (Duddella 2007 Ch.10).

Structure: 3 progressively higher valleys (swing lows) within a rising trendline channel.
Entry: Break above trendline connecting the 3 valley peaks.
Target: 62-78% of AB range (first valley to first peak).
Then short from C (62% retracement from peak to third valley).
"""
```

**File:** `src/patterns/classic/three_valleys.py`

---

### TASK 7.7: Shooting Star / Inverted Hammer (`src/patterns/candlestick/shooting_star.py`)

```python
"""Shooting Star / Inverted Hammer single-candle patterns.

Shooting Star (after uptrend): Upper wick >= 2x body, no lower wick → bearish reversal
Inverted Hammer (after downtrend): Same shape → bullish reversal

Source: Fidelity/Kirkpatrick
"""
```

**File:** `src/patterns/candlestick/shooting_star.py`

---

### TASK 7.8: Key Reversal (`src/patterns/basic/key_reversal.py`)

```python
"""Key Reversal bar — end-of-trend indicator (NCFE).

Bearish Key Reversal: New high but closes lower than previous close, on high volume
Bullish Key Reversal: New low but closes higher than previous close, on high volume
"""
```

**File:** `src/patterns/basic/key_reversal.py`

---

### TASK 7.9: Gap Type Classification — Extend Existing `gap.py`

**Add to `src/patterns/breakout/gap.py`:**

```python
class GapType(enum.Enum):
    COMMON = "common"         # Within congestion, quick fill
    BREAKAWAY = "breakaway"   # At trendline break, most reliable
    CONTINUATION = "measuring"  # Mid-trend, ~halfway
    EXHAUSTION = "exhaustion"  # End of move, blowoff → reversal


def classify_gap_type(ohlc: pd.DataFrame, gap_indices: np.ndarray) -> dict[int, GapType]:
    """Classify each detected gap by type using context rules.

    Rules (Duddella 2007, NCFE):
    - Breakaway: gap at trendline penetration + volume surge + after consolidation
    - Continuation: mid-trend, volume lower than breakaway
    - Exhaustion: wide gap + heavy volume + near end of extended move
    - Common: within trading range, low volume, narrow gap

    Gap validity rule: daily gap > 2.5× 10-day ATR → SKIP (unreliable)
    """
    ...


def is_gap_tradable(ohlc: pd.DataFrame, gap_idx: int, atr: np.ndarray) -> bool:
    """Duddella gap validity check: gap must be <= 2.5× 10-day ATR."""
    gap_size = abs(ohlc.iloc[gap_idx]["Open"] - ohlc.iloc[gap_idx - 1]["Close"])
    avg_atr_10 = atr[gap_idx - 10:gap_idx].mean() if gap_idx >= 10 else atr[gap_idx]
    return gap_size <= 2.5 * avg_atr_10
```

---

### TASK 7.10: Verify All New Patterns

```bash
# Import check
uv run python -c "
from src.patterns.event import island_reversal
from src.patterns.exotic import dragon
from src.patterns.volatility import nr4_inside_bar
from src.patterns.complex import quasimodo
from src.patterns.classic import adam_eve, three_valleys
from src.patterns.candlestick import shooting_star
from src.patterns.basic import key_reversal
print('All new pattern modules import OK')
"

# Quick backtest integration
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-06-30
```

---

## PHASE 8: ADVANCED SMC — SMT Divergence, PD Array Matrix, Fibonacci Enhancements (P1)

**Goal:** Multi-instrument SMT divergence, structural premium/discount tracking, body-to-body Fibonacci.
**Time:** ~90 minutes
**Depends on:** Phases 1-7 complete
**Output:** `src/signals/smc_divergence.py`, `src/patterns/smc/pd_array_matrix.py`, enhanced `src/indicators/ote.py`

---

### TASK 8.1: SMT Divergence — Multi-Asset Correlated Divergence

**ICT MMXM definition:**
- Bullish SMT: Price makes LOWER LOW on one chart but NOT on correlated chart → reversal
- Bearish SMT: Price makes HIGHER HIGH on one chart but NOT on correlated chart → reversal
- Typically used with correlated pairs: ES/NQ, EURUSD/GBPUSD, BTC/ETH

**File:** `src/signals/smc_divergence.py` (~250 lines)

```python
"""SMT (Smart Money Technique) Divergence detector.

Detects when correlated assets diverge at key SMC structural levels,
signaling market manipulation and impending reversal.

Source: ICT MMXM Model, toaz.info glossary
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SMTDivergence:
    bar_index: int
    direction: int          # +1 bullish divergence, -1 bearish divergence
    primary_symbol: str      # Symbol being analyzed
    correlated_symbol: str   # Reference correlated asset
    primary_level: float     # Extreme level on primary
    correlated_level: float  # Corresponding extreme on correlated
    divergence_strength: float  # How far apart the extremes are (normalized)


def detect_smt_divergence(
    primary_df: pd.DataFrame,
    correlated_df: pd.DataFrame,
    lookback_bars: int = 20,
    correlation_window: int = 50,
    min_correlation: float = 0.7,
    divergence_threshold_pct: float = 0.3,
) -> tuple[np.ndarray, list[SMTDivergence]]:
    """Detect SMT divergences between two correlated instruments.

    Args:
        primary_df: OHLCV DataFrame for primary instrument
        correlated_df: OHLCV DataFrame for correlated instrument
        lookback_bars: Lookback window for comparing swing extremes
        correlation_window: Rolling correlation window
        min_correlation: Minimum correlation to consider instruments "correlated"
        divergence_threshold_pct: Minimum divergence as % of ATR

    Returns:
        signals: +1 bullish divergence, -1 bearish divergence
        divergences: list of SMTDivergence dataclasses

    Pairs reference (from ICT MMXM):
        - ES ↔ NQ (S&P 500 ↔ Nasdaq)
        - EURUSD ↔ GBPUSD
        - BTCUSD ↔ ETHUSD
        - XAUUSD ↔ XAGUSD (Gold ↔ Silver)
    """
    p_high = primary_df["High"].to_numpy()
    p_low = primary_df["Low"].to_numpy()
    c_high = correlated_df["High"].to_numpy()
    c_low = correlated_df["Low"].to_numpy()
    n = min(len(p_high), len(c_high))

    signals = np.zeros(n, dtype=np.int8)
    divergences: list[SMTDivergence] = []

    # Validate correlation holds
    p_close = primary_df["Close"].to_numpy()
    c_close = correlated_df["Close"].to_numpy()
    rolling_corr = pd.Series(p_close).rolling(correlation_window).corr(pd.Series(c_close)).fillna(0).to_numpy()

    for i in range(lookback_bars * 2, n):
        if rolling_corr[i] < min_correlation:
            continue

        # Find recent swing highs/lows within lookback
        window_p_high = p_high[i - lookback_bars:i]
        window_p_low = p_low[i - lookback_bars:i]
        window_c_high = c_high[i - lookback_bars:i]
        window_c_low = c_low[i - lookback_bars:i]

        p_recent_high = np.max(window_p_high)
        p_recent_low = np.min(window_p_low)
        c_recent_high = np.max(window_c_high)
        c_recent_low = np.min(window_c_low)

        p_range = p_recent_high - p_recent_low
        if p_range <= 0:
            continue

        # Check for higher high divergence
        p_latest_high = p_high[i]
        c_latest_high = c_high[i]
        if p_latest_high > p_recent_high * (1 + divergence_threshold_pct * 0.01):
            if c_latest_high <= c_recent_high:  # Correlated did NOT make higher high
                strength = (p_latest_high - p_recent_high) / p_recent_high
                divergences.append(SMTDivergence(
                    bar_index=i, direction=-1,
                    primary_symbol="PRIMARY", correlated_symbol="CORRELATED",
                    primary_level=p_latest_high, correlated_level=c_latest_high,
                    divergence_strength=min(strength, 1.0),
                ))
                signals[i] = -1

        # Check for lower low divergence
        p_latest_low = p_low[i]
        c_latest_low = c_low[i]
        if p_latest_low < p_recent_low * (1 - divergence_threshold_pct * 0.01):
            if c_latest_low >= c_recent_low:  # Correlated did NOT make lower low
                strength = (p_recent_low - p_latest_low) / p_recent_low
                divergences.append(SMTDivergence(
                    bar_index=i, direction=1,
                    primary_symbol="PRIMARY", correlated_symbol="CORRELATED",
                    primary_level=p_latest_low, correlated_level=c_latest_low,
                    divergence_strength=min(strength, 1.0),
                ))
                signals[i] = 1

    return signals, divergences


# Predefined correlated pairs
SMC_CORRELATED_PAIRS = {
    "SPY": "QQQ",
    "QQQ": "SPY",
    "ES=F": "NQ=F",
    "NQ=F": "ES=F",
    "EURUSD=X": "GBPUSD=X",
    "GBPUSD=X": "EURUSD=X",
    "BTC-USD": "ETH-USD",
    "ETH-USD": "BTC-USD",
    "GC=F": "SI=F",
    "SI=F": "GC=F",
}
```

---

### TASK 8.2: PD Array Matrix — Multi-Timeframe Premium/Discount

**File:** `src/patterns/smc/pd_array_matrix.py` (~200 lines)

```python
"""PD Array Matrix — ICT 2022 mentorship multi-timeframe Premium/Discount tracking.

The PD (Price Delivery) Array Matrix organizes all SMC structural elements
across timeframes into a hierarchical premium/discount framework.

Premium zone (above equilibrium):
    PDH → Rejection Block → Bearish OB → FVG → LQ Void → Bearish Breaker → Bearish Mitigation Block
Equilibrium (50%):
    Midpoint of dealing range
Discount zone (below equilibrium):
    Bullish Mitigation Block → Bullish Breaker → LQ Void → FVG → Bullish OB → Rejection Block → PDL

Source: toaz.info ICT glossary, David Woods
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class PDZone(enum.Enum):
    PREMIUM_EXTREME = "premium_extreme"
    PREMIUM = "premium"
    EQUILIBRIUM = "equilibrium"
    DISCOUNT = "discount"
    DISCOUNT_EXTREME = "discount_extreme"


@dataclass
class PDArray:
    """A single Price Delivery array (level/zone) with metadata."""
    level: float
    zone: PDZone
    timeframe: str         # "D", "4H", "1H", "15m"
    array_type: str        # "OB", "FVG", "Breaker", "Mitigation", "Rejection", "LiquidityVoid"
    direction: int         # +1 support, -1 resistance
    bar_index: int
    active: bool = True
    strength: float = 0.5  # 0.0-1.0 based on volume/session context


@dataclass
class PDArrayMatrix:
    """Multi-timeframe collection of PD Arrays for a single instrument."""
    daily_range_high: float = 0.0
    daily_range_low: float = 0.0
    equilibrium: float = 0.0   # (Daily high + Daily low) / 2
    arrays: list[PDArray] = field(default_factory=list)

    def is_in_premium(self, price: float) -> bool:
        return price > self.equilibrium

    def is_in_discount(self, price: float) -> bool:
        return price < self.equilibrium

    def get_active_arrays(self, zone: Optional[PDZone] = None) -> list[PDArray]:
        """Get all active arrays, optionally filtered by zone."""
        active = [a for a in self.arrays if a.active]
        if zone:
            active = [a for a in active if a.zone == zone]
        return active

    def get_nearest_array(self, price: float, direction: int) -> Optional[PDArray]:
        """Find the nearest active array in the given price direction."""
        candidates = [a for a in self.arrays if a.active]
        if direction == 1:  # Looking above
            candidates = [a for a in candidates if a.level > price]
            key = lambda a: a.level
        else:  # Looking below
            candidates = [a for a in candidates if a.level < price]
            key = lambda a: -a.level
        candidates.sort(key=key)
        return candidates[0] if candidates else None


def build_pd_array_matrix(
    ohlc: pd.DataFrame,
    fvg_result: pd.DataFrame,
    ob_result: pd.DataFrame,
    breakers: list,
    mitigations: list,
    rejections: list,
    timeframe: str = "1H",
) -> PDArrayMatrix:
    """Build a PD Array Matrix for the given timeframe from SMC detected structures.

    Puts detected zones into the premium/discount hierarchy based on their
    position relative to the dealing range equilibrium.
    """
    high = ohlc["High"]
    low = ohlc["Low"]

    daily_high = high.max()
    daily_low = low.min()
    equilibrium = (daily_high + daily_low) / 2

    matrix = PDArrayMatrix(
        daily_range_high=daily_high,
        daily_range_low=daily_low,
        equilibrium=equilibrium,
    )

    # Classify FVGs
    for i in range(len(ohlc)):
        if not pd.isna(fvg_result["FVG"].iloc[i]):
            direction = int(fvg_result["FVG"].iloc[i])
            top = fvg_result["Top"].iloc[i]
            bottom = fvg_result["Bottom"].iloc[i]
            level = top if direction == 1 else bottom
            zone = _classify_zone(level, daily_high, daily_low)
            matrix.arrays.append(PDArray(
                level=level, zone=zone, timeframe=timeframe, array_type="FVG",
                direction=direction, bar_index=i,
            ))

    # Classify OBs
    for i in range(len(ohlc)):
        if not pd.isna(ob_result["OB"].iloc[i]) and ob_result["OB"].iloc[i] != 0:
            direction = int(ob_result["OB"].iloc[i])
            top = ob_result["Top"].iloc[i]
            bottom = ob_result["Bottom"].iloc[i]
            level = bottom if direction == 1 else top
            zone = _classify_zone(level, daily_high, daily_low)
            strength = ob_result["Percentage"].iloc[i] / 100.0 if not pd.isna(ob_result["Percentage"].iloc[i]) else 0.5
            matrix.arrays.append(PDArray(
                level=level, zone=zone, timeframe=timeframe, array_type="OB",
                direction=direction, bar_index=i, strength=strength,
            ))

    # Classify Breaker Blocks
    for breaker in breakers:
        level = breaker.bottom if breaker.direction == 1 else breaker.top
        zone = _classify_zone(level, daily_high, daily_low)
        matrix.arrays.append(PDArray(
            level=level, zone=zone, timeframe=timeframe, array_type="Breaker",
            direction=breaker.direction, bar_index=breaker.bar_index,
            strength=breaker.strength,
        ))

    # Classify Mitigation Blocks
    for mit in mitigations:
        zone = _classify_zone(mit.level, daily_high, daily_low)
        matrix.arrays.append(PDArray(
            level=mit.level, zone=zone, timeframe=timeframe, array_type="Mitigation",
            direction=mit.direction, bar_index=mit.bar_index,
            strength=mit.strength,
        ))

    # Classify Rejection Blocks
    for rej in rejections:
        zone = _classify_zone(rej.level, daily_high, daily_low)
        matrix.arrays.append(PDArray(
            level=rej.level, zone=zone, timeframe=timeframe, array_type="Rejection",
            direction=rej.direction, bar_index=rej.bar_index,
            strength=rej.confidence,
        ))

    return matrix


def _classify_zone(level: float, range_high: float, range_low: float) -> PDZone:
    """Classify a price level into PD Zone based on dealing range."""
    range_size = range_high - range_low
    if range_size <= 0:
        return PDZone.EQUILIBRIUM
    position = (level - range_low) / range_size

    if position > 0.85:
        return PDZone.PREMIUM_EXTREME
    elif position > 0.60:
        return PDZone.PREMIUM
    elif position > 0.35:
        return PDZone.EQUILIBRIUM
    elif position > 0.15:
        return PDZone.DISCOUNT
    else:
        return PDZone.DISCOUNT_EXTREME
```

---

### TASK 8.3: Fibonacci Enhancements — Body-to-Body + Extended Levels

**Enhance existing `src/indicators/ote.py`:**

```python
# Add to existing calculate_ote_zone():
FIB_LEVELS = [0, 0.50, 0.618, 0.705, 0.79, 1.0, -0.27, -0.62, -1.0]
FIB_TP_LEVELS = [-0.27, -0.62, -1.0]  # TP1, TP2, TP3 (symmetrical swing)


def fibonacci_body_to_body(ohlc: pd.DataFrame, swing_start_idx: int, swing_end_idx: int,
                           direction: int) -> dict[float, float]:
    """Calculate Fibonacci levels using candle BODIES (not wicks).

    ICT Order Block & Fibonacci rule: use body highs/lows to avoid broker variance.

    Bullish: lowest body low → highest body high
    Bearish: highest body high → lowest body low

    Returns dict mapping fib level to price.
    """
    open_ = ohlc["Open"].to_numpy()
    close = ohlc["Close"].to_numpy()

    # Body = max(open, close), min(open, close)
    body_highs = np.maximum(open_, close)
    body_lows = np.minimum(open_, close)

    if direction == 1:  # Uptrend swing
        start_body = np.min(body_lows[swing_start_idx:swing_end_idx + 1])
        end_body = np.max(body_highs[swing_start_idx:swing_end_idx + 1])
        price_range = end_body - start_body
    else:  # Downtrend swing
        start_body = np.max(body_highs[swing_start_idx:swing_end_idx + 1])
        end_body = np.min(body_lows[swing_start_idx:swing_end_idx + 1])
        price_range = start_body - end_body

    levels = {}
    for level in FIB_LEVELS:
        if direction == 1:
            if level >= 0:
                levels[level] = end_body - price_range * level  # Retracement
            else:
                levels[level] = end_body + price_range * abs(level)  # Extension
        else:
            if level >= 0:
                levels[level] = end_body + price_range * level
            else:
                levels[level] = end_body - price_range * abs(level)

    return levels


def get_ote_targets(fib_levels: dict[float, float], direction: int) -> tuple[float, float, float]:
    """Get TP1 (-0.27), TP2 (-0.62), TP3 (-1.0 symmetrical swing) targets."""
    return (
        fib_levels.get(-0.27, 0),
        fib_levels.get(-0.62, 0),
        fib_levels.get(-1.0, 0),
    )
```

---

## PHASE 9: smartmoneyconcepts LIBRARY — Full Integration (P1)

**Goal:** Integrate the 5 remaining library functions not yet wired into the strategy.
**Time:** ~45 minutes
**Depends on:** Phases 1-8 complete, `smartmoneyconcepts` installed

---

### TASK 9.1: Integrate `smc.sessions()` for Killzone Detection

Replace hardcoded session times in `smc_strategy.py` with library's predefined killzones:

```python
def _precompute_sessions(self):
    """Use smartmoneyconcepts sessions() for 9 predefined trading sessions."""
    from smartmoneyconcepts import smc as smc_lib

    # Predefined sessions: Sydney, Tokyo, London, New York,
    # Asian kill zone, London open kill zone, New York kill zone, London close kill zone, Custom
    london_open = smc_lib.sessions(self._df, "London open kill zone", time_zone="UTC+0")
    ny_open = smc_lib.sessions(self._df, "New York kill zone", time_zone="UTC+0")
    london_close = smc_lib.sessions(self._df, "london close kill zone", time_zone="UTC+0")
    asian_kz = smc_lib.sessions(self._df, "Asian kill zone", time_zone="UTC+0")

    # Combine into active trading window
    self._session_active = (
        (london_open["Active"].to_numpy() == 1) |
        (ny_open["Active"].to_numpy() == 1) |
        (london_close["Active"].to_numpy() == 1)
    ).astype(np.int8)

    # Store individual killzone trackers for session-specific logic
    self._in_london_kz = london_open["Active"].to_numpy().astype(bool)
    self._in_ny_kz = ny_open["Active"].to_numpy().astype(bool)
    self._in_asian_kz = asian_kz["Active"].to_numpy().astype(bool)
```

---

### TASK 9.2: Integrate `smc.previous_high_low()` for Period Reference Levels

```python
def _precompute_previous_levels(self):
    """Period-based high/low reference levels for draw-on-liquidity context."""
    from smartmoneyconcepts import smc as smc_lib

    phl = smc_lib.previous_high_low(self._df, time_frame="1D")
    self._prev_daily_high = phl["PreviousHigh"].to_numpy(dtype=np.float64)
    self._prev_daily_low = phl["PreviousLow"].to_numpy(dtype=np.float64)
    self._broke_prev_high = phl["BrokenHigh"].to_numpy(dtype=np.int8)
    self._broke_prev_low = phl["BrokenLow"].to_numpy(dtype=np.int8)

    # Weekly levels
    phl_w = smc_lib.previous_high_low(self._df, time_frame="1W")
    self._prev_weekly_high = phl_w["PreviousHigh"].to_numpy(dtype=np.float64)
    self._prev_weekly_low = phl_w["PreviousLow"].to_numpy(dtype=np.float64)
```

---

### TASK 9.3: Integrate `smc.retracements()` for Dynamic Trend Tracking

```python
def _precompute_retracements(self):
    """Direction-aware Fibonacci retracement tracking."""
    from smartmoneyconcepts import smc as smc_lib

    ret = smc_lib.retracements(self._df, self._swing_highs_lows)
    self._retrace_direction = ret["Direction"].to_numpy(dtype=np.int8)
    self._retrace_current = ret["CurrentRetracement%"].to_numpy(dtype=np.float64)
    self._retrace_deepest = ret["DeepestRetracement%"].to_numpy(dtype=np.float64)

    # Deep retracement (> 78.6%) = potential reversal zone
    self._in_deep_retrace = self._retrace_deepest > 78.6
```

---

### TASK 9.4: Integrate `smc.ob()` Volume Strength for Position Sizing

```python
# In _precompute_indicators():
ob_result = smc.ob(self._df, self._swing_highs_lows, close_mitigation=False)
self._ob_signal = ob_result["OB"].to_numpy(dtype=np.float64)
self._ob_volume = ob_result["OBVolume"].to_numpy(dtype=np.float64)
self._ob_percentage = ob_result["Percentage"].to_numpy(dtype=np.float64)

# Use OB strength for position sizing in position_sizing.py:
# stronger OB → larger position (up to 2x base size)
```

---

## PHASE 10: RISK MANAGEMENT → SMC INTEGRATION (P1)

**Goal:** Wire SMC structural concepts into the risk management system. Currently completely disconnected.
**Time:** ~60 minutes
**Output:** `src/risk/smc_aware.py`

---

### TASK 10.1: Create SMC-Aware Risk Module

**File:** `src/risk/smc_aware.py` (~250 lines)

```python
"""SMC-aware risk management — structural stops, killzone limits, OB-based sizing.

Bridges SMC pattern detection with risk management, replacing generic
ATR-based stops and fixed R-multiples with SMC structural levels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SMCStructuralRisk:
    """Risk parameters derived from SMC structural context."""
    entry_price: float
    stop_price: float               # Behind nearest structural level
    take_profit_1: Optional[float]  # First opposing FVG / liquidity level
    take_profit_2: Optional[float]  # Second opposing structural level
    risk_per_unit: float            # Entry - Stop
    reward_risk_ratio: float        # TP1 - Entry / Risk
    position_risk_pct: float        # 1-2% per trade based on OB strength
    killzone_active: bool           # Amplify/attenuate based on session


def compute_smc_structural_stop(
    entry_price: float,
    entry_direction: int,
    ob_levels: np.ndarray,
    ob_directions: np.ndarray,
    breaker_levels: np.ndarray,
    breaker_directions: np.ndarray,
    atr: float,
    min_atr_buffer: float = 0.5,
) -> float:
    """Place stop loss behind the nearest SMC structural level.

    ICT rules (from SMC docs):
    - Long: SL below nearest Bullish OB (demand zone) or Bullish Breaker
    - Short: SL above nearest Bearish OB (supply zone) or Bearish Breaker
    - If no structural level nearby, fall back to ATR-based stop

    David Woods: "Stop above HVI (High Volume Imbalance)" — highest volume OB
    """
    if entry_direction == 1:  # Long — stop below nearest support
        candidates = []
        for i in range(len(ob_levels)):
            if ob_directions[i] == 1 and ob_levels[i] < entry_price:  # Bullish OB below entry
                candidates.append(ob_levels[i])
        for i in range(len(breaker_levels)):
            if breaker_directions[i] == 1 and breaker_levels[i] < entry_price:
                candidates.append(breaker_levels[i])

        if candidates:
            structural_stop = max(candidates) - min_atr_buffer * atr
            return structural_stop
    else:  # Short — stop above nearest resistance
        candidates = []
        for i in range(len(ob_levels)):
            if ob_directions[i] == -1 and ob_levels[i] > entry_price:
                candidates.append(ob_levels[i])
        for i in range(len(breaker_levels)):
            if breaker_directions[i] == -1 and breaker_levels[i] > entry_price:
                candidates.append(breaker_levels[i])

        if candidates:
            structural_stop = min(candidates) + min_atr_buffer * atr
            return structural_stop

    # Fallback: ATR-based stop
    if entry_direction == 1:
        return entry_price - 3.0 * atr
    else:
        return entry_price + 3.0 * atr


def compute_smc_partial_exit(
    entry_price: float,
    entry_direction: int,
    fvg_top: np.ndarray,
    fvg_bottom: np.ndarray,
    fvg_direction: np.ndarray,
    liquidity_levels: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    """Compute TP1 (first FVG fill) and TP2 (first liquidity level) per ICT MMXM.

    ICT MMXM rules:
    - TP1: First opposing liquidity level providing 2R+
    - TP2: First FVG fill or opposing liquidity level providing 3R+
    """
    tp1 = None
    tp2 = None

    if entry_direction == 1:  # Long
        # TP1: nearest bearish FVG or swing high
        for i in range(len(fvg_top)):
            if fvg_direction[i] == -1 and fvg_top[i] > entry_price:
                if tp1 is None or fvg_top[i] < tp1:
                    tp1 = fvg_top[i]
        # TP2: nearest liquidity level above
        for level in liquidity_levels:
            if level > entry_price:
                if tp2 is None or (level < tp2 and level > (tp1 or entry_price)):
                    tp2 = level
    else:  # Short
        for i in range(len(fvg_bottom)):
            if fvg_direction[i] == 1 and fvg_bottom[i] < entry_price:
                if tp1 is None or fvg_bottom[i] > tp1:
                    tp1 = fvg_bottom[i]
        for level in liquidity_levels:
            if level < entry_price:
                if tp2 is None or (level > tp2 and level < (tp1 or entry_price)):
                    tp2 = level

    return tp1, tp2


def compute_ob_based_position_size(
    base_risk_pct: float,
    ob_strength: float,
    killzone_active: bool,
    confluence_count: int,
) -> float:
    """Scale position size based on OB strength and SMC context.

    SMC rules (Order Block & Fib, Complete SMC):
    - Base: 1% per setup
    - Strong OB (strength > 70%): up to 2%
    - Multiple confluence: +0.25% per additional component
    - Active killzone: +0.25% (higher probability during London/NY)
    - Max: 2% per trade
    """
    risk = base_risk_pct  # 1.0%

    # OB strength scaling
    if ob_strength > 70:
        risk += 0.5
    elif ob_strength > 50:
        risk += 0.25

    # Confluence bonus
    risk += 0.25 * max(0, confluence_count - 1)

    # Killzone bonus
    if killzone_active:
        risk += 0.25

    return min(risk, 2.0)


def compute_daily_smc_limits(
    current_pnl_pct: float,
    daily_profit_target_pct: float = 5.0,
    smc_consecutive_failures: int = 0,
    max_consecutive_losses: int = 3,
) -> dict:
    """SMC-specific daily limits.

    From ICT 1 Year Trading Plan:
    - Daily profit target reached → stop for the day
    - 3 consecutive losses → step away
    - Maximum daily loss = 3%
    """
    limits = {
        "should_stop_trading": False,
        "reason": "",
    }
    if current_pnl_pct >= daily_profit_target_pct:
        limits["should_stop_trading"] = True
        limits["reason"] = f"Daily profit target {daily_profit_target_pct}% reached"
    elif smc_consecutive_failures >= max_consecutive_losses:
        limits["should_stop_trading"] = True
        limits["reason"] = f"{smc_consecutive_failures} consecutive SMC failures"
    return limits
```

---

### TASK 10.2: Create SMC Circuit Breaker

Add to existing `circuit_breakers.py` or in new SMC-aware module:

```python
class SMCCircuitBreaker:
    """SMC-specific circuit breakers.

    - If 3+ consecutive sweeps fail → halt SMC signals
    - If killzone liquidity vanishes (volume < 50% avg) → halt
    - If HTF structure invalidated → halt
    """

    def __init__(self, cooldown_bars: int = 20):
        self.consecutive_sweep_failures = 0
        self.consecutive_bos_failures = 0
        self.cooldown_until = -1
        self.cooldown_bars = cooldown_bars

    def check(self, bar_idx: int, sweep_failed: bool, bos_failed: bool) -> bool:
        """Check if SMC signals should be halted. Returns True if HALT."""
        if bar_idx < self.cooldown_until:
            return True

        if sweep_failed:
            self.consecutive_sweep_failures += 1
        else:
            self.consecutive_sweep_failures = 0

        if bos_failed:
            self.consecutive_bos_failures += 1
        else:
            self.consecutive_bos_failures = 0

        if self.consecutive_sweep_failures >= 3:
            self.cooldown_until = bar_idx + self.cooldown_bars
            self.consecutive_sweep_failures = 0
            return True

        if self.consecutive_bos_failures >= 3:
            self.cooldown_until = bar_idx + self.cooldown_bars
            self.consecutive_bos_failures = 0
            return True

        return False
```

---

## PHASE 11: VOLUME & RELIABILITY INTEGRATION (P2)

**Goal:** Apply empirically-derived pattern reliability weights and volume confirmation rules from Kirkpatrick, NCFE, and Duddella to all pattern detectors.
**Time:** ~45 minutes

---

### TASK 11.1: Pattern Reliability Registry

**File:** `src/signals/pattern_reliability_registry.py` (~100 lines)

```python
"""Pattern reliability weights from authoritative sources.

Sources:
- Kirkpatrick/Fidelity (NCFE quantitative data)
- Duddella 2007 Trade Chart Patterns Guide
- Harmonic Pattern Trading Guides
- NCFE Technical Analysis Price Patterns

Used to weight confluence scoring by pattern reliability.
"""

PATTERN_RELIABILITY = {
    # NCFE-quantified
    "head_and_shoulders": 0.87,       # 86-88% reliability (NCFE)
    "inverse_head_shoulders": 0.80,   # Lower than top (Fidelity)
    "symmetrical_triangle": 0.77,     # 76-78% (NCFE, breakout before 3/4 apex)
    "ascending_triangle": 0.78,       # 75-80% (NCFE)
    "descending_triangle": 0.78,      # 75-80% (NCFE)

    # Fidelity/Kirkpatrick quantified
    "pipe_bottom": 0.85,              # Among best multi-bar for upward
    "pipe_top": 0.85,                 # Symmetric
    "island_reversal_top": 0.88,      # Among best for downward signals
    "island_reversal_bottom": 0.82,   # Slightly weaker than top
    "flag": 0.82,                     # High short-term (NCFE), best for downward
    "pennant": 0.78,                  # High short-term

    # Duddella 2007
    "dragon": 0.85,                   # Very reliable
    "inverse_dragon": 0.85,           # Very reliable
    "three_hills": 0.85,             # Very reliable (already in project)
    "three_valleys": 0.85,            # Very reliable
    "adam_eve": 0.83,                 # Very reliable
    "trader_vic_123": 0.80,           # High
    "msl": 0.70,                      # Moderate, 2/3 success
    "msh": 0.70,                      # Moderate
    "nr4": 0.80,                      # High (Crabel)
    "inside_bar": 0.65,               # Moderate-High
    "crown": 0.68,                    # Moderately reliable

    # Harmonic patterns
    "cypher": 0.88,                   # Highest win rate of all harmonics
    "gartley": 0.82,
    "bat": 0.78,
    "butterfly": 0.76,
    "crab": 0.74,
    "shark": 0.72,
    "abc": 0.75,

    # Standard TA patterns (Fidelity/Duddella)
    "double_top": 0.70,              # Not highly reliable (NCFE)
    "double_bottom": 0.70,
    "triple_top": 0.70,
    "triple_bottom": 0.70,
    "wedge": 0.62,                   # Below average (NCFE)
    "rectangle": 0.72,
    "cup_handle": 0.73,             # Average for bottom patterns

    # SMC components (from David Woods, ICT, experience)
    "sweep_reversal": 0.75,         # Strongest — actual order flow
    "breaker_block": 0.70,          # Failed OB = strong
    "mitigation_block": 0.65,       # Medium
    "fvg_proximity": 0.55,          # Fills are probabilistic
    "rejection_block": 0.60,        # Requires inducement + session
    "order_block": 0.65,            # Supply/demand zones

    # Candlestick patterns
    "doji": 0.40,
    "harami": 0.45,
    "hammer": 0.55,
    "engulfing": 0.60,
    "dark_cloud": 0.58,
    "shooting_star": 0.55,
    "inverted_hammer": 0.50,

    # FMZ / Technical
    "ichimoku": 0.65,
    "keltner_channel": 0.60,
    "williams_r": 0.50,
    "cci": 0.55,
    "bollinger_bands": 0.60,

    # Volatility
    "donchian_channel": 0.60,
    "key_reversal": 0.70,           # End-of-move indicator
    "round_top": 0.40,              # High failure rate (Duddella)
    "round_bottom": 0.45,           # Below average (Duddella)
    "v_top": 0.60,                  # Easy to spot, hard to trade
    "v_bottom": 0.60,
    "scallop_ascending": 0.55,      # Moderate (Duddella)
    "scallop_descending": 0.55,
    "bump_and_run": 0.65,
    "quasimodo": 0.75,              # High-probability reversal
}

# Minimum reliability to include in confluence scoring
MIN_PATTERN_RELIABILITY = 0.40

# Patterns below this threshold still trigger but with warning
WEAK_PATTERN_THRESHOLD = 0.50
```

---

### TASK 11.2: Volume Confirmation Rules

**File:** `src/signals/volume_confirmation_rules.py` (~80 lines)

```python
"""Volume confirmation rules per pattern type.

All rules from: Kirkpatrick/Fidelity, NCFE, Duddella 2007, Warrior Trading.
"""

def get_volume_rule(pattern_type: str) -> dict:
    """Return volume confirmation rule for a given pattern type."""
    rules = {
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
            "gap_size_2_5x_10day_atr_max": True,  # Duddella gap validity rule
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
```

---

## PHASE 12: TIME-BASED ENHANCEMENTS & WEEKLY PROFILES (P2)

**Goal:** Implement ICT time-based concepts: 90-minute cycle, weekly profiles, day-of-week bias, Frankfurt session.
**Time:** ~30 minutes
**Depends on:** Phases 1-11 complete

---

### TASK 12.1: Day-of-Week Gate

```python
# In smc_strategy.py _precompute_gate_arrays():
def _precompute_dow_gate(self):
    """Day of week bias from David Woods weekly profile.

    Mon: MANIPULATION — weekly H/L formed, 0.70 multiplier (caution)
    Tue: CONTINUATION — follow Mon direction, 1.0x
    Wed: REACCUMULATION/REVERSAL — potential change, 0.85x
    Thu: COMPLETE Wed move — continuation, 1.0x
    Fri: DISTRIBUTION — reduce exposure, 0.60x
    """
    weekday_map = {
        0: 0.70,  # Monday — manipulation
        1: 1.00,  # Tuesday — continuation
        2: 0.85,  # Wednesday — potential reversal
        3: 1.00,  # Thursday — completion
        4: 0.60,  # Friday — distribution
        5: 0.50,  # Saturday (crypto) — reduce
        6: 0.50,  # Sunday (crypto) — reduce
    }
    n = len(self._df)
    self._dow_mults = np.ones(n)
    for i in range(n):
        wd = self._df.index[i].weekday()
        self._dow_mults[i] = weekday_map.get(wd, 1.0)
```

---

### TASK 12.2: 90-Minute Cycle Awareness

```python
def _precompute_90min_cycle(self):
    """David Woods: Risk of significant price change every 90 minutes from 00:00 NY time.

    Higher probability of reversal near 90-min markers.
    This serves as a WARNING, not a hard gate — amplifies signal sensitivity at
    90-min cycle points rather than blocking.
    """
    n = len(self._df)
    self._cycle_sensitivity = np.ones(n)
    for i in range(n):
        minutes_since_midnight = self._df.index[i].hour * 60 + self._df.index[i].minute
        cycle_position = minutes_since_midnight % 90
        # Amplify signal sensitivity near cycle boundaries (within 5 min)
        if cycle_position <= 5 or cycle_position >= 85:
            self._cycle_sensitivity[i] = 1.3  # 30% more sensitive
```

---

### TASK 12.3: Frankfurt Fake Move Awareness

```python
# In session gate:
# Frankfurt session (02:00-03:00 EST): "Frankfurt always makes a fake move" — David Woods
# Reduce signal confidence by 0.7 during Frankfurt
def _precompute_frankfurt_gate(self):
    n = len(self._df)
    self._frankfurt_mult = np.ones(n)
    for i in range(n):
        hour = self._df.index[i].hour
        # Frankfurt: 02:00-03:00 EST (subject to timezone, adjust)
        if 7 <= hour < 8:  # Approximate UTC
            self._frankfurt_mult[i] = 0.70  # Fake move expected
```

---

## APPENDIX A: COMPLETE IMPLEMENTATION ORDER

```
Phases 1-5 (existing modernization plan — must be complete)
    │
    ├── Phase 6: Core SMC Concepts (Breaker, Mitigation, Rejection)
    │   T6.1 → T6.2 → T6.3 → T6.4 → T6.5 → T6.6
    │       │
    │       ▼
    ├── Phase 7: New Chart Patterns (Island Reversal, Dragon, NR4, Quasimodo, Adam-Eve, etc.)
    │   T7.1 ∥ T7.2 ∥ T7.3 ∥ T7.4 ∥ T7.5 ∥ T7.6 ∥ T7.7 ∥ T7.8 ∥ T7.9 → T7.10
    │       │
    │       ▼
    ├── Phase 8: Advanced SMC (SMT Divergence ∥ PD Array Matrix ∥ Fibonacci)
    │   T8.1 ∥ T8.2 ∥ T8.3
    │       │
    │       ▼
    ├── Phase 9: Library Full Integration (sessions, previous_high_low, retracements, OB)
    │   T9.1 → T9.2 → T9.3 → T9.4
    │       │
    │       ▼
    ├── Phase 10: Risk/SMC Integration
    │   T10.1 → T10.2
    │       │
    │       ▼
    ├── Phase 11: Volume & Reliability Integration
    │   T11.1 → T11.2
    │       │
    │       ▼
    └── Phase 12: Time-Based Enhancements
        T12.1 → T12.2 → T12.3
```

Parallelizable tasks:
- Phase 7 tasks 7.1-7.9 can all be developed in parallel (independent pattern detectors)
- Phase 8 tasks 8.1, 8.2, 8.3 can be developed in parallel
- Phases 6, 7, 8 can partially overlap (different files, no dependencies between them)

---

## APPENDIX B: COMPLETION CHECKLIST

### Phase 6 (Breaker/Mitigation/Rejection)
- [ ] `src/patterns/smc/__init__.py` created
- [ ] `src/patterns/smc/breaker.py` created (~250 loc)
- [ ] `src/patterns/smc/mitigation.py` created (~200 loc)
- [ ] `src/patterns/smc/rejection.py` created (~150 loc)
- [ ] Detectors integrated into `smc_strategy.py` _precompute_indicators()
- [ ] Signal weights updated (breaker:0.70, mitigation:0.60, rejection:0.55)
- [ ] Pre/Post Phase 6 backtest comparison run
- [ ] BESTS.md updated with Phase 6 results

### Phase 7 (New Chart Patterns)
- [ ] `src/patterns/event/island_reversal.py`
- [ ] `src/patterns/exotic/dragon.py`
- [ ] `src/patterns/volatility/nr4_inside_bar.py`
- [ ] `src/patterns/complex/quasimodo.py`
- [ ] `src/patterns/classic/adam_eve.py`
- [ ] `src/patterns/classic/three_valleys.py`
- [ ] `src/patterns/candlestick/shooting_star.py`
- [ ] `src/patterns/basic/key_reversal.py`
- [ ] Gap type classification added to `src/patterns/breakout/gap.py`
- [ ] Backtest with new patterns integrated

### Phase 8 (Advanced SMC)
- [ ] `src/signals/smc_divergence.py` created (~250 loc)
- [ ] `src/patterns/smc/pd_array_matrix.py` created (~200 loc)
- [ ] Fibonacci body-to-body + extended levels in `src/indicators/ote.py`
- [ ] SMT divergence integrated into strategy (requires correlated pair data)
- [ ] PD Array Matrix integrated into _compute_smc_score()

### Phase 9 (Library Integration)
- [ ] `smc.sessions()` replacing hardcoded session times
- [ ] `smc.previous_high_low()` for daily/weekly reference levels
- [ ] `smc.retracements()` for dynamic trend tracking
- [ ] `smc.ob()` volume strength for position sizing
- [ ] Backtest with all library functions integrated

### Phase 10 (Risk/SMC Integration)
- [ ] `src/risk/smc_aware.py` created (~250 loc)
- [ ] SMC structural stop logic replacing generic ATR stops
- [ ] SMC partial exit (FVG fill / liquidity level)
- [ ] OB strength-based position sizing
- [ ] SMC circuit breaker (consecutive sweep/BOS failures)
- [ ] Daily SMC limits (profit target, consecutive losses)

### Phase 11 (Volume/Reliability)
- [ ] `src/signals/pattern_reliability_registry.py` created
- [ ] `src/signals/volume_confirmation_rules.py` created
- [ ] All pattern detectors reference reliability weights
- [ ] Volume confirmation gating applied to breakouts/gaps

### Phase 12 (Time-Based)
- [ ] Day-of-week gate in multiplicative chain
- [ ] 90-minute cycle awareness
- [ ] Frankfurt fake move gate
- [ ] Backtest comparison with/without time gates

### Documentation (ALL phases)
- [ ] `docs/ict_glossary.md` — 60+ SMC abbreviations
- [ ] `docs/pattern_reliability.md` — all reliability statistics
- [ ] `docs/COMMAND_CHEATSHEET.md` updated with Phase 6-12 commands
- [ ] `BESTS.md` updated with new SMC config sections
- [ ] `MEMORY.md` updated with final state
- [ ] `progress_docs/current.md` updated with all session logs
- [ ] `.kilo/agent/smc-trader.md` — new agent for SMC operations

---

## APPENDIX C: KNOWN PITFALLS (Phase 6-12 specific)

1. **smartmoneyconcepts OB may require accurate volume data** — crypto/forex volume may not be available. Fall back to tick-based proxy.
2. **SMT Divergence requires paired instrument data** — must pre-fetch correlated pair data. If unavailable, skip SMT component.
3. **PD Array Matrix memory usage** — storing all arrays across multiple timeframes. Cap at 1000 active arrays per timeframe.
4. **Breaker Block timing** — breakers require OB mitigation then breach. Can take 50+ bars. May produce too few signals on short backtests.
5. **Pattern reliability registry additions** — must be synchronized with existing `pattern_quality_registry.py` to avoid double-weighting.
6. **Island Reversal rare on intraday** — more common on daily. May produce 0 signals on hourly data.
7. **Fibonacci body-to-body** — wicks vary across brokers. Body-to-body reduces variance but may miss extreme levels.

---

## APPENDIX D: FILES CREATED (by phase)

| Phase | File | Lines (est) |
|-------|------|-------------|
| 6 | `src/patterns/smc/__init__.py` | 5 |
| 6 | `src/patterns/smc/breaker.py` | 250 |
| 6 | `src/patterns/smc/mitigation.py` | 200 |
| 6 | `src/patterns/smc/rejection.py` | 150 |
| 7 | `src/patterns/event/island_reversal.py` | 120 |
| 7 | `src/patterns/exotic/dragon.py` | 180 |
| 7 | `src/patterns/volatility/nr4_inside_bar.py` | 100 |
| 7 | `src/patterns/complex/quasimodo.py` | 120 |
| 7 | `src/patterns/classic/adam_eve.py` | 100 |
| 7 | `src/patterns/classic/three_valleys.py` | 80 |
| 7 | `src/patterns/candlestick/shooting_star.py` | 50 |
| 7 | `src/patterns/basic/key_reversal.py` | 50 |
| 8 | `src/signals/smc_divergence.py` | 250 |
| 8 | `src/patterns/smc/pd_array_matrix.py` | 200 |
| 10 | `src/risk/smc_aware.py` | 250 |
| 11 | `src/signals/pattern_reliability_registry.py` | 100 |
| 11 | `src/signals/volume_confirmation_rules.py` | 80 |
| - | `docs/ict_glossary.md` | 100 |
| - | `docs/pattern_reliability.md` | 80 |
| - | `.kilo/agent/smc-trader.md` | 150 |
| **Total** | **20 files** | **~2,615 loc** |

---

## APPENDIX E: PROJECTED IMPACT

| Dimension | Before | After All Phases | Δ |
|-----------|--------|-----------------|---|
| SMC concepts implemented | 21 | 39 | +18 |
| Pattern detectors total | 54 | ~75 | +21 |
| SMC indicator modules | 8 | 15 | +7 |
| Gate chain components | 3 | 7 | +4 |
| Risk/SMC integration | disconnected | fully integrated | qualitative |
| Signal scoring dimensions | 5 (sweep+MSS+FVG+OB+confluence) | 11 (+breaker+mitigation+rejection+SMT+PD+volume) | +6 |
| Multi-instrument SMC | none | SMT divergence pairs | new |
| Fibonacci levels | standard 6 | body-to-body 9 levels + OTE TPs | +3 levels |
| Volume awareness | binary gate | pattern-specific rules + reliability weights | qualitative |

**Estimated Sharpe improvement:** Phase 6 alone (~+0.2 from Breaker/Mitigation/Rejection). Total Phase 6-12 could add ~+0.5-0.8 to existing (unvalidated estimate — must be confirmed via backtest).
