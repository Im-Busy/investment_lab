# SMC/ICT Gap-Fillers — Full Implementation Plan

> **Source:** Tutorial analysis (`useful_resources/SMC_ICT_TUTORIAL_ANALYSIS.md`) cross-referenced against ~25 SMC/ICT codebase files
> **Date:** 2026-05-20
> **Status:** IMPLEMENTED ✅
> **Total gaps:** 21 improvement opportunities → 10 phases, 28 tasks — all complete
> **Estimated LOC:** ~2,200 new, ~900 modified — actual ~2,500 new, ~1,200 modified across 18 files
> **Target completion:** ✅ All phases complete

---

## Architecture Overview

```
                    ┌──────────────────────────────────────┐
                    │         smc_strategy.py (1110 loc)    │
                    │    Central integration + scoring      │
                    └──────────┬───────────────────────────┘
                               │ _compute_smc_score()
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐   ┌────────▼────────┐   ┌──────▼──────┐
   │  Indicators  │   │ Pattern Detectors│   │    Risk     │
   │ (src/ind/)  │   │ (src/patterns/) │   │ (src/risk/) │
   └──────┬──────┘   └────────┬────────┘   └──────┬──────┘
          │                    │                    │
   ┌──────▼────────────────────▼────────────────────▼──────┐
   │              src/scripts/backtest_smc.py               │
   │              src/scripts/sweep_ict_strategies.py       │
   └────────────────────────────────────────────────────────┘
```

Key principle: **All new detectors → wired into `smc_strategy.py`'s `_compute_smc_score()` with toggle params.** No orphaned modules.

---

## PHASE G1 — FVG Quality Hardening (P0)

**Goal:** Fix FVG detection to match ICT strict rules. ~90 LOC new, ~30 LOC modified.
**Files:** `src/indicators/ifvg.py`, `src/strategies/smc_strategy.py`

### TASK G1.1 — Zero Wick-Overlap Validation in IFVG
**File:** `src/indicators/ifvg.py`
**What:** Currently FVG detection in `_compute_fvg_proximity()` (smc_strategy.py lines 466-490) uses `two_high < prev_low` (bullish) and `two_low > prev_high` (bearish). These compare candle bodies only. ICT tutorial rule is strict: **C1 wick must NOT overlap C3 wick** — zero overlap between the wick of candle-1 and wick of candle-3.

**Change:** Add a validation function `_validate_fvg_zero_overlap()` that:
- For bullish FVG: checks `high[i-2] < low[i]` (C1 full range does NOT touch C3 full range)
- For bearish FVG: checks `low[i-2] > high[i]` (C1 full range does NOT touch C3 full range)
- Returns `True` only if zero overlap exists

**In `smc_strategy.py` `_compute_fvg_proximity()`:** Replace the existing simple body comparison with the new wick-overlap check. FVGs that don't pass are NOT counted.

### TASK G1.2 — Consequent Encroachment (CE) Entry Levels
**File:** `src/indicators/ifvg.py`
**What:** Add CE (50% FVG midpoint) computation to `IFVG` dataclass and `IFVGProximity`. ICT tutorial: price often only retraces to CE before reversing. Entry at CE = tighter stop, better R:R.

**Changes:**
1. Add `ce: float` field to `IFVG` dataclass (`(high + low) / 2.0`)
2. Add `distance_to_ce: float` to `IFVGProximity` dataclass
3. Add `distance_ce_atr_ratio: float` normalized distance to CE
4. Modify `_compute_fvg_proximity()` in `smc_strategy.py` to output proximity to CE when price is within CE range (not just full fill range)

**Scoring integration:** When price is at CE, boost FVG proximity score by 1.3× (tighter stop = higher confidence entry).

### TASK G1.3 — FVG Quality Score
**File:** `src/strategies/smc_strategy.py` (`_compute_fvg_proximity()`)
**What:** Compute a per-bar FVG quality score based on ICT quality filters:
1. **BOS-triggered** (0.0–0.3): FVG that formed during a BOS event — no proximity score if untriggered
2. **Freshness** (0.0–0.3): decaying weight based on bars since formation (exponential decay, half-life=20 bars)
3. **Premium/Discount alignment** (0.0–0.4): FVG in correct zone for the bias direction
4. **Size significance** (0.0–0.2): gap_size / ATR ratio (bigger gaps = stronger)

Output a composite `_fvg_quality` array multiplied into `_fvg_proximity` score.

---

## PHASE G2 — BOS vs CHOCH Differentiation (P0)

**Goal:** Split conflated BOS/CHOCH into distinct concepts with displacement validation. ~120 LOC new, ~60 LOC modified.
**Files:** `src/indicators/mss.py`, `src/strategies/smc_strategy.py`

### TASK G2.1 — Add BOS Detector (Continuation)
**File:** `src/indicators/mss.py`
**What:** Add `detect_bos()` function that detects BOS (trend CONTINUATION, same direction as existing trend). Different from MSS which detects reversal.

**Logic:**
- Bullish BOS: close breaks above most recent confirmed swing high with full candle body (not wick). Previous swing high must have been a HH relative to prior HH.
- Bearish BOS: close breaks below most recent confirmed swing low with full candle body. Previous swing low must have been a LL relative to prior LL.
- **Displacement check:** move size must exceed 0.5× ATR (energetic, not a slow grind)

**Output:** `BOSInfo` dataclass with fields: `detected`, `direction`, `swing_price` (level broken), `break_bar`, `displacement_ratio`, `is_valid`.

### TASK G2.2 — Enhance MSS Detector (Reversal)
**File:** `src/indicators/mss.py`
**What:** Modify existing `detect_mss()` to enforce the ICT CHOCH rules:
1. MSS must break the swing created by the **last BOS**, not any random swing
2. MSS must have **displacement** — the break candle must leave an FVG behind it
3. MSS without displacement = weak signal (confidence × 0.5)

**Change:** Add `last_bos_bar` tracking and verify the broken swing was created by or after the last BOS. Add `displacement_confirmed: bool` to `MSSInfo`.

### TASK G2.3 — Fake CHOCH Detection
**File:** `src/indicators/mss.py`
**What:** Add `detect_fake_choch()` that identifies false reversal signals (liquidity sweeps that look like CHOCH but aren't).

**Logic:**
- Price breaks a swing level with a **wick** (not body close) — this is a sweep, not a structure break
- HTF context contradicts: if daily is bullish and 15m prints bearish CHOCH → likely fake
- No displacement follows → likely fake

**Output:** `FakeCHOCHInfo` dataclass. When fake CHOCH detected at bar i, suppress the MSS signal at that bar (weight × 0.0).

### TASK G2.4 — Wire BOS/CHOCH/MSS into SMC Strategy Scoring
**File:** `src/strategies/smc_strategy.py` (`_compute_bos_choch()` and `_compute_smc_score()`)
**What:** Replace the single `_bos_signals` array with three arrays:
- `_bos_signals` (+1/-1 for continuation with trend)
- `_choch_signals` (+1/-1 for reversal against trend)
- `_fake_choch` (boolean mask to suppress false signals)

**Scoring weights:**
- BOS confirmation: +0.65 (trusted — trend already established)
- CHOCH confirmation: +0.40 (lower — reversals fail more often)
- Sweep confirmed without CHOCH: +0.15 (low confidence without structure confirmation)

---

## PHASE G3 — Judas Swing Detector (P0)

**Goal:** Implement the core ICT entry filter: London session false move sweeping Asian range then reversing. ~180 LOC new, ~50 LOC modified.
**Files:** NEW `src/indicators/judas_swing.py`, `src/strategies/smc_strategy.py`, `src/indicators/asian_range.py`

### TASK G3.1 — Create Judas Swing Detector
**File:** NEW `src/indicators/judas_swing.py`
**What:** Detect the Judas Swing pattern during London Kill Zone (2:00–5:00 AM EST).

**Dataclass:**
```python
@dataclass
class JudasSwingInfo:
    detected: bool
    direction: str  # 'bullish' (fake drop then up) or 'bearish' (fake rise then down)
    asian_high: float
    asian_low: float
    swing_extreme: float  # the price extreme of the false move
    swing_bar: int        # bar where false move completed
    reversal_bar: int     # bar where reversal confirmed
    reversal_confirmed: bool
    displacement_ratio: float  # reversal move size / ATR
    session: str  # 'london' typically
```

**Detection algorithm:**
1. Get Asian session high/low from `asian_range.py`
2. During London KZ window (2:00–5:00 AM EST / 06:00–09:00 UTC):
   - Bullish Judas: price drops below Asian low by ≥ 0.3× ATR (false move down), then closes ABOVE Asian low within N bars (reversal)
   - Bearish Judas: price rises above Asian high by ≥ 0.3× ATR (false move up), then closes BELOW Asian high within N bars
3. Reversal must be energetic: displacement ratio ≥ 1.0× ATR
4. FVG should form during the reversal move

**Add** `JudasSwingInfo` to the detector output.

### TASK G3.2 — Extend Asian Range for London Session Overlap
**File:** `src/indicators/asian_range.py`
**What:** Add a helper `get_asian_range_for_date()` that, given a UTC datetime, returns the most recent Asian session range (high/low/range_size). Used by Judas Swing detector and by PO3/AMD integration.

### TASK G3.3 — Wire Judas Swing into SMC Strategy
**File:** `src/strategies/smc_strategy.py`
**What:** Add `use_judas_swing: bool = False` parameter. When enabled:
1. In `init()`: precompute `_judas_swing_direction` array (per-bar: +1=bullish setup expected, -1=bearish, 0=none)
2. In `_compute_smc_score()`: add Judas Swing gate:
   - If `use_judas_swing` and no Judas Swing direction at bar → score × 0.4 (heavy penalty)
   - If sweep direction aligns with Judas Swing direction → score × 1.3 (bonus)
   - If sweep direction OPPOSES Judas Swing → score × 0.0 (invalid — don't fight the Judas)

**Toggles:** `judas_swing_atr_mult: float = 0.3`, `judas_swing_displacement_mult: float = 1.0`, `judas_swing_reversal_bars: int = 5`

---

## PHASE G4 — Orphaned Indicator Integration (P0)

**Goal:** Wire 5 fully-implemented-but-unused indicators into `SMCStrategy` scoring. ~300 LOC modified, 0 LOC new modules.
**Files:** `src/strategies/smc_strategy.py` (primary), `src/indicators/power_of_3.py`, `src/indicators/ote.py`, `src/indicators/cisd.py`, `src/indicators/crt.py`, `src/signals/smc_divergence.py`

### TASK G4.1 — PO3/AMD Gate
**File:** `src/strategies/smc_strategy.py`
**What:** Wire `power_of_3.py` into scoring. Add `use_po3_gate: bool = False` param.

**Integration:**
- In `init()`: call `detect_po3_daily()` for daily PO3 phases, map to each bar
- Add `_po3_phase` array (0=accumulation, 1=manipulation, 2=distribution, 3=none)
- In `_compute_smc_score()`:
  - During accumulation phase: disable entries entirely (score × 0.0) — ICT rule: "do not enter before manipulation is complete"
  - During manipulation phase: score × 0.5 (caution — manipulation could still be ongoing)
  - During distribution phase: score × 1.0 (full signal — this is the phase to trade)
  - Align sweep direction with PO3: if bullish PO3 (Asian range accumulated then London manipulated down → NY distribution up), bearish sweeps get suppressed

### TASK G4.2 — OTE Fibonacci Confluence
**File:** `src/strategies/smc_strategy.py`
**What:** Wire `ote.py` into scoring. Add `use_ote_confluence: bool = False` param.

**Integration:**
- In `init()`: call `detect_ote_zones()` from `ote.py`, map OTE zone presence to each bar
- Add `_in_ote_zone` boolean array and `_ote_retracement_pct` array
- In `_compute_smc_score()`:
  - If price is in OTE zone (62%–79% retracement) AND sweep signal present → score × 1.25 (confluence bonus)
  - If at 70.5% sweet spot specifically → score × 1.35 (algorithmic compression point)
  - **Equilibrium-first filter:** OTE bonus only applies if price first crossed below 50% equilibrium (deep enough retracement)

### TASK G4.3 — CISD Confirmation
**File:** `src/strategies/smc_strategy.py`
**What:** Wire `cisd.py` into scoring. Add `use_cisd: bool = False` param.

**Integration:**
- In `init()`: call `detect_cisd_vectorized()`, map bullish/bearish CISD signals
- Add `_cisd_direction` array (+1 bullish, -1 bearish, 0 none)
- In `_compute_smc_score()`:
  - If CISD direction aligns with sweep direction → additive confirmation +0.15 to base score
  - No penalty for CISD absence (it's a bonus, not a filter)

### TASK G4.4 — CRT (Candle Range Theory) Reference
**File:** `src/strategies/smc_strategy.py`
**What:** Wire `crt.py` into scoring. Add `use_crt: bool = False` param.

**Integration:**
- In `init()`: call `detect_crt()`, map CRH/CRL reference candles
- Add `_crt_direction` array (bullish/bearish CRT signals)
- In `_compute_smc_score()`:
  - If CRT direction aligns with sweep → bonus +0.10
  - CRT-based premium/discount levels used to validate zone quality

### TASK G4.5 — SMT Divergence
**File:** `src/strategies/smc_strategy.py`
**What:** Wire `smc_divergence.py` into scoring. Add `use_smt: bool = False` and `smt_pair: str = ""` params. Note: SMT requires correlated pair data — this is the heaviest integration.

**Approach:** For backtest simplicity, add `smt_secondary_data: Optional[pd.DataFrame]` param to `SMCStrategy`. If provided and `use_smt=True`, compute SMT divergence between primary and secondary.
- Bullish SMT: primary makes lower low but secondary does NOT → bullish reversal signal
- Bearish SMT: primary makes higher high but secondary does NOT → bearish reversal signal
- Additive bonus to base score: +0.20 when SMT aligns

---

## PHASE G5 — Structural Entry Refinements (P1)

**Goal:** Implement S&D zone patterns, OB+FVG colocation, and Breaker/Mitigation differentiation. ~350 LOC new, ~200 LOC modified.
**Files:** `src/strategies/smc_strategy.py`, `src/patterns/smc/breaker.py`, `src/patterns/smc/mitigation.py`, NEW `src/patterns/smc/sd_zones.py`

### TASK G5.1 — Supply & Demand Zone Patterns (RBD/DBR/RBR/DBD)
**File:** NEW `src/patterns/smc/sd_zones.py` (or add to existing `smc/` module)
**What:** Detect the 4 S&D zone patterns from tutorial Day 11.

**Detection:**
- RBD (Rally-Base-Drop): price rallies → consolidates (base) → drops sharply → bearish supply zone
- DBR (Drop-Base-Rally): price drops → consolidates → rallies sharply → bullish demand zone
- RBR (Rally-Base-Rally): price rallies → pulls back (base) → continues up → bullish continuation demand
- DBD (Drop-Base-Drop): price drops → consolidates → continues down → bearish continuation supply

**Output:** `SDZone` dataclass: `zone_type`, `direction`, `zone_high`, `zone_low`, `strength` (RBD/DBR = strongest reversal, RBR/DBD = continuation, lower confidence).

**Scoring:** Map SD zones to relevant bars. When sweep aligns with a fresh SD zone → +0.25 to base score. Reversal S&D (RBD/DBR) weighted higher than continuation (RBR/DBD).

### TASK G5.2 — OB+FVG Colocation Scoring
**File:** `src/strategies/smc_strategy.py` (`_precompute_order_blocks()` and `_compute_smc_score()`)
**What:** Valid OB should have FVG sitting immediately adjacent. ICT tutorial: "An OB with an FVG sitting next to it carries far more weight."

**Change to `_precompute_order_blocks()`:** When an OB is detected, scan the next 3 bars for an FVG formation. If found within 3 bars, boost `_ob_proximity` at that bar by 1.5×. If no adjacent FVG, `_ob_proximity` stays at computed value.

### TASK G5.3 — Breaker vs Mitigation Sweep Distinction
**File:** `src/patterns/smc/breaker.py`
**What:** Current breaker detector treats all failed OBs as breakers. ICT distinguishes:
- **Breaker** (higher probability): OB failed AFTER price swept liquidity (made a new high/low first). Requires sweep confirmation.
- **Mitigation** (lower probability): OB failed WITHOUT sweeping liquidity first. Price formed a lower high/inverse pattern.

**Change:** Add `sweep_confirmed: bool` to breaker output. If sweep confirmed → classify as "breaker" (strength × 1.0). If no sweep → classify as "mitigation" (strength × 0.6). Use the existing `_sweep_signals` array to verify.

**Scoring update in `_compute_smc_score()`:**
- Breaker (sweep confirmed): weight 0.85 (was 0.70)
- Mitigation (no sweep): weight 0.45 (was 0.60)

### TASK G5.4 — Unicorn Pattern Detection
**File:** `src/strategies/smc_strategy.py`
**What:** Detect when FVG forms inside/overlapping a Breaker Block during displacement. ICT's highest-confluence setup.

**Logic:** At each bar where a breaker is active AND an FVG is detected, check if the FVG's price range overlaps with the breaker zone. If so, mark as Unicorn. Score bonus: +0.30 additive (strong confluence).

---

## PHASE G6 — POI Grading System (P1)

**Goal:** Implement structured POI grading per SMC 4-criteria system. ~150 LOC new.
**Files:** NEW `src/indicators/poi_grader.py`, `src/strategies/smc_strategy.py`

### TASK G6.1 — POI Grader Module
**File:** NEW `src/indicators/poi_grader.py`
**What:** Grade any potential POI against the 4 SMC quality criteria.

**Dataclass:**
```python
@dataclass
class POIGrade:
    criteria_pass: tuple[bool, bool, bool, bool]  # 4 pass/fail
    score: int  # 0-4
    is_tradeable: bool  # all 4 must pass or trade is rejected
    details: dict  # per-criterion explanation
```

**Four criteria:**
1. **BOS-triggered** (0.25): zone must have caused a BOS or market structure shift when price originally left it
2. **Liquidity-protected** (0.25): there must be resting orders (stops) just beyond the zone — equal highs/lows, session extremes
3. **Unmitigated** (0.25): zone must never have been fully retested (fresh = more unfilled orders remaining)
4. **Closest-to-price** (0.25): of all qualified zones, trade from the nearest one first

**Grade function:** `grade_poi(zone_info, bos_history, liquidity_levels, retest_history, distance_to_price) → POIGrade`

### TASK G6.2 — Wire POI Grading into Strategy Scoring
**File:** `src/strategies/smc_strategy.py`
**What:** Add `use_poi_grading: bool = False` param. When enabled:
- Before entering, grade the nearest OB/FVG/breaker against 4 criteria
- If grade < 4 (any criterion fails) → score × 0.3 (soft suppression — not complete block)
- If grade == 4 → score × 1.15 (confidence bonus for fully graded POI)

---

## PHASE G7 — Risk Management Wiring (P1)

**Goal:** Connect `smc_aware.py` risk module to `SMCStrategy.next()` for live risk enforcement. ~150 LOC modified.
**Files:** `src/strategies/smc_strategy.py`, `src/risk/smc_aware.py`

### TASK G7.1 — Daily Loss Limit Circuit Breaker
**File:** `src/strategies/smc_strategy.py`
**What:** Add daily P&L tracking. If daily loss exceeds 3-5% of starting equity, halt trading for the day.

**Implementation:**
- Track `_daily_start_equity` and `_daily_pnl` in `init()`
- In `next()`: if current date differs from last trade date → reset daily P&L
- Before entry: check if `_daily_pnl < -daily_loss_limit * starting_equity` → skip trade
- Add params: `daily_loss_limit: float = 0.03`, `use_daily_loss_limit: bool = False`

### TASK G7.2 — Partial Profit Taking + Break-Even
**File:** `src/strategies/smc_strategy.py` (`next()`)
**What:** Enhance existing `use_multi_tp` logic with ICT-standard partial exit rules:
1. Take 50% off at TP1 (already implemented as `tp1_size`)
2. Move stop to break-even after TP1 hit (already implemented as `move_sl_to_be`)
3. **ADD:** Use structural TP levels instead of fixed ATR multiples:
   - TP1 = nearest opposing FVG or OB (computed at entry time)
   - TP2 = nearest BSL/SSL liquidity level (Draw on Liquidity)

**Change:** When `use_structural_tp: bool = True`, override `tp1_atr`/`tp2_atr` with computed structural levels from `_fvg_proximity` and nearby BSL/SSL.

### TASK G7.3 — 1% Risk Rule Enforcement
**File:** `src/strategies/smc_strategy.py` (`_calculate_size()`)
**What:** Current `_calculate_size()` already uses 1% risk. Keep as-is. Add `max_risk_pct: float = 0.01` param to make it configurable.

---

## PHASE G8 — PD Array Matrix & MTF (P2)

**Goal:** Make PD Array Matrix dynamic and multi-timeframe. ~300 LOC modified.
**Files:** `src/patterns/smc/pd_array_matrix.py`, `src/strategies/smc_strategy.py`

### TASK G8.1 — Multi-Timeframe PD Array Matrix
**File:** `src/patterns/smc/pd_array_matrix.py`
**What:** Extend `build_pd_array_matrix()` to accept multiple timeframe DataFrames and build a hierarchical matrix.

**Changes:**
- Accept `Dict[str, pd.DataFrame]` keyed by timeframe label ("D", "4H", "1H")
- Build separate PD arrays per timeframe
- Add `get_confluent_arrays()` that finds arrays at same price level across multiple timeframes — multi-TF confluence = higher confidence

### TASK G8.2 — Nearest PD Array Selection
**File:** `src/patterns/smc/pd_array_matrix.py`
**What:** Add `select_entry_array()` function that applies ICT selection rule: "find the first PD array price encounters within the correct premium/discount zone."

**Logic:**
- Given current price, determine if in premium or discount zone
- If bullish bias → scan discount zone for nearest PD array below price
- If bearish bias → scan premium zone for nearest PD array above price
- Return `(PDArray, distance_to_price)`

### TASK G8.3 — Wire PD Array Selection into Strategy
**File:** `src/strategies/smc_strategy.py`
**What:** Add `use_pd_array_selection: bool = False` param. When enabled:
- Build multi-TF PD array matrix in `init()` (daily using resampled data, 4H/1H natively)
- In `_compute_smc_score()`: check if current price is near a selected PD array entry point
- If within proximity to PD array entry → score × 1.20
- If price passed THROUGH a PD array without reacting → score × 0.50 (zone breached, confidence reduced)

---

## PHASE G9 — SMC/ICT Trade Plan Checklists (P2)

**Goal:** Implement structured 10-point pre-trade checklists with configurable strictness. ~200 LOC new.
**Files:** NEW `src/strategies/smc_trade_plan.py`, `src/strategies/smc_strategy.py`

### TASK G9.1 — SMC Trade Plan Validator
**File:** NEW `src/strategies/smc_trade_plan.py`
**What:** 10-point checklist validator implementing the SMC 6-step trade plan from tutorial Day 15.

**Checklist function:** `validate_smc_trade(idx, precomputed) → tuple[bool, list[str]]` — returns (pass, failing_checks).

**10 checks:**
1. Weekly bias defined (from `_htf_bias`)
2. Daily BOS confirms weekly direction
3. Unmitigated POI identified and graded ≥ 3
4. POI passes all 4 quality criteria
5. Price swept liquidity at/below POI before entry
6. CHOCH formed on 15M/5M at POI
7. Entry at FVG or OB created during CHOCH
8. Stop below sweep low (not below zone broadly)
9. Target = named liquidity level (not round number)
10. R:R ≥ 1:3

### TASK G9.2 — ICT Trade Plan Validator
**File:** `src/strategies/smc_trade_plan.py`
**What:** 10-point checklist for ICT 6-step trade plan from tutorial Day 16.

**10 checks:**
1. IPDA narrative written (20/40/60-day levels marked)
2. Daily bias set before London open (correct premium/discount)
3. Nearest PD array identified in correct zone
4. Judas Swing completed (liquidity swept, false move confirmed)
5. MSS formed with displacement (large candle + FVG)
6. Entry zone inside active Kill Zone
7. Stop below Judas Swing low
8. T1 at -0.27 extension
9. T2 at named IPDA DOL
10. Risk ≤ 1-2% account

### TASK G9.3 — Wire Trade Plan Validation into Strategy
**File:** `src/strategies/smc_strategy.py`
**What:** Add `use_trade_plan: str = ""` param (options: "smc", "ict", "hybrid", ""). When set:
- Before returning score in `_compute_smc_score()`, run the selected checklist
- If checklist fails → score × 0.0 (hard block on entry)
- `trade_plan_strictness: float = 1.0` — number of checks that must pass (1.0 = all 10, 0.7 = 7/10)

---

## PHASE G10 — Market-Specific Adaptations & Polish (P2)

**Goal:** Market-specific kill zone times, SFP implementation, and documentation updates. ~250 LOC new/modified.
**Files:** `src/strategies/smc_strategy.py`, NEW `src/patterns/smc/sfp.py`, docs

### TASK G10.1 — Market-Specific Kill Zone Times
**File:** `src/strategies/smc_strategy.py` (`_precompute_smc_sessions()`)
**What:** Adjust kill zone detection based on instrument class (forex, crypto, stocks, gold).

**Mapping:**
- `forex`: London 2-5am EST, NY 7-10am EST (existing)
- `crypto`: NYSE open 9:30-11:30am EST (US equity anchor)
- `stocks/indices`: Market open 9:30-11:00am EST
- `gold (XAUUSD)`: London-NY overlap 8am-12pm EST

**Implementation:** Add `instrument_class: str = "auto"` param. Auto-detect from symbol prefix (XAU=gold, BTC/ETH=crypto, ES/NQ=indices, else=forex).

### TASK G10.2 — SFP (Swing Failure Pattern) Implementation
**File:** NEW `src/patterns/smc/sfp.py`
**What:** Implement the missing SFP detector referenced in `__init__.py` docstring.

**SFP Definition:** Price breaks a swing high/low (by wick) but fails to close beyond it, then reverses sharply. Similar to false breakout but specific to swing pivot levels.

**Output:** `SFPInfo` dataclass with direction, swing level, break bar, reversal bar.
**Scoring:** Wire as additive confirmation in `_compute_smc_score()` — SFP at swing level + sweep = +0.20.

### TASK G10.3 — Consequent Encroachment in PD Array
**File:** `src/patterns/smc/pd_array_matrix.py`
**What:** Add CE tracking to PDArray. `ce_level: Optional[float]` — only populated for FVG arrays. When selecting nearest PD array, if an FVG's CE is closer than its edge, use CE as the reference level.

### TASK G10.4 — Documentation & Command Updates
**Files:** `docs/COMMAND_CHEATSHEET.md`, `.useful_commands/smc_commands.txt`
**What:** Document all new `backtest_smc.py` flags. Update SMC backtest command examples with new parameters. Update `ict_glossary.md` with any new terms discovered from tutorial.

---

## Dependency Graph

```
G1 (FVG Hardening) ─────────────────────────────────────────────────────────────┐
G2 (BOS/CHOCH Split) ───────────────────────────────────────────────────────────┤
G3 (Judas Swing) ───── depends on G2 (needs BOS/CHOCH for displacement check) ───┤ ALL feed into
G4 (Orphaned Integration) ─ depends on G1 (uses hardened FVG) ───────────────────┤ SMC Strategy
G5 (Structural Entries) ─ depends on G2,G4 ──────────────────────────────────────┤ scoring
G6 (POI Grading) ─────── depends on G5 (uses SD zones) ──────────────────────────┤
G7 (Risk Mgmt) ───────── independent ────────────────────────────────────────────┤
G8 (PD Array MTF) ────── depends on G1,G5 ───────────────────────────────────────┤
G9 (Trade Plans) ─────── depends on G3,G6,G7 ────────────────────────────────────┤
G10 (Polish) ─────────── depends on G3,G8 ───────────────────────────────────────┘
```

**Independent clusters (can run in parallel):**
- G1 + G2 + G3 (indicator hardening, sequential dependency)
- G7 (risk management, fully independent)
- G4 + G5 + G6 (scoring enhancements, sequential)

G8, G9, G10 depend on everything before them.

---

## File Change Summary

| File | Action | LOC Est. |
|------|--------|----------|
| `src/indicators/judas_swing.py` | **NEW** | ~180 |
| `src/indicators/poi_grader.py` | **NEW** | ~150 |
| `src/patterns/smc/sd_zones.py` | **NEW** | ~200 |
| `src/patterns/smc/sfp.py` | **NEW** | ~100 |
| `src/strategies/smc_trade_plan.py` | **NEW** | ~200 |
| `src/indicators/ifvg.py` | MODIFY | +60 |
| `src/indicators/mss.py` | MODIFY | +120 |
| `src/indicators/asian_range.py` | MODIFY | +30 |
| `src/indicators/ote.py` | MODIFY | +20 |
| `src/patterns/smc/breaker.py` | MODIFY | +50 |
| `src/patterns/smc/mitigation.py` | MODIFY | +30 |
| `src/patterns/smc/pd_array_matrix.py` | MODIFY | +200 |
| `src/strategies/smc_strategy.py` | MODIFY | +400 |
| `src/risk/smc_aware.py` | MODIFY | +30 |
| `src/signals/smc_divergence.py` | MODIFY | +20 |
| `docs/COMMAND_CHEATSHEET.md` | MODIFY | +40 |
| `.useful_commands/smc_commands.txt` | MODIFY | +40 |
| `docs/ict_glossary.md` | MODIFY | +30 |
| **TOTAL** | | **+2,200 new, -900 modified** |

---

## Testing Strategy

### Unit Tests (per new module)
Each new file gets matching test file in `tests/`:
- `tests/test_judas_swing.py` — synthetic OHLCV with known Asian range, verify Judas detection
- `tests/test_poi_grader.py` — mock zones with known criteria pass/fail, verify scoring
- `tests/test_sd_zones.py` — synthetic RBD/DBR/RBR/DBD patterns, verify classification
- `tests/test_sfp.py` — synthetic swing failures, verify detection

### Integration Tests
- Extend `tests/test_ict_new_modules.py` with new module imports and smoke tests
- Run `uv run pytest tests/test_ict_new_modules.py -v` after each phase

### Backtest Validation (CRITICAL — do after each phase)
```bash
# Phase G1-G3: verify no regression on existing SMC backtests
uv run python scripts/backtest_smc.py --all --fast --is-only 2>&1 | tee outputs/smc_g1g3_baseline.txt

# Phase G4+: enable new toggles one at a time, compare OOS Sharpe
uv run python scripts/backtest_smc.py SPY --use-judas-swing --use-po3-gate --fast
```

### Validation Thresholds
- No regression on existing BESTS.md entries for SMCStrategy
- New features must not reduce OOS Sharpe by > 0.05
- New features should increase OOS Sharpe or reduce drawdown
- Each new toggle shown to produce ≥ 1 positive metric change

---

## Implementation Order

```
Week 1 (NOW):
  Day 1-2: G1 (FVG Hardening) — smallest, highest immediate impact
  Day 2-3: G2 (BOS/CHOCH Split) — critical structural fix
  Day 3-4: G3 (Judas Swing) — core ICT concept
  Day 4-5: G4 (Orphaned Integration) — wire 5 existing modules

Week 2:
  Day 1-2: G5 (Structural Entries)
  Day 3: G6 (POI Grading)
  Day 4-5: G7 (Risk Management Wiring)

Week 3:
  Day 1-2: G8 (PD Array MTF)
  Day 3: G9 (Trade Plans)
  Day 4-5: G10 (Polish + Documentation)
```

---

*Plan created 2026-05-20. Source: `useful_resources/SMC_ICT_TUTORIAL_ANALYSIS.md` cross-referenced against codebase audit.*
