# SMC & ICT Tutorial Analysis — Gaps & Improvement Opportunities

> **Source**: 21-day "ICT vs SMC" tutorial series + appendix with 42 illustration descriptions
> **Analyzed**: 2026-05-20 against existing codebase (~25 SMC/ICT files across 7+ directories)

---

## Part 1: Complete Concept Inventory From Tutorial

### Foundational Concepts (Days 1-7)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 1 | **Wyckoff Foundation** | Accumulation/Markup/Distribution/Markdown phases. Wyckoff "Spring" = modern "liquidity grab." Huddleston built ICT directly on Wyckoff. | `po3.py` exists (AMD = Accumulation-Manipulation-Distribution) but NOT wired into any strategy | **Fragment — not integrated** |
| 2 | **Valid Swing Point** | 3-candle formation. Wick alone does NOT confirm structure break — body must CLOSE beyond previous swing point. Wick-through = liquidity sweep, not structural shift. | `mss.py` detects pivot breaks but does not enforce the close-vs-wick distinction | **Gap** |
| 3 | **BOS (Break of Structure)** | Trend continuation. Same direction as trend. Requires: full candle body close, correct structural swing point, displacement behind it. BOS signals "trend is still alive." | Treated identically to MSS in `mss.py` | **Gap — conflated with MSS** |
| 4 | **CHOCH (Change of Character)** | Reversal warning. Against trend direction. Must break the swing created by the LAST BOS — not any random internal swing. Needs displacement to be valid. | Via `smartmoneyconcepts` lib only, not in own detector | **Shallow** |
| 5 | **Fake CHOCH** | Liquidity sweep looks like CHOCH but trend continues. Key differentiator: displacement (real CHOCH has energetic FVG-leaving candle). Also: HTF context. If daily bullish, 15m CHOCH is almost always fake. | Not detected | **Gap** |
| 6 | **Internal vs External Structure** | External = major swing highs/lows (trend definition). Internal = smaller swings inside each leg (entry timing only). Always define trend from external structure first. | Not differentiated in code | **Gap** |
| 7 | **Timeframe Hierarchy** | ICT rigid: Daily→4H/1H→5M. SMC flexible. Higher timeframe ALWAYS has authority over lower. | `multi_timeframe_bias.py` exists but not wired | **Fragment** |

### Liquidity Concepts (Day 4)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 8 | **BSL / SSL** | Buy-Side Liquidity above swing highs/equal highs. Sell-Side below swing lows/equal lows. Institutions target both. | `liquidity_sweep.py` — only Asian session-based sweeps | **Partial — swing-pivot sweeps missing** |
| 9 | **Equal Highs/Lows** | Institutional magnets. Almost always swept before true reversal. Retail sees resistance/support; institutions see liquidity pool. | Not specifically detected or weighted higher | **Gap** |
| 10 | **Premium & Discount** | 50% equilibrium = midpoint of dealing range. Above = premium (sell), below = discount (buy). Buy cheap, sell expensive at structural level. | `pd_array_matrix.py` — static classification only, not used in live scoring | **Static — not dynamic** |
| 11 | **DOL (Draw on Liquidity)** | Nearest significant liquidity pool price is gravitating toward. Used as logical profit target. | In `cameron_model.py` only | **Not in main strategy** |
| 12 | **Stop Hunt Mechanics** | Price spikes through level, triggers stops, absorbs liquidity, then reverses. You weren't wrong about direction — you were on wrong side of liquidity pool. | Understood but not formalized as a separate detector | **Implicit** |

### Order Blocks & FVGs (Days 5-6)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 13 | **OB Validity: 3 Checks** | 1) Clear impulse follows (sharp displacement). 2) FVG should sit next to OB. 3) Must align with HTF structure. | OB detection via `smartmoneyconcepts` lib only; no colocation or HTF checks | **Gap — validity checks missing** |
| 14 | **Breaker Block** | Failed OB after liquidity sweep → polarity flips → higher probability than mitigation block. | `breaker.py` — exists but doesn't differentiate breaker vs mitigation based on sweep | **Gap — sweep distinction missing** |
| 15 | **FVG 3-Candle Rule** | Bullish: C1 high doesn't overlap C3 low. Bearish: C1 low doesn't overlap C3 high. MUST be zero wick overlap. If wicks touch — no FVG. | `ifvg.py` — may not enforce zero-overlap rule | **Verify needed** |
| 16 | **Consequent Encroachment (CE)** | 50% midpoint of FVG. In strong trends, price often only fills to CE then reverses. Entry at CE = tighter stop. | In glossary only — not in detector code | **Gap** |
| 17 | **IFVG (FVG Inversion)** | Broken FVG flips polarity. Bullish FVG broken down = bearish zone. Bearish FVG broken up = bullish zone. | `ifvg.py` — implemented | **OK** |
| 18 | **High-Quality FVG Filters** | Forms after BOS/MSS, sits in correct premium/discount zone, hasn't been fully tested/filled, fresh/untouched. | Not enforced | **Gap** |

### Advanced Concepts (Days 7-14)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 19 | **Inducement Scenarios** | 3 types: S/R inducement, Equal Highs/Lows, Post-BOS inducement (first OB after BOS = inducement, real zone deeper). | `breaker.py` has inducement check; Post-BOS inducement not detected | **Gap** |
| 20 | **Power of Three (AMD)** | Accumulation (Asian range) → Manipulation (London sweeps) → Distribution (NY real move). "Do not enter before manipulation is complete." | `power_of_3.py` exists — NOT integrated into any strategy | **Orphaned** |
| 21 | **Kill Zones (4)** | Asian 8-10pm, London 2-5am, NY 7-10am (forex) / 8:30-11am (indices), London Close 10am-12pm. All EST. | In `silver_bullet.py` and `smc_strategy.py` — hardcoded UTC, no DST handling | **Partial — DST gap** |
| 22 | **Asian Range as Roadmap** | Asian high/low = daily liquidity targets. London sweeps Asian low → bullish day signal; sweeps Asian high → bearish day signal. | `asian_range.py` detects ranges but directional bias logic not used | **Gap** |
| 23 | **MTF Trade Plans** | Layer 1: Weekly/Daily (Bias). Layer 2: 4H/1H (Zone/POI). Layer 3: 15M/5M (Entry). Day/Swing/Position trading combos. | `multi_timeframe_bias.py` — not wired into strategies | **Fragment** |
| 24 | **S&D Zone Patterns** | RBD (Rally-Base-Drop = Supply reversal), DBR (Drop-Base-Rally = Demand reversal), RBR (Rally-Base-Rally = Demand continuation), DBD (Drop-Base-Drop = Supply continuation). | Not classified or detected | **Gap** |
| 25 | **S&D + OB Combined** | S&D zone = broad context (where to look). OB inside zone = precision entry (where to enter). Together: tighter stops, better R:R. | Not combined in strategy logic | **Gap** |
| 26 | **Mitigation Block vs Breaker** | Mitigation = failed OB WITHOUT liquidity sweep (lower probability). Breaker = failed OB WITH liquidity sweep (higher probability). ICT definitions precise, SMC looser. | `breaker.py` and `mitigation.py` — but no sweep-based differentiation | **Gap** |
| 27 | **Unicorn Pattern** | Breaker Block + FVG overlapping during displacement. Dual pressure = faster, more reliable reaction. | Not implemented | **Gap** |
| 28 | **OTE Fibonacci** | 62%-79% retracement zone, 70.5% sweet spot. Fibonacci body-to-body (not wick-to-wick). Must cross equilibrium (50%) first. Targets: -0.27 (T1), -0.62 (T2). | `ote.py` exists — NOT integrated into any strategy | **Orphaned** |
| 29 | **5-Step OTE Checklist** | 1) Confirm HTF bias, 2) Valid impulse swing with displacement, 3) Fib body-to-body, 4) Price crosses below equilibrium, 5) CHoCH/FVG confirmation at OTE. | Not implemented as checklist | **Gap** |
| 30 | **PD Array Matrix** | Premium zone (sell tools): OB, FVG, Breaker, Mitigation, Rejection, IFVG, Volume Imbalance. Discount zone (buy tools): same list. Selection rule: first PD array price encounters in correct zone. | `pd_array_matrix.py` — static builder, single-TF, NOT called live | **Orphaned** |
| 31 | **POI Quality Criteria** | 4 criteria: 1) Triggered BOS or market structure shift, 2) Protected by liquidity, 3) Unmitigated (never fully retested), 4) Closest to current price. | Concept documented but not enforced systematically | **Gap** |

### Trade Plans & Execution (Days 15-16)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 32 | **SMC 6-Step Trade Plan** | 1) Weekly bias + DOL, 2) Daily bias (BOS aligned, POIs), 3) Grade POI (4 criteria), 4) **Wait for sweep at POI — do NOT enter on first touch**, 5) LTF CHoCH (15M/5M) confirmation, 6) Stop below sweep low, target = BSL/SSL, min 1:3 R:R. | Not implemented as workflow. `smc_strategy.py` enters on sweep signal directly without CHoCH confirmation step. | **Gap** |
| 33 | **SMC 10-Checklist** | 10 binary checks before any trade; if any unchecked → no trade. | Not enforced | **Gap** |
| 34 | **ICT 6-Step Trade Plan** | 1) IPDA narrative (20/40/60-day highs/lows), 2) Daily bias BEFORE London open (discount/premium), 3) **Wait for Judas Swing** (London sweeps Asian range and reverses), 4) Confirm MSS with displacement (large candle + FVG), 5) Enter PD array during Kill Zone, 6) Stop below Judas Swing low, T1=-0.27, T2=IPDA DOL. | Not implemented | **Gap** |
| 35 | **ICT 10-Checklist** | Similar to SMC but Kill Zone mandatory, Judas Swing must complete, MSS must have displacement, FVG must be inside Kill Zone. | Not implemented | **Gap** |
| 36 | **Judas Swing** | False move at London open (2-5am EST). Bullish day: drops below Asian low, sweeps SSL, reverses up. Bearish day: rises above Asian high, sweeps BSL, reverses down. "DO NOT ENTER during the Judas Swing." | **Not implemented at all** | **Major Gap** |
| 37 | **IPDA (Interbank Price Delivery Algorithm)** | Two objectives: hunt liquidity pools, fill price imbalances (FVGs). Operates on time cycles (~20/40/60-day). Liquidity pools form every ~20 trading days. | **Not implemented** | **Major Gap** |

### Risk Management (Day 17)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 38 | **1% Rule** | Risk 1-2% of account per trade max. 10 losses at 1% = 90% intact. 10 losses at 10% = account destroyed. | `smc_aware.py` has logic but not wired into any strategy | **Orphaned** |
| 39 | **Position Sizing Formula** | Position = (Account × Risk%) / Stop distance. Wider stop → smaller position. NEVER compress stop to fit desired size. | In `smc_aware.py` but not used live | **Orphaned** |
| 40 | **Break-Even + Partial Profits** | Move stop to BE after first structural level cleared. Take 50% off at T1 (first DOL). Let 50% run to T2. | Not in strategies | **Gap** |
| 41 | **Daily Loss Limit** | Hard cap 3-5% of account. If hit → stop immediately, no exceptions. Prevents revenge trading. | `smc_aware.py` has logic but not active | **Orphaned** |
| 42 | **Drawdown Maths** | 10% loss needs 11.1% gain. 20% needs 25%. 40% needs 66.7%. 50% needs 100%. Capital preservation > aggressive seeking. | Documented but not enforced | **OK** |

### Cross-Market & Mistakes (Days 18-19)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 43 | **Crypto Adaptation** | Use US equity open (9:30am EST) as primary Kill Zone anchor. Asian range 8pm-4am still functions as daily reference. Stick to BTC/ETH majors. | Not adapted | **Gap** |
| 44 | **Gold Specific Timing** | London-NY overlap (8am-12pm EST) = best window for gold. ICT Silver Bullet (10-11am) was developed for gold/indices. | Not adapted | **Gap** |
| 45 | **Stocks Adaptation** | Kill Zone = 9:30-11am EST. Pre-market = Asian equivalent (accumulation). Opening sweep = manipulation. Work best on indices/ETFs (ES, NQ), not individual stocks. | Not adapted | **Gap** |
| 46 | **7 Common Mistakes** | 1) Overcomplicating chart (use 3 tools), 2) Ignoring HTF bias (write daily bias first), 3) Trading outside Kill Zones, 4) Chasing price mid-move, 5) Compressing stop, 6) No journal, 7) Strategy-hopping. | Not enforced in code | **Design guidance** |

### Hybrid & Final (Days 20-21)

| # | Concept | Tutorial Detail | In Codebase? | Status |
|---|---------|----------------|-------------|--------|
| 47 | **Hybrid Framework** | Layer 1: SMC for weekly/daily narrative. Layer 2: SMC for 4H/1H POI identification. Layer 3: ICT for Kill Zone entry (Judas Swing + MSS + FVG). Both filter sets applied rigorously. | Not implemented | **Gap** |
| 48 | **Terminology Translation** | CHoCH = MSS, POI = PD Array, Liquidity sweep = stop hunt/inducement, DOL = IPDA target, etc. | `ict_glossary.md` covers some | **Partial** |
| 49 | **90-Day Learning Plan** | Days 1-30: markup only (no live), 31-60: bar replay backtesting (50+ sessions), 61-90: demo real-time. Commit to ONE model. | Not in project conventions | **Process guidance** |

---

## Part 2: Ranked Improvement Opportunities

### Tier 1 — HIGH Impact, Immediate Wins (30-200 LOC)

| Rank | Improvement | Impact | Cost | What It Is | Source |
|------|-------------|--------|------|------------|--------|
| 1 | **Judas Swing Detector** | High | ~150 LOC | Detect London session false move sweeping Asian range then reversing. Critical ICT entry filter currently missing. | Day 9, 16 |
| 2 | **FVG 3-Candle Zero-Overlap Validation** | High | ~30 LOC | Enforce strict wick non-overlap in `ifvg.py`. Current implementation may accept overlapping wicks as FVGs. | Day 6 |
| 3 | **BOS vs CHOCH/MSF Differentiation** | High | ~80 LOC | Split `mss.py` into distinct BOS (continuation) and CHoCH/MSS (reversal) with displacement requirement. | Day 7 |
| 4 | **FVG Quality: CE Midpoint Entries** | High | ~60 LOC | Add Consequent Encroachment (50% FVG midpoint) to `ifvg.py`. Price often only retraces to CE. Tighter stops. | Day 6 |
| 5 | **PO3/AMD Integration into SMCStrategy** | High | ~100 LOC | Wire existing `power_of_3.py` into composite scoring. Gate: no entry until manipulation phase complete. | Day 8 |
| 6 | **SMC 6-Step: Sweep-then-CHOCH Entry** | High | ~120 LOC | Current code enters directly on sweep. Tutorial says: sweep first, WAIT for CHoCH confirmation on LTF, THEN enter at OB/FVG formed during CHoCH. | Day 15 |
| 7 | **Asian Range Directional Bias** | High | ~50 LOC | If London sweeps Asian low and reverses → bullish day. If sweeps Asian high and reverses → bearish. Use as mandatory daily bias for all entries. | Day 9 |

### Tier 2 — MEDIUM Impact, Structural Improvements (100-400 LOC)

| Rank | Improvement | Impact | Cost | What It Is | Source |
|------|-------------|--------|------|------------|--------|
| 8 | **S&D Zone Patterns (RBD/DBR/RBR/DBD)** | Medium | ~200 LOC | Classify consolidation bases into 4 S&D zone types. RBD/DBD = supply continuation/reversal. DBR/RBR = demand continuation/reversal. Weight zones by type. | Day 11 |
| 9 | **OB Validity: FVG Colocation Check** | Medium | ~50 LOC | Valid OB should have FVG immediately adjacent. Add colocation scoring in `smc_strategy.py`. | Day 5 |
| 10 | **Breaker vs Mitigation: Sweep Distinction** | Medium | ~80 LOC | Modify `breaker.py` to classify as "true breaker" (sweep confirmed) vs "mitigation" (no sweep). Weight breaker higher. | Day 12 |
| 11 | **Unicorn Pattern (Breaker+FVG Overlap)** | Medium | ~100 LOC | Detect when FVG forms inside breaker block during displacement. Dual confluence = highest probability entry. | Day 12 |
| 12 | **OTE Integration into Strategy Scoring** | Medium | ~100 LOC | Wire existing `ote.py` into composite scoring. Add equilibrium-first filter (must cross 50% fib first). | Day 13 |
| 13 | **Risk Management Wiring** | Medium | ~150 LOC | Wire `smc_aware.py` risk rules into strategies: 1% rule, daily loss limit, partial profits at T1, BE move at first structural level. | Day 17 |
| 14 | **POI 4-Criteria Grading System** | Medium | ~150 LOC | Implement structured grading: BOS-triggered, liquidity-protected, unmitigated, closest-to-price. Score 0-4. | Day 14 |

### Tier 3 — LOWER Impact, Architecture Changes (200-500+ LOC)

| Rank | Improvement | Impact | Cost | What It Is | Source |
|------|-------------|--------|------|------------|--------|
| 15 | **SMC 10-Checklist Workflow** | Medium | ~150 LOC | Implement structured pre-trade checklist in `smc_strategy.py` with all 10 checks. No trade if any fail. | Day 15 |
| 16 | **ICT 6-Step Trade Plan** | Medium | ~250 LOC | Full ICT trade plan: IPDA narrative → daily bias → Judas Swing → MSS+displacement → Kill Zone FVG entry → targets. | Day 16 |
| 17 | **IPDA Narrative (20/40/60-day)** | Medium | ~150 LOC | Mark 20/40/60-day highs/lows as DOL targets. Liquidity pools form on ~20-day cycles. | Day 16 |
| 18 | **Multi-TF PD Array Matrix (Live)** | Medium | ~300 LOC | Make `pd_array_matrix.py` dynamic across daily/4H/1H TF hierarchy. Scan premium/discount for nearest PD array. | Day 14 |
| 19 | **SFP (Swing Failure Pattern)** | Low | ~100 LOC | Implement missing SFP detector mentioned in `__init__.py` docstring. | Day 3 (implied) |
| 20 | **Market-Specific Kill Zones** | Low | ~100 LOC | Adjust kill zone times per market: crypto → US equity open, gold → London-NY overlap, stocks → 9:30-11am EST. | Day 18 |
| 21 | **DOL Targeting in Main Strategy** | Low | ~80 LOC | Use nearest BSL/SSL as mandatory take-profit target instead of fixed ATR multiples. | Day 4 |

---

## Part 3: SMC 6-Step Trade Plan — Reference Implementation Design

This is the single most actionable template from the tutorial. Here's how it should work:

```
STEP 1 — WEEKLY BIAS (Sunday/Monday pre-session)
├─ Open weekly chart
├─ Ask: HH+HL (bullish) or LL+LH (bearish)?
├─ Find nearest BSL (above equal highs) or SSL (below equal lows)
└─ Write: "Weekly bullish. DOL = weekly high at X."

STEP 2 — DAILY BIAS (each morning pre-session)
├─ Drop to daily chart
├─ Confirm BOS aligned with weekly
├─ Identify unmitigated POIs between price and weekly DOL
├─ Check: has daily DOL already been reached? If yes → depleted, no trade
└─ Write: "Daily bullish. Last BOS at X. POI = 4H OB at Y."

STEP 3 — GRADE THE POI (4H / 1H)
├─ Apply 4 criteria:
│  ├─ Caused BOS or structural shift?
│  ├─ Protected by liquidity (stops resting beyond)?
│  ├─ Unmitigated (never fully retested)?
│  └─ Closest POI to current price?
├─ All 4 pass → active POI
└─ Any fail → no trade today

STEP 4 — WAIT FOR LIQUIDITY SWEEP AT POI
├─ DO NOT enter on first touch of POI
├─ Wait for wick below zone (sweep)
├─ Wait for aggressive rejection candle
├─ If price slices cleanly through POI with no sweep/rejection → zone invalidated, no trade
└─ ONLY proceed if sweep + rejection confirmed

STEP 5 — LTF CHoCH CONFIRMATION (15M / 5M)
├─ Drop to 15M or 5M chart
├─ Wait for CHoCH (break of recent lower high for bullish)
├─ CHoCH = retracement over, HTF trend resuming
├─ Entry at FVG or OB formed during/after the CHoCH
└─ NO entry without CHoCH confirmation

STEP 6 — STOP + TARGET + R:R
├─ Stop: below the liquidity sweep wick (not below zone broadly)
├─ Target: next BSL (bullish) or SSL (bearish) = named liquidity level
├─ Minimum R:R = 1:3
└─ If R:R < 1:3 → no trade
```

### SMC Pre-Trade Checklist (Every Trade):
```
☐ Weekly bias defined and written down
☐ Daily BOS confirms weekly direction
☐ Unmitigated POI identified and graded
☐ POI passes all 4 quality criteria
☐ Price swept liquidity at/below POI before entry
☐ CHoCH formed on 15M/5M at POI
☐ Entry at FVG or OB created during CHoCH
☐ Stop below sweep low (not zone broadly)
☐ Target = named liquidity level (not round number)
☐ R:R ≥ 1:3
```

---

## Part 4: ICT 6-Step Trade Plan — Reference Implementation Design

```
STEP 1 — IPDA NARRATIVE (weekly, pre-session)
├─ Open weekly chart
├─ Mark 20-day, 40-day, 60-day highs and lows
├─ Identify next DOL (BSL for bullish, SSL for bearish)
└─ Write: "IPDA bullish. 40-day high at X = next DOL."

STEP 2 — DAILY BIAS (each morning, BEFORE London open)
├─ Drop to daily chart
├─ Check: price in discount (buy) or premium (sell)?
├─ Check: price relative to yesterday's high/low
├─ Locate nearest unmitigated PD array in correct zone
└─ Write: "Daily bias bullish. Discount zone. OB at Y."

STEP 3 — WAIT FOR JUDAS SWING (London Kill Zone 2-5am EST)
├─ DO NOT trade before Judas Swing completes
├─ Bullish day expectation: price drops below Asian low (false move)
├─ Bearish day expectation: price rises above Asian high (false move)
├─ Wait for liquidity sweep to complete
└─ DO NOT ENTER during the Judas Swing

STEP 4 — CONFIRM MSS WITH DISPLACEMENT (still in Kill Zone)
├─ Large-bodied candle breaks last lower high (bullish) or higher low (bearish)
├─ FVG left behind by displacement candle
├─ Without displacement → weak MSS → skip
└─ Displacement + FVG = your entry zone

STEP 5 — ENTER PD ARRAY INSIDE KILL ZONE
├─ Price retraces into FVG created by displacement → enter
├─ OR use OTE 62-79% retracement with 70.5% sweet spot
├─ Kill Zone is MANDATORY — setup outside KZ = no trade
└─ London KZ (2-5am) or NY KZ (7-10am forex / 8:30-11am indices)

STEP 6 — STOP + TARGETS
├─ Stop: below Judas Swing low (bullish) or above Judas Swing high (bearish)
├─ Target 1: -0.27 Fibonacci extension of Judas Swing leg
├─ Target 2: next IPDA DOL (20/40/60-day high/low)
└─ Take 50% at T1, move stop to BE, let 50% run to T2
```

### ICT Pre-Trade Checklist:
```
☐ IPDA narrative written — 20/40/60-day levels marked
☐ Daily bias set before London open — discount/premium confirmed
☐ Nearest PD array identified in correct zone
☐ Judas Swing completed — liquidity swept, false move confirmed
☐ MSS formed with displacement — large candle + FVG visible
☐ Entry zone (FVG or OTE) inside active Kill Zone
☐ Stop below Judas Swing low (not below POI broadly)
☐ T1 = -0.27 extension
☐ T2 = named IPDA DOL
☐ Risk ≤ 1-2% of account
```

---

## Part 5: Quick Wins — Top 5 Changes to Make Now

1. **Add Judas Swing detector** (`src/indicators/judas_swing.py`): Detect London session false move of Asian range + reversal. Wire into `SMCStrategy` as mandatory gate before entry.

2. **Enforce CHoCH confirmation before entry**: Current `smc_strategy.py` enters on sweep signal. Change to: sweep first → wait for LTF CHoCH → enter at CHoCH-formed FVG.

3. **Add FVG zero-overlap validation**: In `ifvg.py`, add check that C1 wick and C3 wick do NOT overlap. Currently may be accepting overlaps as FVGs.

4. **Wire PO3/AMD into composite scoring**: `power_of_3.py` already detects AMD phases. Gate all entries: no trade during accumulation or pre-manipulation.

5. **Add CE (Consequent Encroachment) entry logic**: Add 50% FVG midpoint as alternative entry level to `ifvg.py`. Often fills to CE and reverses; currently waiting for full fill misses many trades.

---

## Part 6: Terminology Translation Table (ICT ↔ SMC ↔ Codebase)

| SMC Term | ICT Equivalent | Codebase Reference |
|----------|---------------|-------------------|
| Point of Interest (POI) | PD Array (premium/discount zone) | `pd_array_matrix.py` / `smc_strategy.py` |
| Change of Character (CHoCH) | Market Structure Shift (MSS) | `mss.py` (but conflated with BOS) |
| Liquidity sweep | Stop hunt / Inducement | `liquidity_sweep.py` |
| Manipulation phase (AMD) | Judas Swing | `power_of_3.py` (orphaned) / NOT implemented |
| Draw on Liquidity (DOL) | IPDA DOL target | `cameron_model.py` |
| LTF CHoCH confirmation | Displacement + FVG after MSS | NOT implemented as combined logic |
| Bullish POI in discount | Discount-zone bullish PD array | `pd_array_matrix.py` (static) |
| S&D demand zone | Institutional demand level | NOT implemented |
| Equal highs/lows | Liquidity pool magnet | NOT specifically detected |
| Premium/Discount | P/D in PD Array Matrix | `pd_array_matrix.py` (static) |
| Sweep below POI before entry | Manipulation phase complete | NOT enforced |

---

*Analysis completed 2026-05-20. Total: 49 concepts extracted from tutorial, 21 improvement opportunities ranked by impact/cost.*
