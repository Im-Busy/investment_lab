# Chart Pattern Knowledge Base — Consolidated from PDF Extractions

> **Date:** 2026-05-12
> **Status:** Multi-part (Batch 1 of ~4). Still pending: TRADE-CHART-PATTERNS-86-169, TRADE-CHART-PATTERNS-170-207, TRADE-CHART-PATTERNS-208-293, 151-Trading-Strategies_361Pages, Harmonic Guide (rest).
> **Purpose:** Single-source reference for all quantifiable pattern rules extracted from PDFs in `useful_resources/PDFs_Found_Online/`.

---

## Part 1: Suri Duddella — "Trade Chart Patterns Like The Pros" (2007)

**Source:** `TRADE-CHART-PATTERNS-GUIDE-1-85.md`
**~65 patterns, Fibonacci-centered, bar-chart focused**

### 1.1 Entry Rules (General)
- Find a reversal bar (wider = better), enter 1-2 ticks above high (long) or below low (short)
- Entries valid for next 3-5 bars only; beyond that, avoid setup

### 1.2 Stop Rules
- Fibonacci confluence levels, major swing highs/lows, trendline S/R, or half pattern range

### 1.3 Target Rules
- Fibonacci ratios of prior swing: 0.382, 0.5, 0.618, 0.786, 1.272, 1.618
- Multiples of prior swing range, channel range, pattern depth, prior swing highs/lows

### 1.4 Market Structures (MSL/MSH)
| Pattern | Structure | Entry | Stop | Target |
|---------|-----------|-------|------|--------|
| **MSL** (bullish) | 3 candles: new low, lower low, **higher low of CLOSE** | Above highest close of 3 candles | Below MSL low | Next MSH formation |
| **MSH** (bearish) | 3 candles: new high, higher high, **lower high** | Below low of 3rd candle | Above MSH high | Next MSL formation |

### 1.5 Matching Highs/Lows
- Minimum 3 bars at same level; failure to break = reversal signal
- **Matching Lows (bullish)**: Enter above breakout bar high; Target = 1x/2x breakout bar length
- **Matching Highs (bearish)**: Enter below breakdown bar low; Target = 1x/2x breakdown bar length

### 1.6 n-Bar Rallies/Declines
- ≥21 bar new high/low with ≥3 successive new extremes
- **Entry**: 1 tick above last falling bar's high (long) or below last rising bar's low (short)
- **Target**: 62%-100% of entire n-Bar range

### 1.7 NR7ID (Narrow Range 7 + Inside Day)
- NR7: Daily range narrower than prior 6 days
- ID: Current range entirely within prior day's range
- **Trade**: ORB (Opening Range Breakout) above high + ORB or below low - ORB
- ORB = 10-bar avg distance from open-to-high and open-to-low

### 1.8 WR7OD (Wide Range 7 + Outside Day)
- WR7: Current bar has widest range in last 7 bars
- OD: High > prior high AND Low < prior low
- **Target**: 50%-100% of WR7OD range from breakout level

### 1.9 Floor Pivots
```
Pivot = (H + L + C) / 3
R1 = 2*PP - L       S1 = 2*PP - H
R2 = PP + (R1-S1)   S2 = PP - (R1-S1)
R3 = R1 + (H-L)     S3 = S1 - (H-L)
```

### 1.10 FibZone Pivots
```
R1 = PP + 0.5*DR       S1 = PP - 0.5*DR
R2 = PP + DR           S2 = PP - DR
RB1 = PP + 0.618*DR    SB1 = PP - 0.618*DR
RB2 = PP + 1.382*DR    SB2 = PP - 1.382*DR
```

### 1.11 3-Bar Group Patterns
Six quantifiable OHLC patterns with precise conditions (see source file for exact formulas)

### 1.12 Fractal Patterns
- 5-bar Fractal: 3 bars higher highs + 2 bars lower lows = potential bullish trend change
- Enter above previous bar's high after Fractal formation; stop below Fractal low

---

## Part 2: Harmonic Pattern Trading Guide (TradingStrategyGuides.com)

**Source:** `The Ultimate Harmonic Pattern Trading Guiddes.md`

### 2.1 Pattern Fibonacci Ratios Summary

| Pattern | AB (retrace of XA) | BC (retrace of AB) | CD (target) | Entry Point | Stop Loss | TP1 | TP2 |
|---------|-------------------|-------------------|-------------|-------------|-----------|-----|-----|
| **Butterfly** | 0.786 | 0.382-0.886 | 1.272-1.618 XA | 1.272 XA ext | Below 1.618 XA | Point B | 0.618 CD |
| **Cypher** | 0.382-0.618 | Extend to 1.272-1.414 XA | 0.786 XC | 0.786 XC | Below X | Point A | — |
| **Bat** | 0.382-0.50 | 0.382-0.886 | 0.886 XA | 0.886 XA | Below X | Point C | Point A |
| **Gartley** | 0.618 | 0.382-0.786 | 0.786 XA or 1.27-1.618 AB | 1.272 AB | Below X | XA length from D | — |
| **Crab** | 0.382-0.618 | 0.382-0.886 | 2.24-3.618 AB / 1.618 XA | 2.24 AB ext | Above 3.618 (trail to D) | 0.618 CD | Below A |
| **Shark** | AB extends XA 1.13-1.618 | BC = 113% 0X | CD = 50% BC | At C (not D) | Below 1.15 XA | 50% CD | 100% CD (C) |

### 2.2 Key Differentiators
- **Butterfly**: B at 78.6% XA (deepest B retracement of all)
- **Bat**: B at 38.2-50% XA, D at 88.6% XA (deep D retracement)
- **Gartley**: B at 61.8% XA, AB=CD symmetry common
- **Crab**: Extreme CD extension (2.24-3.618 AB), volatile entries
- **Cypher**: C extends BEYOND X (1.272-1.414 XA), highest win rate claim
- **Shark**: 5-point structure (O-X-A-B-C), entry at C not D

### 2.3 Preferred Timeframes
- 1H, 4H, Daily (avoid lower timeframes)

---

## Part 3: NCFE Technical Analysis Price Patterns

**Source:** `Technical Analysis Price Patterns.md`

### 3.1 8-Step Trading Framework
1. Direction of Major Price Trend
2. Internal Health of Trend (Volume & Open Interest)
3. Closest Support and Resistance Levels
4. Look for Orthodox Price Pattern
5. Determine Measuring Objective and Stop-Out Levels
6. Create Trading Plan
7. Do Not Force a Conclusion
8. Diversify the Analysis

### 3.2 Pattern Reliability Table
| Pattern | Reliability | Key Trigger | Measuring Rule |
|---------|------------|-------------|----------------|
| Head & Shoulders | **86-88%** | Neckline break | Head-to-neckline height |
| Symmetrical Triangle | **76-78%** | Either boundary break before ¾ distance | Height from Point 2 |
| Right Angle Triangle | **75-80%** | Horizontal/sloping line break | Height from Point 2 |
| Double Top/Bottom | Moderate | Intervening trough/peak break | Pattern height |
| Wedge | Moderate | Converging same-slope lines | Point 1 extreme |
| Flag/Pennant | High (short-term) | Consolidation break | Flagpole duplication |

### 3.3 H&S Validation Criteria
- Prior trend exists
- 5-point reversal (LS→H→RS + neckline breaks)
- High volume at Head, declining OI at Head
- Declining volume at Right Shoulder
- Right Shoulder < Head (for Top)
- Close outside pattern (neckline break)
- Symmetry between shoulders (preferred)

### 3.4 Gap Classification
| Gap | Location | Volume | Significance |
|-----|----------|--------|-------------|
| Pattern/Area Gap | Within congestion | Normal | Low; quickly filled |
| Breakaway Gap | Start of new move | High | High; confirms breakout |
| Runaway/Measuring Gap | Mid-trend | Surge | Typically at halfway point |
| Exhaustion Gap | End of trend | Extremely high | Reversal warning |

### 3.5 Minor Trend Indicators
- **Key Reversal**: New high but closes lower + high volume
- **Inside Day/Range**: Entire range within prior bar → momentum pause
- **Outside Day/Range**: High higher AND Low lower → volatility expansion
- **Mid-Range Close**: Close at midpoint → likely adverse move

---

## Part 4: Fidelity / Charles Kirkpatrick — Identifying Chart Patterns

**Source:** `Identifying Chart Patterns with Technical Analysis.md`

### 4.1 Confirmation Filters
- Intrabar, Multiple closes, Time, Percentage/point, Money

### 4.2 Stop-and-Reverse Strategy
1. Enter on breakout
2. Protective stop outside breakout bar
3. Entry stop at same level (stop-and-reverse)
4. If breakout is real → profit from breakout
5. If breakout is false → profit from reversal

### 4.3 Best Multi-Bar Patterns (Kirkpatrick)
| Direction | High-Performance Patterns |
|-----------|--------------------------|
| Upward | Descending Triangle, Rectangle, Pipe Bottom |
| Downward | Flag, Head and Shoulders Top, Island Reversal |

### 4.4 Candlestick Pattern Summary
| Pattern | Bars | Signal | Context |
|---------|------|--------|---------|
| Doji | 1 | Indecision | Can warn of reversal |
| Harami | 2 | Reversal (slight upward bias) | Small body inside large |
| Hammer | 1 | Bullish reversal (below avg perf) | After downtrend |
| Hanging Man | 1 | Continuation (random bias) | After uptrend |
| Shooting Star | 1 | Average performance | After uptrend |
| Inverted Hammer | 1 | Average performance | After downtrend |
| Engulfing | 2 | Good performance in trend | Second engulfs first |
| Dark Cloud Cover | 2 | Bearish reversal | Second opens above, closes >50% into first |
| Piercing Line | 2 | Bullish reversal | Second opens below, closes >50% into first |

### 4.5 Volatility Patterns
- **Inside Bar**: High+Low within prior bar → low volatility, potential breakout
- **NR4**: 4th bar has narrowest range of last 4 → breakout imminent
- **Pipe Bottom**: 2-bar reversal at trend end; first closes at low, second closes in upper half

### 4.6 Gap Trading: Explosion Gap Pivot Strategy
1. After gap up, wait for "throwback" (pullback)
2. If throwback fills the gap → no trade
3. If throwback stops → mark "pivot low"
4. Buy entry above high of gap bar
5. Stop: initially at gap low, then trail below pivot low

---

## Part 5: Chart Patterns Reference Guide

**Source:** `Chart-Pattern-PDF-Free-Download.md`
**Patterns:** 42 patterns cataloged

### 5.1 Reversal Patterns (12)
Double Top/Bottom, Triple Top/Bottom, H&S / Inverse, Rounding Top/Bottom, Island Reversal, V-Top/Bottom

### 5.2 Continuation Patterns (15)
Ascending/Descending/Symmetrical Triangles, Rising/Falling Wedges, Flags & Pennants, Rectangle, Cup & Handle

### 5.3 Complex Patterns (8)
Diamond Top/Bottom, Broadening Top/Bottom/Channel, Pipe Top/Bottom, Spikes, Gaps

### 5.4 Advanced (5)
Harmonic, Elliott Wave, Three Drives, Quasimodo, Bump and Run

### 5.5 Specialty (2)
Dead Cat Bounce, Scallop/Staircase

---

## Part 6: Warrior Trading — Day Trading Rules

**Source:** `ChartPatternsv2_WarriorTrading.md`
**Focus:** Intraday patterns (1-min, 5-min)

### 6.1 Quantifiable Rules
| Setup | Rule |
|-------|------|
| **Bull Flag** | Next candle closes above previous candle's high → buy above that high |
| **Bear Flag** | Short first candle to make new low |
| **Flat Top Breakout** | Buy first candle that breaks flat top resistance |
| **Flat Bottom Breakdown** | Short first candle below flat bottom support |
| **Opening Range Breakout** | Break of first 1-min candle high/low at 9:30am |
| **VWAP Pullback** | First pullback to VWAP after break above → buy |
| **Whole/Half Dollar** | Buy first candle to break .00 or .50 level |
| **R2G (Red to Green)** | Opens below prior close, reverses above ≥ momentum entry |
| **5+ Consecutive Candles** | ≥5 consecutive candles in one direction → reversal alert |
| **10+ Consecutive 1-min** | 10+ candles same direction, first new high candle = exhaustion |
| **ABCD Pattern** | A→B (strong), B→C (pullback), C→D weaker, D breakout |

### 6.2 Key Principles
1. Confirmation over prediction (wait for candle CLOSE)
2. Volume confirmation on breakouts
3. Multi-timeframe: 5-min for context, 1-min for entry
4. VWAP as institutional benchmark
5. False breakout recognition critical

---

## Part 7: 151 Trading Strategies (Chenjie LI, 2024)

**Source:** `151TradingStrategies_49Pages.md`
**Content:** 42 options strategies with Black-Scholes formulas.
**Relevance:** Options-specific. Not directly applicable to equity chart pattern system.
**Possible future use:** Options overlay module for hedging.

---

## Part 8: SRCC Academic Overview

**Source:** `B.Com(Hons)...SRCC.md`
**Content:** Academic introduction to technical analysis. Previously extracted in C7 session.
**Extraction:** `useful_resources/papers_md/SRCC_TECHNICAL_ANALYSIS_extracted.md`

---

## 🎯 Actionable Insights for This Project

### Immediate (for C8 — Pattern Detector Confluence)
1. **Wire Fibonacci-based harmonic patterns** as highest-confidence signals:
   - Gartley (B=61.8% XA, CD=0.786 XA) — most frequent
   - Butterfly (B=78.6% XA, CD=1.272 XA) — deep reversal
   - Bat (B=38.2-50% XA, D=88.6% XA) — tight stop
   - Cypher (C beyond X, D=0.786 XC) — highest win rate

2. **Pattern reliability weights** for signal boosting:
   - H&S: 0.86-0.88 weight
   - Symmetrical Triangle: 0.76-0.78 weight
   - Right Angle Triangle: 0.75-0.80 weight
   - Flag/Pennant: high (short-term)

3. **Volume/OI validation rules** from NCFE:
   - High volume at breakout = confirmation
   - Declining volume during pattern formation = normal
   - OI decline at Head (H&S) = validation

### Short-term (for pattern detector improvements)
4. **Market Structure detection (MSL/MSH)** — 3-candle patterns from Duddella
   - Already partially implemented as `matching_lows.py`
5. **NR7ID pattern** — already implemented as `nr7id.py`
6. **Explosion Gap Pivot** — gap trading with pivot low confirmation
7. **Stop-and-Reverse strategy** — false breakout handling from Kirkpatrick

### Medium-term
8. **Options hedging** — 42 Black-Scholes strategies could complement equity positions
9. **Multiple take-profit methodology** — standard practice in harmonic trading (TP1 at B, TP2 at 0.618 CD)
10. **Fibonacci clusters** — multiple Fib levels converging = stronger S/R

---

## Part 9: Duddella — Geometric Patterns, Channels, Bands (pp. 86-169)

**Source:** `TRADE-CHART-PATTERNS-GUIDE-86-169.md`

### 9.1 ABC Pattern (Duddella)
| Parameter | Range |
|-----------|-------|
| C pivot retracement of AB | 0.382 to 0.618 |
| K (projection from C) | 0.382 to 0.886 of AB |
| BD extension | 1.232 to 2.618 of BC |
| **Entry (Bullish)** | Long above previous bar's high |
| **Stop** | Below level C |
| **Target 1** | 100% of AB range |
| **Target 2** | 127% of BC range |

### 9.2 Gartley Pattern (Duddella)
| Point | Requirement |
|-------|-------------|
| X | Start pivot |
| B | 0.382 to 0.618 of XA |
| C | 0.618 of AB |
| D (PRZ) | 0.618 to 0.786 of XA, OR 1.27 to 1.62 of BC |
| **Entry** | Only after D formation + reversal bar |
| **Stop** | Below D (long) / Above D (short) |
| **Target** | C/A levels; then 1.27-1.62 of AD |

### 9.3 Bat Pattern (Duddella)
| Parameter | Bat | vs Gartley |
|-----------|-----|------------|
| B retracement | 0.382-0.618 (**< 0.618**) | 0.618 |
| D retracement | **Precisely 0.886 XA** | 0.786 XA |
| PRZ | 1.27AB=CD + 1.62BC + 0.886XA | AB=CD |
| **Stop** | 1 tick below X (long) / above X (short) |
| **Target 1** | A level or 1.27 XA |
| **Target 2** | 1.62 to 2.0 XA |

### 9.4 Butterfly Pattern (Duddella)
| Parameter | Requirement |
|-----------|-------------|
| AB retracement | **MUST be 0.786** of XA |
| D point | 0.786 to 0.886 of XA; D **extends beyond X** |
| Perfect pattern | AB = CD |
| **Entry** | 1 tick above high (long) / below low (short) of confirmation bar |
| **Stop** | Below Butterfly low (long) / Above high (short) |
| **Target 1** | 100% of AD from D |
| **Target 2** | 162% of XA from D |

### 9.5 Crab Pattern (Duddella)
| Parameter | Requirement |
|-----------|-------------|
| B retracement | 0.618 of XA |
| D extension | **1.618 of XA** (distinctive) |
| PRZ confluence | 1.27 AB + 1.62 XA + 2.62-3.62 BC |
| **Entry** | Above/below confirmation bar from PRZ |
| **Stop** | Below/above PRZ levels |
| **Target 1** | B level |
| **Target 2** | C level |
| **Target 3** | A level |

### 9.6 Symmetric Triangle
```
Depth = Highest high - Lowest low
Target = Breakout point ± Depth
```
- Prices must intersect each trendline **at least twice**
- **Entry**: 1-2 ticks above/below breakout bar
- **Stop**: Below first major swing low / Above first major swing high
- **Target**: 100% depth (primary), 50% depth (partial exit)

### 9.7 Ascending Triangle
- Flat horizontal resistance + rising support
- Prices intersect each line **at least twice**
- **~75% target achievement**, breakout near 3rd/4th attempt
- **Target**: Depth = Top - Lowest upward slope point; add to breakout

### 9.8 Descending Triangle
- Downward-sloping resistance + flat horizontal support
- Same rules as Ascending, mirrored; favors breakdowns

### 9.9 Rectangle Pattern
```
Depth = Upper trendline - Lower trendline
Target = Breakout point ± Depth
```
- Prices must intersect each boundary **at least twice**
- **Stop**: Middle of rectangle channel
- **Target**: 70-100% of rectangle depth

### 9.10 Bull Flag
```
AB = Distance from swing low (A) to flag formation (B)
C = Flag breakout point
Target_1 = C + (0.70 to 1.00) × AB
Target_2 = C + (1.38 to 1.62) × AB  (bull markets)
```
- Lower highs/lows in tight formation, nearly parallel trendlines

### 9.11 Bear Flag
```
Target_1 = C - (0.76 to 1.00) × AB
Target_2 = C - (1.38 to 1.62) × AB
```

### 9.12 Rising Wedge
- Higher highs + higher lows with converging angled trendlines
- Typically bearish; prices intersect each line **at least twice**
- **Target**: Lowest point in wedge formation

### 9.13 Falling Wedge
- Lower highs + lower lows with converging/diveging angled trendlines
- Typically bullish; **Target**: Higher swing high of wedge pattern

### 9.14 Diamond Pattern
- 4-sided: Inverted Triangle + Symmetric Triangle combination
- **Continuation**: Trade in prior trend direction
- **Reversal**: Trade opposite direction
- **Target**: Prior range before Diamond formation

### 9.15 Donchian Channel
```
Upper = Highest High(20)
Lower = Lowest Low(20)
Mid = (Upper + Lower) / 2
```
- **Entry**: Price exceeds 4-week high/low
- **Stop/Target**: Mid-channel level or 1.5-2× ATR

### 9.16 Broadening Pattern (Megaphone)
- 5 swing points; each swing broader than previous
- **5th Swing Method**: Trade at trendline, target opposite side
- **Breakout Method**: Trade breakout direction, target = full pattern height

### 9.17 Linear Regression Channel (LRC)
```
Center Line = Least-squares regression
Upper/Lower = Center ± X Standard Deviations
```
- Must form ≥12-15 bars
- **Trend change**: Prices outside LRC for ~half LRC length

### 9.18 Andrew's Pitchfork
- 3 pivots A, B, C → median line from A to midpoint of B-C
- **~80% of time**: prices gravitate towards median line
- **Trading**: Fade at upper/lower line, target median; stop outside pitchfork

### 9.19 Bollinger Bands
```
Middle = SMA(Close, 20)
Upper/Lower = Middle ± 2 × StdDev(Close, 20)
```
- **Squeeze**: Bands constrict → impending sharp price change
- **Reversal**: New high/low outside band, then falls inside → potential reversal

### 9.20 Keltner Bands
```
Pivot = (H+L+C)/3
Middle = MA(Pivot, 10)
Upper/Lower = Middle ± ATR(10)
```
- **Trend method**: Buy above upper band, sell below lower band; stop at middle
- **Bollinger-Keltner Squeeze**: BB(21,2SD) inside KC(10) → trade breakout

### 9.21 Fibonacci Bands
```
MA = EMA(Close, 34)
TR = EMA(TrueRange, 8)
Band ±1: MA ± 1.62×TR   (φ)
Band ±2: MA ± 2.62×TR   (φ²)
Band ±3: MA ± 4.23×TR   (φ³)
```
- **Reversal**: Price trades outside band → re-enters → trade reversal
- **Target**: MA (center) or opposite extreme band

---

## Part 10: Duddella — Zigzag, Price-Action, Tops/Bottoms (pp. 170-207)

**Source:** `TRADE-CHART-PATTERNS-GUIDE-170-207.md`

### 10.1 Zigzag Pattern
- Trend indicator only (no predictive power)
- ATR(10) determines minimum move; plots Fib levels 23.6%-78.6%
- Used as input to Elliott Wave, M/W detection, Fib clusters

### 10.2 Elliott Wave
- 5-wave motive (W1-W5) + ABC correction
- W2 NOT below W1 low; W3 longest/strongest; W4 NOT below W2 high
- Fibonacci for time/price: 0.618, 1.618, 2.618

### 10.3 Crown Pattern
| Version | Key Ratio | Target |
|---------|-----------|--------|
| **Bearish** | BC=1.618×AB, DE=0.618×CD, DE=EF | Major swing lows; trail with 2-bar high |
| **Bullish** | CD=1.272×BC, DE=0.618×CD | Major swing highs; trail with 2-bar low |

### 10.4 Cup and Handle
- Handle corrects **25%-38%** of Cup depth
- **Entry**: Close above Cup high
- **Stop**: Below Handle low
- **Target**: 62%-100% of Cup depth from breakout

### 10.5 Head & Shoulders (Detailed)
- Volume: Heavy on LS+Head, dissipating on RS, increases at breakdown
- **Target**: Head-to-neckline distance × 62%-100%
- **H&S Failure**: Close back above neckline → counter-trade long with 100% depth target

### 10.6 Parabolic Arc
- 62% retracement from top is standard correction
- **Entry**: 2B sell at second failed peak test, or trendline break
- **Stop**: Few ticks above arc high

### 10.7 Three Hills and a Mountain (Two-Phase)
| Phase | Entry | Target | Stop |
|-------|-------|--------|------|
| **Phase 1 (Short)** | Close below trendline (3 hills) | 62% of AB | Above B |
| **Phase 2 (Long)** | Above bar high at 62% retracement C | 100%-127% of AB | Below C |

### 10.8 Three Valleys and a River (Two-Phase)
| Phase | Entry | Target | Stop |
|-------|-------|--------|------|
| **Phase 1 (Long)** | 1 tick above breakout bar | 62%-78% of AB | Below trendline |
| **Phase 2 (Short)** | At 62% AB completion at C | 100% of AB | Above C |

### 10.9 Spike and Ledge
- **Spike**: Climax high/low → **Ledge**: ~20 bars matching highs/lows → **Breakout** in opposite direction
- **Entry**: Breakout/breakdown from Ledge in opposite direction of Spike
- **Stop**: Opposite side of Ledge
- **Target**: Prior major swing or gap level
- Best on intra-day timeframes

---

## Part 11: Duddella — Exotic Patterns, Event Patterns (pp. 208-293)

**Source:** `TRADE-CHART-PATTERNS-GUIDE-208-293.md`

### 11.1 Adam-Eve Patterns
- Adam = sharp V spike; Eve = rounded formation
- Variants: Adam-Eve, Eve-Adam, Adam-Adam, Eve-Eve
- Triple sequences (Adam-Eve-Adam) → bigger moves

### 11.2 Trader Vic's 2B Pattern
| Setup | Checklist |
|-------|-----------|
| **2B Buy** | 1) New Low → 2) Retracement → 3) Close below Bar1 Low → 4) Close above Bar3 High → Long |
| **2B Sell** | 1) New High → 2) Pullback → 3) Close above Bar1 High → 4) Close below Bar3 Low → Short |

### 11.3 Trader Vic's 1-2-3 Pattern
1. Trendline breakout from current trend
2. Test and **failure** of previous high/low
3. Breakout of swing high/low prior to Rule 2
- **Entry**: After all 3 conditions met

### 11.4 Pipe Pattern
```
L = max(pipe1_range, pipe2_range)
Target_1 = Entry ± L
Target_2 = Entry ± (2 × L)
```
- Two parallel bars at extremes; second spike opposite direction ≥ first
- **Confirmation**: Close beyond extreme of both pipes

### 11.5 M and W Patterns
- Continuation patterns (NOT termination)
- M-Pattern: Short below neckline breakdown; target = pattern low
- W-Pattern: Long above neckline breakout; target = pattern high

### 11.6 Round Top (Saucer Top)
- Long formation; low volume; fast/sharp breakdowns
- High failure rate
- **Target**: Depth of Round top from trendline breakdown

### 11.7 Round Bottom (Saucer Bottom)
- After prolonged downtrend; better on higher timeframes
- Longer base → more profitable

### 11.8 V-Top Pattern
- Sharp ascent → mirrored descent; retraces to rally start
- **Spike & Ledge variant**: Ledge forms after spike → confirm reversal

### 11.9 V-Bottom Pattern
- Steep decline → fast rally; high volume on exhaustion day
- **Target**: Near pre-decline top

### 11.10 Double Top
```
Depth = Top - Neckline
Target = Breakdown - Depth
Stop = (Top + Neckline) / 2
```
- Volume heavier on first swing + breakdown

### 11.11 Double Bottom
```
Depth = Neckline - Bottom
Target_1 = Neckline + Depth  (100%)
Target_2 = Neckline + (Depth × 1.27) to (Depth × 1.62)
```
- Two bottoms within 2-5% of each other

### 11.12 Triple Top
```
Depth = Highest Top - Lowest Swing Low
Target = Breakdown - Depth
```
- Three failed new highs; volume: heavy→diminished→heavy at breakdown

### 11.13 Triple Bottom
```
Depth = Neckline High - Lowest Low
Target = Entry + (Depth × 0.62) to Entry + (Depth × 1.00)
```

### 11.14 Dragon Pattern (Bullish)
| Point | Fibonacci Constraint |
|-------|---------------------|
| Hump (C) | 38%-50% of AB |
| Second Leg (CD) | 61.8%-127% of AB |
| **Target 1** | 127% of CD range |
| **Target 2** | 88.6%-100% of BC range |
| **Target 3** | 138% of AB range |

### 11.15 Inverse Dragon (Bearish)
- Upside-down Dragon; hump 38%-50% of AB
- **Entry 1**: Close below trendline; **Entry 2**: Close below hump line

### 11.16 Sea Horse Pattern
- ABC variation; BC retracement: **38%-50%** of AB (vs ABC's 61.8%-78.6%)
- Faster descent angle
- **Target**: 100% of CD range from C

### 11.17 Scallops Pattern (J-Shaped)
```
J_Height = Top_of_J - Bottom_of_J
Target = Entry ± J_Height
```
- Ascending retracement: 38%-62% of prior Scallop

### 11.18 Gap Patterns (Four Types)
| Gap Type | Location | Volume | Trading |
|----------|----------|--------|---------|
| **Common** | Congestion | Low | Fade (experienced only) |
| **Breakaway** | Trend start | High | Trade direction; almost never filled |
| **Continuation** | Mid-trend | High | Trade direction |
| **Exhaustion** | Trend end | Extreme | Fade; reversal play |

**Gap size filter**: If daily gap > **2.5×** 10-day ATR → wait ≥1 day

### 11.19 Dead Cat Bounce
```
Gap_Range = |Event_Open - Event_Close_extreme|
Bounce = 0.50 to 0.62 × Gap_Range
Target = Entry ± Gap_Range
```
- Event must move price ≥15%; high failure rate

### 11.20 Island Reversal
- Gaps on **both sides** → isolated price cluster
- Very rare; signals much larger moves; usually news-driven
- **Entry**: Beyond second gap bar extreme

---

## Part 12: 151 Trading Strategies — Technical Analysis Extract

**Source:** `151-Trading-Strategies_361Pages.md`
**Scope:** 18 of 151 strategies involve chart patterns / technical analysis (rest are options, fixed-income, fundamental, etc.)

### 12.1 Trend-Following / Moving Average Strategies
| # | Strategy | Signal |
|---|----------|--------|
| 3.11 | Single MA | `P > MA(T)` → long; `P < MA(T)` → short |
| 3.12 | Two MAs (Golden/Death Cross) | `MA(T₁) > MA(T₂)` where T₁<T₂; SL variant at ±2% adverse move |
| 3.13 | Three MAs | `MA(T₁) > MA(T₂) > MA(T₃)`; exit on middle MA violation |
| 4.1.1 | Sector Momentum + MA Filter | Buy top-decile ETF only if `P > MA(100-200d)` |
| 4.1.2 | Dual-Momentum Sector Rotation | Broad market `P > MA` → risk-on; else risk-off |
| 4.6 | Multi-Asset Trend Following | Multi-ETF momentum + MA filter + vol-adjusted weights |
| 8.1 | MA with HP Filter (FX) | 2-MA crossover on HP-filtered FX; `λ = 100 × n²` |

### 12.2 Channel / Band Strategies
| # | Strategy | Signal |
|---|----------|--------|
| 3.15 | Donchian Channel | `P = floor` → bounce long; `P = ceiling` → fade short; or breakout follow |

### 12.3 Support & Resistance / Pivot Strategies
| # | Strategy | Signal |
|---|----------|--------|
| 3.14 | Pivot Points | `C = (H+L+C)/3`, `R = 2C − L`, `S = 2C − H`; `P > C` → long, target R; `P < C` → short, target S |

### 12.4 Mean-Reversion / Contrarian Strategies
| # | Strategy | Signal |
|---|----------|--------|
| 4.4 | IBS (Internal Bar Strength) | `IBS = (P_C − P_L)/(P_H − P_L)`; buy bottom decile, sell top decile |
| 10.3 | Contrarian (Futures) | `wᵢ = −γ × [Rᵢ − Rₘ]` — buy losers vs index, sell winners |
| 10.3.1 | Contrarian + Volume/OI Filters | High volume change → more overreaction → filter subset, then apply 10.3 |

### 12.5 Cross-Sectional Momentum Strategies
| # | Strategy | Core Formula |
|---|----------|-------------|
| 3.1 | Price-Momentum | `Rᶜᵘᵐᵢ = Pᵢ(S)/Pᵢ(S+T) − 1`; T=12M, S=1M skip |
| 4.1 | Sector Momentum | Same as 3.1 applied to sector ETFs; T=6-12M |
| 10.4 | Futures Trend Following | `ηᵢ = sign(Rᵢ)`, `wᵢ = γ × ηᵢ / σᵢ` |

### 12.6 ML-Based Pattern Recognition
| # | Strategy | Approach |
|---|----------|----------|
| 3.17 | Single-Stock KNN | k-NN using SMA price/volume features; 60/40 train/validation split |
| 18.2 | ANN for BTC | EMA + EMSD + RSI features → K-class softmax → extreme quantile = signal |

### 12.7 PCA Cluster Mean-Reversion
| # | Strategy | Approach |
|---|----------|----------|
| 3.9 | PCA Clustering | Mean-reversion within PCA-derived statistical clusters |

---

## 📋 Extraction Status: COMPLETE

All PDF extractions from `useful_resources/PDFs_Found_Online/` are now consolidated:
- ✅ Parts 1-8: Previously extracted (Duddella 1-85, Harmonic Guide, NCFE, Fidelity, Pattern Reference, Warrior Trading, 151-49pg, SRCC)
- ✅ Part 9: Duddella pp. 86-169 (Geometric, Channels, Bands)
- ✅ Part 10: Duddella pp. 170-207 (Zigzag, Elliott, Crown, C&H, H&S, Parabolic, Spike-Ledge)
- ✅ Part 11: Duddella pp. 208-293 (20 exotic/event patterns: 2B, Pipe, Dragon, Sea Horse, Scallops, Gaps, etc.)
- ✅ Part 12: 151 Trading Strategies (18 technical analysis strategies extracted from 814KB full book)
- ✅ Harmonic Guide (rest from line 656): Illustrations/appendix only — no new quantifiable content
