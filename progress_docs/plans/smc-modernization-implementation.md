# SMC Modernization — Agent Execution Plan

> **Source:** `progress_docs/plans/smc-modernization-proven-enhancements.md` (20 sections, 673 lines)
> **Target:** Rewrite `src/strategies/smc_strategy.py` as a single unified `backtesting.py`-compatible intraday strategy with all proven enhancements.
> **Date:** 2026-05-20 session start
> **Total phases:** 5 (P0–P4). Alloc ~3–4 hours.
> **Output:** `src/strategies/smc_strategy.py` (~500 loc), `scripts/backtest_smc.py` (~200 loc), full OOS validation in BESTS.md

---

## PRE-FLIGHT CHECKLIST (every session)

```
□ Read MEMORY.md — current objectives, system metrics, blockers
□ Read progress_docs/plans/smc-modernization-proven-enhancements.md — the research catalog
□ Read progress_docs/current.md — session log with timestamps
□ Run `uv sync` to ensure environment is current
□ Run `uv add smartmoneyconcepts` to install the SMC indicator library
□ Run `uv run ruff check src/` to verify clean baseline
```

---

## PHASE 1: FOUNDATION — Single Codebase + CLI + First Backtest (P0)

**Goal:** Merge dual codebase into single `smc_strategy.py`, add CLI, run first backtest to establish baseline.
**Time:** ~60 minutes
**Depends on:** `uv add smartmoneyconcepts`

---

### TASK 1.1: Install Dependencies

```bash
uv add smartmoneyconcepts
```

**Verify:** `uv run python -c "from smartmoneyconcepts import smc; print(dir(smc))"`

Expected output: `['bos_choch', 'fvg', 'swing_highs_lows', ...]`

### TASK 1.2: Create the Unified SMC Strategy

**File:** `src/strategies/smc_strategy.py`

**Design:** Single `backtesting.py` Strategy class. Merge logic from `smc_reversal.py` (717 loc) and `smc_reversal_bt.py` (426 loc) into one file. Use `smartmoneyconcepts` library for indicator computation where possible, falling back to existing custom implementations.

**Architecture:**

```
SMCStrategy(backtesting.Strategy)
├── init()
│   ├── _build_df()                    — OHLCV from backtesting.py internal data
│   ├── _precompute_indicators()       — Vectorized precomputation of ALL signals
│   │   ├── smc.swing_highs_lows()     — Swing points (smartmoneyconcepts)
│   │   ├── smc.bos_choch()            — BOS/CHOCH (smartmoneyconcepts)
│   │   ├── smc.fvg()                  — Fair Value Gaps (smartmoneyconcepts)
│   │   ├── _detect_asian_range()      — Session high/low (existing logic)
│   │   ├── _detect_sweep()            — Liquidity sweep from swing extremes
│   │   ├── _detect_order_blocks()    — Institutional OB zones
│   │   └── _compute_atr()             — ATR(14) precomputed array
│   └── _init_quality_gates()          — Optional gate multipliers (deferred to Phase 4)
│
├── next(idx)
│   ├── Warmup check (min_bars)
│   ├── _compute_smc_score(idx)        — Aggregate SMC signal
│   │   ├── BOS/CHOCH score             — Direction + confidence
│   │   ├── FVG proximity score         — How close is price to unfilled FVG
│   │   ├── Sweep confirmation         — Did sweep occur + reverse?
│   │   ├── Confluence bonus            — 2+ components agree
│   │   └── Volume confirmation         — Relative volume multiplier
│   ├── Entry: score > entry_threshold
│   ├── Exit: ATR trailing stop (deferred to Phase 2)
│   └── Exit: score drops below exit_threshold
│
└── Parameters (20 defaults)
    ├── entry_threshold: float = 0.55
    ├── exit_threshold: float = 0.30
    ├── trail_stop_atr: float = 3.0
    ├── min_reliability: float = 0.40
    ├── confluence_bonus: float = 0.10
    ├── volume_confirm: bool = True
    ├── use_multi_tp: bool = False     — Enable in Phase 2
    ├── tp1_atr / tp2_atr / tp1_size
    ├── session_start / session_end     — "00:00" / "08:00" UTC
    ├── sweep_buffer_mult: float = 0.5  — ATR multiplier for sweep detection
    ├── fvg_proximity_mult: float = 2.0 — ATR multiplier for FVG entry zone
    └── ...
```

**Key implementation rules:**

1. **All signals precomputed in `init()`** — no per-bar detection. Store precomputed numpy arrays in `self._signals_cache`.

2. **SMC component weights (initial defaults):**
   ```python
   SMC_COMPONENT_WEIGHTS: dict[str, float] = {
       "sweep_reversal": 0.75,    # Strongest — actual order flow
       "mss_bos_choch": 0.65,     # Structure confirmation
       "fvg_proximity": 0.55,     # Entry precision
       "order_block": 0.50,       # Supply/demand zone
   }
   ```

3. **Use `smartmoneyconcepts` where possible** — fall back to existing implementations where the library doesn't cover:
   - ✅ Use: `smc.swing_highs_lows()`, `smc.bos_choch()`, `smc.fvg()`
   - ❌ Not in library: Asian range, sweep detection → use existing code from `smc_reversal.py`
   - ❌ Not in library: Order blocks → use existing code from `smc_reversal.py`

4. **tanh() normalization on final score** — same as RulesFirst. Score ∈ (-1, +1).

5. **State machine preserved but simplified** — don't need 8 states. Use precomputed arrays with look-ahead-safe indexing:
   ```python
   # Instead of: if state == SWEEP_DETECTED: check for MSS...
   # Use precomputed sweep/MSS arrays with temporal ordering:
   sweep_at_bar = self._sweep_signals[idx]     # 1 = bullish sweep reversal, -1 = bearish
   mss_at_bar   = self._mss_signals[idx]       # 1 = bullish MSS, -1 = bearish
   fvg_near     = self._fvg_proximity[idx]     # 0 = far, 1 = at edge
   ```

**Verify:** File compiles cleanly.
```bash
uv run python -c "from src.strategies.smc_strategy import SMCStrategy; print('OK')"
```

---

### TASK 1.3: Create CLI Script

**File:** `scripts/backtest_smc.py`

**Template:** Copy structure from `scripts/backtest_rules_first.py`, adapt for SMC.

```python
"""SMC Intraday Strategy Backtest CLI.

Usage:
    uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h
    uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --sweep-entry
    uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-12-31
    uv run scripts/backtest_smc.py --symbol GC=F --interval 1h --session-start 00:00 --session-end 08:00
"""
```

**Required features:**
- `--symbol` — data from `data/raw/{symbol}_1h.csv` or fetch via yfinance
- `--interval` — `1h` or `5m` (maps to correct data file)
- `--start` / `--end` — backtest date range
- `--sweep-entry` — sweep entry_threshold 0.35–0.70
- `--sweep-buffer` — sweep sweep_buffer_mult 0.3–1.0
- `--session-start` / `--session-end` — session times (default "00:00" / "08:00")
- `--json` — output results as JSON to `reports/smc/`
- Report: Return%, Sharpe, Trades, Win%, PF, MaxDD, DD%, Exp%, AnnRet
- Print comparison table with B&H baseline

**Data fetching fallback:**
```python
def fetch_data(symbol, interval="1h"):
    path = Path(f"data/raw/{symbol}_{interval}.csv")
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    # Fetch via yfinance
    import yfinance as yf
    period = "730d" if interval == "1h" else "60d"
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    df.to_csv(path)
    return df
```

**Verify:** Script runs without errors (expects the strategy to be working).
```bash
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-06-30
```

---

### TASK 1.4: Run First Baseline Backtest

Run on BTC-USD hourly (Jan–Jun 2024) to establish baseline. Record all metrics.

```bash
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-06-30 --json
```

**Acceptance criteria:**
- Backtest runs to completion (no crashes)
- Strategy generates some trades (even if losing)
- All metrics computed
- JSON output saved to `reports/smc/baseline_BTC-USD_1h.json`

**Deliverable:** Baseline performance metrics table.

---

### TASK 1.5: Delete Redundant Files

Remove old dual-codebase files after confirming new strategy works:
```bash
# DO NOT delete until TASK 1.4 passes
git rm src/strategies/smc_reversal.py
git rm src/strategies/smc_reversal_bt.py
```

Update imports in `src/strategies/__init__.py`:
```python
# Remove:
# from .smc_reversal import SMCReversalStrategy, SMCConfig, TradeSignal, ...
# from .smc_reversal_bt import SMCReversalBacktest

# Replace with:
from .smc_strategy import SMCStrategy
```

Update `src/strategies/strategy_registry.py` — replace `SMCReversalPlugin` with new strategy wrapper.

Update `src/main.py` — replace references to old SMC classes.

---

## PHASE 2: EXIT LOGIC — ATR Trail + Multi-TP (P0)

**Goal:** Apply the two highest-impact enhancements from RulesFirst (+160% Sharpe from trail, +124% Sharpe from multi-TP).
**Time:** ~30 minutes
**Depends on:** Phase 1 complete

---

### TASK 2.1: Implement ATR Trailing Stop

**In `smc_strategy.py` `next()`:**

```python
# ATR trailing stop for LONG positions
if self.position and self.position.is_long:
    self._trail_high = max(self._trail_high, current_close)
    trail_sl = self._trail_high - self.trail_stop_atr * atr

    if current_close <= trail_sl:
        self.position.close()
        self._trail_high = 0.0
    elif score < self.exit_threshold:
        self.position.close()
        self._trail_high = 0.0

# ATR trailing stop for SHORT positions (inverted)
elif self.position and self.position.is_short:
    self._trail_low = min(self._trail_low, current_close)
    trail_sl = self._trail_low + self.trail_stop_atr * atr

    if current_close >= trail_sl:
        self.position.close()
        self._trail_low = float("inf")
    elif score > -self.exit_threshold:
        self.position.close()
        self._trail_low = float("inf")
```

**Replace existing logic:** Remove fixed R-multiple exits (BE@1R, scale@2R, target@2.5R). Keep the daily loss limit (3% equity).

**Verify:** Backtest with trail enabled, compare to Phase 1 baseline. Expected: more trades held longer, larger average win, fewer premature exits.

---

### TASK 2.2: Implement Multi-TP Exit

**In `smc_strategy.py` `next()`:**

```python
# Multi-TP for LONG
if self.use_multi_tp and not self._tp1_hit:
    tp1_price = self._entry_price + self.tp1_atr * atr
    if current_close >= tp1_price:
        self.position.close(portion=self.tp1_size)  # Close 50%
        self._tp1_hit = True
        if self.move_sl_to_be:
            self._trail_high = self._entry_price
            trail_sl = self._entry_price           # SL to breakeven

# Multi-TP for SHORT (inverted)
if self.use_multi_tp and not self._tp1_hit:
    tp1_price = self._entry_price - self.tp1_atr * atr
    if current_close <= tp1_price:
        self.position.close(portion=self.tp1_size)
        self._tp1_hit = True
        if self.move_sl_to_be:
            self._trail_low = self._entry_price
            trail_sl = self._entry_price
```

**Parameters (from RulesFirst defaults):**
```python
use_multi_tp: bool = True      # Enable by default after testing
tp1_atr: float = 1.5           # TP1 at 1.5x ATR
tp2_atr: float = 3.0           # TP2 at 3x ATR (remainder trails)
tp1_size: float = 0.5          # Close 50% at TP1
move_sl_to_be: bool = True     # Move SL to entry after TP1
```

**Verify:** Backtest with multi-TP ON vs OFF. Expected: higher Sharpe, higher win rate, lower MaxDD.

---

### TASK 2.3: Compare Phase 1 vs Phase 2 Results

Run three configs on BTC-USD 1h (Jan–Jun 2024):
1. **Phase 1 baseline:** Trail=OFF, Multi-TP=OFF
2. **+ATR trail only:** Trail=ON, Multi-TP=OFF
3. **+ATR trail +Multi-TP:** Trail=ON, Multi-TP=ON

Report comparison table. Expected: Phase 2 configs significantly outperform Phase 1.

---

## PHASE 3: SIGNAL QUALITY — Replace Subjective SMC with Testable Rules (P1)

**Goal:** Replace ICT-specific terminology with Duddella/Kirkpatrick objectively-defined rules. Add confluence bonus and volume confirmation.
**Time:** ~45 minutes
**Depends on:** Phase 2 complete

---

### TASK 3.1: Replace MSS with Duddella MSL/MSH (3-bar close pattern)

**Current SMC MSS:** Configurable pivot detection with undefined lookback. Subjective.

**Replace with Duddella's objective definition:**
```python
def _detect_msl_msh(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Duddella 2007: 3-bar close pattern for Market Structure Low/High.

    MSL: New low → Lower low → Higher low (of CLOSE)
         Signal: close[i] > highest(close[i-2], close[i-1], close[i])
         Direction: +1 (bullish reversal)

    MSH: New high → Higher high → Lower high (of CLOSE)
         Signal: close[i] < lowest(close[i-2], close[i-1], close[i])
         Direction: -1 (bearish reversal)
    """
    close = df["Close"].to_numpy()
    n = len(close)
    msl = np.zeros(n, dtype=np.int8)
    msh = np.zeros(n, dtype=np.int8)

    for i in range(2, n):
        # MSL check
        if close[i-2] < close[i-1] and close[i-1] < close[i-2]:  # actually: new low then lower low
            if close[i] > max(close[i-2], close[i-1]):  # higher close - reversal
                msl[i] = 1
        # MSH check
        if close[i-2] > close[i-1] and close[i-1] > close[i-2]:
            if close[i] < min(close[i-2], close[i-1]):
                msh[i] = -1

    return msl, msh
```

**Wait — Duddella's exact rule is simpler:**
- **MSL (Market Structure Low):** Three candles where the CLOSES form: new low → lower low → higher low. Entry when close > highest close of the 3-bar group. Stop below the low of the formation.
- **MSH (Market Structure High):** Three candles where CLOSES form: new high → higher high → lower high. Entry when close < lowest close of the 3-bar group. Stop above the high.

```python
def _detect_msl_msh_vectorized(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Duddella 2007 MSL/MSH — vectorized."""
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # MSL: close[i-1] < close[i-2] (lower low) AND close[i] > close[i-1] (higher low)
    #      AND close[i-2] < close[i-1]? Wait, the sequence is:
    #      close[t-2]: starting point
    #      close[t-1]: lower low (close[t-1] < close[t-2])
    #      close[t]:   higher low BUT lower than close[t-2]? No...
    # Duddella: "new low, lower low, higher low of CLOSE"
    # Simplified: close[t-1] < close[t-2] AND close[t] > close[t-1]
    # Entry when close[t] > max(close[t-2], close[t-1]) — above the group high

    msl_condition = (close[2:] > close[1:-1]) & (close[1:-1] < close[:-2])
    msh_condition = (close[2:] < close[1:-1]) & (close[1:-1] > close[:-2])

    msl = np.zeros(n, dtype=np.int8)
    msh = np.zeros(n, dtype=np.int8)
    msl[2:] = msl_condition.astype(np.int8)
    msh[2:] = msh_condition.astype(np.int8)
    return msl, msh
```

**Replace `_check_mss()` call** with Duddella MSL/MSH arrays in `_compute_smc_score()`.

---

### TASK 3.2: Replace Sweep Detection with Kirkpatrick Stop-and-Reverse

**Current SMC sweep:** Complex bidirectional check with volume, configurable buffer.

**Replace with Kirkpatrick's simpler framework:**
```python
def _detect_stop_and_reverse(self, df: pd.DataFrame, session_highs, session_lows) -> np.ndarray:
    """Kirkpatrick (CMT 2011): Stop-and-Reverse false breakout detector.

    1. Identify key level (Asia session high/low)
    2. Wait for price to break beyond level
    3. If price reverses back through level → false breakout → trade reversal
    """
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    n = len(close)

    signals = np.zeros(n, dtype=np.int8)
    breakout_above = np.zeros(n, dtype=bool)
    breakout_below = np.zeros(n, dtype=bool)

    for i in range(2, n):
        # Simple session-range breakout detection
        # (simplified — full implementation tracks per-session)
        session_high = session_highs[i]
        session_low = session_lows[i]
        if session_high == 0 or session_low == 0:
            continue

        # Breakout above session high → bullish? Check if false.
        if close[i-1] > session_high and close[i] < session_high:
            signals[i] = -1  # False breakout above high → bearish reversal
        # Breakout below session low → bearish? Check if false.
        elif close[i-1] < session_low and close[i] > session_low:
            signals[i] = 1   # False breakout below low → bullish reversal

    return signals
```

**Merge with existing sweep logic** — keep volume confirmation (`require_volume_confirmation`), keep ATR buffer (`sweep_buffer_mult`), but simplify the directional logic to the Kirkpatrick pattern.

---

### TASK 3.3: Add Confluence Bonus

**In `_compute_smc_score()`:**
```python
def _compute_smc_score(self, idx: int) -> float:
    sweep_sig = self._sweep_signals[idx] if idx < len(self._sweep_signals) else 0
    msl_sig   = self._msl_signals[idx] if idx < len(self._msl_signals) else 0
    msh_sig   = self._msh_signals[idx] if idx < len(self._msh_signals) else 0
    fvg_sig   = self._fvg_proximity[idx] if idx < len(self._fvg_proximity) else 0
    bos_sig   = self._bos_signals[idx] if idx < len(self._bos_signals) else 0

    # Base score = weighted sum of components
    directional = np.sign(sweep_sig + msl_sig + msh_sig + bos_sig + fvg_sig)
    active_count = sum(s != 0 for s in [sweep_sig, msl_sig, msh_sig, bos_sig, fvg_sig])

    base_score = (
        sweep_sig * self.w_sweep +
        (msl_sig + msh_sig) * self.w_mss +
        bos_sig * self.w_bos +
        fvg_sig * self.w_fvg
    )

    # Confluence bonus: 2+ components agree
    score = base_score
    if active_count >= 2:
        score += self.confluence_bonus * directional

    # Volume confirmation
    if self.volume_confirm:
        score *= self._vol_mult[idx] if idx < len(self._vol_mult) else 1.0

    return np.tanh(score)
```

---

### TASK 3.4: Run Sweep on Entry Threshold

```bash
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-12-31 --sweep-entry
```

**Expected output:** Table of entry_threshold vs Sharpe, Trades, Win%. Identify optimal threshold.

---

## PHASE 4: QUALITY & VALIDATION (P1)

**Goal:** Apply quality registry, differentiated weights, OOS validation with proper CV, and standardized 10-metric evaluation.
**Time:** ~45 minutes
**Depends on:** Phase 3 complete

---

### TASK 4.1: Run First OOS Validation

**Split:** Train 2023-01-01 → 2024-06-30, Test 2024-07-01 → 2025-06-30 (or latest available).

```bash
# IS (training period)
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2023-01-01 --end 2024-06-30 --json -o reports/smc/IS_BTC-USD_1h.json

# OOS (test period)
uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --start 2024-07-01 --end 2025-06-30 --json -o reports/smc/OOS_BTC-USD_1h.json
```

**Acceptance criteria:**
- IS Sharpe > 0
- OOS Sharpe maintained (not collapsing)
- 30+ total trades across IS+OOS
- Document IS vs OOS comparison table

---

### TASK 4.2: Build Quality Sweep for SMC Components

**File:** `scripts/sweep_smc_components.py`

```python
"""Sweep SMC component configurations to find optimal weights.

Evaluates each component independently and in combination:
    sweep_reversal, msl_msh, bos_choch, fvg_proximity

For each component (and combinations):
    - Run solo backtest (component = entry signal)
    - Report: t-stat, IC, return/risk, quantile spread
    - Assign quality tier: PASS / FAIL_STRONG / FAIL_WEAK

Output: reports/smc/component_quality.json
"""
```

**Gate thresholds (from RulesFirst PatternQualityRegistry):**
```
PASS (3-4 steps passed): 1.0x weight, 1 min confluence
FAIL_STRONG (2 steps):   0.5x weight, 2 min confluence
FAIL_WEAK (0-1 steps):   0.3x weight, 3 min confluence
ERROR:                   excluded from signal computation
```

**Update `smc_strategy.py`** to load component quality from JSON and apply weight multipliers.

---

### TASK 4.3: Adopt 10-Metric Evaluation Suite

**In `scripts/backtest_smc.py`:**

Add standardized metric reporting after every backtest:
```python
def compute_full_metrics(stats, trades) -> dict:
    return {
        "return_pct": stats["Return [%]"],
        "sharpe": stats["Sharpe Ratio"],
        "cagr": stats.get("CAGR [%]", 0),
        "calmar": stats.get("Calmar Ratio", 0),
        "max_drawdown": stats["Max. Drawdown [%]"],
        "win_rate": stats["Win Rate [%]"],
        "profit_factor": stats["Profit Factor"],
        "avg_win_loss_ratio": compute_avg_wl(trades),
        "trades": stats["# Trades"],
        "exposure": stats["Exposure Time [%]"],
        "buy_hold_return": stats["Buy & Hold Return [%]"],
        "annualized_return": stats.get("Return (Ann.) [%]", 0),
        "annualized_volatility": stats.get("Volatility (Ann.) [%]", 0),
    }
```

Compare every result side-by-side.

---

### TASK 4.4: Test on Second Instrument

Run backtest on a second instrument for cross-validation:
```bash
uv run scripts/backtest_smc.py --symbol GC=F --interval 1h --start 2024-01-01 --end 2024-12-31
```

Expected: Strategy works on gold futures (another 23-hour market). If results positive (Sharpe > 0), confirms the SMC pipeline generalizes beyond BTC.

---

## PHASE 5: ADVANCED — Gates, Features, Refinement (P2)

**Goal:** Add multiplicative gate chain, tick direction features, intraday crash factors, and IR weighting.
**Time:** ~45 minutes
**Depends on:** Phase 4 complete

---

### TASK 5.1: Multiplicative Gate Chain

**In `smc_strategy.py` `init()`:**

Precompute per-bar multiplier arrays:
```python
def _precompute_gate_arrays(self):
    n = self._n_bars

    # VIX gate (if available — VIX is daily, can approximate intraday)
    self._vix_mults = np.ones(n)

    # Volatility gate (ATR-based)
    atr_pct = self._atr / self._df["Close"].to_numpy()
    self._vol_gate_mults = np.clip(1.5 - atr_pct / (atr_pct.mean() + 1e-10), 0.3, 1.5)

    # Session time gate: reduce signals outside desired session overlap
    self._session_mults = np.ones(n)
    for i in range(n):
        hour = self._df.index[i].hour
        # Only trade during active session overlap (London/NY)
        if hour < 8 or hour > 20:  # Outside 08:00-20:00 UTC
            self._session_mults[i] = 0.5  # Reduce, don't eliminate
```

**In `_compute_smc_score()`:**
```python
score *= self._vol_gate_mults[idx]
score *= self._session_mults[idx]
```

---

### TASK 5.2: Tick Direction Volume Features

**Port from ML4T Ch12** — compute buying/selling pressure features (approximate since we don't have tick data):

```python
def _precompute_volume_pressure(self):
    """Approximate tick direction from OHLCV bars."""
    close = self._df["Close"].to_numpy()
    open_ = self._df["Open"].to_numpy()
    high = self._df["High"].to_numpy()
    low = self._df["Low"].to_numpy()
    volume = self._df["Volume"].to_numpy()
    n = len(close)

    # Direction indicator: was the bar up or down?
    up = np.where(close > open_, 1, 0)
    down = np.where(close < open_, 1, 0)

    # Balance of Power: (Close - Open) / (High - Low)
    bop = np.divide(close - open_, high - low + 1e-10, where=(high - low) > 0)

    # Up/down volume pressure (smoothed)
    up_vol = np.zeros(n)
    down_vol = np.zeros(n)
    for i in range(n):
        lookback = min(i, 10)
        up_vol[i] = np.sum(volume[max(0,i-lookback):i+1] * up[max(0,i-lookback):i+1]) / max(np.sum(volume[max(0,i-lookback):i+1]), 1)
        down_vol[i] = np.sum(volume[max(0,i-lookback):i+1] * down[max(0,i-lookback):i+1]) / max(np.sum(volume[max(0,i-lookback):i+1]), 1)

    self._bop = bop
    self._up_pressure = up_vol
    self._down_pressure = down_vol
```

**Use in scoring:** High up_pressure + sweep_signal bullish → increase confidence. High down_pressure + sweep_signal bearish → increase confidence.

---

### TASK 5.3: Intraday Crash/Risk Factors

**Adapt from crash paper (monthly → intraday rolling):**
```python
def _precompute_crash_factors(self):
    """Rolling 20-bar window crash indicators."""
    close = self._df["Close"].to_numpy()
    returns = np.diff(np.log(close))
    n = len(close)
    window = 20

    self._dturn = np.ones(n)      # Detrended turnover proxy
    self._tskew = np.zeros(n)     # Return skewness
    self._tvol = np.zeros(n)      # Volatility

    for i in range(window, n):
        r = returns[i-window:i]
        self._tvol[i] = np.std(r)
        self._tskew[i] = -1 * pd.Series(r).skew()  # NCSKEW = negative skew
        # DTURN: current volume vs rolling avg (as proxy for turnover)
        vol_window = self._df["Volume"].iloc[i-window:i]
        self._dturn[i] = self._df["Volume"].iloc[i] / max(vol_window.mean(), 1)
```

**Gate logic:** High DTURN + high TVOL → reduce position size to 50%. High NCSKEW → prefer shorts.

---

### TASK 5.4: IR Weighting for SMC Components

**After accumulating OOS history** (requires 252+ bars of signal data):
```python
def _compute_rolling_ir(self):
    """Rolling Information Ratio per component (252-bar window)."""
    for component in ["sweep", "msl_msh", "bos", "fvg", "ob"]:
        ir = mean_return[component] / max(std_return[component], 1e-10)
        if ir > 0:
            self._component_weights[component] = self._base_weights[component] * (0.3 + 0.7 * (ir / max_ir))
        else:
            self._component_weights[component] = self._base_weights[component] * 0.3
```

---

## COMPLETION CHECKLIST

### Phase 1 (P0 — MUST PASS)
- [ ] `uv add smartmoneyconcepts` succeeds
- [ ] `src/strategies/smc_strategy.py` created (~500 loc, single file)
- [ ] `scripts/backtest_smc.py` created (~200 loc)
- [ ] Baseline backtest runs on BTC-USD 1h (any result, even negative)
- [ ] Old files deleted: `smc_reversal.py`, `smc_reversal_bt.py`
- [ ] Imports updated in `__init__.py`, `strategy_registry.py`, `main.py`

### Phase 2 (P0 — MUST PASS)
- [ ] ATR trailing stop implemented
- [ ] Multi-TP exit implemented
- [ ] Phase 1 vs Phase 2 comparison table generated
- [ ] Trail + Multi-TP produces higher Sharpe than baseline

### Phase 3 (P1 — SHOULD PASS)
- [ ] Duddella MSL/MSH replaces SMC MSS
- [ ] Kirkpatrick stop-and-reverse replaces sweep
- [ ] Confluence bonus wired in
- [ ] Volume confirmation wired in
- [ ] Entry threshold sweep completed, optimal value found

### Phase 4 (P1 — SHOULD PASS)
- [ ] IS vs OOS split backtest on BTC-USD
- [ ] Quality sweep script created and run
- [ ] 10-metric eval in CLI output
- [ ] Second instrument tested (GC=F)
- [ ] BESTS.md updated with SMC section

### Phase 5 (P2 — NICE TO HAVE)
- [ ] Multiplicative gate chain wired
- [ ] Tick direction volume features added
- [ ] Intraday crash factors added
- [ ] Component weight optimization via IR

### Documentation
- [ ] `docs/COMMAND_CHEATSHEET.md` updated (+SMC section)
- [ ] `BESTS.md` updated with SMC leaderboard
- [ ] `MEMORY.md` updated (Session handover)
- [ ] `progress_docs/current.md` updated (session log)
- [ ] `.useful_commands/` updated with SMC workflow commands

---

## KNOWN PITFALLS

1. **`smartmoneyconcepts` may not work on hourly data** — the library may expect daily OHLCV arrays. If it fails, fall back to existing custom implementations from `smc_reversal.py`.

2. **backtesting.py Position API** — `self.position.close(portion=0.5)` may not be supported in backtesting.py 0.6.5. Test first. If not, implement partial close manually via position size reduction.

3. **`self.sell()` for shorts** — backtesting.py 0.6.5 supports `self.sell()` for short positions. Verify the position tracking works correctly with both long and short entries.

4. **Yahoo Finance 1h data limit** — Only 730 days available. This limits OOS windows. For BTC, consider switching to Binance via CCXT (already available via `src/data_ingestion/crypto_provider.py`).

5. **Session timezone** — Current SMC uses UTC. Ensure session_start/end parameters use correct timezone. Yahoo Finance data may be in local exchange timezone, not UTC.

6. **Look-ahead bias in precomputed arrays** — When computing sweep/MSS/FVG, ensure each bar only uses data available at that bar (no forward-looking). This is the #1 backtest killer.

---

## IMPLEMENTATION ORDER (DO NOT DEVIATE)

```
Phase 1 (Foundation)
  T1.1 → T1.2 → T1.3 → T1.4 → T1.5
         │
Phase 2 (Exit Logic)          Phase 3 (Signal Quality)
  T2.1 → T2.2 → T2.3             T3.1 ∥ T3.2 → T3.3 → T3.4
         │                              │
         └──────────┬───────────────────┘
                    ▼
Phase 4 (Quality & Validation)
  T4.1 → T4.2 → T4.3 → T4.4
                    │
                    ▼
Phase 5 (Advanced — optional, do only if Phases 1-4 complete)
  T5.1 → T5.2 → T5.3 → T5.4
```

---

## HANDOVER FORMAT

After completing each phase, update:

1. **`MEMORY.md`** — Update "Current Objective" with phase status, key metrics
2. **`progress_docs/current.md`** — Append session log entry with timestamp
3. **`BESTS.md`** — Add SMC leaderboard section with best results per config
4. **`docs/COMMAND_CHEATSHEET.md`** — Add SMC CLI commands with examples
