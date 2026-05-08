# Strategy Implementation Fixes Plan

**Version:** 1.0
**Date:** 2026-03-25
**Status:** Planning
**Based on:** Code analysis comparing trading_strategy.md and smc_strategy_implementation_plan.md with actual implementation

---

## 📋 Executive Summary

This plan addresses 7 critical discrepancies found between the documented trading strategies and the actual code implementation. The fixes ensure the code correctly implements the documented strategy logic.

---

## 🎯 Issues to Fix

| # | Issue | Severity | File(s) |
|---|-------|----------|---------|
| 1 | SMC Trade Management Missing | HIGH | smc_reversal.py |
| 2 | Position Sizing Broken for Shorts | HIGH | position_sizing.py |
| 3 | Confluence Regime Weights Incorrect | MEDIUM | confluence.py |
| 4 | Signal Decay Not Implemented | MEDIUM | confluence.py, signal_generator.py |
| 5 | Multi-Pattern Entry Logic Incomplete | HIGH | multi_pattern_strategy.py |
| 6 | Portfolio Heat Calculation Missing | MEDIUM | daily_limits.py |
| 7 | Pattern Correlation Limits Missing | LOW | confluence.py |

---

## 🔧 Implementation Tasks

### Task 1: Implement SMC Trade Management

**File:** `src/strategies/smc_reversal.py`

**Current State:** Strategy generates signals with targets but never manages the trade after entry.

**Required Changes:**

1. Add trade management method to `SMCReversalStrategy` class:

```python
def _manage_active_trade(self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int) -> None:
    """
    Manage active trade with breakeven, scaling, and target exits.

    Rules (from documentation):
    - Move stop to breakeven at 1R profit
    - Scale out 50% at 2R profit
    - Close remaining at 2.5R target
    """
    if self.state is None or self.state.active_signal is None:
        return

    signal = self.state.active_signal
    current_price = df.iloc[global_idx]['Close']

    # Calculate current profit in R-multiples
    risk_distance = abs(signal.entry_price - signal.stop_loss)

    if signal.direction == 'long':
        current_profit = current_price - signal.entry_price
    else:
        current_profit = signal.entry_price - current_price

    r_multiple = current_profit / risk_distance if risk_distance > 0 else 0

    # Breakeven at 1R
    if r_multiple >= self.config.target_1r and not signal.metadata.get('be_moved', False):
        signal.stop_loss = signal.entry_price
        signal.metadata['be_moved'] = True
        logger.info(f"Stop moved to breakeven at {idx}")

    # Scale out at 2R
    if r_multiple >= self.config.target_2r and not signal.metadata.get('scaled', False):
        signal.metadata['scaled'] = True
        signal.metadata['scale_price'] = current_price
        logger.info(f"Scaled out 50% at {idx}, price={current_price:.4f}")

    # Final target at 2.5R
    if r_multiple >= self.config.target_final:
        self.state.active_signal = None
        self.state.state = StrategyState.TRACKING_ASIA
        logger.info(f"Position closed at target {idx}, price={current_price:.4f}")

    # Stop loss hit
    if signal.direction == 'long' and current_price <= signal.stop_loss:
        self.state.active_signal = None
        self.state.state = StrategyState.TRACKING_ASIA
        logger.info(f"Stop loss hit at {idx}, price={current_price:.4f}")
    elif signal.direction == 'short' and current_price >= signal.stop_loss:
        self.state.active_signal = None
        self.state.state = StrategyState.TRACKING_ASIA
        logger.info(f"Stop loss hit at {idx}, price={current_price:.4f}")
```

2. Call this method in `_process_bar` when state is `IN_TRADE`:

```python
elif self.state.state == StrategyState.IN_TRADE:
    # Manage active trade
    self._manage_active_trade(df, idx, global_idx)
```

---

### Task 2: Fix Position Sizing for Short Positions

**File:** `src/risk/position_sizing.py`

**Current State:** Validation assumes long-only, crashes on short positions.

**Required Changes:**

1. Update `calculate` method to detect position direction:

```python
def calculate(
    self,
    equity: float,
    entry_price: float,
    stop_price: Optional[float] = None,
    atr: Optional[float] = None,
    direction: str = 'long',  # ADD THIS PARAMETER
    **kwargs
) -> PositionSizeResult:
```

2. Update stop price validation:

```python
# Ensure stop price is valid for direction
if direction == 'long':
    if stop_price >= entry_price:
        raise ValueError("Stop price must be below entry price for long positions")
    risk_per_share = entry_price - stop_price
else:  # short
    if stop_price <= entry_price:
        raise ValueError("Stop price must be above entry price for short positions")
    risk_per_share = stop_price - entry_price
```

3. Update default stop calculation:

```python
if stop_price is None:
    if atr is not None:
        if direction == 'long':
            stop_price = entry_price - (atr * self.atr_multiplier)
        else:
            stop_price = entry_price + (atr * self.atr_multiplier)
    else:
        # Default to 5% stop
        if direction == 'long':
            stop_price = entry_price * 0.95
        else:
            stop_price = entry_price * 1.05
```

---

### Task 3: Implement Correct Regime Weights

**File:** `src/strategies/confluence.py`

**Current State:** Uses simplified binary logic instead of documented 4-regime system.

**Required Changes:**

Replace `_get_regime_weights` method with documented weights:

```python
def _get_regime_weights(self, regime: RegimeState) -> Dict[str, Dict[str, float]]:
    """
    Get regime-based weight multipliers matching documentation.

    Documented weights from trading_strategy.md:
    | Pattern  | Trending | Ranging | Volatile | Quiet |
    |----------|----------|---------|----------|-------|
    | Basic    | 0.8      | 1.0     | 0.7      | 1.0   |
    | Harmonic | 1.0      | 0.8     | 0.6      | 1.2   |
    | Complex  | 1.2      | 0.6     | 0.8      | 0.7   |
    | Classic  | 1.0      | 1.0     | 1.0      | 1.0   |
    """
    weights = {
        'category': {
            'basic': 1.0,
            'harmonic': 1.0,
            'complex': 1.0,
            'classic': 1.0
        },
        'type': {
            'Reversal': 1.0,
            'Continuation': 1.0,
            'Breakout': 1.0,
            'Counter-Trend': 1.0,
            'Volatility': 1.0
        }
    }

    # Determine regime category
    is_trending = regime.trend_direction in [TrendDirection.UPTREND, TrendDirection.DOWNTREND]
    is_high_vol = regime.volatility_regime == VolatilityRegime.HIGH
    is_low_vol = regime.volatility_regime == VolatilityRegime.LOW

    if is_trending and not is_high_vol:
        # Trending regime
        weights['category']['basic'] = 0.8
        weights['category']['harmonic'] = 1.0
        weights['category']['complex'] = 1.2
        weights['type']['Continuation'] = 1.3
        weights['type']['Reversal'] = 0.7
    elif is_high_vol:
        # Volatile regime
        weights['category']['basic'] = 0.7
        weights['category']['harmonic'] = 0.6
        weights['category']['complex'] = 0.8
        weights['type']['Breakout'] = 1.2
        weights['type']['Reversal'] = 0.8
    elif is_low_vol:
        # Quiet regime
        weights['category']['basic'] = 1.0
        weights['category']['harmonic'] = 1.2
        weights['category']['complex'] = 0.7
        weights['type']['Breakout'] = 0.7
    else:
        # Ranging regime
        weights['category']['basic'] = 1.0
        weights['category']['harmonic'] = 0.8
        weights['category']['complex'] = 0.6
        weights['type']['Reversal'] = 1.2
        weights['type']['Continuation'] = 0.6

    return weights
```

---

### Task 4: Implement Signal Decay

**File:** `src/strategies/confluence.py`

**Current State:** Signals never expire or lose confidence.

**Required Changes:**

1. Add signal validity periods to `ConfluenceScorer`:

```python
# Signal validity periods (bars)
SIGNAL_VALIDITY = {
    'basic': 5,
    'harmonic': 10,
    'complex': 20,
    'classic': 15
}

# Confidence decay rate per bar
DECAY_RATE = 0.05

# Minimum confidence threshold
MIN_CONFIDENCE = 0.40
```

2. Add decay calculation method:

```python
def apply_signal_decay(
    self,
    score: ConfluenceScore,
    bars_since_detection: int
) -> ConfluenceScore:
    """
    Apply time-based confidence decay to signal.

    Args:
        score: Original confluence score
        bars_since_detection: Bars elapsed since pattern detected

    Returns:
        Updated ConfluenceScore with decayed confidence
    """
    if bars_since_detection <= 0:
        return score

    # Get validity period based on pattern category
    category = self._get_pattern_category(score.patterns[0]) if score.patterns else 'classic'
    validity_period = self.SIGNAL_VALIDITY.get(category, 10)

    # No decay within validity period
    if bars_since_detection <= validity_period:
        return score

    # Apply decay after validity period
    excess_bars = bars_since_detection - validity_period
    decay = excess_bars * self.DECAY_RATE

    new_score = max(self.MIN_CONFIDENCE, score.score - decay)

    return ConfluenceScore(
        score=new_score,
        direction=score.direction,
        confidence_level=score.confidence_level,
        pattern_count=score.pattern_count,
        patterns=score.patterns,
        regime_alignment=score.regime_alignment,
        risk_reward_ratio=score.risk_reward_ratio,
        quality_score=score.quality_score,
        metadata={**score.metadata, 'decay_applied': decay, 'bars_since_detection': bars_since_detection}
    )
```

---

### Task 5: Complete Multi-Pattern Entry Logic

**File:** `src/strategies/backtest_py/multi_pattern_strategy.py`

**Current State:** `next()` method detects patterns but doesn't execute trades.

**Required Changes:**

Complete the `next()` method:

```python
def next(self):
    """Execute trading logic for current bar."""
    # Check position limit
    if len(self.trades) >= self.max_open_positions:
        return

    # Get current DataFrame
    df = self._get_dataframe()
    current_idx = len(df) - 1

    # Skip if not enough bars
    min_bars = max(p.min_bars_required for p in self.patterns)
    if current_idx < min_bars:
        return

    # Detect patterns at current bar
    results = []
    for pattern in self.patterns:
        try:
            result = pattern.detect(df, current_idx)
            if result.detected and result.signal is not None:
                if result.signal.confidence >= self.min_confidence:
                    results.append(result)
        except Exception as e:
            continue

    if not results:
        return

    # Get market regime if enabled
    regime = None
    if self.use_regime_filter:
        regime = self.regime_detector.detect(df, current_idx)

    # Calculate confluence
    confluence = self.confluence_scorer.calculate_confluence(results, regime)

    # Check minimum confluence count
    if confluence.pattern_count < self.min_confluence_count:
        return

    # Check minimum confidence
    if confluence.score < self.min_confidence:
        return

    # Determine entry direction
    if confluence.direction == SignalDirection.LONG:
        direction = 'long'
    elif confluence.direction == SignalDirection.SHORT:
        direction = 'short'
    else:
        return

    # Get current price
    current_price = df.iloc[current_idx]['Close']

    # Calculate stop loss using ATR
    atr_value = self._get_atr(df, current_idx)
    stop_distance = atr_value * self.stop_loss_atr_mult

    if direction == 'long':
        stop_loss = current_price - stop_distance
        # Ensure stop is valid
        if stop_loss >= current_price:
            return
    else:
        stop_loss = current_price + stop_distance
        # Ensure stop is valid
        if stop_loss <= current_price:
            return

    # Calculate position size
    risk_amount = self.equity * self.risk_per_trade
    risk_per_share = abs(current_price - stop_loss)
    position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0

    # Calculate take profit targets
    if direction == 'long':
        tp1 = current_price + (risk_per_share * self.take_profit_1_ratio)
        tp2 = current_price + (risk_per_share * self.take_profit_2_ratio) if self.use_take_profit_2 else None
        tp3 = current_price + (risk_per_share * self.take_profit_3_ratio) if self.use_take_profit_3 else None
    else:
        tp1 = current_price - (risk_per_share * self.take_profit_1_ratio)
        tp2 = current_price - (risk_per_share * self.take_profit_2_ratio) if self.use_take_profit_2 else None
        tp3 = current_price - (risk_per_share * self.take_profit_3_ratio) if self.use_take_profit_3 else None

    # Execute trade
    if direction == 'long':
        self.buy(size=position_size, sl=stop_loss, tp=tp1)
    else:
        self.sell(size=position_size, sl=stop_loss, tp=tp1)

    # Log signal
    self.signal_count += 1
    self.signals.append({
        'timestamp': df.index[current_idx],
        'direction': direction,
        'entry_price': current_price,
        'stop_loss': stop_loss,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'confidence': confluence.score,
        'patterns': confluence.patterns,
        'pattern_count': confluence.pattern_count
    })
```

Add helper method for ATR:

```python
def _get_atr(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
    """Calculate ATR at given index."""
    if idx < period:
        return df.iloc[idx]['High'] - df.iloc[idx]['Low']

    high = df['High'].iloc[idx-period:idx+1]
    low = df['Low'].iloc[idx-period:idx+1]
    close = df['Close'].iloc[idx-period:idx+1]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]

    return atr if not pd.isna(atr) else 0.01
```

---

### Task 6: Implement Portfolio Heat Tracking

**File:** `src/risk/daily_limits.py`

**Current State:** Individual position limits tracked but not aggregate portfolio risk.

**Required Changes:**

1. Add portfolio heat tracking to `DailyLossState`:

```python
@dataclass
class DailyLossState:
    # ... existing fields ...
    open_position_risks: Dict[str, float] = field(default_factory=dict)

    @property
    def portfolio_heat(self) -> float:
        """Calculate total portfolio heat as fraction of equity."""
        if self.starting_equity == 0:
            return 0.0
        total_risk = sum(self.open_position_risks.values())
        return total_risk / self.starting_equity
```

2. Add portfolio heat check to `DailyLossLimiter`:

```python
def check_portfolio_heat(self, state: DailyLossState) -> tuple:
    """
    Check if portfolio heat exceeds limits.

    Documented limits:
    - Maximum Portfolio Heat: 6%
    - Warning Level: 4%
    """
    heat = state.portfolio_heat

    if heat >= 0.06:
        return (
            True,
            LimitType.PORTFOLIO_HEAT,
            f"Portfolio heat limit reached: {heat:.2%} >= 6.00%"
        )

    if heat >= 0.04:
        return (
            False,
            LimitType.PORTFOLIO_HEAT,
            f"Portfolio heat warning: {heat:.2%} >= 4.00%"
        )

    return (False, None, None)
```

3. Add method to update position risk:

```python
def update_position_risk(
    self,
    state: DailyLossState,
    position_id: str,
    risk_amount: float
) -> None:
    """Update risk for a specific position."""
    state.open_position_risks[position_id] = risk_amount

def remove_position_risk(
    self,
    state: DailyLossState,
    position_id: str
) -> None:
    """Remove position risk when position is closed."""
    state.open_position_risks.pop(position_id, None)
```

---

### Task 7: Implement Pattern Correlation Limits

**File:** `src/strategies/confluence.py`

**Current State:** Only checks conflicting patterns, not correlated patterns.

**Required Changes:**

1. Add correlation groups to `ConfluenceScorer`:

```python
# Pattern correlation groups (from documentation)
CORRELATION_GROUPS = {
    'double_patterns': ['Double Top', 'Double Bottom', 'Triple Top', 'Triple Bottom'],
    'harmonic': ['Gartley', 'ABC'],
    'breakout': ['NR7 Inside Day', 'Donchian Channel', 'Bollinger Bands'],
    'reversal_tops': ['Head and Shoulders', 'Double Top', "Trader Vic's 2B"],
    'reversal_bottoms': ['Double Bottom', 'Market Structure Low', 'Matching Lows']
}

# Maximum active patterns per correlation group
MAX_CORRELATED = {
    'double_patterns': 1,
    'harmonic': 1,
    'breakout': 2,
    'reversal_tops': 1,
    'reversal_bottoms': 1
}
```

2. Add correlation filtering method:

```python
def _filter_correlated_patterns(
    self,
    results: List[PatternResult]
) -> List[PatternResult]:
    """
    Filter patterns to enforce correlation limits.

    Only applies to patterns in the same direction.
    """
    if len(results) <= 1:
        return results

    # Group by direction
    long_results = [r for r in results if r.signal and r.signal.direction == SignalDirection.LONG]
    short_results = [r for r in results if r.signal and r.signal.direction == SignalDirection.SHORT]

    def filter_by_correlation(group: List[PatternResult]) -> List[PatternResult]:
        if len(group) <= 1:
            return group

        # Track which correlation groups have been used
        used_groups = {}
        filtered = []

        for result in group:
            pattern_name = result.pattern_name

            # Find which correlation group this pattern belongs to
            pattern_group = None
            for group_name, patterns in self.CORRELATION_GROUPS.items():
                if any(p.lower() in pattern_name.lower() for p in patterns):
                    pattern_group = group_name
                    break

            if pattern_group is None:
                # Not in any correlation group, always include
                filtered.append(result)
                continue

            # Check if we've hit the limit for this group
            current_count = used_groups.get(pattern_group, 0)
            max_allowed = self.MAX_CORRELATED.get(pattern_group, 999)

            if current_count < max_allowed:
                filtered.append(result)
                used_groups[pattern_group] = current_count + 1

        return filtered

    return filter_by_correlation(long_results) + filter_by_correlation(short_results)
```

3. Apply correlation filter in `calculate_confluence`:

```python
def calculate_confluence(
    self,
    results: List[PatternResult],
    regime: Optional[RegimeState] = None
) -> ConfluenceScore:
    # ... existing code ...

    # Apply correlation filtering
    results = self._filter_correlated_patterns(results)

    # ... rest of existing code ...
```

---

## 📅 Implementation Order

| Order | Task | Dependencies | Estimated Complexity |
|-------|------|--------------|---------------------|
| 1 | Fix Position Sizing for Shorts | None | Low |
| 2 | Implement Correct Regime Weights | None | Low |
| 3 | Implement Signal Decay | None | Medium |
| 4 | Complete Multi-Pattern Entry Logic | Task 2 | Medium |
| 5 | Implement SMC Trade Management | None | Medium |
| 6 | Implement Portfolio Heat Tracking | None | Medium |
| 7 | Implement Pattern Correlation Limits | None | Low |

---

## ✅ Testing Requirements

Each fix should include:

1. **Unit tests** for the specific function/method
2. **Integration tests** with existing code
3. **Backtest validation** to ensure strategy behavior matches documentation

### Test Files to Create/Update:

- `tests/test_position_sizing_shorts.py` - Test short position calculations
- `tests/test_signal_decay.py` - Test signal decay logic
- `tests/test_portfolio_heat.py` - Test portfolio heat tracking
- `tests/test_pattern_correlation.py` - Test correlation limits
- `tests/test_smc_trade_management.py` - Test SMC trade exits

---

## 📝 Notes

- All changes should maintain backward compatibility with existing tests
- Log messages should use loguru for consistency
- Follow existing code style and patterns in each file
- Update docstrings to reflect any parameter changes

---

*This plan should be reviewed and approved before implementation begins.*
