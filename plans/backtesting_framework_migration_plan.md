# Backtesting Framework Migration Plan

## Current State Analysis

### Dual Framework Architecture
The project currently uses **two parallel backtesting frameworks**:

1. **Homemade BacktestEngine** (`src/backtest/engine.py`)
   - Custom-built engine with position management, signal generation, and performance metrics
   - Used by `src/main.py` for CLI backtesting
   - Has 465 lines of code with comprehensive features

2. **backtesting.py Integration** (`src/strategies/backtest_py/`)
   - Wrapper around the popular `backtesting.py` library
   - Used by pattern selection framework, notebooks, and visualization
   - Has 506 lines of code in runner.py alone

### Usage Distribution

| Component | Framework Used | Notes |
|-----------|----------------|-------|
| `src/main.py` CLI | Homemade BacktestEngine | Primary command-line interface |
| Pattern Selection Framework | backtesting.py | All analysis (solo, correlation, ablation) |
| Notebooks (05, 07) | backtesting.py | SPY long-term backtest and pattern selection |
| Visualization (`src/visualization/`) | Both | Unified interface supports both |
| Tests | Minimal coverage | No comprehensive backtest engine tests |

### Dependencies
- `backtesting>=0.6.5` already in `pyproject.toml`
- No additional dependencies needed for either framework

## Pros and Cons Analysis

### Homemade BacktestEngine

**Advantages:**
- Full control over execution logic and position management
- Tight integration with custom pattern detection system
- No external library dependencies (beyond pandas/numpy)
- Customizable risk management and exit logic
- Direct access to internal signal and position objects

**Disadvantages:**
- Limited optimization capabilities
- No built-in walk-forward analysis
- Less battle-tested than backtesting.py
- Manual handling of edge cases
- No built-in visualization integration
- Requires maintenance of custom code

### backtesting.py Framework

**Advantages:**
- Industry-standard, well-tested library
- Built-in optimization with `Backtest.optimize()`
- Excellent visualization capabilities
- Walk-forward analysis support
- Active community and documentation
- Automatic handling of many edge cases
- Already integrated with pattern selection framework

**Disadvantages:**
- Less control over execution logic
- Must adapt pattern detection to Strategy interface
- Some features (multiple take profits) require workarounds
- External dependency
- Learning curve for custom integration

## Migration Effort Assessment

### Compatibility Assessment
- **Pattern Detection**: Both frameworks use the same pattern detectors (`BasePattern`)
- **Signal Generation**: Same `SignalGenerator` can be used
- **Position Management**: backtesting.py handles positions internally
- **Metrics**: Both produce similar metrics (Sharpe, CAGR, etc.)
- **Visualization**: Already unified via `ReportGenerator`

### Migration Complexity
| Component | Migration Effort | Risk |
|-----------|-----------------|------|
| CLI (`src/main.py`) | Medium | Must rewrite backtest execution logic |
| Pattern Selection | None | Already uses backtesting.py |
| Notebooks | None | Already uses backtesting.py |
| Visualization | None | Already supports both |
| Tests | High | Need new test suite for backtesting.py integration |

### Estimated Timeline
- **Phase 1 (Analysis & Planning)**: 1-2 days
- **Phase 2 (Core Migration)**: 3-5 days  
- **Phase 3 (Testing & Validation)**: 2-3 days
- **Phase 4 (Deprecation & Cleanup)**: 1-2 days

## Recommendation

### Standardize on backtesting.py

**Primary Reasons:**
1. **Already Dominant Usage**: Pattern selection framework, notebooks, and visualization already use backtesting.py
2. **Industry Standard**: More reliable, tested, and feature-rich
3. **Optimization Capabilities**: Built-in parameter optimization is critical for strategy development
4. **Maintenance Burden**: Reduces custom code maintenance
5. **Future Extensibility**: Supports walk-forward analysis, Monte Carlo simulations, etc.

**Migration Strategy**: **Progressive Migration** with backward compatibility
1. Create unified adapter interface
2. Migrate CLI to use backtesting.py with fallback
3. Gradually deprecate homemade engine
4. Maintain dual support during transition

## Migration Plan

### Phase 1: Create Unified Interface
1. **Create `BacktestAdapter` class** (`src/backtest/adapter.py`)
   - Unified interface for both engines
   - Factory pattern to select engine
   - Common result format

2. **Update `ReportGenerator`** to use adapter
   - Single entry point for all backtests

3. **Create migration tests** to verify equivalence

### Phase 2: Migrate Core Components
1. **Update `src/main.py` CLI**
   - Add `--engine` flag (backtesting.py, homemade, auto)
   - Default to backtesting.py
   - Maintain backward compatibility

2. **Create `HomemadeToBacktestingAdapter`**
   - Translate homemade config to backtesting.py parameters
   - Ensure identical behavior

3. **Update documentation** with migration guide

### Phase 3: Enhance backtesting.py Integration
1. **Improve pattern integration**
   - Optimize signal caching for backtesting.py
   - Add multiple take-profit support

2. **Add optimization workflows**
   - Parameter optimization templates
   - Walk-forward analysis scripts

3. **Enhance visualization**
   - Better pattern markers on charts
   - Interactive optimization reports

### Phase 4: Deprecation and Cleanup
1. **Mark homemade engine as deprecated**
   - Add deprecation warnings
   - Update documentation

2. **Remove unused code** after 3-month transition
   - Delete `src/backtest/engine.py` (or keep as reference)
   - Remove homemade-only dependencies

3. **Final validation**
   - Compare results between engines
   - Performance benchmarking

## Risk Mitigation

### Technical Risks
1. **Result Discrepancies**: Run parallel backtests during transition
2. **Performance Regression**: Profile both engines, optimize bottlenecks
3. **Feature Gaps**: Identify missing features, implement workarounds

### Project Risks
1. **Timeline Overruns**: Start with minimal viable migration
2. **Team Knowledge Gap**: Provide training on backtesting.py
3. **Regression Bugs**: Comprehensive test suite before deprecation

## Success Metrics

1. **Functionality**: All existing CLI commands work with backtesting.py
2. **Performance**: No significant speed regression (<20% slower acceptable)
3. **Accuracy**: Results within 1% of homemade engine (allowing for rounding differences)
4. **Adoption**: All notebooks and scripts use backtesting.py by default
5. **Maintenance**: Reduced codebase by ~400 lines (homemade engine)

## Implementation Status

### Completed ✅
- [x] Create `BacktestAdapter` interface (`src/backtest/adapter.py`)
- [x] Add engine selection to CLI (`--engine` flag)
- [x] Migrate `src/main.py` to use adapter
- [x] Update `src/backtest/__init__.py` to export adapter components
- [x] Fix import issues (NR7ID, ThreeHillsMountain)
- [x] Test adapter imports successfully

### Remaining Tasks
- [ ] Run comparative backtests for validation
- [ ] Update all notebooks to use adapter
- [ ] Optimize backtesting.py integration
- [ ] Add optimization examples
- [ ] Performance benchmarking
- [ ] Deprecation warnings for homemade engine
- [ ] Final validation and sign-off

## Conclusion

Migrating from the homemade BacktestEngine to backtesting.py is **recommended** due to:

1. **Reduced maintenance burden**
2. **Access to advanced features** (optimization, walk-forward)
3. **Industry-standard reliability**
4. **Already predominant usage** in the codebase

The migration should be **progressive** with careful validation to ensure no regression in functionality or performance. The unified adapter approach minimizes risk while providing a clear path to standardization.