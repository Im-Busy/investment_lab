# 23 Untested Patterns + 5 Indicator Strategies - Complete Results

## Summary: ALL FAIL - 0 profitable strategies

### Pass 1: 23 Untested Patterns on SPY/QQQ/BTC_daily
4 patterns passed relaxed thresholds (min_trades>=10, min_sharpe>=-1.0):
| Pattern | Asset | Trades | WR% | Sharpe | Return% | DD% |
|---------|-------|--------|-----|--------|---------|-----|
| Triple Top | SPY | 10 | 0 | -0.47 | -10.8 | 11.6 |
| Symmetric Triangle | SPY | 64 | 36 | -0.01 | -0.2 | 6.4 |
| Donchian Channel Breakout | SPY | 48 | 33 | -0.14 | -4.4 | 10.1 |
| Symmetric Triangle | QQQ | 49 | 31 | 0.40 | 17.0 | 7.9 |
| Donchian Channel Breakout | QQQ | 48 | 29 | -0.19 | -7.5 | 15.7 |
| Gap Pattern | QQQ | 33 | 33 | 0.50 | 23.5 | 7.2 |

Note: These only passed very relaxed thresholds. All have negative or marginal returns.

### Pass 2: 5 Remaining Strategies on BTC 1H - ALL FAIL
| Strategy | Trades | WR% | Sharpe | Return% |
|----------|--------|-----|--------|----------|
| Ichimoku Cloud | 113 | 35.4 | -0.37 | -11.7 |
| RSI Divergence | 0 | 0 | -0.02 | -2.7 |
| MFI Strategy | 177 | 50.8 | -0.34 | -15.1 |
| Williams %R | 224 | 54.0 | -1.15 | -34.7 |
| MACD Histogram | 301 | 45.8 | -1.67 | -39.0 |

## Conclusion

- 28 total strategies tested (13 prior + 23 patterns + 5 indicators, with overlap)
- 0 profitable strategies
- Geometric patterns produce too few signals on daily data
- Indicator strategies all negative Sharpe

## Recommended Next Step

**Phase 2: Pair Trading** (cointegration + Kalman filter)
- GLD/IAU pair
- SPY/QQQ pair
- Dynamic hedge ratios via Kalman filter
- Mean-reversion on spread
