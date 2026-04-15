# Project Rules: investment_trying

## Financial Data & Backtesting Guardrails
- NEVER use future data in backtests (look-ahead bias). Validate signal generation only uses data available at that point in time.
- Handle stock splits and dividends correctly — always use adjusted close prices for backtesting.
- Validate that backtest results are sane: check for impossible returns, negative prices, or trades on non-existent data.
- When using `backtesting.py` or custom engines: verify signal timing aligns with trade execution (signals at close execute next bar open).
- For crypto: use correct ticker format `CRYPTO::SYMBOL//USD`. 24/7 trading has no market hours constraints.
- For equities: check market hours before executing trades. Use `get_market_hours` tool.

## Trading System Context
- This is a rule-based multi-pattern trading system with 34+ chart pattern detectors.
- Pattern detectors live in `src/patterns/` across 7 categories.
- Strategy wrappers for `backtesting.py` live in `src/strategies/`.
- The custom event-driven backtest engine is in `src/backtest/engine.py`.
- Signal aggregation and confluence scoring is in `src/signals/`.
- Risk management (position sizing, loss limits, circuit breakers) is in `src/risk/`.

## Backtesting Standards
- Every strategy must produce valid signal logs before being considered complete.
- Report key metrics in comparison tables: return, Sharpe, max drawdown, win rate, profit factor.
- Compare against SPY baseline (buy and hold) and BTC baseline for context.
- A strategy is considered overfit if OOS performance is significantly worse than training performance.
- Minimum viable thresholds: 30+ trades, Sharpe > 0.5, profit factor > 1.2 (unless user specifies otherwise).
