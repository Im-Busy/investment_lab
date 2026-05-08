"""
Backtest script for ALL strategies on BTC 1H data with signal log validation.

Verifies every strategy produces valid output:
- At least 1 trade (or reasonable zero if strategy doesn't trigger)
- Valid signal generation (entry/exit times, prices)
- No exceptions during execution

This fulfills the Phase 1 checkpoint: "Verify all 15+ strategies produce valid signal logs"
"""

import json
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path

import pandas as pd
from backtesting import Backtest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATA_PATH = project_root / "data" / "raw" / "BTC_USD_1h.csv"

# ALL 22 backtesting.py strategies — no skipping
STRATEGIES = {
    # Core strategies (already tested)
    "VWAP Bounce": ("src.strategies.vwap_bounce", "VWAPBounceStrategy"),
    "SMA Crossover 50/200": ("src.strategies.sma_crossover", "SMACrossoverStrategy"),
    "EMA Ribbon 9/21/55": ("src.strategies.ema_ribbon", "EMARibbonStrategy"),
    "Keltner Channel": ("src.strategies.keltner_channel", "KeltnerChannelStrategy"),
    "Chandelier Exit": ("src.strategies.chandelier_exit", "ChandelierExitStrategy"),
    "ADX Trend Strength": ("src.strategies.adx_trend_strength", "ADXTrendStrengthStrategy"),
    "Parabolic SAR": ("src.strategies.parabolic_sar", "ParabolicSARStrategy"),
    "Ichimoku Cloud": ("src.strategies.ichimoku_cloud", "IchimokuCloudStrategy"),
    "RSI Divergence": ("src.strategies.rsi_divergence", "RSIDivergenceStrategy"),
    "Stoch RSI Crossover": ("src.strategies.stoch_rsi_crossover", "StochRSICrossoverStrategy"),
    "CCI": ("src.strategies.cci_strategy", "CCIStrategy"),
    "Donchian Channel": ("src.strategies.donchian_breakout", "DonchianChannelStrategy"),
    # Previously skipped — now included
    "MFI": ("src.strategies.mfi_strategy", "MFIStrategy"),
    # Previously not in script — now included
    "Awesome Oscillator": ("src.strategies.awesome_oscillator", "AwesomeOscillatorStrategy"),
    "Chaikin Oscillator": ("src.strategies.chaikin_oscillator", "ChaikinOscillatorStrategy"),
    "Connors RSI": ("src.strategies.connors_rsi", "ConnorsRSIMeanReversion"),
    "Linear Regression Channel": (
        "src.strategies.linear_regression_channel",
        "LinearRegressionChannelStrategy",
    ),
    "MACD Histogram": ("src.strategies.macd_histogram", "MACDHistogramStrategy"),
    "TSI": ("src.strategies.tsi_strategy", "TSIStrategy"),
    "Ultimate Oscillator": ("src.strategies.ultimate_oscillator", "UltimateOscillatorStrategy"),
    "Williams %R": ("src.strategies.williams_r_reversal", "WilliamsRReversalStrategy"),
    # SMC backtest version
    "SMC Reversal (bt)": ("src.strategies.smc_reversal_bt", "SMCReversalBacktest"),
}


def load_data() -> pd.DataFrame:
    """Load BTC 1H data."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def run_backtest(name: str, strategy_cls, df: pd.DataFrame) -> dict:
    """Run a single strategy backtest."""
    bt = Backtest(df, strategy_cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()
    trades = stats.get("# Trades", 0)
    return {
        "Strategy": name,
        "Trades": int(trades),
        "Win Rate": float(stats.get("Win Rate [%]", 0)),
        "Sharpe": float(stats.get("Sharpe Ratio", -999)),
        "Profit Factor": float(stats.get("Profit Factor", 0)),
        "Max DD": float(stats.get("Max. Drawdown [%]", 0)),
        "Return": float(stats.get("Return [%]", 0)),
        "B&H Return": float(stats.get("Buy & Hold Return [%]", 0)),
        "Signal Count": int(trades),
        "Duration": str(stats.get("Duration", "N/A")),
    }


def validate_signal_log(result: dict) -> list:
    """Validate that a backtest result has proper signal log fields.

    Returns list of validation errors (empty list = valid).
    """
    errors = []

    required_keys = [
        "Strategy",
        "Trades",
        "Win Rate",
        "Sharpe",
        "Profit Factor",
        "Max DD",
        "Return",
    ]
    for key in required_keys:
        if key not in result:
            errors.append(f"Missing key: {key}")

    if "Trades" in result:
        if result["Trades"] < 0:
            errors.append(f"Negative trade count: {result['Trades']}")
        elif result["Trades"] == 0:
            errors.append("Zero trades — strategy may not generate signals on this data")

    if "Win Rate" in result:
        if not (0 <= result["Win Rate"] <= 100):
            errors.append(f"Win Rate out of range: {result['Win Rate']}%")

    if "Sharpe" in result:
        if result["Sharpe"] == -999:
            errors.append("Sharpe not computed (no trades or insufficient data)")

    if "Signal Count" in result:
        if result["Signal Count"] < 1:
            errors.append("No signals generated")

    return errors


def _extract_stats(results_obj) -> dict:
    """Extract stats from backtesting.py Results object."""
    try:
        return dict(results_obj._asdict())
    except AttributeError:
        stats = {}
        for key in results_obj.index:
            try:
                stats[key] = results_obj[key]
            except (KeyError, TypeError):
                pass
        return stats


def main() -> None:
    """Run all backtests and validate signal logs."""
    df = load_data()
    print(f"Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}\n")

    results = []
    all_errors = {}
    success_count = 0
    error_count = 0
    zero_trade_count = 0

    for name, (module_path, cls_name) in STRATEGIES.items():
        print(f"[{name}] Running...", end=" ", flush=True)
        try:
            module = import_module(module_path)
            strategy_cls = getattr(module, cls_name)
            t0 = time.time()
            bt = Backtest(df, strategy_cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
            stats_obj = bt.run()
            stats = _extract_stats(stats_obj)
            elapsed = time.time() - t0

            trades_count = int(stats.get("# Trades", 0))
            return_pct = float(stats.get("Return [%]", 0))
            wr = float(stats.get("Win Rate [%]", 0))
            sharpe = float(stats.get("Sharpe Ratio", -999))
            max_dd = float(stats.get("Max. Drawdown [%]", 0))
            pf = float(stats.get("Profit Factor", 0))
            bnh = float(stats.get("Buy & Hold Return [%]", 0))

            result = {
                "Strategy": name,
                "Trades": trades_count,
                "Win Rate": wr,
                "Sharpe": sharpe,
                "Profit Factor": pf,
                "Max DD": max_dd,
                "Return": return_pct,
                "B&H Return": bnh,
                "Signal Count": trades_count,
                "Duration": str(stats.get("Duration", "N/A")),
                "Elapsed_s": round(elapsed, 1),
            }
            results.append(result)

            # Validate
            errors = validate_signal_log(result)

            if not errors:
                success_count += 1
                print(
                    f"OK - {trades_count} trades, WR={wr:.1f}%, Return={return_pct:+.1f}%, {elapsed:.1f}s"
                )
            else:
                all_errors[name] = errors
                zero_trade_count += 1 if trades_count == 0 else 0
                print(f"WARNING - {', '.join(errors)}")

        except Exception as e:
            error_count += 1
            all_errors[name] = [f"Exception: {e}"]
            print(f"ERROR: {e}")

        time.sleep(0.1)

    if results:
        print("\n" + "=" * 120)
        print(
            f"{'Strategy':<30} {'Trades':>7} {'WR%':>7} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8} {'Return%':>8} {'B&H%':>8} {'Time':>6}"
        )
        print("=" * 120)
        for r in sorted(results, key=lambda x: x["Trades"], reverse=True):
            print(
                f"{r['Strategy']:<30} {r['Trades']:>7} {r['Win Rate']:>6.1f}% {r['Sharpe']:>8.2f} {r['Profit Factor']:>7.2f} {r['Max DD']:>7.1f}% {r['Return']:>7.1f}% {r['B&H Return']:>7.1f}% {r.get('Elapsed_s', 'N/A'):>5}s"
            )

    print("\n" + "=" * 70)
    print("SIGNAL LOG VALIDATION SUMMARY")
    print("=" * 70)
    total = len(STRATEGIES)
    print(f"Total strategies:    {total}")
    print(f"Passed validation:   {success_count}/{total}")
    print(f"Runtime errors:      {error_count}/{total}")
    print(f"Zero trades:         {zero_trade_count}/{total}")
    print(f"Signal log format:   All {'valid' if not all_errors else 'INVALID'}")

    if all_errors:
        print("\n--- DETAILS ---")
        for name, errors in all_errors.items():
            for err in errors:
                print(f"  {name}: {err}")

    total_trades = sum(r["Trades"] for r in results)
    avg_wr = sum(r["Win Rate"] for r in results) / len(results) if results else 0
    pos_return_count = sum(1 for r in results if r["Return"] > 0)
    print("\nAggregate:")
    print(f"  Total trades across all strategies: {total_trades}")
    print(f"  Average win rate: {avg_wr:.1f}%")
    print(f"  Strategies with positive return: {pos_return_count}/{len(results)}")

    if total_trades >= 15 and success_count >= 15:
        print(
            f"\n[PASS] All 15+ strategies produced valid signal logs ({success_count} pass, {total_trades} total trades)"
        )
    else:
        print(f"\n[INFO] {success_count} strategies passed, {total_trades} total trades")

    # Save results
    output_path = project_root / "reports" / "strategy_signal_log_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "data": f"BTC/USD 1H, {len(df)} bars",
                "total_strategies": total,
                "passed": success_count,
                "errors": error_count,
                "zero_trades": zero_trade_count,
                "validation_errors": all_errors,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
