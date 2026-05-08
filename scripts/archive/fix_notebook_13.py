"""Fix issues in 13_ml_validation.ipynb for local execution."""

import nbformat


def fix_notebook(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    # Fix 1: Cell 17 - Wrong Strategy/runner usage
    code_cells[16].source = (
        "runner = BacktestPyRunner(data=df, cash=100000, verbose=False)\n"
        "backtest_result = runner.run(ConnorsRSIMeanReversion)\n"
        "trades = backtest_result.get('trades', [])\n"
        'print(f"Generated {len(trades)} trades")\n'
        "\n"
        "if len(trades) > 0:\n"
        "    trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades\n"
        "    if 'pnl' in trades_df.columns:\n"
        '        trades_df["is_profitable"] = (trades_df["pnl"] > 0).astype(int)\n'
        "        print(f\"Profitable trades: {trades_df['is_profitable'].sum()}/{len(trades_df)} \"\n"
        "              f\"({trades_df['is_profitable'].mean():.1%})\")\n"
        '        print("\\nTrade P&L summary:")\n'
        "        print(trades_df['pnl'].describe())\n"
        "    else:\n"
        '        print("No pnl column in trades - skipping signal scorer")\n'
        "else:\n"
        '    print("No trades generated - skipping signal scorer")\n'
    )

    # Fix 2: Cell 18 - Handle case where signal scoring fails due to insufficient trades
    code_cells[17].source = (
        "if len(trades) > 0 and 'pnl' in trades_df.columns:\n"
        "    trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades\n"
        "    \n"
        "    signal_feat_cols = ['entry_price', 'stop_loss', 'take_profit', 'atr_at_entry']\n"
        "    available_cols = [c for c in signal_feat_cols if c in trades_df.columns]\n"
        "    \n"
        "    if available_cols:\n"
        "        signal_X = trades_df[available_cols].copy()\n"
        "        signal_y = (trades_df['pnl'] > 0).astype(int)\n"
        "        \n"
        "        signal_X = signal_X.ffill().bfill().dropna()\n"
        "        signal_y = signal_y.loc[signal_X.index]\n"
        "        \n"
        "        if len(signal_X) >= 10:\n"
        "            scorer = SignalScorer(\n"
        '                model_type=CONFIG["ml"]["signal_model"],\n'
        '                n_estimators=CONFIG["ml"]["n_estimators"],\n'
        '                max_depth=CONFIG["ml"]["max_depth"],\n'
        '                random_state=CONFIG["ml"]["random_state"],\n'
        "            )\n"
        "            try:\n"
        "                signal_results = scorer.train(signal_X, signal_y)\n"
        '                print("Signal Scorer Training Results:")\n'
        "                for k, v in signal_results.items():\n"
        "                    if isinstance(v, (int, float)):\n"
        '                        print(f"  {k}: {v:.4f}")\n'
        "                    elif isinstance(v, str):\n"
        '                        print(f"  {k}: {v}")\n'
        "            except Exception as e:\n"
        '                print(f"Signal scorer training failed: {e}")\n'
        "        else:\n"
        '            print(f"Insufficient trades with features: {len(signal_X)} (need >=10)")\n'
        "    else:\n"
        '        print("No signal feature columns available in trades")\n'
        "else:\n"
        '    print("Skipping signal scorer - no valid trades")\n'
    )

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[OK] Created {output_path}")


if __name__ == "__main__":
    fix_notebook(
        "notebooks/13_ml_validation.ipynb",
        "notebooks/13_ml_validation.ipynb",
    )
