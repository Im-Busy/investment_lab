"""Run notebook 02 fixing cells to handle missing 5min data and skip SMC intraday checks."""

import json

with open("notebooks/02_smc_backtest.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Cell that loads data - fix to use SPY_daily.csv directly
        if (
            "data_config = CONFIG" in source
            and "data_path" in source
            and "5-minute data not found" in source
        ):
            cell["source"] = [
                "from pathlib import Path\n",
                "\n",
                "# Force loading daily data since 5min isn't available\n",
                "data_path = project_root / CONFIG['data']['directory'] / 'SPY_daily.csv'\n",
                "assert data_path.exists(), f'Need data file for SMC notebook'\n",
                "\n",
                "print('WARNING: 5-minute data not found. Using daily data for demonstration.')\n",
                "print('For proper SMC backtesting, please provide 5-minute OHLCV data.')\n",
                "\n",
                "fallback_config = CONFIG.copy()\n",
                "fallback_config['data'] = CONFIG['data'].copy()\n",
                "fallback_config['data']['file'] = 'SPY_daily.csv'\n",
                "df = load_price_data(fallback_config, project_root)\n",
                "data_freq = 'daily'\n",
                "\n",
                "# Display data summary\n",
                "print_data_summary(df, title=f'SMC Backtest Data ({data_freq})')\n",
                "\n",
                "# Check data frequency\n",
                "median_diff = df.index.to_series().diff().median()\n",
                "print(f'Data frequency: {median_diff}')\n",
            ]
            print(f"Fixed data loading cell {i}")
        # Cell that calls detect_asian_range - keep the if check but verify
        elif "detect_asian_range(df)" in source:
            # Already has the if/else check, just make sure it's fine
            print(f"Cell {i}: Asian range check is present - OK")
        # Cell that calls detect_mss in a loop - this takes too long on 2500 rows
        elif "for i in range(50, len(df)):" in source and "detect_mss" in source:
            cell["source"] = [
                "# Detect Market Structure Shifts (limited sample for daily data)\n",
                "print('Detecting Market Structure Shifts (sampling every 10th day)')\n",
                "mss_list = []\n",
                "\n",
                "# For daily data, sample every 10th bar to keep it fast\n",
                "step = 10 if data_freq == 'daily' else 1\n",
                "for i in range(50, len(df), step):\n",
                "    mss = detect_mss(df, i)\n",
                "    if mss.detected and getattr(mss, 'is_valid', False):\n",
                "        mss_list.append({\n",
                "            'index': i,\n",
                "            'timestamp': df.index[i],\n",
                "            'direction': mss.direction,\n",
                "            'break_price': mss.break_price\n",
                "        })\n",
                "\n",
                "print(f'Found {len(mss_list)} valid MSS signals')\n",
                "if mss_list:\n",
                "    print('\\nRecent MSS signals:')\n",
                "    for mss in mss_list[-5:]:\n",
                "        print(f\"  {mss['timestamp']}: {mss['direction']} at {mss['break_price']:.2f}\")\n",
            ]
            print(f"Fixed MSS detection cell {i} - now sampling for daily data")

with open("notebooks/02_smc_backtest.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4)
print("Saved notebook 02 fixes")
