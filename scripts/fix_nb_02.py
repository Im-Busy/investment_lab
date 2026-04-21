"""Fix and execute notebook 02_smc_backtest with daily data."""

import json
import subprocess
import sys

with open("notebooks/02_smc_backtest.ipynb", "r") as f:
    nb = json.load(f)

# Fix cell 7 - data loading: properly handle the SPY_5min.csv vs SPY_daily.csv issue
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Data loading cell - fix the path resolution
        if "Try to load 5-minute data, fallback to daily" in source:
            new_source = [
                "from pathlib import Path\n",
                "\n",
                "# Try to load 5-minute data, fallback to daily\n",
                "data_config = CONFIG['data']\n",
                "data_path = project_root / data_config['directory'] / data_config['file']\n",
                "\n",
                "if data_path.exists() and 'SPY_5min.csv' in data_config['file']:\n",
                "    df = load_price_data(CONFIG, project_root)\n",
                "    data_freq = '5-minute'\n",
                "    print('Loaded 5-minute data')\n",
                "elif data_path.exists() and 'SPY_daily.csv' in data_config['file']:\n",
                "    df = load_price_data(CONFIG, project_root)\n",
                "    data_freq = 'daily'\n",
                "    print('Loaded daily data')\n",
                "else:\n",
                "    # Try the specific file\n",
                "    data_path = project_root / data_config['directory'] / 'SPY_daily.csv'\n",
                "    if data_path.exists():\n",
                "        print('WARNING: 5-minute data not found. Using daily data for demonstration.')\n",
                "        fallback_config = CONFIG.copy()\n",
                "        fallback_config['data'] = CONFIG['data'].copy()\n",
                "        fallback_config['data']['file'] = 'SPY_daily.csv'\n",
                "        # Remove fallback_file key from data_config if it was there\n",
                "        df = load_price_data(fallback_config, project_root)\n",
                "        data_freq = 'daily'\n",
                "    else:\n",
                "        raise FileNotFoundError(f'No data found at {data_path}')\n",
                "\n",
                "# Display data summary\n",
                "print_data_summary(df, title=f'SMC Backtest Data ({data_freq})')\n",
                "\n",
                "# Check data frequency\n",
                "median_diff = df.index.to_series().diff().median()\n",
                "print(f'Data frequency: {median_diff}')\n",
            ]
            cell["source"] = new_source
            print(f"Fixed cell {i} (data loading)")
            break

with open("notebooks/02_smc_backtest.ipynb", "w") as f:
    json.dump(nb, f, indent=4)
print("Fixed notebook 02 data loading")
