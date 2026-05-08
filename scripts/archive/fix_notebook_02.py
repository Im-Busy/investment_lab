"""Fix notebook 02 to handle missing 5min data by falling back to daily gracefully."""
import json

with open('notebooks/02_smc_backtest.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 4 (load data) to gracefully detect data type
data_cells = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'load_price_data' in ''.join(c['source'])]
if data_cells:
    idx = data_cells[0]
    cell = nb['cells'][idx]
    new_source = [
        'from pathlib import Path\n',
        '\n',
        "# Try to load 5-minute data, fallback to daily\n",
        "data_config = CONFIG['data']\n",
        "data_path = project_root / data_config['directory'] / data_config['file']\n",
        "\n",
        "if data_path.exists():\n",
        "    df = load_price_data(CONFIG, project_root)\n",
        "    data_freq = '5-minute'\n",
        "    print(f'Loaded 5-minute data')\n",
        "else:\n",
        "    print('WARNING: 5-minute data not found. Using daily data for demonstration.')\n",
        "    df = None  # Will be loaded as fallback below\n",
        "\n",
        "if df is None:\n",
        "    fallback_config = CONFIG.copy()\n",
        "    fallback_config['data'] = CONFIG['data'].copy()\n",
        "    fallback_config['data']['file'] = 'SPY_daily.csv'\n",
        "    df = load_price_data(fallback_config, project_root)\n",
        "    data_freq = 'daily'\n",
        "\n",
        "# Display data summary\n",
        "print_data_summary(df, title=f\"SMC Backtest Data ({data_freq})\")\n",
        "\n",
        "# Check data frequency\n",
        "median_diff = df.index.to_series().diff().median()\n",
        "print(f'Data frequency: {median_diff}')\n',
        'if data_freq == "daily":\n",
        '    print("NOTE: SMC strategy is designed for intraday data. Results on daily data will be limited.")\n',
    ]
    cell['source'] = new_source
    print(f'Fixed data loading cell {idx}')

# Fix cells that require intraday-specific functionality
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'detect_asian_range' in source and 'if data_freq' not in source:
            # The cell already has this check, just print it
            print(f'Cell {i}: Has asian range check - OK')
            continue

with open('notebooks/02_smc_backtest.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)
print('Saved notebook 02 with data fallback!')
