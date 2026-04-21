"""Fix notebook 04 pattern visualization to handle plotting errors gracefully."""

import json

with open("notebooks/04_pattern_visualization.ipynb", "r") as f:
    nb = json.load(f)

# Find and fix the plot_pattern_type cell
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def plot_pattern_type" in source:
            new_source = [
                "def plot_pattern_type(pattern_name, df, signals, project_root, chart_gen, output_dir):\n",
                '    """Plot chart with specific pattern type."""\n',
                "    filtered = [s for s in signals if s['pattern_name'] == pattern_name]\n",
                "    \n",
                "    if not filtered:\n",
                '        print(f"No {pattern_name} signals found")\n',
                "        return\n",
                "    \n",
                "    # Generate output filename\n",
                "    filename = f\"{pattern_name.lower().replace(' ', '_')}.png\"\n",
                "    save_path = project_root / output_dir / filename\n",
                "    \n",
                "    try:\n",
                "        chart_gen.plot_with_patterns(\n",
                "            df=df,\n",
                "            signals=filtered,\n",
                "            title=f'SPY - {pattern_name} Signals ({len(filtered)} detected)',\n",
                "            save_path=str(save_path),\n",
                "            show=True\n",
                "        )\n",
                "    except Exception as e:\n",
                "        print(f'    Error plotting {pattern_name}: {e}')\n",
                "\n",
                "# Plot each pattern type\n",
                "for pattern_name in patterns.keys():\n",
                "    print(f'Plotting {pattern_name}...')\n",
                "    plot_pattern_type(\n",
                "        pattern_name,\n",
                "        df,\n",
                "        all_signals,\n",
                "        project_root,\n",
                "        chart_gen,\n",
                "        CONFIG['output']['directory']\n",
                "    )\n",
            ]
            cell["source"] = new_source
            print(f"Fixed cell {i}")
            break

with open("notebooks/04_pattern_visualization.ipynb", "w") as f:
    json.dump(nb, f, indent=4)
print("Saved notebook 04 with fixes!")
