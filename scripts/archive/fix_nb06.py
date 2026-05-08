"""Fix notebook 06 to handle empty DataFrames gracefully."""

import json
from pathlib import Path

nb_path = Path("notebooks/06_pattern_contribution.ipynb")
nb = json.loads(nb_path.read_text())

# Fix cell 24 - plot pattern participation
cell = nb["cells"][24]
src = cell["source"]
new_src = []
for line in src:
    new_src.append(line)

# Check if there's a guard already
code = "".join(new_src)
if "if pattern_trade_stats.empty" not in code:
    guard = '# Plot pattern participation\nif pattern_trade_stats.empty:\n    print("No pattern trade data available")\nelse:\n'
    # Find where the comment is and add guard
    new_code = code.replace(
        "# Plot pattern participation\n",
        guard,
    )
    # Now indent everything after the guard
    parts = new_code.split("\n")
    indented_parts = []
    after_guard = False
    for p in parts:
        if after_guard and p.strip():
            if not p.startswith("    "):
                indented_parts.append("    " + p)
            else:
                indented_parts.append(p)
        else:
            indented_parts.append(p)
            if "else:" in p:
                after_guard = True
    cell["source"] = [p + "\n" for p in indented_parts[:-1]] + [indented_parts[-1]]
    print("Fixed cell 24")

# Fix cell 28 - pattern combination stats
cell28 = nb["cells"][28]
code28 = "".join(cell28["source"])
if "pattern_trade_stats.empty" not in code28 and "get_pattern_combination_stats" in code28:
    for ci, cell_idx in enumerate([28, 29, 30, 31, 32, 33]):
        if cell_idx < len(nb["cells"]):
            c = nb["cells"][cell_idx]
            code_c = "".join(c["source"])
            empty_check_needed = any(
                x in code_c
                for x in [
                    "get_pattern_combination_stats",
                    "get_best_trade_recipes",
                    "get_worst_trade_recipes",
                    "get_confluence_vs_performance",
                    "create_confluence_performance_chart",
                    "get_pattern_participation_rate",
                    "create_frequency_quality_scatter",
                    "get_unattributed_trades",
                ]
            )
            if empty_check_needed:
                lines = code_c.split("\n")
                fixed = []
                added_guard = False
                for li, l in enumerate(lines):
                    if li == 0 and not added_guard:
                        fixed.append("trade_count = trade_attributor.get_pattern_trade_stats()")
                        fixed.append("if trade_count.empty:")
                        fixed.append('    print("No attributed trade data")')
                        fixed.append("else:")
                        fixed.append("    " + l)
                        added_guard = True
                    elif added_guard:
                        fixed.append("    " + l)
                    else:
                        fixed.append(l)
                c["source"] = [f + "\n" for f in fixed[:-1]] + [fixed[-1]]
                print(f"Fixed cell {cell_idx}")

nb_path.write_text(json.dumps(nb, indent=4))
print("Done saving fixed notebook")
