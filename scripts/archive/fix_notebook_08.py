"""Fix empty DataFrame guards in 08_benchmark_ranking.ipynb."""

import nbformat


def fix_notebook(input_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    # Cell [5] (index 4): Rewrite with proper guard
    cell5 = code_cells[4]
    cell5.source = (
        'if len(df_results) > 0 and "Sharpe" in df_results.columns:\n'
        '    passing = df_results[(df_results["Sharpe"] >= 0.3) & (df_results["PF"] >= 1.0)]\n'
        '    print(f"Passing: {len(passing)} of {len(df_results)} combos\\n")\n'
        "    passing.style.format({\n"
        '        "WR": "{:.1f}%", "Sharpe": "{:.2f}", "PF": "{:.2f}",\n'
        '        "Return": "{:.1f}%", "BnH": "{:.1f}%", "MaxDD": "{:.1f}%"\n'
        "    })\n"
        "else:\n"
        '    print("No results to display - skipping")\n'
    )

    # Cell [6] (index 5): Rewrite with proper guard
    cell6 = code_cells[5]
    cell6.source = (
        'if len(df_results) > 0 and "Sharpe" in df_results.columns:\n'
        '    out_path = project_root / "reports" / "benchmark_ranking.csv"\n'
        "    df_results.to_csv(out_path, index=False)\n"
        '    print(f"Saved to: {out_path}")\n'
        "else:\n"
        '    print("No results to save")\n'
    )

    with open(input_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[OK] Fixed {input_path}")


if __name__ == "__main__":
    fix_notebook("notebooks/08_benchmark_ranking.ipynb")
