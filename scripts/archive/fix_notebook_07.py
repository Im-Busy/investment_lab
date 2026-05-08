"""Fix import errors in 07_pattern_selection_framework.ipynb."""

import nbformat


def fix_notebook(input_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    # Fix: PatternSelectionConfig is in src.utils.notebook_helpers, not src.analysis.pattern_selector
    for cell in code_cells:
        if "from src.analysis.pattern_selector import" in cell.source:
            cell.source = cell.source.replace(
                "from src.analysis.pattern_selector import (\n"
                "    PatternSelectionConfig,\n"
                "    PatternSelector,\n"
                ")",
                "from src.analysis.pattern_selector import PatternSelector\n"
                "from src.utils.notebook_helpers import PatternSelectionConfig",
            )
            break

    # Fix: remove __file__ references that don't work in papermill
    for cell in nb.cells:
        if cell.cell_type == "code" and "__file__" in cell.source:
            cell.source = cell.source.replace(
                "Path(__file__).parent.parent.resolve() if '__file__' in dir() else None,",
                "Path('.').resolve().parent.parent,",
            )
            break

    with open(input_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[OK] Fixed {input_path}")


if __name__ == "__main__":
    fix_notebook("notebooks/07_pattern_selection_framework.ipynb")
