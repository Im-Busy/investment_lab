"""Fix Path('..').resolve() -> Path('.').resolve() in all notebooks."""

import nbformat
from pathlib import Path


def fix_notebooks():
    notebooks_dir = Path("notebooks")
    for nb_path in notebooks_dir.glob("*.ipynb"):
        if "_executed" in nb_path.name or "_local" in nb_path.name or "_temp" in nb_path.name:
            continue

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        fixed = False
        for cell in nb.cells:
            if cell.cell_type == "code" and "Path('..').resolve()" in cell.source:
                cell.source = cell.source.replace("Path('..').resolve()", "Path('.').resolve()")
                fixed = True

        if fixed:
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"[OK] Fixed {nb_path.name}")


if __name__ == "__main__":
    fix_notebooks()
