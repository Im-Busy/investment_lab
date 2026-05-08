"""Transform Google Colab notebook to local-executable version."""

import nbformat
import re


def transform_colab_to_local(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Cell index mapping (0-indexed, counting only code cells):
    # Cell 0: markdown "Step 1"
    # Cell 1: code: pip install
    # Cell 2: markdown "Step 2"
    # Cell 3: code: google.colab drive mount
    # Cell 4: markdown "Step 3"
    # Cell 5: code: CONFIG
    # Cell 6: markdown "Step 4"
    # Cell 7: code: yfinance download
    # Cell 8: markdown "Step 5"
    # Cell 9: code: create_features
    # Cell 10: markdown "Step 6"
    # Cell 11: code: train_test_split
    # Cell 12: markdown "Step 7"
    # Cell 13: code: LightGBM train
    # Cell 14: markdown "Step 8"
    # Cell 15: code: evaluation
    # Cell 16: markdown "Step 9"
    # Cell 17: code: save model
    # Cell 18: markdown "Step 10"
    # Cell 19: code: google.colab.files download
    # Cell 20: markdown "Step 11"
    # Cell 21: code: usage instructions (print only)

    # Fix code bugs in the notebook:

    # Bug 1: bullish_engulfing .astype(int) fails with NaN -> use .fillna(0).astype(int)
    # Bug is in cell 10 (create_features code cell)
    src = nb.cells[10].source
    src = src.replace(")).astype(int)", ")).fillna(0).astype(int)")
    nb.cells[10].source = src

    # Fix Cell [2]: pip install code -> skip
    nb.cells[2].source = (
        "# Dependencies already installed via uv/pyproject.toml\n"
        'print("[OK] Dependencies already installed")'
    )

    # Fix Cell [4]: google.colab.drive.mount code -> local dirs
    nb.cells[4].source = (
        "import os\n"
        'os.makedirs("models", exist_ok=True)\n'
        'os.makedirs("outputs", exist_ok=True)\n'
        'print("[OK] Local directories created")'
    )

    # Fix Cell 3: google.colab.drive.mount -> local dirs
    nb.cells[3].source = (
        "import os\n"
        'os.makedirs("models", exist_ok=True)\n'
        'os.makedirs("outputs", exist_ok=True)\n'
        'print("[OK] Local directories created")'
    )

    # Fix Cell 16 (evaluation): replace /content/ save paths
    cell15 = nb.cells[16]
    source = cell15.source
    source = source.replace(
        "plt.savefig('/content/confusion_matrix.png', dpi=150)",
        "plt.savefig('outputs/confusion_matrix.png', dpi=150)",
    )
    source = source.replace(
        "plt.savefig('/content/feature_importance.png', dpi=150, bbox_inches='tight')",
        "plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')",
    )
    cell15.source = source

    # Fix Cell 18 (save model): replace /content/ paths and remove Google Drive copy
    cell17 = nb.cells[18]
    source = cell17.source
    source = source.replace(
        "model_path = f'/content/pattern_classifier_{timestamp}.pkl'",
        'model_path = f"outputs/pattern_classifier_{timestamp}.pkl"',
    )
    source = source.replace(
        "scaler_path = f'/content/scaler_{timestamp}.pkl'",
        'scaler_path = f"outputs/scaler_{timestamp}.pkl"',
    )
    source = source.replace(
        "metadata_path = f'/content/metadata_{timestamp}.json'",
        'metadata_path = f"outputs/metadata_{timestamp}.json"',
    )
    # Remove Google Drive backup section
    drive_pattern = (
        r"\n# Copy to Google Drive.*shutil\.copy\(metadata_path, drive_metadata_path\).*?\n\s*\n"
    )
    source = re.sub(drive_pattern, "\n", source, flags=re.DOTALL)
    # Remove remaining drive paths lines
    source = re.sub(r'drive_\w+_path = f"/content/drive/MyDrive/ML_Models/.*?"\n', "", source)
    # Clean up the Google Drive print section
    source = source.replace(
        'print(f"\\n✓ All files backed up to Google Drive!")\n'
        'print(f"Google Drive folder: /content/drive/MyDrive/ML_Models/")',
        "",
    )
    cell17.source = source

    # Fix Cell 20: google.colab.files download -> local zip
    cell19 = nb.cells[20]
    cell19.source = (
        "import zipfile\n"
        "import os\n"
        "\n"
        "# Create zip file with all artifacts\n"
        "zip_name = f\"outputs/ml_model_{CONFIG['symbol']}_{timestamp}.zip\"\n"
        'os.makedirs("outputs", exist_ok=True)\n'
        "with zipfile.ZipFile(zip_name, 'w') as zipf:\n"
        "    if os.path.exists(model_path):\n"
        "        zipf.write(model_path, os.path.basename(model_path))\n"
        "    if os.path.exists(scaler_path):\n"
        "        zipf.write(scaler_path, os.path.basename(scaler_path))\n"
        "    if os.path.exists(metadata_path):\n"
        "        zipf.write(metadata_path, os.path.basename(metadata_path))\n"
        "    if os.path.exists('outputs/confusion_matrix.png'):\n"
        "        zipf.write('outputs/confusion_matrix.png', 'confusion_matrix.png')\n"
        "    if os.path.exists('outputs/feature_importance.png'):\n"
        "        zipf.write('outputs/feature_importance.png', 'feature_importance.png')\n"
        "\n"
        'print(f"\\nCreated: {zip_name}")\n'
        'print("\\nModel files saved to outputs/ directory:")\n'
        'print(f"  - Model: outputs/pattern_classifier_{timestamp}.pkl")\n'
        'print(f"  - Scaler: outputs/scaler_{timestamp}.pkl")\n'
        'print(f"  - Metadata: outputs/metadata_{timestamp}.json")'
    )

    # Remove Colab-specific metadata
    nb.metadata.pop("colab", None)
    nb.metadata.pop("accelerator", None)

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[OK] Created {output_path}")


if __name__ == "__main__":
    transform_colab_to_local(
        "notebooks/ML_Training_Colab.ipynb",
        "notebooks/ML_Training_Colab_local.ipynb",
    )
