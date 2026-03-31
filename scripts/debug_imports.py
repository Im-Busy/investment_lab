"""
Diagnostic script to identify the import causing the kernel crash.
Run this script to test imports incrementally.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("Import Diagnostic Script")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Project root: {project_root}")
print()

# Test 1: Basic imports
print("[1/6] Testing basic imports...")
try:
    import pandas as pd

    print(f"  [OK] pandas {pd.__version__}")
except Exception as e:
    print(f"  [FAIL] pandas failed: {e}")

try:
    import numpy as np

    print(f"  [OK] numpy {np.__version__}")
except Exception as e:
    print(f"  [FAIL] numpy failed: {e}")

# Test 2: backtesting library
print("\n[2/6] Testing backtesting library...")
try:
    print("  [OK] backtesting.Strategy")
except Exception as e:
    print(f"  [FAIL] backtesting failed: {e}")

# Test 3: Pattern detectors - Basic
print("\n[3/6] Testing basic pattern detectors...")
basic_patterns = [
    ("src.patterns.basic.msl", "MarketStructureLow"),
    ("src.patterns.basic.matching_lows", "MatchingLows"),
    ("src.patterns.basic.nr7id", "NR7ID"),
    ("src.patterns.basic.n_bar_decline", "NBarDecline"),
    ("src.patterns.basic.floor_pivot", "FloorPivotBreakout"),
]

for module_name, class_name in basic_patterns:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  [OK] {class_name}")
    except Exception as e:
        print(f"  [FAIL] {class_name}: {e}")

# Test 4: Pattern detectors - Harmonic
print("\n[4/6] Testing harmonic pattern detectors...")
harmonic_patterns = [
    ("src.patterns.harmonic.gartley", "GartleyPattern"),
    ("src.patterns.harmonic.abc", "ABCPattern"),
    ("src.patterns.harmonic.symmetric_triangle", "SymmetricTriangle"),
    ("src.patterns.harmonic.donchian", "DonchianChannel"),
    ("src.patterns.harmonic.bollinger", "BollingerBands"),
]

for module_name, class_name in harmonic_patterns:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  [OK] {class_name}")
    except Exception as e:
        print(f"  [FAIL] {class_name}: {e}")

# Test 5: Pattern detectors - Complex & Classic
print("\n[5/6] Testing complex/classic pattern detectors...")
complex_patterns = [
    ("src.patterns.complex.cup_handle", "CupAndHandle"),
    ("src.patterns.complex.head_shoulders", "HeadAndShoulders"),
    ("src.patterns.complex.spike_ledge", "SpikeAndLedge"),
    ("src.patterns.complex.three_hills", "ThreeHillsMountain"),
    ("src.patterns.complex.parabolic_arc", "ParabolicArc"),
    ("src.patterns.classic.double_top", "DoubleTop"),
    ("src.patterns.classic.double_bottom", "DoubleBottom"),
    ("src.patterns.classic.trader_vic_2b", "TraderVic2B"),
    ("src.patterns.classic.triple_top", "TripleTop"),
    ("src.patterns.classic.dead_cat_bounce", "DeadCatBounce"),
]

for module_name, class_name in complex_patterns:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  [OK] {class_name}")
    except Exception as e:
        print(f"  [FAIL] {class_name}: {e}")

# Test 6: Full import
print("\n[6/6] Testing full module import...")
try:
    print("  [OK] BacktestPyRunner")
    print("  [OK] MultiPatternStrategy")
except Exception as e:
    print(f"  [FAIL] Full import failed: {e}")
    import traceback

    traceback.print_exc()

# Test 7: Visualization import
print("\n[7/7] Testing visualization import...")
try:
    print("  [OK] ReportGenerator")
except Exception as e:
    print(f"  [FAIL] visualization failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)
