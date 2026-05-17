"""Identify flipped correlation features from retrained model."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(".").absolute()))
logging.basicConfig(level=logging.WARNING)
from scripts.model_health import ModelHealthMonitor

m = ModelHealthMonitor("models/pattern_classifier_v3_SPY_20260514_124612.pkl", "SPY")
result = m.check_feature_drift("2025-01-01")

s = result["summary"]
print(f"Features: {s['n_features']}, Shifted: {s['n_shifted']}, Flipped: {s['n_flipped']}")
print()

print("=== CORRELATION-FLIPPED FEATURES ===")
for col in result["flipped_features"]:
    info = result["drift_by_feature"][col]
    print(f"  {col}:")
    print(f"    IS_corr={info['ref_corr']:.4f}, OOS_corr={info['mon_corr']:.4f}")
    print(f"    KS_stat={info['ks_stat']:.4f}, p={info['p_value']}")

print()
print("=== ALL SHIFTED FEATURES (warn+failed) ===")
for col in result["shifted_features"]:
    info = result["drift_by_feature"][col]
    flipped = info.get("corr_flipped", False)
    extra = " [FLIP]" if flipped else ""
    print(f"  {col}: KS={info['ks_stat']:.4f}, p={info['p_value']}{extra}")
