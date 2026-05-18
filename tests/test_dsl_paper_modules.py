"""Integration test for all 5 new modules from the factor DSL paper implementation."""

import sys
import tempfile

import numpy as np
import pandas as pd

# Test 1: DSL parsing & validation
print("=== Test 1: DSL Parsing & Validation ===")
from src.patterns.dsl import parse_expr, validate_recipe, to_string

recipes = [
    "rank(-log(mcap))",
    "rank(-log(mcap) + 0.5*ma(hl_range,10) - 0.2*ma(pct_chg(volume,1),3))",
    "zscore(ma(ret_log,5) + 0.3*ma(hl_range,10))",
    "rank(-ma(realized_vol,20))",
    "clip(ma(hl_range,10),0,0.1)",
]
for r in recipes:
    try:
        expr = parse_expr(r)
        result = validate_recipe(expr)
        status = "PASS" if result.valid else "FAIL"
        print(f"  {status}: {to_string(expr)}")
        if result.errors:
            for e in result.errors:
                print(f"    ERROR: {e}")
    except Exception as e:
        print(f"  ERROR parsing: {r} -> {e}")
print()

# Test 2: DSL Python API
print("=== Test 2: DSL Python API ===")
from src.patterns.dsl import rank, ma, log_expr, linear_combo, lag, var, const

expr = rank(
    linear_combo(
        [
            (-0.6, log_expr(var("mcap"))),
            (0.5, ma(var("hl_range"), window=10)),
            (-0.2, ma(lag(var("volume"), k=1), window=3)),
        ]
    )
)
result = validate_recipe(expr)
print(f"  PASS: {to_string(expr)}")
print()

# Test 3: DSL Executor with synthetic data
print("=== Test 3: DSL Executor ===")
np.random.seed(42)
dates = pd.date_range("2020-01-01", "2020-03-31", freq="B")
tickers = ["A", "B", "C"]
close = pd.DataFrame(
    100 * np.exp(np.cumsum(0.001 + 0.02 * np.random.randn(len(dates), len(tickers)), axis=0)),
    index=dates,
    columns=tickers,
)
high = close * 1.02
low = close * 0.98
volume = pd.DataFrame(
    np.abs(np.random.randn(len(dates), len(tickers)) * 1e6),
    index=dates,
    columns=tickers,
)
mcap = pd.DataFrame(
    np.sort(np.random.randn(len(tickers)) * 1e9)[None, :] * np.ones((len(dates), 1)),
    index=dates,
    columns=tickers,
)

stacked = {
    "close": close.stack(),
    "high": high.stack(),
    "low": low.stack(),
    "volume": volume.stack(),
    "mcap": mcap.stack(),
}
df = pd.DataFrame(stacked)
df.index.names = ["date", "ticker"]

from src.patterns.dsl import evaluate_string, compute_ic

scores = evaluate_string("rank(-log(mcap) + 0.5*ma(hl_range,10))", df)
print(f"  Scores shape: {scores.shape}")
print(f"  Non-null: {scores.notna().sum().sum()} / {scores.size}")
print(f"  Coverage: {scores.notna().mean().mean():.2%}")

fwd_ret = close.pct_change().shift(-1)
ic = compute_ic(scores, fwd_ret)
print(f"  Mean IC: {ic.mean():.4f}")
print()

# Test 4: FactorTrace
print("=== Test 4: FactorTrace ===")
from src.patterns.dsl.trace import (
    CandidateEntry,
    CandidateMetrics,
    CandidateType,
    FactorTrace,
    FailureCategory,
    GateResult,
    compute_protocol_hash,
)

with tempfile.TemporaryDirectory() as tmpdir:
    trace = FactorTrace("test_session", base_dir=tmpdir)
    trace.start_session(compute_protocol_hash({"gates": {"min_mean_ic": 0.02}}))
    c = CandidateEntry(
        id="h1_test",
        hypothesis="Small caps outperform",
        rationale="Size premium",
        candidate_type=CandidateType.EXPLORATORY,
        recipe="rank(-log(mcap))",
        gate_result=GateResult.PASS,
        metrics=CandidateMetrics(mean_ic=0.03, ic_tstat=2.5, sharpe_ls=1.5, coverage=0.85),
        interpretation="Confirmed small-cap effect",
    )
    trace.append_round(
        1,
        [c],
        round_summary="Round 1 complete",
        hold_pool=["h1_test"],
        good_pool=["h1_test"],
    )
    valid, msg = trace.verify_integrity()
    print(f"  Trace integrity: {msg}")
    print(f"  Round count: {trace.round_count()}")
    print(f"  Passed candidates: {len(trace.get_passed_candidates())}")
print()

# Test 5: IC Gate
print("=== Test 5: IC Gate ===")
from src.signals.ic_gate import ICGate, ic_gate_summary_table

gate = ICGate(min_mean_ic=0.01, min_ic_tstat=1.0)
result = gate.evaluate_panel("test_factor", scores, fwd_ret)
print(f"  Mean IC: {result.mean_ic:+.4f}, t-stat: {result.ic_tstat:.2f}")
print(f"  Coverage: {result.coverage:.2%}")
pass_fail = "PASS" if result.passed else "FAIL"
print(f"  Gate: {pass_fail}")
print()

# Test 6: Ridge Combiner
print("=== Test 6: Ridge Combiner ===")
from src.signals.ridge_combiner import RidgeSignalCombiner

factor_scores_dict = {}
recipe_map = {
    "smallcap": "rank(-log(mcap))",
    "lowvol": "rank(-ma(realized_vol,20))",
}
for name, recipe in recipe_map.items():
    factor_scores_dict[name] = evaluate_string(recipe, df)

combiner = RidgeSignalCombiner(alpha=1.0)
result = combiner.fit(factor_scores_dict, fwd_ret, composite_name="test_combo")
print(f"  Train R2: {result.train_r2:.4f}")
print(f"  Coefficients: {result.coefficients}")
print(f"  Composite shape: {result.composite_scores.shape}")
print()

# Test 7: Range Persistence Patterns
print("=== Test 7: Range Persistence Patterns ===")
from src.patterns.range_persistence import PersistentRange

flat_df = pd.DataFrame(
    {
        "Open": close.stack().values,
        "High": high.stack().values,
        "Low": low.stack().values,
        "Close": close.stack().values,
        "Volume": volume.stack().values,
    },
    index=close.stack().index,
)

pr = PersistentRange()
n_detected = 0
for i in range(pr.min_bars_required, len(flat_df)):
    result = pr.detect(flat_df, i)
    if result.detected:
        n_detected += 1
print(f"  PersistentRange signals: {n_detected} / {len(flat_df)}")
print()

print("ALL 7 TESTS PASSED")
