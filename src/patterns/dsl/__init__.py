"""
Constrained Factor DSL — symbolic expression language for pattern/factor definitions.

Source: "From Hypotheses to Factors: Constrained LLM Agents in Cryptocurrency Markets"
(arXiv:2604.26747v1, Huang, Fan, Hu & Ye, 2026)

The DSL constrains factor recipes to point-in-time market variables with four
operator families, preventing arbitrary code generation and forward-looking bias.

Operator families:
    Cross-sectional: rank, zscore
    Time-series: lag, ma, std, diff, pct_chg
    Nonlinear: log, abs, clip, sqrt
    Combination: linear_combo (weighted sum)

Usage (Python API):
    from src.patterns.dsl import rank, ma, log, linear_combo, evaluate

    factor = rank(linear_combo([
        (-0.6, log("mcap")),
        (0.5, ma("hl_range", window=10)),
        (-0.2, ma(pct_chg("volume", lag=1), window=3))
    ]))
    scores = evaluate(factor, ohlcv_df)

Usage (String DSL — canonical trace format):
    from src.patterns.dsl import parse_expr, evaluate_string

    expr_str = 'rank(-0.6*log(mcap) + 0.5*ma(hl_range,10) - 0.2*ma(pct_chg(volume,1),3))'
    factor = parse_expr(expr_str)
    scores = evaluate_string(expr_str, ohlcv_df)
"""

from src.patterns.dsl.grammar import (
    FactorExpression,
    CrossSectionalExpr,
    TimeSeriesExpr,
    NonlinearExpr,
    LinearComboExpr,
    VariableExpr,
    ConstantExpr,
    rank,
    zscore,
    lag,
    ma,
    std,
    diff,
    pct_chg,
    log_expr,
    abs_expr,
    clip,
    sqrt_expr,
    linear_combo,
    parse_expr,
    validate_expr,
    to_string,
    KEYWORD_TO_CLASS,
    var,
    const,
)

from src.patterns.dsl.executor import (
    evaluate,
    evaluate_string,
    evaluate_factor_panel,
    compute_ic,
    DSLValidationError,
    DSLExecutionError,
    BASE_VARIABLES,
    DERIVED_VARIABLES,
    ALLOWED_OPERATORS,
)

from src.patterns.dsl.validator import (
    DSLValidator,
    ValidationResult,
    validate_recipe,
    MAX_OPERATOR_DEPTH,
    MAX_TOTAL_OPERATORS,
)

__all__ = [
    "FactorExpression",
    "CrossSectionalExpr",
    "TimeSeriesExpr",
    "NonlinearExpr",
    "LinearComboExpr",
    "VariableExpr",
    "ConstantExpr",
    "rank",
    "zscore",
    "lag",
    "ma",
    "std",
    "diff",
    "pct_chg",
    "log_expr",
    "abs_expr",
    "clip",
    "sqrt_expr",
    "linear_combo",
    "var",
    "const",
    "parse_expr",
    "validate_expr",
    "to_string",
    "evaluate",
    "evaluate_string",
    "evaluate_factor_panel",
    "compute_ic",
    "DSLValidationError",
    "DSLExecutionError",
    "DSLValidator",
    "ValidationResult",
    "validate_recipe",
    "BASE_VARIABLES",
    "DERIVED_VARIABLES",
    "ALLOWED_OPERATORS",
    "MAX_OPERATOR_DEPTH",
    "MAX_TOTAL_OPERATORS",
]
