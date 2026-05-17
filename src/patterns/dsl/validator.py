"""
DSL Validator — structural and safety validation for factor recipes.

Enforces the DSL constraints from the paper:
    - No forward-looking variables (all inputs must be ≤ t)
    - No arbitrary code (only DSL expression trees allowed)
    - Nested time-series ops max depth 2
    - Must include ≥1 time-series OR nonlinear transform
    - Must be manually inspectable (max 15 operators)
    - No data leakage via look-ahead rolling windows
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.patterns.dsl.grammar import (
    FactorExpression,
    CrossSectionalExpr,
    TimeSeriesExpr,
    NonlinearExpr,
    LinearComboExpr,
    VariableExpr,
    ConstantExpr,
    validate_expr as _validate_grammar,
)
from src.patterns.dsl.executor import ALLOWED_VARIABLES

MAX_OPERATOR_DEPTH: int = 3
MAX_TOTAL_OPERATORS: int = 15
MAX_TS_NESTING: int = 2


@dataclass
class ValidationResult:
    """Result of DSL recipe validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    variable_count: int = 0
    operator_count: int = 0
    ts_nesting_depth: int = 0
    has_ts_or_nonlinear: bool = False
    canonical_form: str = ""

    def report(self) -> str:
        lines: list[str] = []
        lines.append(f"Validation: {'PASS' if self.valid else 'FAIL'}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        lines.append(f"  Variables: {self.variable_count}")
        lines.append(f"  Operators: {self.operator_count} (max {MAX_TOTAL_OPERATORS})")
        lines.append(f"  TS nesting: {self.ts_nesting_depth} (max {MAX_TS_NESTING})")
        lines.append(f"  Has TS/nonlinear: {self.has_ts_or_nonlinear}")
        if self.canonical_form:
            lines.append(f"  Canonical: {self.canonical_form}")
        return "\n".join(lines)


class DSLValidator:
    """Validates factor recipes against all DSL constraints.

    Usage:
        validator = DSLValidator()
        result = validator.validate(expr)
        if result.valid:
            print("Recipe is valid")
        else:
            print(result.report())
    """

    def validate(self, expr: FactorExpression) -> ValidationResult:
        """Run full validation suite on an expression tree.

        Args:
            expr: Parsed FactorExpression tree.

        Returns:
            ValidationResult with errors, warnings, and metadata.
        """
        errors: list[str] = []
        warnings: list[str] = []

        grammar_errors = _validate_grammar(expr)
        errors.extend(grammar_errors)

        vs = expr.variables()
        for v in vs:
            if v not in ALLOWED_VARIABLES:
                errors.append(f"Unknown variable: '{v}'. Allowed: {sorted(ALLOWED_VARIABLES)}")

        op_count = expr.operator_count()
        if op_count > MAX_TOTAL_OPERATORS:
            errors.append(f"Operator count {op_count} exceeds maximum {MAX_TOTAL_OPERATORS}")

        def _has_ts_or_nl(e: FactorExpression) -> bool:
            if isinstance(e, (TimeSeriesExpr, NonlinearExpr, CrossSectionalExpr)):
                return True
            if isinstance(e, LinearComboExpr):
                return any(_has_ts_or_nl(t) for _, t in e.terms)
            return False

        has_transform = _has_ts_or_nl(expr)
        if not has_transform:
            errors.append("Must include at least one time-series or nonlinear transform")

        def _ts_depth(e: FactorExpression) -> int:
            if isinstance(e, TimeSeriesExpr):
                return 1 + _ts_depth(e.arg)
            if isinstance(e, (CrossSectionalExpr, NonlinearExpr)):
                return _ts_depth(e.arg)
            if isinstance(e, LinearComboExpr):
                return max((_ts_depth(t) for _, t in e.terms), default=0)
            return 0

        ts_depth = _ts_depth(expr)
        if ts_depth > MAX_TS_NESTING:
            errors.append(f"TS nesting depth {ts_depth} exceeds max {MAX_TS_NESTING}")

        depth = expr.depth()
        if depth > MAX_OPERATOR_DEPTH:
            warnings.append(
                f"Expression depth {depth} exceeds recommendation of {MAX_OPERATOR_DEPTH}"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            variable_count=len(vs),
            operator_count=op_count,
            ts_nesting_depth=ts_depth,
            has_ts_or_nonlinear=has_transform,
            canonical_form=expr.to_dsl_string(),
        )


def validate_recipe(expr: FactorExpression | str) -> ValidationResult:
    """Convenience function: validate an expression or DSL string.

    Args:
        expr: FactorExpression tree or DSL string.

    Returns:
        ValidationResult.
    """
    from src.patterns.dsl.grammar import parse_expr

    if isinstance(expr, str):
        expr = parse_expr(expr)
    return DSLValidator().validate(expr)
