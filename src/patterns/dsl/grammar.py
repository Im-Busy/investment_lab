"""
DSL Grammar — expression tree types, Python API constructors, and string DSL parser.

Defines the abstract syntax tree (AST) for factor expressions. Each node type
corresponds to one of the four operator families plus leaves (variables/constants).

The string DSL grammar (PEG-like):
    expr        := cs_expr | ts_expr | nl_expr | combo_expr | variable | constant
    cs_expr     := "rank(" ws expr ws ")" | "zscore(" ws expr ws ")"
    ts_expr     := op_name "(" ws expr ws "," ws number ws ")"
    nl_expr     := op_name "(" ws expr ws ["," ws number ws ["," ws number ws]] ")"
    combo_expr  := term (ws sign ws term)*
    term        := [coeff "*"] atom
    atom        := cs_expr | ts_expr | nl_expr | "(" ws expr ws ")" | variable | constant
    variable    := [a-z_][a-z_0-9]* (validated against BASE_VARIABLES + DERIVED_VARIABLES)
    constant    := number
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

_KEYWORDS: dict[str, type["FactorExpression"]] = {}


def _register(cls: type["FactorExpression"]) -> type["FactorExpression"]:
    if cls.KEYWORD:
        _KEYWORDS[cls.KEYWORD] = cls
    return cls


@dataclass(frozen=True)
class FactorExpression:
    """Abstract base for all DSL expression nodes."""

    KEYWORD: ClassVar[str] = ""

    def depth(self) -> int:
        return 1

    def operator_count(self) -> int:
        return 0

    def variables(self) -> set[str]:
        return set()

    def to_dsl_string(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class VariableExpr(FactorExpression):
    """Leaf node: a base or derived market variable."""

    name: str

    def depth(self) -> int:
        return 0

    def operator_count(self) -> int:
        return 0

    def variables(self) -> set[str]:
        return {self.name}

    def to_dsl_string(self) -> str:
        return self.name


@dataclass(frozen=True)
class ConstantExpr(FactorExpression):
    """Leaf node: a numeric constant."""

    value: float

    def depth(self) -> int:
        return 0

    def operator_count(self) -> int:
        return 0

    def variables(self) -> set[str]:
        return set()

    def to_dsl_string(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:.6g}"


@_register
@dataclass(frozen=True)
class CrossSectionalExpr(FactorExpression):
    """Cross-sectional transform: rank(expr) or zscore(expr)."""

    KEYWORD: ClassVar[str] = ""
    method: str
    arg: FactorExpression

    def depth(self) -> int:
        return 1 + self.arg.depth()

    def operator_count(self) -> int:
        return 1 + self.arg.operator_count()

    def variables(self) -> set[str]:
        return self.arg.variables()

    def to_dsl_string(self) -> str:
        return f"{self.method}({self.arg.to_dsl_string()})"


@_register
@dataclass(frozen=True)
class TimeSeriesExpr(FactorExpression):
    """Time-series transform: lag, ma, std, diff, pct_chg."""

    KEYWORD: ClassVar[str] = ""
    method: str
    arg: FactorExpression
    param: int

    def depth(self) -> int:
        return 1 + self.arg.depth()

    def operator_count(self) -> int:
        return 1 + self.arg.operator_count()

    def variables(self) -> set[str]:
        return self.arg.variables()

    def to_dsl_string(self) -> str:
        return f"{self.method}({self.arg.to_dsl_string()},{self.param})"


@_register
@dataclass(frozen=True)
class NonlinearExpr(FactorExpression):
    """Nonlinear transform: log, abs, clip, sqrt."""

    KEYWORD: ClassVar[str] = ""
    method: str
    arg: FactorExpression
    param1: int | float | None = None
    param2: int | float | None = None

    def depth(self) -> int:
        return 1 + self.arg.depth()

    def operator_count(self) -> int:
        return 1 + self.arg.operator_count()

    def variables(self) -> set[str]:
        return self.arg.variables()

    def to_dsl_string(self) -> str:
        if self.param1 is None:
            return f"{self.method}({self.arg.to_dsl_string()})"
        if self.param2 is None:
            return f"{self.method}({self.arg.to_dsl_string()},{self.param1})"
        return f"{self.method}({self.arg.to_dsl_string()},{self.param1},{self.param2})"


@dataclass(frozen=True)
class LinearComboExpr(FactorExpression):
    """Linear combination: coeff1*expr1 + coeff2*expr2 + ..."""

    KEYWORD: ClassVar[str] = ""
    terms: tuple[tuple[float, FactorExpression], ...] = field(default_factory=tuple)

    def depth(self) -> int:
        if not self.terms:
            return 0
        return 1 + max(t.depth() for _, t in self.terms)

    def operator_count(self) -> int:
        return sum(t.operator_count() for _, t in self.terms) + len(self.terms) - 1

    def variables(self) -> set[str]:
        vs: set[str] = set()
        for _, t in self.terms:
            vs |= t.variables()
        return vs

    def to_dsl_string(self) -> str:
        parts: list[str] = []
        for i, (coeff, term) in enumerate(self.terms):
            term_str = term.to_dsl_string()
            if isinstance(term, LinearComboExpr):
                term_str = f"({term_str})"
            if i == 0:
                if coeff == 1.0:
                    parts.append(term_str)
                elif coeff == -1.0:
                    parts.append(f"-{term_str}")
                else:
                    parts.append(f"{coeff:.6g}*{term_str}")
            else:
                if coeff == 1.0:
                    parts.append(f" + {term_str}")
                elif coeff == -1.0:
                    parts.append(f" - {term_str}")
                elif coeff > 0:
                    parts.append(f" + {coeff:.6g}*{term_str}")
                else:
                    parts.append(f" - {abs(coeff):.6g}*{term_str}")
        return "".join(parts)


# --- Python API constructors ---


def rank(arg: FactorExpression) -> CrossSectionalExpr:
    return CrossSectionalExpr(method="rank", arg=arg)


def zscore(arg: FactorExpression) -> CrossSectionalExpr:
    return CrossSectionalExpr(method="zscore", arg=arg)


def lag(arg: FactorExpression, k: int) -> TimeSeriesExpr:
    return TimeSeriesExpr(method="lag", arg=arg, param=k)


def ma(arg: FactorExpression, window: int) -> TimeSeriesExpr:
    return TimeSeriesExpr(method="ma", arg=arg, param=window)


def std(arg: FactorExpression, window: int) -> TimeSeriesExpr:
    return TimeSeriesExpr(method="std", arg=arg, param=window)


def diff(arg: FactorExpression, k: int) -> TimeSeriesExpr:
    return TimeSeriesExpr(method="diff", arg=arg, param=k)


def pct_chg(arg: FactorExpression, k: int = 1) -> TimeSeriesExpr:
    return TimeSeriesExpr(method="pct_chg", arg=arg, param=k)


def log_expr(arg: FactorExpression, offset: float = 1.0) -> NonlinearExpr:
    return NonlinearExpr(method="log", arg=arg, param1=offset)


def abs_expr(arg: FactorExpression) -> NonlinearExpr:
    return NonlinearExpr(method="abs", arg=arg)


def clip(arg: FactorExpression, lo: float, hi: float) -> NonlinearExpr:
    return NonlinearExpr(method="clip", arg=arg, param1=lo, param2=hi)


def sqrt_expr(arg: FactorExpression) -> NonlinearExpr:
    return NonlinearExpr(method="sqrt", arg=arg)


def linear_combo(terms: list[tuple[float, FactorExpression]]) -> LinearComboExpr:
    return LinearComboExpr(terms=tuple(terms))


def var(name: str) -> VariableExpr:
    return VariableExpr(name=name)


def const(value: float) -> ConstantExpr:
    return ConstantExpr(value=value)


# --- String DSL Parser ---

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number>-?\d+\.?\d*(?:[eE][+-]?\d+)?)
        |(?P<keyword>rank|zscore|lag|ma|std|diff|pct_chg|log|abs|clip|sqrt)
        |(?P<lparen>\()
        |(?P<rparen>\))
        |(?P<comma>,)
        |(?P<star>\*)
        |(?P<plus>\+)
        |(?P<minus>-)
        |(?P<ident>[a-z_][a-z_0-9]*)
    )
    """,
    re.VERBOSE,
)

KEYWORD_TO_CLASS: dict[str, type[FactorExpression]] = {
    "rank": CrossSectionalExpr,
    "zscore": CrossSectionalExpr,
    "lag": TimeSeriesExpr,
    "ma": TimeSeriesExpr,
    "std": TimeSeriesExpr,
    "diff": TimeSeriesExpr,
    "pct_chg": TimeSeriesExpr,
    "log": NonlinearExpr,
    "abs": NonlinearExpr,
    "clip": NonlinearExpr,
    "sqrt": NonlinearExpr,
}


class DSLParseError(ValueError):
    """Raised when a DSL string cannot be parsed."""


def _tokenize(expr: str) -> list[dict[str, str]]:
    tokens: list[dict[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise DSLParseError(f"Unexpected character at position {pos}: '{expr[pos]}'")
        kind = m.lastgroup
        value = m.group(kind)
        tokens.append({"kind": kind, "value": value})
        pos = m.end()
    return tokens


def _parse_tokens(tokens: list[dict[str, str]], pos: int = 0) -> tuple[FactorExpression, int]:
    """Recursive descent parser. Returns (expr, next_pos)."""

    def _peek() -> dict[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def _consume(expected_kind: str | None = None) -> dict[str, str]:
        nonlocal pos
        if pos >= len(tokens):
            raise DSLParseError("Unexpected end of expression")
        tk = tokens[pos]
        if expected_kind is not None and tk["kind"] != expected_kind:
            raise DSLParseError(
                f"Expected {expected_kind} at position {pos}, got {tk['kind']} ('{tk['value']}')"
            )
        pos += 1
        return tk

    def _parse_combo() -> FactorExpression:
        """Parse linear combination: term (('+'|'-') term)*"""
        lhs = _parse_term()
        terms: list[tuple[float, FactorExpression]] = [(1.0, lhs)]
        while True:
            tk = _peek()
            if tk is None or tk["kind"] not in ("plus", "minus"):
                break
            sign = 1.0 if _consume()["kind"] == "plus" else -1.0
            rhs = _parse_term()
            terms.append((sign, rhs))
        if len(terms) == 1:
            return terms[0][1]
        return LinearComboExpr(terms=tuple(terms))

    def _parse_term() -> FactorExpression:
        """Parse term: [coeff '*'] atom"""
        tk = _peek()
        if tk is None:
            raise DSLParseError("Unexpected end of expression")

        coeff: float | None = None
        sign: float = 1.0
        if tk["kind"] == "minus":
            next_tk = tokens[pos + 1] if pos + 1 < len(tokens) else None
            is_unary = next_tk is not None and next_tk["kind"] in (
                "keyword",
                "lparen",
                "ident",
                "number",
            )
            if is_unary and next_tk["kind"] == "number":
                _consume("minus")
                val = -float(_consume("number")["value"])
                tk = _peek()
                if tk is not None and tk["kind"] == "star":
                    _consume("star")
                    coeff = val
                else:
                    return ConstantExpr(value=val)
            elif is_unary:
                sign = -1.0
                _consume("minus")
        elif tk["kind"] == "number":
            coeff_val = float(_consume()["value"])
            tk = _peek()
            if tk is not None and tk["kind"] == "star":
                _consume("star")
                coeff = coeff_val
            else:
                return ConstantExpr(value=coeff_val)

        atom = _parse_atom()
        if sign != 1.0:
            atom = LinearComboExpr(terms=((-1.0, atom),))
        if coeff is not None and coeff != 1.0:
            return LinearComboExpr(terms=((coeff, atom),))
        return atom

    def _parse_atom() -> FactorExpression:
        """Parse atom: keyword_call | '(' expr ')' | ident | number"""
        tk = _peek()
        if tk is None:
            raise DSLParseError("Unexpected end of expression")

        if tk["kind"] == "lparen":
            _consume("lparen")
            inner = _parse_combo()
            _consume("rparen")
            return inner

        if tk["kind"] == "keyword":
            return _parse_keyword_call()

        if tk["kind"] == "ident":
            name = _consume("ident")["value"]
            return VariableExpr(name=name)

        if tk["kind"] == "number":
            val = float(_consume("number")["value"])
            return ConstantExpr(value=val)

        raise DSLParseError(f"Unexpected token: {tk['kind']} ('{tk['value']}')")

    def _parse_keyword_call() -> FactorExpression:
        kw = _consume("keyword")
        method = kw["value"]
        _consume("lparen")
        inner = _parse_combo()

        params: list[float] = []
        while True:
            tk = _peek()
            if tk is None:
                break
            if tk["kind"] == "rparen":
                break
            if tk["kind"] == "comma":
                _consume("comma")
                num_tk = _peek()
                if num_tk is None or num_tk["kind"] not in ("number", "minus"):
                    raise DSLParseError(f"Expected number after comma in {method}()")
                sign = 1.0
                if num_tk["kind"] == "minus":
                    _consume("minus")
                    num_tk = _consume("number")
                    sign = -1.0
                else:
                    num_tk = _consume("number")
                params.append(sign * float(num_tk["value"]))
            else:
                break

        _consume("rparen")

        if method in ("rank", "zscore"):
            return CrossSectionalExpr(method=method, arg=inner)
        elif method in ("lag", "ma", "std", "diff", "pct_chg"):
            if not params:
                raise DSLParseError(f"{method}() requires a numeric parameter")
            return TimeSeriesExpr(method=method, arg=inner, param=int(params[0]))
        elif method in ("log", "abs", "clip", "sqrt"):
            p1 = params[0] if len(params) > 0 else None
            p2 = params[1] if len(params) > 1 else None
            return NonlinearExpr(method=method, arg=inner, param1=p1, param2=p2)
        else:
            raise DSLParseError(f"Unknown keyword: {method}")

    result, final_pos = _parse_combo(), pos
    if final_pos < len(tokens):
        raise DSLParseError(
            f"Trailing tokens at position {final_pos}: {tokens[final_pos]['value']}"
        )
    return result, final_pos


def parse_expr(expr_str: str) -> FactorExpression:
    """Parse a DSL string into an expression tree.

    Args:
        expr_str: Factor expression in the constrained DSL syntax.

    Returns:
        FactorExpression tree.

    Raises:
        DSLParseError: If the string is not valid DSL syntax.
    """
    tokens = _tokenize(expr_str)
    result, _ = _parse_tokens(tokens)
    return result


def validate_expr(expr: FactorExpression) -> list[str]:
    """Validate an expression tree against DSL constraints.

    Checks:
        - All variables are in the allowed set
        - No nested time-series ops deeper than 2 levels
        - At least one time-series or nonlinear transform
        - Max 15 total operators

    Returns:
        List of validation errors (empty = valid).
    """
    errors: list[str] = []

    from src.patterns.dsl.executor import ALLOWED_VARIABLES

    vs = expr.variables()
    for v in vs:
        if v not in ALLOWED_VARIABLES:
            errors.append(f"Unknown variable: '{v}'. Allowed: {sorted(ALLOWED_VARIABLES)}")

    if 0 < expr.depth() < 1:
        pass

    def _has_time_series_or_nonlinear(e: FactorExpression) -> bool:
        if isinstance(e, (TimeSeriesExpr, NonlinearExpr, CrossSectionalExpr)):
            return True
        if isinstance(e, LinearComboExpr):
            return any(_has_time_series_or_nonlinear(t) for _, t in e.terms)
        return False

    if not _has_time_series_or_nonlinear(expr):
        errors.append("Expression must include at least one time-series or nonlinear transform")

    op_count = expr.operator_count()
    if op_count > 15:
        errors.append(f"Too many operators ({op_count}). Maximum is 15.")

    def _max_ts_depth(e: FactorExpression) -> int:
        if isinstance(e, TimeSeriesExpr):
            return 1 + _max_ts_depth(e.arg)
        if isinstance(e, CrossSectionalExpr):
            return _max_ts_depth(e.arg)
        if isinstance(e, NonlinearExpr):
            return _max_ts_depth(e.arg)
        if isinstance(e, LinearComboExpr):
            return max((_max_ts_depth(t) for _, t in e.terms), default=0)
        return 0

    ts_depth = _max_ts_depth(expr)
    if ts_depth > 2:
        errors.append(f"Nested time-series depth {ts_depth} exceeds maximum of 2")

    return errors


def to_string(expr: FactorExpression) -> str:
    """Convert an expression tree to its canonical DSL string representation."""
    return expr.to_dsl_string()
