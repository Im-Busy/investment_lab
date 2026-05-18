"""
H2: Pattern Quality Registry — gate-based weight modulation for pattern detectors.

Consumes output from H7 (pattern evaluation sweep) and maps each pattern to a
quality tier: PASS (1.0x weight), FAIL_STRONG (0.5x weight, require 2+ confluence),
FAIL_WEAK (0.3x weight, require 3+ confluence), ERROR (excluded).

Used by RulesFirstStrategy to dynamically adjust pattern weights based on empirical
gate results before aggregating composite signals.

Usage:
    from src.signals.pattern_quality_registry import PatternQualityRegistry

    registry = PatternQualityRegistry.load("reports/pattern_gate/all_patterns.json")
    weight_mult = registry.get_multiplier("Head and Shoulders")  # 0.5 if FAIL
    min_confluence = registry.get_min_confluence("Head and Shoulders")  # 2 if FAIL
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class PatternQualityEntry:
    """Quality assessment for a single pattern detector."""

    name: str
    cls_path: str
    passed: bool
    steps_passed: int
    total_steps: int
    rank_ic: float | None
    tstat: float | None
    pvalue: float | None
    weight_multiplier: float
    required_confluence: int
    error: str | None = None


@dataclass
class PatternQualityRegistry:
    """Registry mapping pattern names to quality tiers and weight multipliers.

    Quality tiers:
        PASS (3-4 steps):   1.0x weight,  1 min confluence
        FAIL_STRONG (2):    0.5x weight,  2 min confluence
        FAIL_WEAK (0-1):    0.3x weight,  3 min confluence
        ERROR:              excluded from signal computation
    """

    entries: dict[str, PatternQualityEntry] = field(default_factory=dict)
    _meta: dict = field(default_factory=dict)

    DEFAULT_MULTIPLIER: ClassVar[float] = 0.7
    DEFAULT_CONFLUENCE: ClassVar[int] = 1

    @classmethod
    def load(cls, path: str | Path) -> PatternQualityRegistry:
        """Load sweep results and build quality registry."""
        raw = json.loads(Path(path).read_text())
        entries: dict[str, PatternQualityEntry] = {}

        for r in raw.get("results", []):
            name = r["pattern_name"]
            passed = r.get("passed", False)
            steps_passed = r.get("steps_passed", 0)
            total_steps = r.get("total_steps", 4)
            error = r.get("error")

            if error:
                entries[name] = PatternQualityEntry(
                    name=name,
                    cls_path=r.get("cls_path", ""),
                    passed=False,
                    steps_passed=0,
                    total_steps=total_steps,
                    rank_ic=None,
                    tstat=None,
                    pvalue=None,
                    weight_multiplier=0.0,
                    required_confluence=999,
                    error=error,
                )
            elif passed:
                entries[name] = PatternQualityEntry(
                    name=name,
                    cls_path=r.get("cls_path", ""),
                    passed=True,
                    steps_passed=steps_passed,
                    total_steps=total_steps,
                    rank_ic=r.get("step3_ic"),
                    tstat=r.get("step1_tstat"),
                    pvalue=r.get("step1_pvalue"),
                    weight_multiplier=1.0,
                    required_confluence=1,
                )
            elif steps_passed >= 2:
                entries[name] = PatternQualityEntry(
                    name=name,
                    cls_path=r.get("cls_path", ""),
                    passed=False,
                    steps_passed=steps_passed,
                    total_steps=total_steps,
                    rank_ic=r.get("step3_ic"),
                    tstat=r.get("step1_tstat"),
                    pvalue=r.get("step1_pvalue"),
                    weight_multiplier=0.5,
                    required_confluence=2,
                )
            else:
                entries[name] = PatternQualityEntry(
                    name=name,
                    cls_path=r.get("cls_path", ""),
                    passed=False,
                    steps_passed=steps_passed,
                    total_steps=total_steps,
                    rank_ic=r.get("step3_ic"),
                    tstat=r.get("step1_tstat"),
                    pvalue=r.get("step1_pvalue"),
                    weight_multiplier=0.3,
                    required_confluence=3,
                )

        return cls(entries=entries, _meta=raw.get("config", {}))

    def get_multiplier(self, pattern_name: str) -> float:
        """Get weight multiplier for a pattern (1.0 = full weight, 0.0 = excluded)."""
        entry = self.entries.get(pattern_name)
        if entry is None:
            return self.DEFAULT_MULTIPLIER
        return entry.weight_multiplier

    def get_min_confluence(self, pattern_name: str) -> int:
        """Get minimum number of concurrent patterns required for this pattern."""
        entry = self.entries.get(pattern_name)
        if entry is None:
            return self.DEFAULT_CONFLUENCE
        return entry.required_confluence

    def is_excluded(self, pattern_name: str) -> bool:
        """Check if pattern should be completely excluded from signal computation."""
        entry = self.entries.get(pattern_name)
        if entry is None:
            return False
        return entry.weight_multiplier == 0.0

    def summary_table(self) -> str:
        """Format a summary table of all entries."""
        lines = []
        lines.append("\n  Pattern Quality Registry Summary")
        lines.append(f"  {'Pattern':<32} {'Pass':<6} {'Wt':>5} {'MinC':>5} {'IC':>8}")
        lines.append(f"  {'-' * 32} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 8}")

        for name, entry in sorted(self.entries.items()):
            ic = f"{entry.rank_ic:.4f}" if entry.rank_ic is not None else "N/A"
            wt = f"{entry.weight_multiplier:.1f}x"
            if entry.error:
                wt = "ERR"
            lines.append(
                f"  {name:<32} {str(entry.passed):<6} {wt:>5} {entry.required_confluence:>5} {ic:>8}"
            )

        pass_count = sum(1 for e in self.entries.values() if e.passed)
        fail_count = sum(1 for e in self.entries.values() if not e.passed and not e.error)
        err_count = sum(1 for e in self.entries.values() if e.error)
        lines.append(
            f"\n  Pass: {pass_count}  Fail: {fail_count}  Error: {err_count}  Total: {len(self.entries)}"
        )
        return "\n".join(lines)

    @property
    def meta(self) -> dict:
        return self._meta
