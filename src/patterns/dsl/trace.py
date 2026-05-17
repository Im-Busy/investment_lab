"""
Factor Discovery Trace — append-only experiment trace for agentic factor search.

Extends the existing ExperimentLogger with a JSONL trace format that records
every candidate factor: its hypothesis, rationale, DSL recipe, empirical metrics,
gate result, interpretation, and next-step decision.

Schema per trace entry (one JSON object per line):
{
    "session_id": "uuid",
    "round": 1,
    "timestamp": "2026-05-17T15:30:00Z",
    "protocol_hash": "sha256_of_config",
    "candidates": [
        {
            "id": "h1_smallcap_lowvol",
            "hypothesis": "...",
            "rationale": "...",
            "candidate_type": "exploratory",
            "recipe": "rank(-log(mcap) - ma(realized_vol,20))",
            "metrics": {
                "mean_ic": 0.031,
                "ic_tstat": 2.8,
                "sharpe_ls": 1.88,
                "coverage": 0.82
            },
            "gate_result": "PASS",
            "failure_category": null,
            "interpretation": "...",
            "next_action": "add_to_hold_pool"
        }
    ],
    "round_summary": "...",
    "hold_pool": ["h1_smallcap_lowvol", ...],
    "good_pool": ["h1_smallcap_lowvol", ...],
    "search_direction": "Focus on range + liquidity interaction"
}

Source: "From Hypotheses to Factors" by Huang, Fan, Hu & Ye (arXiv:2604.26747v1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRACE_DIR = Path("experiments") / "factor_traces"


class CandidateType(str, Enum):
    EXPLORATORY = "exploratory"
    EXPLOITATIVE = "exploitative"
    MECHANICAL = "mechanical"
    HYBRID = "hybrid"


class GateResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class FailureCategory(str, Enum):
    NOISE = "NOISE"
    REGIME_DEPENDENT = "REGIME_DEPENDENT"
    CAPACITY_LIMITED = "CAPACITY_LIMITED"
    REDUNDANT = "REDUNDANT"
    DATA_ISSUE = "DATA_ISSUE"
    HYPOTHESIS_INVALID = "HYPOTHESIS_INVALID"


@dataclass
class CandidateMetrics:
    """Empirical metrics for a single factor candidate."""

    mean_ic: float | None = None
    ic_tstat: float | None = None
    sharpe_ls: float | None = None
    coverage: float | None = None
    ic_std: float | None = None
    n_observations: int | None = None

    def to_dict(self) -> dict:
        return {
            "mean_ic": self.mean_ic,
            "ic_tstat": self.ic_tstat,
            "sharpe_ls": self.sharpe_ls,
            "coverage": self.coverage,
            "ic_std": self.ic_std,
            "n_observations": self.n_observations,
        }


@dataclass
class CandidateEntry:
    """A single candidate factor entry in the trace."""

    id: str
    hypothesis: str
    rationale: str
    candidate_type: CandidateType
    recipe: str
    metrics: CandidateMetrics | None = None
    gate_result: GateResult = GateResult.UNCERTAIN
    failure_category: FailureCategory | None = None
    interpretation: str = ""
    next_action: str = ""
    dsl_canonical: str = ""
    correlation_with_existing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "candidate_type": self.candidate_type.value,
            "recipe": self.recipe,
            "gate_result": self.gate_result.value,
            "failure_category": self.failure_category.value if self.failure_category else None,
            "interpretation": self.interpretation,
            "next_action": self.next_action,
        }
        if self.metrics:
            d["metrics"] = self.metrics.to_dict()
        if self.dsl_canonical:
            d["dsl_canonical"] = self.dsl_canonical
        if self.correlation_with_existing:
            d["correlation_with_existing"] = self.correlation_with_existing
        return d


@dataclass
class RoundEntry:
    """A single round in the factor discovery session."""

    round_num: int
    timestamp: str
    candidates: list[CandidateEntry] = field(default_factory=list)
    round_summary: str = ""
    hold_pool: list[str] = field(default_factory=list)
    good_pool: list[str] = field(default_factory=list)
    search_direction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "timestamp": self.timestamp,
            "candidates": [c.to_dict() for c in self.candidates],
            "round_summary": self.round_summary,
            "hold_pool": self.hold_pool,
            "good_pool": self.good_pool,
            "search_direction": self.search_direction,
        }


class FactorTrace:
    """Append-only trace for factor discovery sessions.

    Usage:
        trace = FactorTrace(session_id="sess_001")
        trace.start_session(config_hash="abc123")

        candidate = CandidateEntry(
            id="h1_smallcap",
            hypothesis="Small-cap tokens outperform",
            rationale="Prior lit shows small-cap effect in crypto",
            candidate_type=CandidateType.EXPLORATORY,
            recipe="rank(-log(mcap))",
        )
        trace.append_round(
            round_num=1,
            candidates=[candidate],
            round_summary="Round 1: explored size, vol, range",
            hold_pool=["h1_smallcap"],
            search_direction="Refine small-cap with range interaction"
        )

        is_valid, msg = trace.verify_integrity()
    """

    def __init__(
        self,
        session_id: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_TRACE_DIR
        self.trace_path = self.base_dir / f"{self.session_id}.jsonl"
        self._config_hash: str | None = None
        self._rounds: list[RoundEntry] = []
        self._all_candidate_ids: set[str] = set()

    def start_session(self, config_hash: str, config_snapshot: dict | None = None) -> None:
        """Initialize the trace with protocol config hash."""
        self._config_hash = config_hash
        self.base_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = self.base_dir / f"{self.session_id}_config.json"
        if config_snapshot:
            with open(snapshot_path, "w") as f:
                json.dump(config_snapshot, f, indent=2, default=str)

    def append_round(
        self,
        round_num: int,
        candidates: list[CandidateEntry],
        round_summary: str = "",
        hold_pool: list[str] | None = None,
        good_pool: list[str] | None = None,
        search_direction: str = "",
    ) -> None:
        """Append a discovery round to the trace. Raises if duplicate candidate IDs."""
        for c in candidates:
            if c.id in self._all_candidate_ids:
                raise ValueError(
                    f"Duplicate candidate ID '{c.id}'. Each candidate must have a unique ID. "
                    "The trace is append-only and immutable."
                )
            self._all_candidate_ids.add(c.id)

        entry = RoundEntry(
            round_num=round_num,
            timestamp=datetime.now(timezone.utc).isoformat(),
            candidates=candidates,
            round_summary=round_summary,
            hold_pool=hold_pool or [],
            good_pool=good_pool or [],
            search_direction=search_direction,
        )
        self._rounds.append(entry)

        line = {
            "session_id": self.session_id,
            "protocol_hash": self._config_hash,
            **entry.to_dict(),
        }

        with open(self.trace_path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")

        logger.info(
            "Round %d appended: %d candidates, hold=%d, good=%d",
            round_num,
            len(candidates),
            len(entry.hold_pool),
            len(entry.good_pool),
        )

    def load(self) -> list[dict]:
        """Load all trace entries from disk."""
        if not self.trace_path.exists():
            return []
        entries: list[dict] = []
        with open(self.trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def get_hold_pool(self) -> list[str]:
        """Get current hold pool from the latest round."""
        entries = self.load()
        if not entries:
            return []
        return entries[-1].get("hold_pool", [])

    def get_good_pool(self) -> list[str]:
        """Get current good pool from the latest round."""
        entries = self.load()
        if not entries:
            return []
        return entries[-1].get("good_pool", [])

    def get_passed_candidates(self) -> list[dict]:
        """Get all candidates that passed the gate across all rounds."""
        entries = self.load()
        passed: list[dict] = []
        for entry in entries:
            for c in entry.get("candidates", []):
                if c.get("gate_result") == "PASS":
                    passed.append(c)
        return passed

    def get_failed_candidates(self) -> list[dict]:
        """Get all candidates that failed the gate, with failure categories."""
        entries = self.load()
        failed: list[dict] = []
        for entry in entries:
            for c in entry.get("candidates", []):
                if c.get("gate_result") == "FAIL":
                    failed.append(c)
        return failed

    def get_round_summary(self, round_num: int) -> dict | None:
        """Get summary for a specific round."""
        entries = self.load()
        for entry in entries:
            if entry.get("round") == round_num:
                return entry
        return None

    def candidate_count(self) -> int:
        """Total number of candidates across all rounds."""
        return len(self._all_candidate_ids)

    def round_count(self) -> int:
        """Total number of rounds."""
        return len(self._rounds) or len(self.load())

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify trace integrity: monotonic rounds, no duplicate IDs, consistent hashes.

        Returns:
            (is_valid, message)
        """
        entries = self.load()
        if not entries:
            return True, "Empty trace (no entries)"

        errors: list[str] = []

        # Check monotonic round numbers
        rounds = [e.get("round", 0) for e in entries]
        if rounds != sorted(rounds):
            errors.append("Round numbers are not monotonic")

        # Check no duplicate candidate IDs
        all_ids: list[str] = []
        for entry in entries:
            for c in entry.get("candidates", []):
                cid = c.get("id", "")
                if cid:
                    all_ids.append(cid)
        if len(all_ids) != len(set(all_ids)):
            errors.append("Duplicate candidate IDs found")

        # Check protocol hash consistency
        if self._config_hash:
            for entry in entries:
                if entry.get("protocol_hash") != self._config_hash:
                    errors.append(
                        f"Protocol hash mismatch at round {entry.get('round')}: "
                        f"expected {self._config_hash}, got {entry.get('protocol_hash')}"
                    )

        if errors:
            return False, "; ".join(errors)
        return True, "Trace integrity verified"

    def summary_table(self) -> str:
        """Generate a formatted summary table of the trace."""
        entries = self.load()
        lines: list[str] = []
        lines.append(f"Trace: {self.session_id}")
        lines.append(f"Rounds: {len(entries)}")
        lines.append("-" * 80)

        total_pass = 0
        total_fail = 0
        for entry in entries:
            r = entry.get("round", "?")
            n = len(entry.get("candidates", []))
            n_pass = sum(1 for c in entry.get("candidates", []) if c.get("gate_result") == "PASS")
            n_fail = n - n_pass
            total_pass += n_pass
            total_fail += n_fail
            direction = entry.get("search_direction", "")[:50]
            lines.append(
                f"Round {r}: {n} candidates ({n_pass} pass, {n_fail} fail) | "
                f"Hold={len(entry.get('hold_pool', []))} | "
                f"Good={len(entry.get('good_pool', []))} | "
                f"→ {direction}"
            )

        lines.append("-" * 80)
        lines.append(
            f"Total: {total_pass + total_fail} candidates ({total_pass} pass, {total_fail} fail)"
        )
        return "\n".join(lines)


def compute_protocol_hash(config: dict) -> str:
    """Compute a deterministic hash of the evaluation protocol config."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def curate_good_pool(
    hold_pool: list[dict],
    max_factors: int = 10,
    min_corr_threshold: float = 0.7,
    corr_matrix: np.ndarray | None = None,
) -> list[str]:
    """Curate a good pool from the hold pool using diversity enforcement.

    Sorts candidates by training Sharpe (descending), then greedily selects
    candidates that are not highly correlated with already-selected ones.

    Args:
        hold_pool: List of candidate dicts with 'id' and metrics.sharpe_ls keys.
        max_factors: Maximum number of factors in the good pool.
        min_corr_threshold: Skip candidates with |corr| > this to any selected.
        corr_matrix: (N, N) correlation matrix of candidate scores. If None,
            assumes candidates are sorted and no correlation check is done.

    Returns:
        List of candidate IDs in the good pool.
    """
    sorted_pool = sorted(
        hold_pool,
        key=lambda c: (
            c.get("metrics", {}).get("sharpe_ls", 0)
            if isinstance(c.get("metrics"), dict)
            else getattr(c.get("metrics", None), "sharpe_ls", 0) or 0
        ),
        reverse=True,
    )

    good_pool: list[str] = []
    for idx, candidate in enumerate(sorted_pool):
        cid = candidate.get("id", "") if isinstance(candidate, dict) else candidate.id
        if corr_matrix is not None and len(good_pool) > 0:
            candidate_idx = idx
            too_correlated = False
            for selected_idx, selected_id in enumerate(good_pool):
                selected_pool_idx = next(
                    (i for i, c in enumerate(sorted_pool) if c.get("id") == selected_id),
                    None,
                )
                if (
                    selected_pool_idx is not None
                    and abs(corr_matrix[candidate_idx, selected_pool_idx]) > min_corr_threshold
                ):
                    too_correlated = True
                    break
            if too_correlated:
                continue

        good_pool.append(cid)
        if len(good_pool) >= max_factors:
            break

    return good_pool
