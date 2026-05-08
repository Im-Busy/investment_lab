"""
Epistemic Autopsy Module - Post-mortem analysis of why strategies fail.

This module performs systematic analysis of trade outcomes to classify
failure modes, identify regime mismatches, and distinguish systematic
from random losses.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# Failure mode constants
FAILURE_REGIME_MISMATCH = "regime_mismatch"
FAILURE_WHIPSAW = "whipsaw"
FAILURE_FAT_TAIL = "fat_tail"
FAILURE_SIGNAL_DECAY = "signal_decay"
FAILURE_CORRELATION_COLLAPSE = "correlation_collapse"
FAILURE_EXECUTION_SLIPPAGE = "execution_slippage"
FAILURE_OVERFIT = "overfit"

ALL_FAILURE_MODES = [
    FAILURE_REGIME_MISMATCH,
    FAILURE_WHIPSAW,
    FAILURE_FAT_TAIL,
    FAILURE_SIGNAL_DECAY,
    FAILURE_CORRELATION_COLLAPSE,
    FAILURE_EXECUTION_SLIPPAGE,
    FAILURE_OVERFIT,
]

# Threshold constants
PNL_PCT_THRESHOLD_FAT_TAIL = -5.0
SLIPPAGE_RATIO_THRESHOLD = 2.0
VOLATILITY_REGIME_THRESHOLD = 0.02
HOLDING_PERIOD_DECAY_BARS = 10
CONFIDENCE_LOW_THRESHOLD = 0.5
REGIME_INCOMPATIBLE_PENALTY = 0.3
WHIPSAW_REVERSAL_THRESHOLD = 0.6
MAX_DRAWDOWN_FROM_ENTRY = -3.0
SYSTEMATIC_LOSS_FREQUENCY_THRESHOLD = 0.4


@dataclass
class AutopsyResult:
    """Result of a single trade autopsy."""

    trade_id: str
    outcome: str
    expected_outcome: str
    actual_pnl: float
    failure_mode: Optional[str]
    failure_evidence: Dict[str, float]
    regime_at_entry: Optional[str]
    holding_period: int
    confidence_at_entry: float


class EpistemicAutopsy:
    """Performs epistemic autopsies on trade outcomes to classify failure modes."""

    def __init__(self, regime_incompatible_map: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the autopsy analyzer.

        Args:
            regime_incompatible_map: Optional dict mapping regime names to
                incompatible pattern categories. If None, uses default mapping.
        """
        self._regime_incompatible_map = regime_incompatible_map or self._build_default_regime_map()

    @staticmethod
    def _build_default_regime_map() -> Dict[str, List[str]]:
        """Build default regime incompatibility map."""
        return {
            "Ranging": ["breakout", "momentum", "trend_following"],
            "Trending": ["mean_reversion", "range_bound"],
            "Volatile": ["tight_stop", "low_volatility_breakout"],
            "Transition": ["continuation", "breakout"],
        }

    def analyze_trade_result(self, trade: dict, market_state: dict) -> AutopsyResult:
        """
        Analyze a single closed trade and determine why it won or lost.

        Args:
            trade: Trade dict with entry_price, exit_price, pnl, pnl_pct,
                direction, entry_time, exit_time, patterns, confidence, regime.
            market_state: Dict with regime, volatility, volume, max_adverse_excursion,
                max_favorable_excursion, slippage_bps.

        Returns:
            AutopsyResult with classification and evidence.
        """
        trade_id = trade.get("id", trade.get("trade_id", "unknown"))
        actual_pnl = trade.get("pnl", 0.0)
        pnl_pct = trade.get("pnl_pct", 0.0)
        direction = trade.get("direction", "Long")
        confidence = trade.get("confidence", 0.5)
        regime_at_entry = trade.get("regime", market_state.get("regime"))

        holding_period = self._calculate_holding_period(trade)
        expected_outcome = self._determine_expected_outcome(trade, market_state)
        outcome = self._classify_outcome(actual_pnl, pnl_pct)

        failure_mode = None
        failure_evidence: Dict[str, float] = {}

        if outcome == "LOSS":
            failure_mode, failure_evidence = self._classify_failure_mode(
                trade, market_state, pnl_pct, direction, confidence, regime_at_entry
            )

        return AutopsyResult(
            trade_id=trade_id,
            outcome=outcome,
            expected_outcome=expected_outcome,
            actual_pnl=actual_pnl,
            failure_mode=failure_mode,
            failure_evidence=failure_evidence,
            regime_at_entry=regime_at_entry,
            holding_period=holding_period,
            confidence_at_entry=confidence,
        )

    def analyze_strategy_performance(self, trades: list, benchmark_return: float) -> dict:
        """
        Overall strategy post-mortem comparing against benchmark.

        Args:
            trades: List of trade dicts with pnl, entry_time, exit_time.
            benchmark_return: Benchmark return (e.g., SPY) as percentage.

        Returns:
            Dict with strategy_return, excess_return, failure_mode_breakdown,
            win_rate, avg_loss_when_wrong, avg_win_when_right,
            systematic_loss_pct, random_loss_pct, regime_performance.
        """
        if not trades:
            return self._empty_strategy_report()

        autopsies = []
        for trade in trades:
            market_state = self._extract_market_state(trade)
            autopsy = self.analyze_trade_result(trade, market_state)
            autopsies.append(autopsy)

        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        initial_equity = self._estimate_initial_equity(trades)
        strategy_return = (total_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0
        excess_return = strategy_return - benchmark_return

        wins = [a for a in autopsies if a.outcome == "WIN"]
        losses = [a for a in autopsies if a.outcome == "LOSS"]
        breakevens = [a for a in autopsies if a.outcome == "BREAKEVEN"]

        win_rate = len(wins) / len(autopsies) if autopsies else 0.0

        failure_breakdown = self._count_failure_modes(autopsies)
        systematic_losses, random_losses = self._classify_loss_types(autopsies)

        total_losses = len(losses)
        systematic_pct = systematic_losses / total_losses if total_losses > 0 else 0.0
        random_pct = random_losses / total_losses if total_losses > 0 else 0.0

        avg_loss = sum(a.actual_pnl for a in losses) / len(losses) if losses else 0.0
        avg_win = sum(a.actual_pnl for a in wins) / len(wins) if wins else 0.0

        regime_performance = self._analyze_regime_performance(autopsies)

        return {
            "total_trades": len(trades),
            "strategy_return_pct": round(strategy_return, 4),
            "benchmark_return_pct": round(benchmark_return, 4),
            "excess_return_pct": round(excess_return, 4),
            "win_rate": round(win_rate, 4),
            "num_wins": len(wins),
            "num_losses": len(losses),
            "num_breakevens": len(breakevens),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "failure_mode_breakdown": failure_breakdown,
            "systematic_loss_pct": round(systematic_pct, 4),
            "random_loss_pct": round(random_pct, 4),
            "regime_performance": regime_performance,
        }

    def identify_failure_modes(self, trades: list) -> dict:
        """
        Categorize trades into failure modes with frequency analysis.

        Args:
            trades: List of trade dicts to analyze.

        Returns:
            Dict with mode_counts, mode_frequencies, dominant_failure_mode,
            mode_details (list of trades per mode), systematic_vs_random.
        """
        if not trades:
            return self._empty_failure_report()

        autopsies = []
        for trade in trades:
            market_state = self._extract_market_state(trade)
            autopsy = self.analyze_trade_result(trade, market_state)
            autopsies.append(autopsy)

        losses = [a for a in autopsies if a.outcome == "LOSS"]
        mode_counts = self._count_failure_modes(autopsies)
        total_trades = len(autopsies)
        total_losses = len(losses)

        mode_frequencies = {
            mode: count / total_trades if total_trades > 0 else 0.0
            for mode, count in mode_counts.items()
        }

        dominant_mode = (
            max(mode_counts, key=mode_counts.get) if mode_counts and total_losses > 0 else None
        )

        mode_details = self._group_trades_by_failure_mode(autopsies)
        systematic, random = self._classify_loss_types(autopsies)

        return {
            "total_trades_analyzed": total_trades,
            "total_losses": total_losses,
            "mode_counts": mode_counts,
            "mode_frequencies_pct": {k: round(v * 100, 2) for k, v in mode_frequencies.items()},
            "dominant_failure_mode": dominant_mode,
            "mode_details": mode_details,
            "systematic_losses": systematic,
            "random_losses": random,
            "systematic_pct": round(systematic / total_losses if total_losses > 0 else 0.0, 4)
            * 100,
        }

    def generate_autopsy_report(self, trades: list, strategy_name: str) -> str:
        """
        Generate human-readable autopsy report.

        Args:
            trades: List of trade dicts to analyze.
            strategy_name: Name of the strategy being analyzed.

        Returns:
            Formatted markdown report string.
        """
        if not trades:
            return f"# Epistemic Autopsy Report: {strategy_name}\n\nNo trades to analyze."

        autopsies = []
        for trade in trades:
            market_state = self._extract_market_state(trade)
            autopsy = self.analyze_trade_result(trade, market_state)
            autopsies.append(autopsy)

        losses = [a for a in autopsies if a.outcome == "LOSS"]
        wins = [a for a in autopsies if a.outcome == "WIN"]
        breakevens = [a for a in autopsies if a.outcome == "BREAKEVEN"]

        mode_counts = self._count_failure_modes(autopsies)
        systematic, random = self._classify_loss_types(autopsies)
        total_losses = len(losses)
        systematic_pct = systematic / total_losses * 100 if total_losses > 0 else 0.0
        random_pct = random / total_losses * 100 if total_losses > 0 else 0.0

        win_rate = len(wins) / len(autopsies) * 100 if autopsies else 0.0
        total_pnl = sum(a.actual_pnl for a in autopsies)

        lines = [
            f"# Epistemic Autopsy Report: {strategy_name}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Trades:** {len(autopsies)}",
            f"- **Win Rate:** {win_rate:.1f}%",
            f"- **Total P&L:** {total_pnl:,.2f}",
            f"- **Wins:** {len(wins)} | **Losses:** {len(losses)} | **Breakevens:** {len(breakevens)}",
            "",
            "## Failure Mode Breakdown",
            "",
        ]

        if mode_counts:
            lines.append("| Failure Mode | Count | % of Losses |")
            lines.append("|---|---|---|")
            for mode, count in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True):
                pct_of_losses = count / total_losses * 100 if total_losses > 0 else 0.0
                display_name = mode.replace("_", " ").title()
                lines.append(f"| {display_name} | {count} | {pct_of_losses:.1f}% |")
        else:
            lines.append("No failure modes detected.")

        lines.extend(
            [
                "",
                "## Systematic vs Random Losses",
                "",
                f"- **Systematic Losses:** {systematic} ({systematic_pct:.1f}%)",
                f"- **Random Losses:** {random} ({random_pct:.1f}%)",
                "",
            ]
        )

        if systematic_pct > 50:
            lines.append(
                "> **WARNING:** Majority of losses are systematic. Strategy logic requires review."
            )
        elif systematic_pct > 30:
            lines.append(
                "> **CAUTION:** Significant systematic component detected. "
                "Consider regime filters or position sizing adjustments."
            )

        lines.extend(self._build_regime_section(autopsies))
        lines.extend(self._build_low_confidence_section(losses))
        lines.extend(self._build_top_losses_section(losses))

        return "\n".join(lines)

    def _classify_failure_mode(
        self,
        trade: dict,
        market_state: dict,
        pnl_pct: float,
        direction: str,
        confidence: float,
        regime_at_entry: Optional[str],
    ) -> tuple:
        """
        Classify the failure mode for a losing trade.

        Args:
            trade: Trade dict with pattern and price data.
            market_state: Market context at trade time.
            pnl_pct: Trade P&L as percentage.
            direction: Trade direction (Long/Short).
            confidence: Signal confidence at entry.
            regime_at_entry: Market regime at entry.

        Returns:
            Tuple of (failure_mode_name, evidence_dict).
        """
        evidence: Dict[str, float] = {}

        regime_score = self._score_regime_mismatch(trade, market_state, regime_at_entry)
        evidence["regime_mismatch_score"] = regime_score

        whipsaw_score = self._score_whipsaw(trade, direction)
        evidence["whipsaw_score"] = whipsaw_score

        fat_tail_score = self._score_fat_tail(pnl_pct, market_state)
        evidence["fat_tail_score"] = fat_tail_score

        signal_decay_score = self._score_signal_decay(trade)
        evidence["signal_decay_score"] = signal_decay_score

        slippage_score = self._score_slippage(trade, market_state)
        evidence["slippage_score"] = slippage_score

        scores = {
            FAILURE_REGIME_MISMATCH: regime_score,
            FAILURE_WHIPSAW: whipsaw_score,
            FAILURE_FAT_TAIL: fat_tail_score,
            FAILURE_SIGNAL_DECAY: signal_decay_score,
            FAILURE_EXECUTION_SLIPPAGE: slippage_score,
        }

        primary_mode = max(scores, key=scores.get)
        primary_score = scores[primary_mode]

        if primary_score < 0.3:
            evidence["classification_confidence"] = 0.0
            return None, evidence

        evidence["classification_confidence"] = primary_score
        return primary_mode, evidence

    def _score_regime_mismatch(
        self,
        trade: dict,
        market_state: dict,
        regime_at_entry: Optional[str],
    ) -> float:
        """Score how much a regime mismatch contributed to the loss (0.0-1.0)."""
        if not regime_at_entry:
            return 0.0

        patterns = trade.get("patterns", [])
        pattern_name = trade.get("pattern", "")
        if pattern_name and pattern_name not in patterns:
            patterns = [pattern_name] + patterns

        if not patterns:
            return 0.0

        incompatible = self._regime_incompatible_map.get(regime_at_entry, [])
        mismatches = sum(1 for p in patterns if any(inc in p.lower() for inc in incompatible))

        return min(mismatches / max(len(patterns), 1), 1.0)

    def _score_whipsaw(self, trade: dict, direction: str) -> float:
        """Score how likely a whipsaw caused the loss (0.0-1.0)."""
        max_adverse = trade.get("max_adverse_excursion", 0.0)
        max_favorable = trade.get("max_favorable_excursion", 0.0)

        if max_favorable == 0 and max_adverse == 0:
            return 0.0

        pnl_pct = trade.get("pnl_pct", 0.0)
        if pnl_pct >= 0:
            return 0.0

        if direction == "Long":
            if max_favorable > 0 and max_adverse < 0:
                reversal_ratio = abs(max_adverse) / (max_favorable + abs(max_adverse))
                return reversal_ratio
        else:
            if max_adverse > 0 and max_favorable < 0:
                reversal_ratio = max_adverse / (max_adverse + abs(max_favorable))
                return reversal_ratio

        return 0.0

    def _score_fat_tail(self, pnl_pct: float, market_state: dict) -> float:
        """Score how likely a fat tail event caused the loss (0.0-1.0)."""
        if pnl_pct > PNL_PCT_THRESHOLD_FAT_TAIL:
            return 0.0

        volatility = market_state.get("volatility", 0.01)
        z_score = abs(pnl_pct) / (volatility * 100) if volatility > 0 else 0

        if z_score > 3.0:
            return min(z_score / 5.0, 1.0)

        return 0.0

    def _score_signal_decay(self, trade: dict) -> float:
        """Score how likely signal decay caused the loss (0.0-1.0)."""
        holding_period = self._calculate_holding_period(trade)

        if holding_period <= 0:
            return 0.0

        decay_ratio = holding_period / HOLDING_PERIOD_DECAY_BARS
        return min(decay_ratio, 1.0) if decay_ratio > 1.0 else 0.0

    def _score_slippage(self, trade: dict, market_state: dict) -> float:
        """Score how much slippage contributed to the loss (0.0-1.0)."""
        expected_slippage = trade.get("expected_slippage_bps", 10)
        actual_slippage = market_state.get("slippage_bps", expected_slippage)

        if expected_slippage <= 0:
            return 0.0

        slippage_ratio = actual_slippage / expected_slippage
        if slippage_ratio > SLIPPAGE_RATIO_THRESHOLD:
            return min((slippage_ratio - 1) / SLIPPAGE_RATIO_THRESHOLD, 1.0)

        return 0.0

    def _classify_outcome(self, actual_pnl: float, pnl_pct: float) -> str:
        """Classify trade outcome as WIN, LOSS, or BREAKEVEN."""
        pnl_threshold = 0.01
        if abs(actual_pnl) < pnl_threshold and abs(pnl_pct) < 0.001:
            return "BREAKEVEN"
        if actual_pnl > 0 or pnl_pct > 0:
            return "WIN"
        return "LOSS"

    def _determine_expected_outcome(self, trade: dict, market_state: dict) -> str:
        """
        Determine what outcome the strategy predicted based on signals.

        Args:
            trade: Trade dict with confidence and patterns.
            market_state: Market context.

        Returns:
            "WIN_EXPECTED" or "LOSS_EXPECTED" or "NEUTRAL".
        """
        confidence = trade.get("confidence", 0.5)

        if confidence > CONFIDENCE_LOW_THRESHOLD:
            return "WIN_EXPECTED"
        elif confidence < 0.3:
            return "LOSS_EXPECTED"
        return "NEUTRAL"

    def _calculate_holding_period(self, trade: dict) -> int:
        """Calculate holding period in bars/trading days."""
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")

        if entry_time and exit_time:
            try:
                import pandas as pd

                entry = pd.to_datetime(entry_time)
                exit_dt = pd.to_datetime(exit_time)
                delta = exit_dt - entry
                trading_days = delta.days
                return max(trading_days, 0)
            except (ValueError, TypeError):
                pass

        return trade.get("holding_period_bars", 0)

    def _extract_market_state(self, trade: dict) -> dict:
        """Extract or synthesize market state from trade dict."""
        return {
            "regime": trade.get("regime"),
            "volatility": trade.get("volatility", 0.015),
            "volume": trade.get("volume_ratio", 1.0),
            "max_adverse_excursion": trade.get("max_adverse_excursion", 0.0),
            "max_favorable_excursion": trade.get("max_favorable_excursion", 0.0),
            "slippage_bps": trade.get("slippage_bps", 10),
        }

    def _count_failure_modes(self, autopsies: list) -> Dict[str, int]:
        """Count occurrences of each failure mode."""
        counts: Dict[str, int] = {}
        for a in autopsies:
            if a.failure_mode:
                counts[a.failure_mode] = counts.get(a.failure_mode, 0) + 1
        return counts

    def _classify_loss_types(self, autopsies: list) -> tuple:
        """
        Classify losses into systematic vs random.

        Systematic losses have a clear failure mode with high evidence.
        Random losses are unexplained regime-independent losses.

        Returns:
            Tuple of (systematic_count, random_count).
        """
        systematic = 0
        random = 0

        for a in autopsies:
            if a.outcome != "LOSS":
                continue

            if (
                a.failure_mode
                and a.failure_evidence.get("classification_confidence", 0)
                > SYSTEMATIC_LOSS_FREQUENCY_THRESHOLD
            ):
                systematic += 1
            else:
                random += 1

        return systematic, random

    def _group_trades_by_failure_mode(self, autopsies: list) -> Dict[str, List[Dict[str, Any]]]:
        """Group trade IDs by their failure mode."""
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for a in autopsies:
            if a.failure_mode:
                if a.failure_mode not in groups:
                    groups[a.failure_mode] = []
                groups[a.failure_mode].append(
                    {
                        "trade_id": a.trade_id,
                        "pnl": a.actual_pnl,
                        "confidence": a.confidence_at_entry,
                        "regime": a.regime_at_entry,
                        "evidence": a.failure_evidence,
                    }
                )

        return groups

    def _analyze_regime_performance(self, autopsies: list) -> Dict[str, Dict[str, Any]]:
        """Analyze performance broken down by regime."""
        regime_stats: Dict[str, Dict[str, Any]] = {}

        for a in autopsies:
            regime = a.regime_at_entry or "Unknown"
            if regime not in regime_stats:
                regime_stats[regime] = {
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "failure_modes": {},
                }

            stats = regime_stats[regime]
            stats["total_trades"] += 1
            stats["total_pnl"] += a.actual_pnl

            if a.outcome == "WIN":
                stats["wins"] += 1
            elif a.outcome == "LOSS":
                stats["losses"] += 1
                if a.failure_mode:
                    fm = a.failure_mode
                    stats["failure_modes"][fm] = stats["failure_modes"].get(fm, 0) + 1

        for stats in regime_stats.values():
            total = stats["total_trades"]
            stats["win_rate"] = stats["wins"] / total if total > 0 else 0.0
            stats["avg_pnl"] = stats["total_pnl"] / total if total > 0 else 0.0

        return regime_stats

    def _estimate_initial_equity(self, trades: list) -> float:
        """Estimate initial equity from trade data."""
        if not trades:
            return 0.0

        total_abs_pnl = sum(abs(t.get("pnl", 0.0)) for t in trades)
        avg_position_size = total_abs_pnl / len(trades) if trades else 0.0

        return avg_position_size * 10 if avg_position_size > 0 else 10000.0

    def _empty_strategy_report(self) -> dict:
        """Return empty strategy report structure."""
        return {
            "total_trades": 0,
            "strategy_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "excess_return_pct": 0.0,
            "win_rate": 0.0,
            "num_wins": 0,
            "num_losses": 0,
            "num_breakevens": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "failure_mode_breakdown": {},
            "systematic_loss_pct": 0.0,
            "random_loss_pct": 0.0,
            "regime_performance": {},
        }

    def _empty_failure_report(self) -> dict:
        """Return empty failure report structure."""
        return {
            "total_trades_analyzed": 0,
            "total_losses": 0,
            "mode_counts": {},
            "mode_frequencies_pct": {},
            "dominant_failure_mode": None,
            "mode_details": {},
            "systematic_losses": 0,
            "random_losses": 0,
            "systematic_pct": 0.0,
        }

    def _build_regime_section(self, autopsies: list) -> List[str]:
        """Build regime performance section for report."""
        regime_perf = self._analyze_regime_performance(autopsies)
        if not regime_perf:
            return []

        lines = [
            "",
            "## Performance by Regime",
            "",
            "| Regime | Trades | Win Rate | Avg P&L |",
            "|---|---|---|---|",
        ]

        for regime, stats in sorted(
            regime_perf.items(), key=lambda x: x[1]["win_rate"], reverse=True
        ):
            wr = stats["win_rate"] * 100
            avg = stats["avg_pnl"]
            lines.append(f"| {regime} | {stats['total_trades']} | {wr:.1f}% | {avg:,.2f} |")

        return lines

    def _build_low_confidence_section(self, losses: list) -> List[str]:
        """Build section for losses with low entry confidence."""
        low_conf = [a for a in losses if a.confidence_at_entry < CONFIDENCE_LOW_THRESHOLD]

        if not low_conf:
            return []

        lines = [
            "",
            "## Low-Confidence Losses",
            "",
            f"{len(low_conf)} losses occurred with confidence "
            f"below {CONFIDENCE_LOW_THRESHOLD:.0%}. "
            "Consider raising entry confidence thresholds.",
            "",
        ]

        return lines

    def _build_top_losses_section(self, losses: list) -> List[str]:
        """Build section showing top losses by magnitude."""
        if not losses:
            return []

        sorted_losses = sorted(losses, key=lambda a: a.actual_pnl)[:5]

        lines = [
            "",
            "## Top Losses",
            "",
            "| Trade ID | P&L | Failure Mode | Confidence | Regime |",
            "|---|---|---|---|---|",
        ]

        for loss in sorted_losses:
            mode = loss.failure_mode or "N/A"
            mode_display = mode.replace("_", " ").title()
            lines.append(
                f"| {loss.trade_id} | {loss.actual_pnl:,.2f} | "
                f"{mode_display} | {loss.confidence_at_entry:.2f} | "
                f"{loss.regime_at_entry or 'N/A'} |"
            )

        return lines
