"""
GPU-Accelerated Signal Aggregation

Applies GPU-friendly principles to multi-pattern signal aggregation:
1. Data parallelism: Conflict resolution and aggregation across ALL bars simultaneously
2. Minimal branching: Mask-based direction selection, tensor where() replaces if/else
3. Coalesced memory: Direction/confidence matrices are contiguous [N, P] tensors
4. Batched transfers: Aggregation happens entirely on-device; only final results copied
5. Tensor cores: Weighted averages via matmul-friendly operations
6. Maximized occupancy: PyTorch native reductions (sum, mean, max) use optimized kernels

This replaces the CPU SignalGenerator._aggregate_signals() and
EventWeightedAggregator.aggregate() loops with batched tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from .pattern_detector import GPUPatternDetector
from .utils import get_device, to_numpy


@dataclass
class GPUAggregatedSignals:
    """Batch of aggregated signals across all bars."""

    direction: torch.Tensor  # [N] int8: -1, 0, 1
    confidence: torch.Tensor  # [N] float32
    entry_price: torch.Tensor
    stop_loss: torch.Tensor
    take_profit_1: torch.Tensor
    take_profit_2: torch.Tensor
    take_profit_3: torch.Tensor
    pattern_count: torch.Tensor  # [N] int32: number of agreeing patterns
    dominant_event_count: torch.Tensor  # [N] int32: max same-event patterns
    event_weighted: torch.Tensor  # [N] float32: average event weight
    confluence_boost: torch.Tensor  # [N] float32

    def to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Export to pandas DataFrame for backtesting."""
        dir_np = to_numpy(self.direction)
        conf_np = to_numpy(self.confidence)
        entry_np = to_numpy(self.entry_price)
        stop_np = to_numpy(self.stop_loss)
        tp1_np = to_numpy(self.take_profit_1)
        tp2_np = to_numpy(self.take_profit_2)
        tp3_np = to_numpy(self.take_profit_3)
        count_np = to_numpy(self.pattern_count)
        event_np = to_numpy(self.event_weighted)
        boost_np = to_numpy(self.confluence_boost)

        records = []
        for i, d in enumerate(dir_np):
            if d != 0 and conf_np[i] > 0:
                records.append(
                    {
                        "date": df.index[i],
                        "bar_index": i,
                        "direction": "Long" if d > 0 else "Short",
                        "confidence": conf_np[i],
                        "entry_price": entry_np[i],
                        "stop_loss": stop_np[i],
                        "take_profit_1": tp1_np[i],
                        "take_profit_2": tp2_np[i],
                        "take_profit_3": tp3_np[i],
                        "pattern_count": count_np[i],
                        "dominant_event_count": to_numpy(self.dominant_event_count)[i],
                        "event_weighted": event_np[i],
                        "confluence_boost": boost_np[i],
                    }
                )

        return pd.DataFrame(records)


class GPUSignalAggregator:
    """
    GPU-accelerated signal aggregation with event-type weighting.

    Takes a signal matrix [N, P] (N bars, P patterns) where each entry
    is -1 (short), 0 (neutral), or 1 (long), plus a confidence matrix [N, P].

    Operates entirely on GPU — no per-bar Python loops.

    Usage:
        detector = GPUPatternDetector(device)
        detector.load_data(df)
        detector.detect_all()

        aggregator = GPUSignalAggregator(device)
        aggregator.load_signals(detector)
        result = aggregator.aggregate(min_confidence=0.3)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        event_type_weights: Optional[Dict[str, float]] = None,
        pattern_event_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            device: PyTorch device
            event_type_weights: Custom event weight overrides
            pattern_event_map: Custom pattern-to-event mapping
        """
        self.device = device or get_device()
        self._signal_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._confidence_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._entry_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._stop_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._tp1_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._tp2_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._tp3_matrix: Optional[torch.Tensor] = None  # [N, P]
        self._pattern_names: List[str] = []
        self._n_bars: int = 0
        self._event_type_weights = event_type_weights or {}
        self._pattern_event_map = pattern_event_map or {}

        self._default_event_weights = {
            "REVERSAL": 0.65,
            "BREAKOUT": 0.70,
            "CONTINUATION": 0.60,
            "TREND_INITIATION": 0.75,
            "MEAN_REVERSION": 0.55,
            "MOMENTUM": 0.65,
            "VOLATILITY": 0.50,
            "STRUCTURAL": 0.70,
        }
        self._default_pattern_events = {
            "MSL": "REVERSAL",
            "MSH": "REVERSAL",
            "Doji": "REVERSAL",
            "Hammer": "REVERSAL",
            "Engulfing": "REVERSAL",
            "Harami": "REVERSAL",
            "DarkCloudCover": "REVERSAL",
            "TwoBarReversal": "REVERSAL",
            "DonchianBreakout": "BREAKOUT",
            "Gap": "BREAKOUT",
            "NR7ID": "VOLATILITY",
        }

    def load_signals(self, detector: GPUPatternDetector) -> None:
        """
        Load all pattern signals from GPU detector into contiguous matrices.

        Principle 3 (Coalesced Memory): Signal/confidence/entry/stop matrices
        are all contiguous [N, P] tensors for optimal GPU memory access.
        """
        signal_batches = detector.detect_all()

        self._pattern_names = []
        signals_list = []
        conf_list = []
        entry_list = []
        stop_list = []
        tp1_list = []
        tp2_list = []
        tp3_list = []

        self._n_bars = detector._n_bars

        for sb in signal_batches:
            self._pattern_names.append(sb.pattern_name)
            signals_list.append(sb.direction.float())
            conf_list.append(sb.confidence)
            entry_list.append(sb.entry_price)
            stop_list.append(sb.stop_loss)
            tp1_list.append(sb.take_profit_1)
            tp2 = sb.take_profit_2
            tp3 = sb.take_profit_3
            tp2_list.append(
                tp2 if tp2 is not None else torch.zeros(self._n_bars, device=self.device)
            )
            tp3_list.append(
                tp3 if tp3 is not None else torch.zeros(self._n_bars, device=self.device)
            )

        self._signal_matrix = torch.stack(signals_list, dim=1)  # [N, P]
        self._confidence_matrix = torch.stack(conf_list, dim=1)
        self._entry_matrix = torch.stack(entry_list, dim=1)
        self._stop_matrix = torch.stack(stop_list, dim=1)
        self._tp1_matrix = torch.stack(tp1_list, dim=1)
        self._tp2_matrix = torch.stack(tp2_list, dim=1)
        self._tp3_matrix = torch.stack(tp3_list, dim=1)

    def _get_event_weight_vector(self) -> torch.Tensor:
        """Build [P] tensor of event type base weights for each pattern."""
        weights = torch.zeros(len(self._pattern_names), device=self.device)
        for i, name in enumerate(self._pattern_names):
            event = self._pattern_event_map.get(name) or self._default_pattern_events.get(
                name, "REVERSAL"
            )
            w = self._event_type_weights.get(event) or self._default_event_weights.get(event, 0.5)
            weights[i] = w
        return weights

    def _get_event_type_indices(self) -> torch.Tensor:
        """Build [P] tensor of event type index (0-7) for each pattern."""
        event_types = list(self._default_event_weights.keys())
        indices = torch.zeros(len(self._pattern_names), dtype=torch.int32, device=self.device)
        for i, name in enumerate(self._pattern_names):
            event = self._pattern_event_map.get(name) or self._default_pattern_events.get(
                name, "REVERSAL"
            )
            try:
                indices[i] = event_types.index(event)
            except ValueError:
                indices[i] = 0
        return indices

    @property
    def signal_matrix(self) -> torch.Tensor:
        if self._signal_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._signal_matrix

    @property
    def confidence_matrix(self) -> torch.Tensor:
        if self._confidence_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._confidence_matrix

    @property
    def entry_matrix(self) -> torch.Tensor:
        if self._entry_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._entry_matrix

    @property
    def stop_matrix(self) -> torch.Tensor:
        if self._stop_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._stop_matrix

    @property
    def tp1_matrix(self) -> torch.Tensor:
        if self._tp1_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._tp1_matrix

    @property
    def tp2_matrix(self) -> torch.Tensor:
        if self._tp2_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._tp2_matrix

    @property
    def tp3_matrix(self) -> torch.Tensor:
        if self._tp3_matrix is None:
            raise RuntimeError("load_signals() must be called first")
        return self._tp3_matrix

    def aggregate(
        self,
        min_confidence: float = 0.3,
        conflict_resolution: str = "highest_confidence",
        use_event_weighting: bool = True,
        max_signals_per_bar: int = 3,
    ) -> GPUAggregatedSignals:
        """
        Aggregate all pattern signals into final trading signals.

        Principle 2 (Minimal Branching): Direction splits use boolean masks
        and torch.where() instead of Python if/else per bar.

        Principle 1 (Data Parallelism): Every operation spans [N, P] tensors.
        No per-bar loops — all bars processed in parallel.

        Args:
            min_confidence: Minimum confidence threshold
            conflict_resolution: 'highest_confidence', 'majority', or 'none'
            use_event_weighting: Apply event-type weighting to aggregation
            max_signals_per_bar: Maximum signals per bar (always 1 in GPU path)

        Returns:
            GPUAggregatedSignals with final entry/stop/targets per bar
        """
        S = self.signal_matrix  # [N, P]
        C = self.confidence_matrix  # [N, P]
        n_bars, n_patterns = S.shape

        active = (S != 0) & (C >= min_confidence)
        active_s = torch.where(active, S, torch.zeros_like(S))
        active_c = torch.where(active, C, torch.zeros_like(C))

        long_mask = active_s > 0
        short_mask = active_s < 0

        long_s = torch.where(long_mask, active_s, torch.zeros_like(S))
        short_s = torch.where(short_mask, active_s, torch.zeros_like(S))
        long_c = torch.where(long_mask, active_c, torch.zeros_like(C))
        short_c = torch.where(short_mask, active_c, torch.zeros_like(C))

        long_count = long_mask.sum(dim=1)
        short_count = short_mask.sum(dim=1)

        # ---- Conflict resolution (per bar, batched) ----
        conflict = (long_count > 0) & (short_count > 0)

        if conflict_resolution == "highest_confidence":
            best_long_conf = long_c.max(dim=1).values
            best_short_conf = short_c.max(dim=1).values
            long_wins = best_long_conf >= best_short_conf

            short_s[conflict & long_wins] = 0
            short_c[conflict & long_wins] = 0
            short_mask[conflict & long_wins] = False
            long_s[conflict & ~long_wins] = 0
            long_c[conflict & ~long_wins] = 0
            long_mask[conflict & ~long_wins] = False

        elif conflict_resolution == "majority":
            long_wins = long_count >= short_count
            short_s[conflict & long_wins] = 0
            short_c[conflict & long_wins] = 0
            short_mask[conflict & long_wins] = False
            long_s[conflict & ~long_wins] = 0
            long_c[conflict & ~long_wins] = 0
            long_mask[conflict & ~long_wins] = False

        elif conflict_resolution == "none":
            long_s[conflict] = 0
            long_c[conflict] = 0
            long_mask[conflict] = False
            short_s[conflict] = 0
            short_c[conflict] = 0
            short_mask[conflict] = False

        # ---- Aggregation per direction ----
        final_dir = torch.zeros(n_bars, dtype=torch.int8, device=self.device)
        final_conf = torch.zeros(n_bars, device=self.device)
        final_entry = torch.zeros(n_bars, device=self.device)
        final_stop = torch.zeros(n_bars, device=self.device)
        final_tp1 = torch.zeros(n_bars, device=self.device)
        final_tp2 = torch.zeros(n_bars, device=self.device)
        final_tp3 = torch.zeros(n_bars, device=self.device)
        final_count = torch.zeros(n_bars, dtype=torch.int32, device=self.device)
        final_event_count = torch.zeros(n_bars, dtype=torch.int32, device=self.device)
        final_event_weight = torch.zeros(n_bars, device=self.device)
        final_confluence = torch.zeros(n_bars, device=self.device)

        event_type_idx = self._get_event_type_indices() if use_event_weighting else None
        event_weights = self._get_event_weight_vector() if use_event_weighting else None

        for direction, d_mask, d_s, d_c in [
            (1, long_mask, long_s, long_c),
            (-1, short_mask, short_s, short_c),
        ]:
            has_signal = d_mask.sum(dim=1) > 0
            if not has_signal.any():
                continue

            final_dir[has_signal] = direction

            combined = d_s + d_c
            total_weight = d_c.sum(dim=1)
            total_weight_safe = torch.where(
                total_weight > 0, total_weight, torch.ones_like(total_weight)
            )

            weighted_entry = (self.entry_matrix * d_c).sum(dim=1) / total_weight_safe
            weighted_tp1 = (self.tp1_matrix * d_c).sum(dim=1) / total_weight_safe

            has_tp2 = (self.tp2_matrix * d_c).sum(dim=1) > 0
            weighted_tp2 = torch.where(
                has_tp2,
                (self.tp2_matrix * d_c).sum(dim=1) / total_weight_safe,
                torch.zeros(n_bars, device=self.device),
            )
            has_tp3 = (self.tp3_matrix * d_c).sum(dim=1) > 0
            weighted_tp3 = torch.where(
                has_tp3,
                (self.tp3_matrix * d_c).sum(dim=1) / total_weight_safe,
                torch.zeros(n_bars, device=self.device),
            )

            dir_sign = float(direction)
            if dir_sign > 0:
                stop_combined = torch.where(
                    d_mask, self.stop_matrix, torch.full_like(self.stop_matrix, float("inf"))
                )
                stop_vals = stop_combined.min(dim=1).values
            else:
                stop_combined = torch.where(
                    d_mask, self.stop_matrix, torch.full_like(self.stop_matrix, float("-inf"))
                )
                stop_vals = stop_combined.max(dim=1).values

            base_conf = d_c.sum(dim=1) / total_weight_safe
            count = d_mask.sum(dim=1).int()
            pattern_bonus = torch.clamp(count.float() * 0.05, max=0.15)
            agg_conf = torch.clamp(base_conf + pattern_bonus, max=1.0)

            final_entry[has_signal] = weighted_entry[has_signal]
            final_stop[has_signal] = stop_vals[has_signal]
            final_tp1[has_signal] = weighted_tp1[has_signal]
            final_tp2[has_signal] = weighted_tp2[has_signal]
            final_tp3[has_signal] = weighted_tp3[has_signal]
            final_conf[has_signal] = agg_conf[has_signal]
            final_count[has_signal] = count[has_signal]

            if use_event_weighting and event_type_idx is not None and event_weights is not None:
                event_confluence = self._compute_event_confluence(
                    d_mask, event_type_idx, event_weights, n_bars, n_patterns
                )
                final_event_count[has_signal] = event_confluence["max_same_event"][has_signal]
                final_event_weight[has_signal] = event_confluence["avg_event_weight"][has_signal]
                final_confluence[has_signal] = event_confluence["confluence_boost"][has_signal]

                event_boost = final_confluence * final_event_weight
                final_conf[has_signal] = torch.clamp(
                    final_conf[has_signal] + event_boost[has_signal], max=1.0
                )

        return GPUAggregatedSignals(
            direction=final_dir,
            confidence=final_conf,
            entry_price=final_entry,
            stop_loss=final_stop,
            take_profit_1=final_tp1,
            take_profit_2=final_tp2,
            take_profit_3=final_tp3,
            pattern_count=final_count,
            dominant_event_count=final_event_count,
            event_weighted=final_event_weight,
            confluence_boost=final_confluence,
        )

    def _compute_event_confluence(
        self,
        active_mask: torch.Tensor,  # [N, P]
        event_type_idx: torch.Tensor,  # [P]
        event_weights: torch.Tensor,  # [P]
        n_bars: int,
        n_patterns: int,
    ) -> Dict[str, torch.Tensor]:
        """
        GPU-parallel event type confluence computation.

        Principle 2 (Minimal Branching): Uses scatter_add_ to count per-event-type
        for all bars simultaneously instead of iterating per event type.

        Replaces the CPU for-loop in EventWeightedAggregator with batched
        tensor operations using index broadcasting and scatter add.
        """
        n_event_types = 8

        safe_event_idx = event_type_idx.unsqueeze(0).expand(n_bars, -1).clone()  # [N, P]
        active_float = active_mask.float()

        active_counts = torch.zeros(n_bars, n_event_types, device=self.device)
        active_counts.scatter_add_(1, safe_event_idx, active_float)

        max_same_event = active_counts.max(dim=1).values.int()

        total_active_weight = (active_float * event_weights.unsqueeze(0)).sum(dim=1)
        total_active = active_mask.sum(dim=1)
        avg_event_weight = torch.where(
            total_active > 0,
            total_active_weight / total_active.float(),
            torch.zeros(n_bars, device=self.device),
        )

        boost = torch.where(
            max_same_event >= 3,
            torch.tensor(0.20, device=self.device),
            torch.where(
                max_same_event >= 2,
                torch.tensor(0.12, device=self.device),
                torch.zeros(n_bars, device=self.device),
            ),
        )
        boost = boost * avg_event_weight
        boost = torch.clamp(boost, max=0.30)

        return {
            "max_same_event": max_same_event,
            "avg_event_weight": avg_event_weight,
            "confluence_boost": boost,
        }

    def get_active_pattern_names(self, bar_idx: int) -> List[Tuple[str, float, int]]:
        """Get active pattern names, confidences, and directions for a specific bar."""
        if self._signal_matrix is None:
            raise RuntimeError("load_signals() must be called first")

        active = (self._signal_matrix[bar_idx] != 0) & (self._confidence_matrix[bar_idx] >= 0)
        indices = torch.where(active)[0]

        results = []
        for idx in indices.tolist():
            name = self._pattern_names[idx]
            conf = float(self._confidence_matrix[bar_idx, idx].item())
            direction = int(self._signal_matrix[bar_idx, idx].item())
            results.append((name, conf, direction))

        return results
