"""
GPU-Accelerated Pattern Detection and Signal Processing

This module provides PyTorch-based GPU acceleration for:
- Batch pattern detection across all bars simultaneously
- Signal aggregation with conflict resolution
- Feature engineering with parallel rolling windows
- Exponential smoothing via truncated convolution (zero Python loops)
- End-to-end GPU backtest engine

Design Principles:
1. Data parallelism: All bars processed simultaneously via tensor operations
2. Minimal branching: Mask-based selection instead of if/else
3. Coalesced memory: Contiguous tensor layouts for optimal GPU throughput
4. Batched transfers: Single CPU↔GPU transfer, accumulate results on-device
5. Tensor cores: unfold + dot products for matmul-friendly smoothing
6. Maximized occupancy: Native PyTorch reductions handle thread scheduling
"""

from .backtest import GPUBacktestEngine, GPUBacktestResult, GPUEquitySimulator, GPUTrade
from .features import GPUFeatureEngineer
from .pattern_detector import GPUPatternDetector, GPUSignalBatch
from .signal_aggregator import GPUAggregatedSignals, GPUSignalAggregator
from .smooth import (
    adx_parallel,
    atr_parallel,
    ema_parallel,
    ema_parallel_batch,
    rsi_parallel,
    wilder_smooth_parallel,
)
from .utils import get_device, to_numpy, to_tensor

__all__ = [
    "GPUPatternDetector",
    "GPUSignalBatch",
    "GPUSignalAggregator",
    "GPUAggregatedSignals",
    "GPUFeatureEngineer",
    "GPUBacktestEngine",
    "GPUBacktestResult",
    "GPUEquitySimulator",
    "GPUTrade",
    "get_device",
    "to_tensor",
    "to_numpy",
    "ema_parallel",
    "ema_parallel_batch",
    "wilder_smooth_parallel",
    "rsi_parallel",
    "atr_parallel",
    "adx_parallel",
]
