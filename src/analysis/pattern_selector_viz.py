"""
Pattern Selection Visualization

Generates comprehensive visualizations for the pattern selection framework:
- Correlation heat map with hierarchical clustering
- Pattern contribution bar chart with role classification
- Solo performance summary dashboard
- Selection funnel visualization
- Equity curve comparison

Author: Pattern Selection Framework
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .pattern_selector import (
    AblationContribution,
    PatternRole,
    SelectionResult,
    SoloBacktestResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt

        return True
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return False


def _check_seaborn() -> bool:
    """Check if seaborn is available."""
    try:
        import seaborn as sns

        return True
    except ImportError:
        logger.warning("seaborn not available for advanced visualizations")
        return False


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Pattern Equity Curve Correlation Matrix",
    figsize: Tuple[int, int] = (14, 12),
    threshold: float = 0.8,
    cmap: str = "RdBu_r",
    annot: bool = True,
    fmt: str = ".2f",
) -> Optional[str]:
    """
    Generate a correlation heat map for pattern equity curves.

    Args:
        correlation_matrix: NxN correlation matrix
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
        threshold: Threshold for highlighting redundant pairs
        cmap: Colormap name
        annot: Whether to annotate cells with values
        fmt: Format string for annotations

    Returns:
        Path to saved figure or None if failed
    """
    if not _check_matplotlib() or not _check_seaborn():
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    if correlation_matrix.empty:
        logger.warning("Empty correlation matrix")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Mask for upper triangle (to show only lower)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create heat map
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
    )

    # Highlight cells above threshold
    for i in range(len(correlation_matrix)):
        for j in range(i):
            val = correlation_matrix.iloc[i, j]
            try:
                val_float = float(val)  # type: ignore[arg-type]
                if abs(val_float) > threshold:
                    ax.add_patch(
                        Rectangle((j, i), 1, 1, fill=False, edgecolor="red", linewidth=2.5)
                    )
            except (ValueError, TypeError):
                pass

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Pattern", fontsize=11)
    ax.set_ylabel("Pattern", fontsize=11)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Heat map saved to: {output_path}")
        plt.close(fig)

    return output_path


def plot_contribution_ranking(
    ablation_results: List[AblationContribution],
    output_path: Optional[str] = None,
    title: str = "Pattern Marginal Contribution (Ablation Analysis)",
    figsize: Tuple[int, int] = (14, 10),
    show_values: bool = True,
) -> Optional[str]:
    """
    Generate a bar chart showing pattern contribution ranking.

    Args:
        ablation_results: List of ablation contribution results
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
        show_values: Whether to show value labels on bars

    Returns:
        Path to saved figure or None if failed
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not ablation_results:
        logger.warning("No ablation results to plot")
        return None

    # Create DataFrame
    df = pd.DataFrame(
        [
            {
                "pattern": r.pattern_name,
                "delta_sharpe": r.delta_sharpe,
                "delta_return": r.delta_return,
                "role": r.role.value if isinstance(r.role, PatternRole) else r.role,
                "keep": r.keep,
            }
            for r in ablation_results
        ]
    )

    # Sort by delta_sharpe
    df = df.sort_values("delta_sharpe", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars by contribution
    color_map = {
        "Primary Signal": "#2ecc71",  # Green
        "Confirmation Filter": "#3498db",  # Blue
        "Neutral": "#95a5a6",  # Gray
        "Noise Generator": "#e74c3c",  # Red
    }

    colors = [color_map.get(role, "#95a5a6") for role in df["role"]]

    # Plot horizontal bars
    bars = ax.barh(
        df["pattern"],
        df["delta_sharpe"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)

    # Add threshold lines
    ax.axvline(
        x=0.1,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Primary Signal threshold",
    )
    ax.axvline(
        x=-0.05, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Noise threshold"
    )

    # Add labels
    ax.set_xlabel("Delta Sharpe Ratio (Baseline - Ablated)", fontsize=12)
    ax.set_ylabel("Pattern", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add legend
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Primary Signal (δ > 0.1)"),
        Patch(facecolor="#3498db", edgecolor="black", label="Confirmation Filter (0 < δ ≤ 0.1)"),
        Patch(facecolor="#95a5a6", edgecolor="black", label="Neutral (-0.05 ≤ δ ≤ 0)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Noise Generator (δ < -0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    # Add value labels on bars
    if show_values:
        for bar, val in zip(bars, df["delta_sharpe"]):
            if val >= 0:
                ax.text(
                    val + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=9,
                )
            else:
                ax.text(
                    val - 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    ha="right",
                    fontsize=9,
                )

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Contribution chart saved to: {output_path}")
        plt.close(fig)

    return output_path


def plot_solo_performance_summary(
    solo_results: List[SoloBacktestResult],
    output_path: Optional[str] = None,
    title: str = "Pattern Solo Performance Summary",
    figsize: Tuple[int, int] = (16, 12),
) -> Optional[str]:
    """
    Generate a summary chart of solo performance metrics.

    Args:
        solo_results: List of solo backtest results
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size

    Returns:
        Path to saved figure or None if failed
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    if not solo_results:
        logger.warning("No solo results to plot")
        return None

    # Create DataFrame
    df = pd.DataFrame(
        [
            {
                "pattern": r.pattern_name,
                "sharpe": r.sharpe_ratio,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "total_return": r.total_return_pct,
                "max_dd": r.max_drawdown_pct,
                "trades": r.total_trades,
                "passed": r.passed_filter,
            }
            for r in solo_results
        ]
    )

    # Sort by Sharpe ratio
    df = df.sort_values("sharpe", ascending=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Color scheme
    colors = ["#2ecc71" if p else "#e74c3c" for p in df["passed"]]

    # Plot 1: Sharpe Ratio
    ax1 = axes[0, 0]
    ax1.barh(df["pattern"], df["sharpe"], color=colors, edgecolor="black", linewidth=0.5)
    ax1.axvline(x=0.5, color="red", linestyle="--", linewidth=2, label="Threshold (0.5)")
    ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Sharpe Ratio", fontsize=11)
    ax1.set_title("Sharpe Ratio by Pattern", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # Plot 2: Win Rate
    ax2 = axes[0, 1]
    ax2.barh(df["pattern"], df["win_rate"] * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(x=40, color="red", linestyle="--", linewidth=2, label="Threshold (40%)")
    ax2.set_xlabel("Win Rate (%)", fontsize=11)
    ax2.set_title("Win Rate by Pattern", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    # Plot 3: Profit Factor
    ax3 = axes[1, 0]
    ax3.barh(df["pattern"], df["profit_factor"], color=colors, edgecolor="black", linewidth=0.5)
    ax3.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Threshold (1.0)")
    ax3.set_xlabel("Profit Factor", fontsize=11)
    ax3.set_title("Profit Factor by Pattern", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(axis="x", alpha=0.3, linestyle="--")

    # Plot 4: Total Return
    ax4 = axes[1, 1]
    ax4.barh(df["pattern"], df["total_return"], color=colors, edgecolor="black", linewidth=0.5)
    ax4.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax4.set_xlabel("Total Return (%)", fontsize=11)
    ax4.set_title("Total Return by Pattern", fontsize=12, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Solo performance chart saved to: {output_path}")
        plt.close(fig)

    return output_path


def plot_solo_performance(
    result: SelectionResult,
    output_path: Optional[str] = None,
    title: str = "Pattern Solo Performance Summary",
    figsize: Tuple[int, int] = (16, 12),
) -> Optional[str]:
    """
    Generate a summary chart of solo performance metrics from a SelectionResult.

    This is a convenience wrapper around plot_solo_performance_summary that
    accepts a SelectionResult directly, matching the signature style of
    plot_selection_funnel and plot_role_distribution.

    Args:
        result: SelectionResult from pattern selection
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size

    Returns:
        Path to saved figure or None if failed
    """
    if not result.solo_results:
        logger.warning("No solo results in SelectionResult")
        return None

    return plot_solo_performance_summary(
        solo_results=result.solo_results,
        output_path=output_path,
        title=title,
        figsize=figsize,
    )


def plot_selection_funnel(
    result: SelectionResult,
    output_path: Optional[str] = None,
    title: str = "Pattern Selection Funnel",
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[str]:
    """
    Generate a funnel visualization showing pattern filtering at each phase.

    Args:
        result: SelectionResult from pattern selection
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size

    Returns:
        Path to saved figure or None if failed
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # Data for funnel
    phases = [
        ("Total Patterns", len(result.solo_results)),
        ("Passed Phase 1 Filter", len(result.filtered_patterns)),
        ("After Redundancy Removal", len(result.non_redundant_patterns)),
        ("Final Selected", len(result.final_patterns)),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors for each phase
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    # Calculate widths based on values
    max_val = phases[0][1]
    if max_val == 0:
        max_val = 1

    y_positions = [8, 6, 4, 2]

    for i, (label, value) in enumerate(phases):
        width = (value / max_val) * 7 + 1  # Scale width
        x_center = 5
        x_left = x_center - width / 2
        y = y_positions[i]

        # Draw box
        box = FancyBboxPatch(
            (x_left, y - 0.5),
            width,
            1,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=colors[i],
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(box)

        # Add text
        ax.text(
            x_center,
            y,
            f"{label}\n{value} patterns",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if i < 3 else "black",
        )

        # Add arrow between phases
        if i < len(phases) - 1:
            ax.annotate(
                "",
                xy=(5, y - 0.6),
                xytext=(5, y - 1.4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2),
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Funnel chart saved to: {output_path}")
        plt.close(fig)

    return output_path


def plot_role_distribution(
    result: SelectionResult,
    output_path: Optional[str] = None,
    title: str = "Pattern Role Distribution",
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[str]:
    """
    Generate a pie chart showing distribution of pattern roles.

    Args:
        result: SelectionResult from pattern selection
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size

    Returns:
        Path to saved figure or None if failed
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    role_dist = result.get_role_distribution()

    if not role_dist:
        logger.warning("No role distribution to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Colors for each role
    color_map = {
        "Primary Signal": "#2ecc71",
        "Confirmation Filter": "#3498db",
        "Neutral": "#95a5a6",
        "Noise Generator": "#e74c3c",
    }

    labels = list(role_dist.keys())
    sizes = list(role_dist.values())
    colors = [color_map.get(label, "#95a5a6") for label in labels]

    # Create pie chart (with autopct, always returns 3-tuple)
    wedges, texts, autotexts = ax.pie(  # type: ignore[assignment,misc]
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05 if label == "Primary Signal" else 0 for label in labels],
        shadow=True,
    )

    # Enhance text
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add legend with counts
    legend_labels = [f"{label}: {role_dist[label]}" for label in labels]
    ax.legend(wedges, legend_labels, loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Role distribution chart saved to: {output_path}")
        plt.close(fig)

    return output_path


def plot_equity_curves_comparison(
    solo_results: Union[List[SoloBacktestResult], SelectionResult],
    output_path: Optional[str] = None,
    title: str = "Pattern Equity Curves Comparison",
    figsize: Tuple[int, int] = (14, 10),
    max_curves: int = 10,
) -> Optional[str]:
    """
    Generate a comparison plot of equity curves for top patterns.

    Args:
        solo_results: List of solo backtest results or SelectionResult
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
        max_curves: Maximum number of curves to display

    Returns:
        Path to saved figure or None if failed
    """
    # Handle SelectionResult input
    if isinstance(solo_results, SelectionResult):
        solo_results = solo_results.solo_results

    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    # Filter patterns with equity curves
    valid_results = [r for r in solo_results if r.equity_curve is not None and r.passed_filter]

    if not valid_results:
        logger.warning("No equity curves available for comparison")
        return None

    # Sort by Sharpe and take top N
    valid_results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
    top_results = valid_results[:max_curves]

    fig, ax = plt.subplots(figsize=figsize)

    # Color map
    cmap = plt.cm.get_cmap("tab10")

    for i, result in enumerate(top_results):
        equity = np.array(result.equity_curve)
        # Normalize to start at 100
        equity_normalized = (equity / equity[0]) * 100

        ax.plot(
            equity_normalized,
            label=f"{result.pattern_name} (Sharpe: {result.sharpe_ratio:.2f})",
            color=cmap(i % 10),
            linewidth=1.5,
            alpha=0.8,
        )

    ax.axhline(y=100, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Break-even")

    ax.set_xlabel("Trading Days", fontsize=11)
    ax.set_ylabel("Normalized Equity (Starting = 100)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Equity curves comparison saved to: {output_path}")
        plt.close(fig)

    return output_path


def generate_selection_report(
    result: SelectionResult,
    output_dir: str,
) -> str:
    """
    Generate a comprehensive markdown report for the pattern selection.

    Args:
        result: SelectionResult from the pattern selection
        output_dir: Directory to save the report

    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("# Pattern Selection Report\n")
    lines.append("*Generated by Pattern Selection Framework*\n")

    # Executive Summary
    summary = result.get_summary()
    lines.append("## Executive Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Total Patterns Tested | {summary.get('total_patterns_tested', summary.get('total_patterns', 'N/A'))} |"
    )
    lines.append(
        f"| Passed Phase 1 Filter | {summary.get('passed_phase1_filter', summary.get('patterns', 'N/A'))} |"
    )
    lines.append(f"| Excluded (Phase 1) | {summary.get('excluded_phase1', 0)} |")
    lines.append(
        f"| Redundant Pairs Found | {summary.get('redundant_pairs_found', len(result.redundant_patterns))} |"
    )
    lines.append(f"| After Redundancy Removal | {summary.get('after_redundancy_removal', 'N/A')} |")
    lines.append(
        f"| **Final Selected Patterns** | **{summary.get('final_selected_patterns', len(result.final_selection))}** |"
    )
    lines.append(f"| Execution Time | {summary.get('execution_time_seconds', 0)}s |")
    lines.append("")

    # Phase Timings
    if result.phase_timings:
        lines.append("### Phase Timings\n")
        lines.append("| Phase | Duration |")
        lines.append("|-------|----------|")
        for phase, timing in result.phase_timings.items():
            lines.append(f"| {phase.upper()} | {timing:.1f}s |")
        lines.append("")

    # Phase 1: Solo Performance
    lines.append("## Phase 1: Isolated Performance Baseline\n")
    lines.append("### Thresholds Applied\n")
    config = result.config
    if config is not None:
        lines.append("| Metric | Threshold |")
        lines.append("|--------|-----------|")
        lines.append(f"| Minimum Trades | {getattr(config, 'min_trades', 'N/A')} |")
        lines.append(f"| Minimum Sharpe Ratio | {getattr(config, 'min_sharpe', 'N/A')} |")
        lines.append(f"| Minimum Profit Factor | {getattr(config, 'min_profit_factor', 'N/A')} |")
        lines.append(f"| Minimum Win Rate | {getattr(config, 'min_win_rate', 0)} |")
        lines.append(f"| Maximum Drawdown | {getattr(config, 'max_drawdown', 0)} |")
        lines.append("")

    # Solo Results Table
    lines.append("### Solo Backtest Results\n")
    lines.append(
        "| Pattern | Trades | Win Rate | Sharpe | Profit Factor | Return | Max DD | Status |"
    )
    lines.append(
        "|---------|--------|----------|--------|---------------|--------|--------|--------|"
    )

    for r in sorted(result.solo_results, key=lambda x: x.sharpe_ratio, reverse=True):
        status = "[PASS]" if r.passed_filter else "[FAIL]"
        lines.append(
            f"| {r.pattern_name} | {r.total_trades} | {r.win_rate:.1%} | "
            f"{r.sharpe_ratio:.3f} | {r.profit_factor:.2f} | {r.total_return_pct:.2%} | "
            f"{r.max_drawdown_pct:.1%} | {status} |"
        )
    lines.append("")

    # Phase 2: Correlation Analysis
    lines.append("## Phase 2: Statistical Correlation Analysis\n")

    if result.redundant_pairs:
        lines.append("### Redundant Pairs Identified\n")
        lines.append("| Pattern A | Pattern B | Correlation | Selected | Reason |")
        lines.append("|-----------|-----------|-------------|----------|--------|")
        for pair in result.redundant_pairs:
            lines.append(
                f"| {pair.pattern_a} | {pair.pattern_b} | {pair.correlation:.3f} | "
                f"{pair.selected_pattern or 'N/A'} | {pair.reason} |"
            )
        lines.append("")
    else:
        lines.append("No redundant pairs found.\n")

    # Phase 3: Ablation Analysis
    lines.append("## Phase 3: Ablation Testing (Leave-One-Out)\n")

    lines.append("### Marginal Contribution Ranking\n")
    lines.append("| Rank | Pattern | Delta Sharpe | Delta Return | Role | Keep |")
    lines.append("|------|---------|---------------|--------------|------|------|")

    for c in result.ablation_contributions:
        keep_val = c.keep if hasattr(c, "keep") else c.get("keep", False)
        keep_icon = "[KEEP]" if keep_val else "[REMOVE]"
        role_val = c.role if hasattr(c, "role") else c.get("role", "Unknown")
        if isinstance(role_val, PatternRole):
            role_name = role_val.value
        else:
            role_name = str(role_val)
        rank = (
            c.contribution_rank
            if hasattr(c, "contribution_rank")
            else c.get("contribution_rank", 0)
        )
        name = c.pattern_name if hasattr(c, "pattern_name") else c.get("pattern_name", "Unknown")
        delta_s = c.delta_sharpe if hasattr(c, "delta_sharpe") else c.get("delta_sharpe", 0)
        delta_r = c.delta_return if hasattr(c, "delta_return") else c.get("delta_return", 0)
        lines.append(
            f"| {rank} | {name} | {delta_s:.4f} | {delta_r:.2%} | {role_name} | {keep_icon} |"
        )
    lines.append("")

    # Pattern Roles Summary
    lines.append("### Pattern Role Classification\n")
    role_dist = result.get_role_distribution()

    lines.append("| Role | Count | Description |")
    lines.append("|------|-------|-------------|")
    lines.append(
        f"| Primary Signal | {role_dist.get('Primary Signal', 0)} | High solo edge, high marginal contribution |"
    )
    lines.append(
        f"| Confirmation Filter | {role_dist.get('Confirmation Filter', 0)} | Low solo edge, positive marginal contribution |"
    )
    lines.append(f"| Neutral | {role_dist.get('Neutral', 0)} | No significant impact |")
    lines.append(
        f"| Noise Generator | {role_dist.get('Noise Generator', 0)} | Negative marginal contribution (remove) |"
    )
    lines.append("")

    # Final Selection
    lines.append("## Final Selected Patterns\n")
    lines.append(f"**{len(result.final_patterns)} patterns selected for production:**\n")

    for i, pattern in enumerate(result.final_patterns, 1):
        c = next((c for c in result.ablation_contributions if c.pattern_name == pattern), None)  # type: ignore[arg-type]
        if c:
            role_name = c.role.value if isinstance(c.role, PatternRole) else c.role
            lines.append(f"{i}. **{pattern}** — {role_name} (δ Sharpe: {c.delta_sharpe:.4f})")
        else:
            lines.append(f"{i}. **{pattern}**")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")

    # Noise generators to remove
    noise = []
    for c in result.ablation_contributions:
        role = c.role if hasattr(c, "role") else c.get("role", "")
        if isinstance(role, PatternRole):
            role = role.value
        if role == "Noise Generator" or role == PatternRole.NOISE_GENERATOR.value:
            noise.append(c)
    if noise:
        lines.append("### Patterns to Remove\n")
        for c in noise:
            name = (
                c.pattern_name if hasattr(c, "pattern_name") else c.get("pattern_name", "Unknown")
            )
            delta = c.delta_sharpe if hasattr(c, "delta_sharpe") else c.get("delta_sharpe", 0)
            lines.append(f"- **{name}** — Delta Sharpe: {delta:.4f}")
        lines.append("")

    # Primary signals to prioritize
    primary = []
    for c in result.ablation_contributions:
        role = c.role if hasattr(c, "role") else c.get("role", "")
        if isinstance(role, PatternRole):
            role = role.value
        if role == "Primary Signal" or role == PatternRole.PRIMARY_SIGNAL.value:
            primary.append(c)
    if primary:
        lines.append("### Primary Signals (High Priority)\n")
        for c in primary:
            name = (
                c.pattern_name if hasattr(c, "pattern_name") else c.get("pattern_name", "Unknown")
            )
            delta = c.delta_sharpe if hasattr(c, "delta_sharpe") else c.get("delta_sharpe", 0)
            lines.append(f"- **{name}** — Delta Sharpe: {delta:.4f}")
        lines.append("")

    # Configuration
    lines.append("## Configuration Used\n")
    if config is not None:
        lines.append("```json")
        try:
            if hasattr(config, "to_dict"):
                lines.append(json.dumps(config.to_dict(), indent=2))
            else:
                lines.append(json.dumps({"config_type": type(config).__name__}, indent=2))
        except Exception:
            lines.append('{ "error": "Could not serialize config" }')
        lines.append("```\n")

    # Write report
    report_path = os.path.join(output_dir, "selection_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"  Report saved to: {report_path}")

    return report_path


def generate_all_visualizations(
    result: SelectionResult,
    output_dir: str,
    generate_equity_curves: bool = True,
) -> Dict[str, str]:
    """
    Generate all visualizations and report.

    Args:
        result: SelectionResult from pattern selection
        output_dir: Directory to save outputs
        generate_equity_curves: Whether to generate equity curve comparison

    Returns:
        Dictionary mapping visualization name to file path
    """

    os.makedirs(output_dir, exist_ok=True)

    outputs = {}

    print(f"\n{'=' * 60}")
    print("Generating Visualizations and Reports")
    print(f"{'=' * 60}")

    # Correlation heat map
    correlation_matrix = getattr(result, "correlation_matrix", None)
    correlation_threshold = (
        getattr(getattr(result, "config", None), "correlation_threshold", 0.7) or 0.7
    )
    if correlation_matrix is not None and not correlation_matrix.empty:
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        outputs["correlation_heatmap"] = (
            plot_correlation_heatmap(
                correlation_matrix,
                heatmap_path,
                threshold=correlation_threshold,
            )
            or ""
        )

    # Contribution ranking
    if result.ablation_contributions:
        contrib_path = os.path.join(output_dir, "contribution_ranking.png")
        outputs["contribution_ranking"] = (
            plot_contribution_ranking(
                result.ablation_contributions,
                contrib_path,
            )
            or ""
        )

    # Solo performance summary
    solo_results = getattr(result, "solo_results", None)
    if solo_results:
        solo_path = os.path.join(output_dir, "solo_performance.png")
        outputs["solo_performance"] = (
            plot_solo_performance_summary(
                solo_results,
                solo_path,
            )
            or ""
        )

    # Selection funnel
    funnel_path = os.path.join(output_dir, "selection_funnel.png")
    outputs["selection_funnel"] = plot_selection_funnel(result, funnel_path) or ""

    # Role distribution
    role_path = os.path.join(output_dir, "role_distribution.png")
    outputs["role_distribution"] = plot_role_distribution(result, role_path) or ""

    # Equity curves comparison
    if generate_equity_curves and solo_results:
        equity_path = os.path.join(output_dir, "equity_curves_comparison.png")
        outputs["equity_curves_comparison"] = (
            plot_equity_curves_comparison(solo_results, equity_path) or ""
        )

    # Markdown report
    outputs["report"] = generate_selection_report(result, output_dir)

    return outputs


# Import json for report generation
import json
