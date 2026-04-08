"""
Contribution Charts - Visualization for Pattern Contribution Analysis

All visualizations use matplotlib/seaborn (already in dependencies). No new dependencies needed.

Charts included:
1. Pattern Leaderboard (Horizontal Bar Chart)
2. Contribution Waterfall
3. Co-occurrence Heatmap
4. Frequency vs. Quality Scatter
5. Confluence Count vs. Performance
6. Synergy Network Graph
7. Pattern Category Performance
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .signal_event_log import SignalEventLog
from .trade_attributor import TradeAttributor


def create_contribution_charts(
    signal_log: SignalEventLog,
    trade_attributor: TradeAttributor,
    ablation_results: Optional[pd.DataFrame] = None,
    synergy_results: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Figure]:
    """
    Create all contribution analysis charts.

    Args:
        signal_log: SignalEventLog from backtest
        trade_attributor: TradeAttributor with attributed trades
        ablation_results: Optional DataFrame from AblationEngine
        synergy_results: Optional DataFrame from SynergyAnalyzer
        output_dir: Optional directory to save figures

    Returns:
        Dictionary mapping chart name to Figure object
    """
    figures = {}

    # Chart 3: Co-occurrence Heatmap (requires only L1 data)
    if signal_log:
        fig = create_co_occurrence_heatmap(signal_log)
        figures["co_occurrence_heatmap"] = fig

    # Chart 4: Frequency vs. Quality Scatter (requires L1+L2 data)
    if signal_log and trade_attributor:
        fig = create_frequency_quality_scatter(signal_log, trade_attributor)
        figures["frequency_quality_scatter"] = fig

    # Chart 5: Confluence Count vs. Performance (requires L2 data)
    if trade_attributor:
        fig = create_confluence_performance_chart(trade_attributor)
        figures["confluence_performance"] = fig

    # Chart 1: Pattern Leaderboard (requires L3 data)
    if ablation_results is not None and not ablation_results.empty:
        fig = create_pattern_leaderboard(ablation_results)
        figures["pattern_leaderboard"] = fig

    # Chart 2: Contribution Waterfall (requires L3 data)
    if ablation_results is not None and not ablation_results.empty:
        fig = create_contribution_waterfall(ablation_results)
        figures["contribution_waterfall"] = fig

    # Chart 6: Synergy Network Graph (requires L4 data)
    if synergy_results is not None and not synergy_results.empty:
        fig = create_synergy_network(synergy_results)
        figures["synergy_network"] = fig

    # Chart 7: Pattern Category Performance (requires L3 data)
    if ablation_results is not None and not ablation_results.empty:
        fig = create_category_performance(ablation_results)
        figures["category_performance"] = fig

    # Save figures if output directory provided
    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figures.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")

    return figures


def create_co_occurrence_heatmap(signal_log: SignalEventLog) -> Figure:
    """
    Create co-occurrence heatmap.

    Args:
        signal_log: SignalEventLog instance

    Returns:
        matplotlib Figure
    """
    co_occurrence = signal_log.get_co_occurrence_matrix()

    if co_occurrence.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Limit to top 20 patterns by total co-occurrences
    if len(co_occurrence) > 20:
        # Get diagonal (self co-occurrences = detection count)
        diag = np.diag(co_occurrence.values)
        top_indices = np.argsort(diag)[-20:]
        co_occurrence = co_occurrence.iloc[top_indices, top_indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        co_occurrence,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Co-occurrence Count"},
    )

    ax.set_title("Pattern Co-occurrence Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Pattern")

    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig


def create_frequency_quality_scatter(
    signal_log: SignalEventLog, trade_attributor: TradeAttributor
) -> Figure:
    """
    Create frequency vs. quality scatter plot.

    X-axis: Detection frequency (% of bars where pattern fires)
    Y-axis: Avg P&L when pattern participates in a trade
    Bubble size: Number of trades pattern participated in
    Color: Pattern category

    Args:
        signal_log: SignalEventLog instance
        trade_attributor: TradeAttributor instance

    Returns:
        matplotlib Figure
    """
    # Get detection frequency
    freq_df = signal_log.get_detection_frequency()

    # Get pattern trade stats
    trade_stats = trade_attributor.get_pattern_trade_stats()

    if freq_df.empty or trade_stats.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Merge data
    merged = freq_df.merge(trade_stats, on="pattern_name", how="inner")

    if merged.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No matching data", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Get pattern categories from signal log
    categories = {}
    for event in signal_log.events:
        if event.pattern_name not in categories:
            categories[event.pattern_name] = event.pattern_category

    merged["category"] = merged["pattern_name"].map(categories)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot with different colors per category
    category_colors = {
        "basic": "#1f77b4",
        "harmonic": "#ff7f0e",
        "complex": "#2ca02c",
        "classic": "#d62728",
        "continuation": "#9467bd",
        "breakout": "#8c564b",
        "candlestick": "#e377c2",
    }

    for category, color in category_colors.items():
        cat_data = merged[merged["category"] == category]
        if not cat_data.empty:
            ax.scatter(
                cat_data["detection_frequency_pct"],
                cat_data["avg_pnl"],
                s=cat_data["trade_count"] * 10,  # Scale bubble size by trade count
                c=color,
                alpha=0.6,
                label=category,
                edgecolors="black",
                linewidth=0.5,
            )

    # Add quadrant lines
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(
        x=merged["detection_frequency_pct"].median(), color="gray", linestyle="--", alpha=0.5
    )

    # Add quadrant labels
    x_mid = merged["detection_frequency_pct"].median()
    y_mid = 0

    ax.text(
        merged["detection_frequency_pct"].max() * 0.8,
        merged["avg_pnl"].max() * 0.8,
        "Frequent &\nProfitable",
        ha="center",
        fontsize=10,
        color="green",
        fontweight="bold",
    )
    ax.text(
        merged["detection_frequency_pct"].min() * 1.2,
        merged["avg_pnl"].max() * 0.8,
        "Rare &\nProfitable",
        ha="center",
        fontsize=10,
        color="blue",
        fontweight="bold",
    )
    ax.text(
        merged["detection_frequency_pct"].max() * 0.8,
        merged["avg_pnl"].min() * 0.8,
        "Frequent &\nUnprofitable",
        ha="center",
        fontsize=10,
        color="red",
        fontweight="bold",
    )
    ax.text(
        merged["detection_frequency_pct"].min() * 1.2,
        merged["avg_pnl"].min() * 0.8,
        "Rare &\nUnprofitable",
        ha="center",
        fontsize=10,
        color="gray",
        fontweight="bold",
    )

    ax.set_xlabel("Detection Frequency (% of bars)", fontsize=12)
    ax.set_ylabel("Average P&L per Trade", fontsize=12)
    ax.set_title("Pattern Frequency vs. Quality", fontsize=14, fontweight="bold")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_confluence_performance_chart(trade_attributor: TradeAttributor) -> Figure:
    """
    Create confluence count vs. performance chart.

    X-axis: Number of patterns in confluence (1, 2, 3, 4, 5+)
    Y-axis (left): Win rate (line)
    Y-axis (right): Average P&L (bar)
    Annotation: Number of trades per group

    Args:
        trade_attributor: TradeAttributor instance

    Returns:
        matplotlib Figure
    """
    confluence_perf = trade_attributor.get_confluence_vs_performance()

    if confluence_perf.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for average P&L
    x = range(len(confluence_perf))
    bars = ax1.bar(x, confluence_perf["avg_pnl"], color="steelblue", alpha=0.7, label="Avg P&L")
    ax1.set_xlabel("Confluence Count", fontsize=12)
    ax1.set_ylabel("Average P&L", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Line chart for win rate
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        confluence_perf["win_rate"],
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Win Rate",
    )
    ax2.set_ylabel("Win Rate", fontsize=12, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 1)

    # Add trade count annotations
    for i, (_, row) in enumerate(confluence_perf.iterrows()):
        ax1.text(
            i,
            row["avg_pnl"],
            f"n={int(row['trade_count'])}",
            ha="center",
            va="bottom" if row["avg_pnl"] >= 0 else "top",
            fontsize=9,
        )

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(confluence_perf["confluence_count"])

    ax1.set_title("Confluence Count vs. Performance", fontsize=14, fontweight="bold")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def create_pattern_leaderboard(ablation_results: pd.DataFrame) -> Figure:
    """
    Create pattern leaderboard horizontal bar chart.

    X-axis: Ablation delta Sharpe ratio
    Y-axis: Pattern names (sorted)
    Color: Green = positive contributor, Red = negative contributor

    Args:
        ablation_results: DataFrame from AblationEngine

    Returns:
        matplotlib Figure
    """
    if ablation_results.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Sort by delta_sharpe
    df = ablation_results.sort_values("delta_sharpe")

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))

    # Color bars based on delta_sharpe
    colors = ["green" if x < 0 else "red" for x in df["delta_sharpe"]]

    y = range(len(df))
    ax.barh(y, df["delta_sharpe"], color=colors, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["pattern_name"])
    ax.set_xlabel("Delta Sharpe (Baseline - Ablated)", fontsize=12)
    ax.set_title("Pattern Contribution Leaderboard", fontsize=14, fontweight="bold")

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["delta_sharpe"],
            i,
            f"{row['delta_sharpe']:.3f}",
            va="center",
            ha="left" if row["delta_sharpe"] < 0 else "right",
            fontsize=8,
        )

    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def create_contribution_waterfall(ablation_results: pd.DataFrame) -> Figure:
    """
    Create contribution waterfall chart.

    X-axis: Cumulative P&L
    Y-axis: Steps (baseline → +pattern → +pattern → ...)
    Bars: Green for positive contribution, Red for negative

    Args:
        ablation_results: DataFrame from AblationEngine

    Returns:
        matplotlib Figure
    """
    if ablation_results.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Sort by contribution (delta_return)
    df = ablation_results.sort_values("delta_return", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))

    # Create waterfall
    # delta_return = baseline_return - ablated_return
    # delta > 0: removing pattern hurts (pattern helps) -> green
    # delta < 0: removing pattern helps (pattern hurts) -> red
    cumulative = 0
    for i, (_, row) in enumerate(df.iterrows()):
        delta = row["delta_return"]
        color = "green" if delta > 0 else "red"

        ax.barh(i, delta, left=cumulative, color=color, alpha=0.7)
        cumulative += delta

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["pattern_name"])
    ax.set_xlabel("Cumulative Return Impact", fontsize=12)
    ax.set_title("Contribution Waterfall", fontsize=14, fontweight="bold")

    # Add baseline and final labels
    ax.text(0, -0.5, "Baseline", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        cumulative,
        len(df) - 0.5,
        f"Final: {cumulative:.2%}",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def create_synergy_network(synergy_results: pd.DataFrame) -> Figure:
    """
    Create synergy network graph.

    Nodes: Patterns (sized by ablation contribution, colored by category)
    Edges: Connected if synergy_score is significant
      Green edges: Positive synergy (complementary)
      Red edges: Negative synergy (conflicting)
      Edge thickness: |synergy_score|

    Args:
        synergy_results: DataFrame from SynergyAnalyzer

    Returns:
        matplotlib Figure
    """
    try:
        import networkx as nx
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "NetworkX not installed.\nInstall with: pip install networkx",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    if synergy_results.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Create graph
    G = nx.Graph()

    # Add nodes (unique patterns)
    patterns = set()
    for _, row in synergy_results.iterrows():
        patterns.add(row["pattern_a"])
        patterns.add(row["pattern_b"])

    for pattern in patterns:
        G.add_node(pattern)

    # Add edges with significant synergy
    for _, row in synergy_results.iterrows():
        synergy = row.get("synergy_score", 0)
        if abs(synergy) > 0.01:  # Only show significant edges
            G.add_edge(
                row["pattern_a"],
                row["pattern_b"],
                weight=abs(synergy),
                synergy=synergy,
                color="green" if synergy > 0 else "red",
            )

    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        edge_colors = [d["color"] for _, _, d in edges]
        edge_widths = [d["weight"] * 10 for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title("Pattern Synergy Network", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.6, label="Positive Synergy"),
        Patch(facecolor="red", alpha=0.6, label="Negative Synergy"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    return fig


def create_category_performance(ablation_results: pd.DataFrame) -> Figure:
    """
    Create pattern category performance chart.

    Grouped bar chart:
    X-axis: Categories (basic, harmonic, complex, classic, continuation, breakout, candlestick)
    Y-axis: Avg contribution metrics
    Groups: Solo return, Ablation delta, Avg synergy

    Args:
        ablation_results: DataFrame from AblationEngine

    Returns:
        matplotlib Figure
    """
    if ablation_results.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Check if category column exists
    if "category" not in ablation_results.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Category data not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Group by category
    category_stats = (
        ablation_results.groupby("category")
        .agg(
            {
                "delta_return": "mean",
                "delta_sharpe": "mean",
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(category_stats))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        category_stats["delta_return"],
        width,
        label="Delta Return",
        color="steelblue",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        category_stats["delta_sharpe"],
        width,
        label="Delta Sharpe",
        color="darkorange",
    )

    ax.set_xlabel("Pattern Category", fontsize=12)
    ax.set_ylabel("Average Impact", fontsize=12)
    ax.set_title("Pattern Category Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(category_stats["category"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
