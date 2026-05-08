"""
Contribution Report - Aggregation & Reporting

Aggregates all analysis layers into a unified report.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .signal_event_log import SignalEventLog
from .trade_attributor import TradeAttributor


class ContributionReport:
    """Aggregates all analysis layers into a unified report."""

    def __init__(
        self,
        signal_log: SignalEventLog,
        attributor: TradeAttributor,
        ablation_results: Optional[pd.DataFrame] = None,
        synergy_results: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            signal_log: SignalEventLog from the backtest
            attributor: TradeAttributor with attributed trades
            ablation_results: Optional DataFrame from AblationEngine
            synergy_results: Optional DataFrame from SynergyAnalyzer
        """
        self.signal_log = signal_log
        self.attributor = attributor
        self.ablation_results = ablation_results
        self.synergy_results = (
            self._flatten_synergy(synergy_results)
            if synergy_results is not None and not synergy_results.empty
            else synergy_results
        )

    def generate_summary(self) -> Dict[str, Any]:
        """
        Executive summary of pattern contributions.

        Returns:
            Dictionary with summary statistics
        """
        # Signal log stats
        signal_stats = self.signal_log.get_summary_stats()

        # Trade attribution stats
        attribution_stats = self.attributor.get_summary_stats()

        # Pattern trade stats
        pattern_stats = self.attributor.get_pattern_trade_stats()

        # Ablation stats
        ablation_summary = {}
        if self.ablation_results is not None and not self.ablation_results.empty:
            ablation_summary = {
                "total_patterns": len(self.ablation_results),
                "positive_contributors": len(
                    self.ablation_results[self.ablation_results["delta_sharpe"] > 0]
                ),
                "negative_contributors": len(
                    self.ablation_results[self.ablation_results["delta_sharpe"] < 0]
                ),
                "avg_delta_sharpe": self.ablation_results["delta_sharpe"].mean(),
                "best_pattern": self.ablation_results.iloc[0]["pattern_name"]
                if len(self.ablation_results) > 0
                else None,
                "worst_pattern": self.ablation_results.iloc[-1]["pattern_name"]
                if len(self.ablation_results) > 0
                else None,
            }

        # Synergy stats
        synergy_summary = {}
        if self.synergy_results is not None and not self.synergy_results.empty:
            synergy_flat = self._flatten_synergy(self.synergy_results)
            synergy_summary = {
                "total_pairs": len(synergy_flat),
                "complementary_pairs": len(synergy_flat[synergy_flat["synergy_score"] > 0]),
                "conflicting_pairs": len(synergy_flat[synergy_flat["synergy_score"] < 0]),
                "avg_synergy_score": synergy_flat["synergy_score"].mean(),
            }

        return {
            "signal_log": signal_stats,
            "trade_attribution": attribution_stats,
            "pattern_trade_stats": pattern_stats.to_dict() if not pattern_stats.empty else {},
            "ablation": ablation_summary,
            "synergy": synergy_summary,
        }

    def _flatten_synergy(self, synergy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert synergy matrix (NxN with pattern names as index/columns) to flat format
        with columns: pattern_a, pattern_b, synergy_score.

        Args:
            synergy_df: NxN DataFrame from SynergyAnalyzer.get_synergy_matrix()

        Returns:
            Flat DataFrame with synergy_score column
        """
        # Check if already flat
        if "synergy_score" in synergy_df.columns:
            return synergy_df

        # Convert matrix to flat
        patterns = synergy_df.index.tolist()
        rows = []
        for i, pa in enumerate(patterns):
            for j, pb in enumerate(patterns):
                if i < j:
                    rows.append(
                        {
                            "pattern_a": pa,
                            "pattern_b": pb,
                            "synergy_score": synergy_df.iloc[i, j],
                        }
                    )
        return pd.DataFrame(rows)

    def get_pattern_leaderboard(self) -> pd.DataFrame:
        """
        Combined ranking: solo edge + ablation contribution + synergy.

        Returns:
            DataFrame with columns: pattern, category, solo_return, solo_sharpe,
                     ablation_delta_sharpe, avg_synergy_score,
                     participation_rate, composite_score, role
        """
        # Get pattern trade stats
        pattern_stats = self.attributor.get_pattern_trade_stats()

        if pattern_stats.empty:
            return pd.DataFrame()

        # Build leaderboard
        leaderboard = pattern_stats.copy()

        # Add ablation data if available
        if self.ablation_results is not None and not self.ablation_results.empty:
            ablation_data = self.ablation_results[
                ["pattern_name", "delta_sharpe", "delta_return"]
            ].copy()
            ablation_data = ablation_data.rename(
                columns={
                    "delta_sharpe": "ablation_delta_sharpe",
                    "delta_return": "ablation_delta_return",
                }
            )
            leaderboard = leaderboard.merge(ablation_data, on="pattern_name", how="left")
        else:
            leaderboard["ablation_delta_sharpe"] = np.nan
            leaderboard["ablation_delta_return"] = np.nan

        # Add synergy data if available
        if self.synergy_results is not None and not self.synergy_results.empty:
            # Calculate average synergy score per pattern
            synergy_per_pattern: Dict[str, List[float]] = {}
            for _, row in self.synergy_results.iterrows():
                for pattern in [row["pattern_a"], row["pattern_b"]]:
                    if pattern not in synergy_per_pattern:
                        synergy_per_pattern[pattern] = []
                    synergy_per_pattern[pattern].append(row["synergy_score"])

            avg_synergy = {k: np.mean(v) for k, v in synergy_per_pattern.items()}
            leaderboard["avg_synergy_score"] = leaderboard["pattern_name"].map(avg_synergy)
        else:
            leaderboard["avg_synergy_score"] = np.nan

        # Calculate composite score
        # Normalize metrics to 0-1 range
        def normalize(series):
            if series.isna().all():
                return series
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return series * 0
            return (series - min_val) / (max_val - min_val)

        leaderboard["norm_win_rate"] = normalize(leaderboard["win_rate"])
        leaderboard["norm_trade_count"] = normalize(leaderboard["trade_count"])
        leaderboard["norm_ablation"] = normalize(leaderboard["ablation_delta_sharpe"])
        leaderboard["norm_synergy"] = normalize(leaderboard["avg_synergy_score"])

        # Composite score: weighted combination
        # Weights: win_rate (0.3), trade_count (0.2), ablation (0.3), synergy (0.2)
        leaderboard["composite_score"] = (
            leaderboard["norm_win_rate"].fillna(0) * 0.3
            + leaderboard["norm_trade_count"].fillna(0) * 0.2
            + leaderboard["norm_ablation"].fillna(0) * 0.3
            + leaderboard["norm_synergy"].fillna(0) * 0.2
        )

        # Sort by composite score
        leaderboard = leaderboard.sort_values("composite_score", ascending=False).reset_index(
            drop=True
        )

        # Add rank
        leaderboard["rank"] = range(1, len(leaderboard) + 1)

        # Select columns
        result = leaderboard[
            [
                "rank",
                "pattern_name",
                "trade_count",
                "win_rate",
                "avg_pnl",
                "ablation_delta_sharpe",
                "avg_synergy_score",
                "composite_score",
            ]
        ].copy()

        return result

    def get_pattern_roles(self) -> Dict[str, str]:  # type: ignore[return-value]
        """
        Classify each pattern into a role:
        - "Primary Signal" — high solo edge, high ablation contribution
        - "Confirmation Filter" — low solo edge, positive ablation contribution
        - "Noise Generator" — negative ablation contribution (should be removed)
        - "Neutral" — no significant impact

        Returns:
            Dictionary mapping pattern name to role
        """
        roles: Dict[str, str] = {}

        if self.ablation_results is None or self.ablation_results.empty:
            return roles

        # Get pattern trade stats
        pattern_stats = self.attributor.get_pattern_trade_stats()

        for _, row in self.ablation_results.iterrows():
            pattern = row["pattern_name"]
            delta_sharpe = row["delta_sharpe"]

            # Get win rate from pattern stats
            win_rate = 0.0
            if not pattern_stats.empty:
                pattern_row = pattern_stats[pattern_stats["pattern_name"] == pattern]
                if not pattern_row.empty:
                    win_rate = pattern_row.iloc[0]["win_rate"]

            # Classify based on delta_sharpe and win_rate
            if delta_sharpe > 0.1 and win_rate > 0.5:
                roles[pattern] = "Primary Signal"
            elif delta_sharpe > 0 and win_rate > 0.45:
                roles[pattern] = "Confirmation Filter"
            elif delta_sharpe < -0.05:
                roles[pattern] = "Noise Generator"
            else:
                roles[pattern] = "Neutral"

        return roles

    def get_recommendations(self) -> List[str]:
        """
        Actionable recommendations:
        - "Remove Doji — ablation shows +1.7% improvement without it"
        - "Keep H&S — highest marginal contributor (-6.9% delta)"
        - "Investigate H&S+MSL pair — highest synergy score"

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Ablation-based recommendations
        if self.ablation_results is not None and not self.ablation_results.empty:
            # Patterns to remove (positive delta_sharpe)
            removal_candidates = self.ablation_results[self.ablation_results["delta_sharpe"] > 0.05]
            for _, row in removal_candidates.iterrows():
                pattern = row["pattern_name"]
                delta = row["delta_sharpe"]
                recommendations.append(
                    f"Remove {pattern} — ablation shows +{delta:.1%} Sharpe improvement without it"
                )

            # Top contributors to keep
            top_contributors = self.ablation_results[self.ablation_results["delta_sharpe"] < -0.1]
            for _, row in top_contributors.iterrows():
                pattern = row["pattern_name"]
                delta = row["delta_sharpe"]
                recommendations.append(
                    f"Keep {pattern} — highest marginal contributor ({delta:.1%} delta Sharpe)"
                )

        # Synergy-based recommendations
        if self.synergy_results is not None and not self.synergy_results.empty:
            # Best synergy pairs
            best_pairs = self.synergy_results.nlargest(3, "synergy_score")
            for _, row in best_pairs.iterrows():
                pattern_a = row["pattern_a"]
                pattern_b = row["pattern_b"]
                synergy = row["synergy_score"]
                recommendations.append(
                    f"Investigate {pattern_a}+{pattern_b} pair — synergy score: {synergy:.3f}"
                )

            # Worst synergy pairs
            worst_pairs = self.synergy_results.nsmallest(3, "synergy_score")
            for _, row in worst_pairs.iterrows():
                pattern_a = row["pattern_a"]
                pattern_b = row["pattern_b"]
                synergy = row["synergy_score"]
                recommendations.append(
                    f"Consider separating {pattern_a}+{pattern_b} — negative synergy: {synergy:.3f}"
                )

        # Trade attribution recommendations
        pattern_stats = self.attributor.get_pattern_trade_stats()
        if not pattern_stats.empty:
            # Low win rate patterns
            low_win_rate = pattern_stats[pattern_stats["win_rate"] < 0.4]
            for _, row in low_win_rate.iterrows():
                pattern = row["pattern_name"]
                win_rate = row["win_rate"]
                recommendations.append(f"Review {pattern} — low win rate: {win_rate:.1%}")

        return recommendations

    def to_markdown(self) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown string
        """
        lines = []
        lines.append("# Pattern Contribution Analysis Report\n")

        # Summary
        summary = self.generate_summary()
        lines.append("## Executive Summary\n")

        signal_stats = summary["signal_log"]
        lines.append(f"- **Total Signal Events**: {signal_stats['total_events']:,}")
        lines.append(f"- **Unique Patterns**: {signal_stats['unique_patterns']}")
        lines.append(f"- **Unique Bars**: {signal_stats['unique_bars']:,}")
        lines.append(f"- **Avg Events per Bar**: {signal_stats['avg_events_per_bar']:.2f}\n")

        attribution_stats = summary["trade_attribution"]
        lines.append(f"- **Total Trades**: {attribution_stats['total_trades']}")
        lines.append(f"- **Attributed Trades**: {attribution_stats['attributed_trades']}")
        lines.append(f"- **Attribution Rate**: {attribution_stats['attribution_rate']:.1%}\n")

        # Pattern Leaderboard
        leaderboard = self.get_pattern_leaderboard()
        if not leaderboard.empty:
            lines.append("## Pattern Leaderboard\n")
            lines.append(
                "| Rank | Pattern | Trades | Win Rate | Avg P&L | Ablation ΔSharpe | Synergy | Score |"
            )
            lines.append(
                "|------|---------|--------|----------|---------|------------------|---------|-------|"
            )
            for _, row in leaderboard.head(20).iterrows():
                lines.append(
                    f"| {int(row['rank'])} | {row['pattern_name']} | {int(row['trade_count'])} | "
                    f"{row['win_rate']:.1%} | {row['avg_pnl']:.2f} | "
                    f"{row['ablation_delta_sharpe']:.3f} | {row['avg_synergy_score']:.3f} | "
                    f"{row['composite_score']:.3f} |"
                )
            lines.append("")

        # Pattern Roles
        roles = self.get_pattern_roles()
        if roles:
            lines.append("## Pattern Roles\n")
            role_counts: Dict[str, int] = {}
            for pattern, role in roles.items():
                role_counts[role] = role_counts.get(role, 0) + 1

            for role, count in sorted(role_counts.items()):
                lines.append(f"- **{role}**: {count} patterns")

            lines.append("")

        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            lines.append("## Recommendations\n")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str) -> Dict[str, str]:
        """
        Save all results to files.

        Args:
            output_dir: Directory to save results

        Returns:
            Dictionary mapping file name to file path
        """
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Save summary
        summary = self.generate_summary()
        summary_path = os.path.join(output_dir, "summary.json")
        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files["summary"] = summary_path

        # Save leaderboard
        leaderboard = self.get_pattern_leaderboard()
        if not leaderboard.empty:
            leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
            leaderboard.to_csv(leaderboard_path, index=False)
            saved_files["leaderboard"] = leaderboard_path

        # Save roles
        roles = self.get_pattern_roles()
        if roles:
            roles_path = os.path.join(output_dir, "roles.json")
            with open(roles_path, "w") as f:
                json.dump(roles, f, indent=2)
            saved_files["roles"] = roles_path

        # Save recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            rec_path = os.path.join(output_dir, "recommendations.txt")
            with open(rec_path, "w", encoding="utf-8") as f:
                f.write("\n".join(recommendations))
            saved_files["recommendations"] = rec_path

        # Save markdown report
        markdown = self.to_markdown()
        markdown_path = os.path.join(output_dir, "report.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        saved_files["report"] = markdown_path

        print(f"Report saved to {output_dir}")
        return saved_files

    def __repr__(self) -> str:
        """String representation."""
        summary = self.generate_summary()
        signal_stats = summary["signal_log"]
        attribution_stats = summary["trade_attribution"]
        return (
            f"ContributionReport("
            f"events={signal_stats['total_events']}, "
            f"trades={attribution_stats['total_trades']}, "
            f"attributed={attribution_stats['attributed_trades']})"
        )
