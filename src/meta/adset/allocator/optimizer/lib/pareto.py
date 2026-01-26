"""
Pareto optimization for multi-objective budget allocation.

This module provides tools for finding and analyzing Pareto-optimal solutions
when optimizing multiple conflicting objectives (e.g., ROAS vs CTR).

A solution is Pareto-optimal if no other solution is better in all objectives.
The Pareto frontier represents the set of all Pareto-optimal solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.meta.adset.allocator.optimizer.tuning import OptimizationObjectives, TuningResult


@dataclass(frozen=True)
class ParetoPoint:
    """A single point on the Pareto frontier."""

    result: TuningResult  # Full tuning result
    roas_score: float  # ROAS objective value
    ctr_score: float  # CTR objective value
    stability_score: float  # Stability objective value
    dominated_by: int  # Number of solutions that dominate this one
    dominates_count: int  # Number of solutions this one dominates


@dataclass(frozen=True)
class ParetoFrontier:
    """Pareto frontier with analysis and recommendations."""

    frontier_points: List[ParetoPoint]  # All Pareto-optimal solutions
    all_results: List[TuningResult]  # All evaluated solutions
    objectives: OptimizationObjectives  # Objectives used for optimization

    # Frontier characteristics
    roas_range: Tuple[float, float]  # (min, max) ROAS on frontier
    ctr_range: Tuple[float, float]  # (min, max) CTR on frontier
    stability_range: Tuple[float, float]  # (min, max) stability on frontier

    # Trade-off analysis
    roas_ctr_correlation: float  # Correlation between ROAS and CTR
    knee_point: Optional[ParetoPoint]  # Point of diminishing returns
    recommended_point: Optional[ParetoPoint]  # Best point based on objectives

    def get_frontier_df(self) -> pd.DataFrame:
        """Get Pareto frontier as DataFrame for analysis.

        Returns:
            DataFrame with frontier metrics.
        """
        data = []
        for point in self.frontier_points:
            data.append(
                {
                    "weighted_avg_roas": point.result.weighted_avg_roas,
                    "weighted_avg_ctr": point.result.weighted_avg_ctr,
                    "roas_std": point.result.roas_std,
                    "ctr_std": point.result.ctr_std,
                    "budget_utilization": point.result.budget_utilization,
                    "change_rate": point.result.change_rate,
                    "roas_score": point.roas_score,
                    "ctr_score": point.ctr_score,
                    "stability_score": point.stability_score,
                    "dominated_by": point.dominated_by,
                    "dominates_count": point.dominates_count,
                }
            )
        return pd.DataFrame(data)

    def get_trade_off_summary(self) -> Dict[str, any]:
        """Get summary of trade-offs on the frontier.

        Returns:
            Dictionary with trade-off analysis.
        """
        if len(self.frontier_points) < 2:
            return {"error": "Need at least 2 points for trade-off analysis"}

        df = self.get_frontier_df()

        # Calculate opportunity costs
        max_roas_idx = df["weighted_avg_roas"].idxmax()
        max_ctr_idx = df["weighted_avg_ctr"].idxmax()

        max_roas_point = df.loc[max_roas_idx]
        max_ctr_point = df.loc[max_ctr_idx]

        roas_to_ctr_cost = (
            (max_roas_point["weighted_avg_roas"] - max_ctr_point["weighted_avg_roas"])
            / max_roas_point["weighted_avg_roas"]
            * 100
        )
        ctr_to_roas_cost = (
            (max_ctr_point["weighted_avg_ctr"] - max_roas_point["weighted_avg_ctr"])
            / max_ctr_point["weighted_avg_ctr"]
            * 100
        )

        return {
            "roas_range": self.roas_range,
            "ctr_range": self.ctr_range,
            "correlation": self.roas_ctr_correlation,
            "max_roas_point": {
                "roas": max_roas_point["weighted_avg_roas"],
                "ctr": max_roas_point["weighted_avg_ctr"],
            },
            "max_ctr_point": {
                "roas": max_ctr_point["weighted_avg_roas"],
                "ctr": max_ctr_point["weighted_avg_ctr"],
            },
            "opportunity_cost": {
                "roas_to_ctr_loss_pct": roas_to_ctr_cost,
                "ctr_to_roas_loss_pct": ctr_to_roas_cost,
            },
            "frontier_size": len(self.frontier_points),
            "total_solutions": len(self.all_results),
        }


def is_dominated(
    candidate: Tuple[float, float, float],
    other: Tuple[float, float, float],
    objectives: OptimizationObjectives,
) -> bool:
    """Check if candidate solution is dominated by other solution.

    A solution dominates another if it is better or equal in ALL objectives
    and strictly better in at least ONE objective.

    Args:
        candidate: (roas_score, ctr_score, stability_score) for candidate
        other: (roas_score, ctr_score, stability_score) for other
        objectives: Objectives defining preference weights

    Returns:
        True if candidate is dominated by other.
    """
    roas_c, ctr_c, stab_c = candidate
    roas_o, ctr_o, stab_o = other

    # Higher scores are better for all objectives
    at_least_one_better = False

    # ROAS comparison
    if roas_o < roas_c:
        return False  # Other is worse in ROAS, doesn't dominate
    elif roas_o > roas_c:
        at_least_one_better = True

    # CTR comparison
    if ctr_o < ctr_c:
        return False  # Other is worse in CTR, doesn't dominate
    elif ctr_o > ctr_c:
        at_least_one_better = True

    # Stability comparison
    if stab_o < stab_c:
        return False  # Other is worse in stability, doesn't dominate
    elif stab_o > stab_c:
        at_least_one_better = True

    # Dominate only if better in at least one and not worse in any
    return at_least_one_better


def compute_pareto_frontier(
    results: List[TuningResult],
    objectives: OptimizationObjectives,
) -> ParetoFrontier:
    """Compute Pareto frontier from tuning results.

    Args:
        results: List of tuning results to analyze.
        objectives: Optimization objectives used for scoring.

    Returns:
        ParetoFrontier with frontier points and analysis.
    """
    if not results:
        return ParetoFrontier(
            frontier_points=[],
            all_results=[],
            objectives=objectives,
            roas_range=(0.0, 0.0),
            ctr_range=(0.0, 0.0),
            stability_range=(0.0, 0.0),
            roas_ctr_correlation=0.0,
            knee_point=None,
            recommended_point=None,
        )

    # Calculate scores for each result
    scored_results = []
    for result in results:
        roas_score = float(result.weighted_avg_roas) * objectives.roas_weight
        ctr_score = float(result.weighted_avg_ctr) * 100 * objectives.ctr_weight
        stability_score = objectives.stability_weight - (
            result.roas_std * 0.5 if result.roas_std > 1.5 else 0
        )
        scored_results.append((result, roas_score, ctr_score, stability_score))

    # Find dominance relationships
    dominated_counts = {i: 0 for i in range(len(scored_results))}
    dominates_counts = {i: 0 for i in range(len(scored_results))}

    for i in range(len(scored_results)):
        for j in range(i + 1, len(scored_results)):
            _, roas_i, ctr_i, stab_i = scored_results[i]
            _, roas_j, ctr_j, stab_j = scored_results[j]

            score_i = (roas_i, ctr_i, stab_i)
            score_j = (roas_j, ctr_j, stab_j)

            if is_dominated(score_i, score_j, objectives):
                dominated_counts[i] += 1
                dominates_counts[j] += 1
            elif is_dominated(score_j, score_i, objectives):
                dominated_counts[j] += 1
                dominates_counts[i] += 1

    # Pareto-optimal solutions are those with dominated_count == 0
    frontier_indices = [i for i, count in dominated_counts.items() if count == 0]

    # Create ParetoPoints
    frontier_points = []
    for idx in frontier_indices:
        result, roas_score, ctr_score, stability_score = scored_results[idx]
        point = ParetoPoint(
            result=result,
            roas_score=roas_score,
            ctr_score=ctr_score,
            stability_score=stability_score,
            dominated_by=dominated_counts[idx],
            dominates_count=dominates_counts[idx],
        )
        frontier_points.append(point)

    # Sort frontier by ROAS (descending) for easier analysis
    frontier_points.sort(key=lambda p: p.result.weighted_avg_roas, reverse=True)

    # Calculate ranges
    frontier_roas = [p.result.weighted_avg_roas for p in frontier_points]
    frontier_ctr = [p.result.weighted_avg_ctr for p in frontier_points]
    frontier_stability = [
        (
            p.result.stability_score
            if hasattr(p.result, "stability_score")
            else (3.0 - p.result.roas_std)
        )  # Approximate stability
        for p in frontier_points
    ]

    # Calculate correlation
    if len(frontier_points) > 1:
        roas_ctr_correlation = float(np.corrcoef(frontier_roas, frontier_ctr)[0, 1])
    else:
        roas_ctr_correlation = 0.0

    # Find knee point (point of diminishing returns)
    knee_point = None
    if len(frontier_points) > 2:
        # Use angle-based knee detection
        # Knee is where the curve bends most sharply
        roas_vals = np.array(frontier_roas)
        ctr_vals = np.array(frontier_ctr)

        # Normalize to [0, 1]
        roas_norm = (roas_vals - roas_vals.min()) / (
            roas_vals.max() - roas_vals.min() + 1e-6
        )
        ctr_norm = (ctr_vals - ctr_vals.min()) / (
            ctr_vals.max() - ctr_vals.min() + 1e-6
        )

        # Calculate angles at each point
        max_angle = -1
        knee_idx = 0
        for i in range(1, len(roas_norm) - 1):
            v1 = np.array(
                [roas_norm[i - 1] - roas_norm[i], ctr_norm[i - 1] - ctr_norm[i]]
            )
            v2 = np.array(
                [roas_norm[i + 1] - roas_norm[i], ctr_norm[i + 1] - ctr_norm[i]]
            )

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            if angle > max_angle:
                max_angle = angle
                knee_idx = i

        knee_point = frontier_points[knee_idx]

    # Find recommended point based on objectives
    recommended_point = None
    if frontier_points:
        # Weighted score
        best_score = -float("inf")
        best_idx = 0
        for i, point in enumerate(frontier_points):
            total_score = (
                point.roas_score * objectives.roas_weight
                + point.ctr_score * objectives.ctr_weight
                + point.stability_score * objectives.stability_weight
            )
            if total_score > best_score:
                best_score = total_score
                best_idx = i
        recommended_point = frontier_points[best_idx]

    return ParetoFrontier(
        frontier_points=frontier_points,
        all_results=results,
        objectives=objectives,
        roas_range=(
            (min(frontier_roas), max(frontier_roas)) if frontier_roas else (0.0, 0.0)
        ),
        ctr_range=(
            (min(frontier_ctr), max(frontier_ctr)) if frontier_ctr else (0.0, 0.0)
        ),
        stability_range=(
            (min(frontier_stability), max(frontier_stability))
            if frontier_stability
            else (0.0, 0.0)
        ),
        roas_ctr_correlation=roas_ctr_correlation,
        knee_point=knee_point,
        recommended_point=recommended_point,
    )


def visualize_pareto_frontier(
    frontier: ParetoFrontier,
    save_path: Optional[str] = None,
) -> None:
    """Visualize Pareto frontier (requires matplotlib).

    Args:
        frontier: ParetoFrontier to visualize.
        save_path: Optional path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]  matplotlib not installed, skipping visualization")
        return

    df = frontier.get_frontier_df()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: ROAS vs CTR frontier
    ax1 = axes[0]
    ax1.scatter(
        df["weighted_avg_roas"],
        df["weighted_avg_ctr"],
        c=range(len(df)),
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )
    ax1.set_xlabel("Weighted Avg ROAS", fontsize=12)
    ax1.set_ylabel("Weighted Avg CTR (%)", fontsize=12)
    ax1.set_title("Pareto Frontier: ROAS vs CTR", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Highlight knee point and recommended point
    if frontier.knee_point:
        ax1.scatter(
            [frontier.knee_point.result.weighted_avg_roas],
            [frontier.knee_point.result.weighted_avg_ctr],
            s=200,
            marker="*",
            c="gold",
            edgecolors="black",
            linewidths=2,
            label="Knee Point",
            zorder=10,
        )
    if frontier.recommended_point:
        ax1.scatter(
            [frontier.recommended_point.result.weighted_avg_roas],
            [frontier.recommended_point.result.weighted_avg_ctr],
            s=200,
            marker="*",
            c="red",
            edgecolors="black",
            linewidths=2,
            label="Recommended",
            zorder=10,
        )
    ax1.legend()

    # Plot 2: Objective scores
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.25
    ax2.bar(x - width, df["roas_score"], width, label="ROAS", alpha=0.8)
    ax2.bar(x, df["ctr_score"], width, label="CTR", alpha=0.8)
    ax2.bar(x + width, df["stability_score"], width, label="Stability", alpha=0.8)
    ax2.set_xlabel("Pareto Point", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Objective Scores on Frontier", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{i+1}" for i in range(len(df))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved Pareto frontier visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def explain_pareto_frontier(frontier: ParetoFrontier) -> str:
    """Generate human-readable explanation of Pareto frontier.

    Args:
        frontier: ParetoFrontier to explain.

    Returns:
        String explanation.
    """
    if not frontier.frontier_points:
        return "No Pareto frontier found (all solutions may be equal)."

    summary = frontier.get_trade_off_summary()

    # Handle case where trade-off summary returns an error (e.g., < 2 points)
    if "error" in summary:
        # Build partial explanation with available information
        lines = [
            "=" * 70,
            "PARETO FRONTIER ANALYSIS",
            "=" * 70,
            "",
            f"Frontier Size: {len(frontier.frontier_points)} point(s)",
            f"Total Solutions Evaluated: {len(frontier.all_results)}",
            "",
            f"Note: Insufficient data for detailed trade-off analysis ({summary['error']})",
            "",
        ]
    else:
        lines = [
            "=" * 70,
            "PARETO FRONTIER ANALYSIS",
            "=" * 70,
            "",
            f"Frontier Size: {summary['frontier_size']} optimal solutions",
            f"Total Solutions Evaluated: {summary['total_solutions']}",
            "",
            "OBJECTIVE RANGES ON FRONTIER:",
            f"  ROAS:  {summary['roas_range'][0]:.4f} → {summary['roas_range'][1]:.4f}",
            f"  CTR:   {summary['ctr_range'][0]:.4f}% → {summary['ctr_range'][1]:.4f}%",
            "",
            f"ROAS-CTR Correlation: {summary['correlation']:.3f}",
            "  (negative = trade-off, positive = aligned)",
            "",
        ]

        if "opportunity_cost" in summary:
            oc = summary["opportunity_cost"]
            lines.extend(
                [
                    "OPPORTUNITY COST (Choosing one objective over the other):",
                    f"  Maximizing ROAS sacrifices {oc['roas_to_ctr_loss_pct']:.1f}% of CTR",
                    f"  Maximizing CTR sacrifices {oc['ctr_to_roas_loss_pct']:.1f}% of ROAS",
                    "",
                ]
            )

    # Add knee point and recommended point (available even with trade-off error)
    if frontier.knee_point:
        lines.extend(
            [
                "KNEE POINT (Point of diminishing returns):",
                f"  ROAS: {frontier.knee_point.result.weighted_avg_roas:.4f}",
                f"  CTR:  {frontier.knee_point.result.weighted_avg_ctr:.4f}%",
                "  → Good balance between objectives",
                "",
            ]
        )

    if frontier.recommended_point:
        lines.extend(
            [
                "RECOMMENDED POINT (Based on your objectives):",
                f"  ROAS: {frontier.recommended_point.result.weighted_avg_roas:.4f}",
                f"  CTR:  {frontier.recommended_point.result.weighted_avg_ctr:.4f}%",
                f"  → Matches weights: ROAS={frontier.objectives.roas_weight:.0%}, "
                f"CTR={frontier.objectives.ctr_weight:.0%}, "
                f"Stability={frontier.objectives.stability_weight:.0%}",
                "",
            ]
        )

    lines.extend(
        [
            "INTERPRETATION:",
            "  - Pareto frontier shows all 'non-dominated' solutions",
            "  - No solution on the frontier is strictly better than another",
            "  - Choose based on your preference: ROAS vs CTR vs Stability",
            "  - Knee point = good compromise, Recommended = matches your weights",
            "",
            "=" * 70,
        ]
    )

    return "\n".join(lines)
