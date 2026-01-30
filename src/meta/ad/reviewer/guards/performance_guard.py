"""
Performance Guard - Optimization scoring.

This guard scores images on psychology alignment, saliency, and realism
for A/B test prioritization. This is NOT a filter - all images reaching
this stage are valid.
"""

import time
from typing import Dict

from ..schemas.audit_report import GuardStatus, PerformanceScore
from ..vlms.base import VLMClient


class PerformanceGuard:
    """
    Scores images on performance optimization potential.

    This guard does NOT filter - it scores images for ranking.
    All images that reach this guard have passed previous validation.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        weights: Dict[str, float] = None
    ):
        """
        Initialize PerformanceGuard.

        Args:
            vlm_client: VLM client for image analysis
            weights: Dict with 'psychology', 'saliency', 'consistency' weights
        """
        self.vlm_client = vlm_client

        # Default weights
        if weights is None:
            weights = {
                'psychology': 0.40,
                'saliency': 0.30,
                'consistency': 0.30
            }

        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.weights = weights

    def score(
        self,
        image_path: str,
        psychology_goal: str
    ) -> PerformanceScore:
        """
        Score an image on performance dimensions.

        Args:
            image_path: Path to generated image
            psychology_goal: Target psychological driver (e.g., "trust")

        Returns:
            PerformanceScore with dimension scores and overall score
        """
        start_time = time.time()

        try:
            # Use VLM client's built-in method
            result = self.vlm_client.score_performance(
                image_path=image_path,
                psychology_goal=psychology_goal
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract scores
            psychology = result.get("psychology_alignment", 0)
            saliency = result.get("saliency_clarity", 0)
            consistency = result.get("consistency_realism", 0)
            overall = result.get("overall_score", 0)
            reasoning = result.get("reasoning", "")
            suggestions = result.get("suggestions", [])

            # Calculate weighted overall score if not provided
            if overall == 0:
                overall = int(
                    psychology * self.weights['psychology'] +
                    saliency * self.weights['saliency'] +
                    consistency * self.weights['consistency']
                )

            # Performance guard never fails - it only scores
            return PerformanceScore(
                guard_name="performance",
                status=GuardStatus.PASS,  # Always PASS - this is a scorer, not filter
                reasoning=f"Performance score: {overall}/100. {reasoning}",
                metrics={
                    "psychology_alignment": psychology,
                    "saliency_clarity": saliency,
                    "consistency_realism": consistency,
                    "overall_score": overall,
                    "psychology_goal": psychology_goal,
                    "weights": self.weights,
                    "suggestions": suggestions
                },
                execution_time_ms=execution_time,
                overall_score=overall,
                psychology_alignment=psychology,
                saliency_clarity=saliency,
                consistency_realism=consistency
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_reasoning = f"Scoring error: {str(e)}"
            # Return a low score on error
            return PerformanceScore(
                guard_name="performance",
                status=GuardStatus.PASS,  # Still pass - don't block on scoring errors
                reasoning=f"Performance scoring error: {error_reasoning}",
                metrics={"error": str(e)},
                execution_time_ms=execution_time,
                overall_score=0,
                psychology_alignment=0,
                saliency_clarity=0,
                consistency_realism=0
            )
