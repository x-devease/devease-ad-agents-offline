"""
Aesthetic Guard - Visual quality validation.

This guard checks for AI artifacts, composition issues,
and violations of negative prompts.
"""

import time
from typing import Optional, List

from ..schemas.audit_report import GuardStatus, AestheticResult
from ..vlms.base import VLMClient


class AestheticGuard:
    """
    Validates aesthetic quality of generated images.

    Uses VLM to detect artifacts, check composition,
    and verify negative prompt compliance.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        min_score: float = 7.0
    ):
        """
        Initialize AestheticGuard.

        Args:
            vlm_client: VLM client for image analysis
            min_score: Minimum aesthetic score (0-10 scale)
        """
        self.vlm_client = vlm_client
        self.min_score = min_score

    def check(
        self,
        image_path: str,
        negative_prompts: Optional[List[str]] = None
    ) -> AestheticResult:
        """
        Check aesthetic quality of an image.

        Args:
            image_path: Path to generated image
            negative_prompts: List of features to avoid

        Returns:
            AestheticResult with validation status
        """
        start_time = time.time()

        try:
            # Use VLM client's built-in method
            result = self.vlm_client.check_aesthetics(
                image_path=image_path,
                negative_prompts=negative_prompts or []
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract score and issues
            score = result.get("score", 0.0)
            issues = result.get("issues", [])
            has_negative = result.get("has_negative_feature", False)
            detected_features = result.get("detected_features", [])
            reasoning = result.get("reasoning", "")

            # Determine status
            # FAIL if score too low OR if negative feature detected
            if has_negative:
                status = GuardStatus.FAIL
                fail_reason = f"Negative feature detected: {detected_features}"
            elif score < self.min_score:
                status = GuardStatus.FAIL
                fail_reason = f"Score {score:.1f} below minimum {self.min_score}"
            else:
                status = GuardStatus.PASS
                fail_reason = ""

            # Build reasoning
            if status == GuardStatus.PASS:
                reasoning = f"Aesthetic quality PASSED with score {score:.1f}/10"
                if issues:
                    reasoning += f". Minor issues: {', '.join(issues[:3])}"
            else:
                reasoning = fail_reason
                if issues:
                    reasoning += f". Issues: {', '.join(issues[:3])}"

            return AestheticResult(
                guard_name="aesthetic",
                status=status,
                reasoning=reasoning,
                metrics={
                    "score": score,
                    "min_score": self.min_score,
                    "issues": issues,
                    "negative_prompts_checked": len(negative_prompts or []),
                    "negative_features_detected": detected_features
                },
                execution_time_ms=execution_time,
                score=score,
                issues=issues,
                negative_features_detected=detected_features
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return AestheticResult(
                guard_name="aesthetic",
                status=GuardStatus.FAIL,
                reasoning=f"Error during aesthetic check: {str(e)}",
                metrics={"error": str(e)},
                execution_time_ms=execution_time,
                score=0.0,
                issues=[str(e)],
                negative_features_detected=[]
            )
