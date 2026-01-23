"""
Image Feature Validator

Validates if features from creative scorer recommendations are present in
generated images using vision models (Claude Vision API).

This closes the loop on feature reproduction:
Formula → Prompt → Image → Validation
"""

import base64
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx


logger = logging.getLogger(__name__)


@dataclass
class ImageFeatureValidationResult:
    """Result of validating a single feature in an image."""

    feature_name: str
    expected_value: str
    detected_value: Optional[str]
    is_present: bool
    confidence: float  # 0.0 - 1.0
    reasoning: str
    image_region: Optional[str] = None  # e.g., "top-left corner"


class FeatureValidator:
    """
    Validates feature reproduction in generated images using Claude Vision API.

    Usage:
        validator = FeatureValidator(api_key="anthropic_api_key")
        results = validator.validate_features_in_image(
            image_path="generated.jpg",
            expected_features=[
                {"feature_name": "lighting", "expected_value": "studio"},
                {"feature_name": "color_balance", "expected_value": "warm"},
            ],
            prompt_text="Professional studio lighting with warm tones...",
        )

        for result in results:
            print(f"{result.feature_name}: {'[OK]' if result.is_present else '[FAIL]'} "
                  f"({result.confidence:.0%}) - {result.reasoning}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: int = 120,
    ):
        """
        Initialize validator.

        Args:
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
            model: Claude model to use for vision
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Set environment variable or pass api_key parameter."
            )

        self.model = model
        self.timeout = timeout
        self.api_url = "https://api.anthropic.com/v1/messages"

    @classmethod
    def from_env(cls, model: str = "claude-3-5-sonnet-20241022", timeout: int = 120):
        """
        Create validator from environment variables.

        Args:
            model: Claude model to use for vision
            timeout: Request timeout in seconds

        Returns:
            FeatureValidator instance configured with ANTHROPIC_API_KEY from environment

        Raises:
            ValueError: If ANTHROPIC_API_KEY environment variable is not set
        """
        return cls(api_key=None, model=model, timeout=timeout)

    def validate_features_in_image(
        self,
        image_path: str,
        expected_features: List[Dict[str, str]],
        prompt_text: Optional[str] = None,
    ) -> List[ImageFeatureValidationResult]:
        """
        Validate if expected features are present in the generated image.

        Args:
            image_path: Path to generated image
            expected_features: List of {feature_name, expected_value}
            prompt_text: Original prompt used to generate image (for context)

        Returns:
            List of ImageFeatureValidationResult for each feature
        """
        logger.info(
            "Validating %d features in image: %s",
            len(expected_features),
            image_path,
        )
        # Encode image
        image_base64 = self.encode_image(image_path)
        image_media_type = self.get_media_type(image_path)
        # Build validation prompt
        validation_prompt = self._build_validation_prompt(
            expected_features,
            prompt_text,
        )
        # Call Claude Vision API
        try:
            response_text = self._call_claude_vision(
                image_base64,
                image_media_type,
                validation_prompt,
            )
            # Parse JSON response
            response_data = json.loads(response_text)
            validations = response_data.get("validations", [])
            # Convert to dataclass objects
            results = [
                ImageFeatureValidationResult(
                    feature_name=v.get("feature_name", ""),
                    expected_value=v.get("expected_value", ""),
                    detected_value=v.get("detected_value"),
                    is_present=v.get("is_present", False),
                    confidence=float(v.get("confidence", 0.0)),
                    reasoning=v.get("reasoning", ""),
                    image_region=v.get("image_region"),
                )
                for v in validations
            ]
            # Log summary
            present_count = sum(1 for r in results if r.is_present)
            logger.info(
                "Feature validation complete: %d/%d features present (%.0f%%)",
                present_count,
                len(results),
                (present_count / len(results) * 100) if results else 0,
            )

            return results

        except json.JSONDecodeError as e:
            logger.error("Failed to parse Claude response as JSON: %s", e)
            logger.debug("Response text: %s", response_text)
            raise ValueError(
                "Claude returned invalid JSON. "
                "Check API response logs for details."
            ) from e

    def encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    def get_media_type(self, image_path: str) -> str:
        """
        Get media type (MIME type) for image file.

        Args:
            image_path: Path to image file

        Returns:
            Media type string (e.g., "image/jpeg")
        """
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/jpeg")

    def _build_validation_prompt(
        self,
        expected_features: List[Dict[str, str]],
        prompt_text: Optional[str],
    ) -> str:
        """Build prompt for Claude Vision API."""

        feature_list = "\n".join(
            [
                f"  - {f['feature_name']}: {f['expected_value']}"
                for f in expected_features
            ]
        )

        context_section = ""
        if prompt_text:
            context_section = f"""

**Original Prompt Used:**
{prompt_text}
"""

        prompt = f"""You are analyzing a product marketing image to validate if
specific features from a creative scorer recommendation are present.

**Features to Validate:**
{feature_list}
{context_section}
**Your Task:**
For EACH feature listed above, determine:
1. **Is it present?** (true/false) - Is the feature clearly visible in the image?
2. **Confidence** (0.0-1.0) - How confident are you in this assessment?
3. **Detected Value** - What do you actually see in the image? (be specific)
4. **Reasoning** - One sentence explaining your assessment
5. **Image Region** (optional) - Where in the image is this feature? (e.g., "center", "top-left")

**Validation Rules:**
- Be STRICT: If a feature is ambiguous or barely visible, mark as not present
- Check the ENTIRE image: Features can be in background, lighting, composition, etc.
- Use semantic understanding: "warm tones" can include orange, yellow, soft red
- Consider artistic interpretation: "studio lighting" might mean bright, even lighting without visible fixtures

**Response Format:**
Return ONLY a valid JSON object (no markdown, no extra text):

```json
{{
  "validations": [
    {{
      "feature_name": "<feature name from list>",
      "expected_value": "<expected value from list>",
      "detected_value": "<what you actually see or null>",
      "is_present": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "<brief explanation>",
      "image_region": "<location or null>"
    }}
  ]
}}
```

Analyze the image now and provide the JSON response."""

        return prompt

    def _call_claude_vision(
        self,
        image_base64: str,
        media_type: str,
        prompt: str,
    ) -> str:
        """Call Claude Vision API."""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                return data["content"][0]["text"]

        except httpx.HTTPStatusError as e:
            logger.error("Claude API error: %s", e.response.text)
            raise RuntimeError(
                f"Claude API request failed: {e.response.status_code}"
            ) from e
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Claude API request timed out after {self.timeout}s"
            ) from exc
        except Exception:
            logger.exception("Unexpected error calling Claude API")
            raise
