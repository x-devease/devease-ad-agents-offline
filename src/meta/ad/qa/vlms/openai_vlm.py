"""
OpenAI Vision Language Model implementation.

This module provides GPT-4o / GPT-4o-mini vision support.
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path

from .base import VLMClient


class OpenAIVLM(VLMClient):
    """
    OpenAI GPT-4o Vision API client.

    Supports GPT-4o, GPT-4o-mini, and other OpenAI vision models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize OpenAI VLM client.

        Args:
            model: Model identifier (default: "gpt-4o-mini")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            timeout: Request timeout in seconds
        """
        super().__init__(model, temperature, max_tokens)

        self.api_key = api_key
        self.timeout = timeout

        # Lazy import to avoid hard dependency
        try:
            import openai
            self.openai = openai
            self.client = None  # Created on first use
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

    def _get_client(self):
        """Get or create OpenAI client."""
        if self.client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self.client = self.openai.OpenAI(**kwargs)
        return self.client

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Analyze an image using OpenAI's vision API.

        Args:
            image_path: Path to image file
            prompt: Text prompt for analysis
            response_format: Expected response format ("json" or "text")

        Returns:
            Parsed response as dictionary

        Raises:
            Exception: If API call fails
        """
        # Validate image path
        image_path = self.validate_image_path(image_path)

        # Encode image
        base64_image = self.encode_image_base64(str(image_path))

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        try:
            client = self._get_client()

            start_time = time.time()

            # Make API call
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            elapsed = time.time() - start_time

            # Extract response text
            response_text = response.choices[0].message.content

            # Parse response
            if response_format == "json":
                parsed = self.parse_json_response(response_text)
            else:
                parsed = {"response": response_text}

            # Add metadata
            parsed["_metadata"] = {
                "model": self.model,
                "response_time_seconds": elapsed,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }

            return parsed

        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    def check_aesthetics(
        self,
        image_path: str,
        negative_prompts: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Check image aesthetic quality.

        Args:
            image_path: Path to image
            negative_prompts: List of features to avoid

        Returns:
            Dict with score, issues, and negative_features
        """
        negative_list = negative_prompts or []
        negative_str = ", ".join(f'"{item}"' for item in negative_list) if negative_list else "None"

        prompt = f"""You are an expert visual quality analyst for advertising.

Analyze this image for:
1. **Technical Quality**: Glitches, noise, blur, artifacts, seams?
2. **Layout & Composition**: Balanced? Good whitespace usage?
3. **Negative Features**: Any of these forbidden elements?
   {negative_str}

Score the image 0-10 and explain issues.

Return JSON:
{{
  "score": float (0-10),
  "issues": ["issue1", "issue2"],
  "has_negative_feature": bool,
  "detected_features": ["feature1"],
  "reasoning": "explanation"
}}
"""

        return self.analyze_image(image_path, prompt, response_format="json")

    def check_culture(
        self,
        image_path: str,
        region: str,
        taboos: list
    ) -> Dict[str, Any]:
        """
        Check image for cultural compliance issues.

        Args:
            image_path: Path to image
            region: Target region
            taboos: List of forbidden elements

        Returns:
            Dict with risk_level, detected_issues, confidence
        """
        taboo_str = "\n".join(f"- {t}" for t in taboos)

        prompt = f"""You are a cultural compliance officer for {region}.

Check this image for these forbidden elements:
{taboo_str}

Return JSON:
{{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "detected_issues": ["issue1"],
  "confidence": float (0-1),
  "reasoning": "explanation"
}}
"""

        return self.analyze_image(image_path, prompt, response_format="json")

    def score_performance(
        self,
        image_path: str,
        psychology_goal: str
    ) -> Dict[str, Any]:
        """
        Score image on performance dimensions.

        Args:
            image_path: Path to image
            psychology_goal: Target psychological driver (e.g., "trust")

        Returns:
            Dict with dimension scores and overall score
        """
        prompt = f"""You are a creative performance analyst scoring an advertisement.

Psychology Goal: {psychology_goal}

Score on three dimensions (0-100 each):

1. **PSYCHOLOGY ALIGNMENT**:
   - Does it convey "{psychology_goal}"?
   - Are visual cues consistent?
   - Any conflicting emotions?

2. **SALIENCY & CLARITY**:
   - Is product the focal point?
   - Are CTA/text elements readable?
   - Is visual hierarchy clear?

3. **CONSISTENCY & REALISM**:
   - Do lighting/shadows appear natural?
   - Do foreground/background lighting match?
   - Does it look like a real photograph?

Return JSON:
{{
  "psychology_alignment": int (0-100),
  "saliency_clarity": int (0-100),
  "consistency_realism": int (0-100),
  "overall_score": int (0-100),
  "reasoning": "explanation",
  "suggestions": ["suggestion1"]
}}
"""

        return self.analyze_image(image_path, prompt, response_format="json")
