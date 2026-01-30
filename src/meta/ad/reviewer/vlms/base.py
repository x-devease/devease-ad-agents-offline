"""
Base class for Vision Language Model (VLM) clients.

This module defines the abstract interface for VLM implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import base64


class VLMClient(ABC):
    """
    Abstract base class for Vision Language Model clients.

    Implementations must support image analysis with text prompts.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize VLM client.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Analyze an image with a text prompt.

        Args:
            image_path: Path to image file
            prompt: Text prompt for analysis
            response_format: Expected response format ("json" or "text")

        Returns:
            Parsed response as dictionary

        Raises:
            Exception: If API call fails
        """
        pass

    def encode_image_base64(self, image_path: str) -> str:
        """
        Encode image as base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def validate_image_path(self, image_path: str) -> Path:
        """
        Validate that image path exists and is accessible.

        Args:
            image_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If image doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return path

    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from VLM.

        Args:
            response_text: Raw response text

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        import json

        # Try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                try:
                    return json.loads(response_text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        if "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                try:
                    # Skip language identifier if present
                    content = response_text[start:end].strip()
                    if content.startswith(("json", "JSON")):
                        content = "\n".join(content.split("\n")[1:])
                    return json.loads(content.strip())
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to parse JSON from response: {response_text}")
