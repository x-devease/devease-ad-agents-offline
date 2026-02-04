"""
Text Extractor - Extract product description information from prompts.

Extracts product name, key features, and benefit text from prompts
for automatic text overlay on generated ad creatives.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ExtractedText:
    """Extracted text elements for overlay."""
    product_name: str
    key_features: List[str]
    benefit_text: Optional[str] = None
    cta_text: Optional[str] = None
    brand_name: Optional[str] = None


class TextExtractor:
    """
    Extract product description text from prompts.

    Analyzes prompt text and feature dictionaries to extract
    relevant text elements for overlay on generated images.
    """

    # Patterns for extracting product information
    PRODUCT_NAME_PATTERNS = [
        r"(?:product|showing|featuring)\s+([A-Z][a-zA-Z\s]+?)(?:\s+in\s+detail)",
        r"(?:displaying|showcasing)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+,|\s+\.)",
        r"([A-Z][a-zA-Z]+)\s+displays?\s+(?:\w+|\w+\s+\w+)",
    ]

    FEATURE_PATTERNS = [
        r"displays?\s+(.+?)(?:,|\.|;)",
        r"features?\s+(?:the\s+)?([^.]+\.)",
        r"including\s+([^.]+\.)",
    ]

    BENEFIT_PHRASES = [
        "premium quality",
        "professional",
        "durable",
        "effective",
        "easy to use",
        "efficient",
        "reliable",
    ]

    CTA_PATTERNS = [
        "Shop Now",
        "Buy Now",
        "Learn More",
        "Order Today",
        "Get Yours",
    ]

    def __init__(self):
        """Initialize text extractor."""
        pass

    def extract_from_prompt(
        self,
        prompt: str,
        features: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> ExtractedText:
        """
        Extract text elements from prompt and features.

        Args:
            prompt: The prompt text
            features: Optional features dict with product info
            metadata: Optional metadata from patterns.yaml

        Returns:
            ExtractedText with product name, features, benefits, CTA
        """
        # Extract from metadata first (most reliable)
        if metadata:
            product_name = metadata.get("product", "")
        else:
            product_name = self._extract_product_name(prompt)

        # Extract key features
        key_features = self._extract_key_features(prompt, features)

        # Extract benefit text
        benefit_text = self._extract_benefit_text(prompt)

        # Extract CTA
        cta_text = self._extract_cta(prompt, features)

        # Get brand name
        brand_name = metadata.get("customer", "") if metadata else ""

        return ExtractedText(
            product_name=product_name,
            key_features=key_features[:3],  # Max 3 features
            benefit_text=benefit_text,
            cta_text=cta_text,
            brand_name=brand_name,
        )

    def _extract_product_name(self, prompt: str) -> str:
        """Extract product name from prompt text."""
        # Try regex patterns first
        for pattern in self.PRODUCT_NAME_PATTERNS:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up name
                name = re.sub(r"\s+", " ", name)
                return name

        # Fallback: look for first capitalized phrase
        words = prompt.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 3:
                # Get next 2-3 words as potential product name
                end_idx = min(i + 3, len(words))
                name = " ".join(words[i:end_idx])
                # Remove trailing punctuation
                name = re.sub(r"[,.;:!?\-]+$", "", name)
                return name

        return "Product"

    def _extract_key_features(
        self,
        prompt: str,
        features: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Extract key features from prompt and features dict."""
        feature_list = []

        # Extract from features dict if available
        if features:
            # Surface material
            if "surface_material" in features:
                material = features["surface_material"]
                feature_list.append(f"{material} surface")

            # Lighting style
            if "lighting_style" in features:
                lighting = features["lighting_style"]
                feature_list.append(f"{lighting}")

            # Camera angle
            if "camera_angle" in features:
                angle = features["camera_angle"]
                feature_list.append(f"{angle} view")

            # Color temperature
            if "color_temperature" in features:
                temp = features["color_temperature"]
                feature_list.append(f"{temp.lower()} tones")

        # Extract from prompt text using regex
        for pattern in self.FEATURE_PATTERNS:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Clean up feature text
                feature = match.strip()
                feature = re.sub(r"^[,\s]+|[,\s.]+$", "", feature)
                if len(feature) > 3 and feature not in feature_list:
                    feature_list.append(feature)

        # If still no features, use generic ones
        if not feature_list:
            feature_list = [
                "Professional quality",
                "Premium design",
                "Durable construction",
            ]

        return feature_list

    def _extract_benefit_text(self, prompt: str) -> Optional[str]:
        """Extract benefit/emotional text from prompt."""
        # Look for benefit-related words
        prompt_lower = prompt.lower()

        for benefit in self.BENEFIT_PHRASES:
            if benefit in prompt_lower:
                return benefit.capitalize()

        # Look for emotional/quality keywords
        quality_keywords = [
            "luxurious", "elegant", "professional", "premium",
            "reliable", "effective", "powerful", "efficient"
        ]

        for keyword in quality_keywords:
            if keyword in prompt_lower:
                return keyword.capitalize()

        return None

    def _extract_cta(
        self,
        prompt: str,
        features: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Extract call-to-action text."""
        # Check if prompt mentions shopping/buying
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["shop", "buy", "order", "purchase"]):
            return "Shop Now"

        if any(word in prompt_lower for word in ["learn", "discover", "explore"]):
            return "Learn More"

        # Default CTA based on context
        if features and features.get("campaign_goal") == "conversion":
            return "Shop Now"

        return None

    def format_for_overlay(
        self,
        extracted: ExtractedText,
        template: str = "minimal"
    ) -> Dict[str, any]:
        """
        Format extracted text for overlay based on template.

        Args:
            extracted: ExtractedText object
            template: Template name ('minimal', 'detailed', 'feature_focus')

        Returns:
            Dict with formatted text elements ready for overlay
        """
        if template == "minimal":
            return {
                "headline": extracted.product_name,
                "subline": "",
                "features": [],
                "cta": extracted.cta_text or "",
            }

        elif template == "detailed":
            # Select top 2 features
            top_features = extracted.key_features[:2]

            return {
                "headline": extracted.product_name,
                "subline": extracted.benefit_text or "",
                "features": top_features,
                "cta": extracted.cta_text or "Shop Now",
            }

        elif template == "feature_focus":
            # Show more features
            return {
                "headline": extracted.product_name,
                "subline": "",
                "features": extracted.key_features[:3],
                "cta": extracted.cta_text or "",
            }

        else:
            # Default to minimal
            return self.format_for_overlay(extracted, "minimal")
