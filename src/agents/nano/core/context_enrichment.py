"""
Context Enrichment Engine for Nano Banana Pro Agent.

Enriches prompts with product context, brand guidelines, ROAS features,
and psychology drivers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from src.agents.nano.core.types import (
    ProductContext,
    BrandGuidelines,
    AgentInput,
)


logger = logging.getLogger(__name__)


# Mock product database (in production, this would come from a real database)
PRODUCT_DATABASE = {
    "moprobo mop": {
        "name": "moprobo mop",
        "category": "cleaning_tools",
        "key_features": [
            "360° rotating head",
            "Built-in spray mechanism",
            "Microfiber bristles",
            "Ergonomic handle",
            "Lightweight design",
        ],
        "materials": ["microfiber", "plastic", "metal"],
        "colors": ["red", "black", "white"],
        "brand_colors": ["#FF0000", "#000000", "#FFFFFF"],
    },
    "mop": {
        "name": "mop",
        "category": "cleaning_tools",
        "key_features": [
            "Cleaning tool",
            "Absorbent head",
            "Handle for reaching",
        ],
        "materials": ["microfiber", "cotton", "plastic"],
        "colors": ["varies"],
        "brand_colors": ["#FF0000"],
    },
}

# Mock brand guidelines database
BRAND_DATABASE = {
    "moprobo": BrandGuidelines(
        brand_name="moprobo",
        primary_colors=["#FF0000", "#000000"],
        secondary_colors=["#FFFFFF", "#333333"],
        typography_system="Clean, modern sans-serif (Inter, Roboto, or Arial)",
        visual_language="Clean, modern, technical precision with professional photography style",
        logo_requirements="Logo should be clearly visible, top-right or top-left, 5-8% of image width",
    ),
}


class ContextEnrichmentEngine:
    """
    Enrich generic prompts with context from product and brand databases.

    Adds:
    - Product context (features, materials, colors)
    - Brand guidelines (colors, typography, visual language)
    - ROAS-optimized features (if available)
    - Psychology driver detection
    """

    def __init__(self):
        """Initialize the context enrichment engine."""
        self.product_db = PRODUCT_DATABASE
        self.brand_db = BRAND_DATABASE

    def enrich(self, agent_input: AgentInput) -> AgentInput:
        """
        Enrich the agent input with context.

        Args:
            agent_input: The input agent_input (may have None context)

        Returns:
            Enriched agent_input with filled ProductContext and BrandGuidelines
        """

        # Enrich product context if not provided
        if agent_input.product_context is None:
            agent_input.product_context = self._extract_product_context(
                agent_input.generic_prompt
            )

        # Enrich brand guidelines if not provided
        if agent_input.brand_guidelines is None:
            agent_input.brand_guidelines = self._extract_brand_guidelines(
                agent_input.generic_prompt,
                agent_input.product_context,
            )

        logger.info(
            f"Enriched context: product={agent_input.product_context.name if agent_input.product_context else 'None'}, "
            f"brand={agent_input.brand_guidelines.brand_name if agent_input.brand_guidelines else 'None'}"
        )

        return agent_input

    def _extract_product_context(self, prompt: str) -> Optional[ProductContext]:
        """
        Extract product context from prompt using product database.

        Args:
            prompt: The generic input prompt

        Returns:
            ProductContext if found, None otherwise
        """

        prompt_lower = prompt.lower()

        # Try to find product in database
        for product_name, product_data in self.product_db.items():
            if product_name in prompt_lower:
                logger.info(f"Found product in database: {product_name}")

                return ProductContext(
                    name=product_data["name"],
                    category=product_data["category"],
                    key_features=product_data["key_features"],
                    materials=product_data["materials"],
                    colors=product_data["colors"],
                    brand_colors=product_data.get("brand_colors"),
                )

        # If no exact match, try to extract from keywords
        keywords = self._extract_product_keywords(prompt)

        if keywords:
            # Create a generic product context
            return ProductContext(
                name=keywords[0] if keywords else "product",
                category="unknown",
                key_features=[],
                materials=[],
                colors=[],
            )

        return None

    def _extract_brand_guidelines(
        self,
        prompt: str,
        product_context: Optional[ProductContext],
    ) -> Optional[BrandGuidelines]:
        """
        Extract brand guidelines from prompt or product context.

        Args:
            prompt: The generic input prompt
            product_context: Product context (may contain brand info)

        Returns:
            BrandGuidelines if found, default guidelines otherwise
        """

        prompt_lower = prompt.lower()

        # Try to find brand in prompt
        for brand_name, brand_guidelines in self.brand_db.items():
            if brand_name in prompt_lower:
                logger.info(f"Found brand in database: {brand_name}")
                return brand_guidelines

        # Try to infer from product context
        if product_context and product_context.brand_colors:
            # Create brand guidelines from product context
            return BrandGuidelines(
                brand_name="unknown",
                primary_colors=product_context.brand_colors,
                secondary_colors=[],
                typography_system="Modern sans-serif",
                visual_language="Professional photography",
            )

        # Return default guidelines
        logger.info("Using default brand guidelines")
        return BrandGuidelines(
            brand_name="default",
            primary_colors=["#333333", "#666666"],
            secondary_colors=["#FFFFFF"],
            typography_system="Clean, modern sans-serif",
            visual_language="Professional, clean, modern",
        )

    def _extract_product_keywords(self, prompt: str) -> List[str]:
        """
        Extract product-related keywords from prompt.

        Args:
            prompt: The generic input prompt

        Returns:
            List of product keywords
        """

        # Simple keyword extraction
        import re

        # Look for common product patterns
        patterns = [
            r"for ([a-z]+(?: [a-z]+)?)",
            r"our ([a-z]+(?: [a-z]+)?)",
            r"the ([a-z]+(?: [a-z]+)?)",
        ]

        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, prompt.lower())
            keywords.extend(matches)

        # Filter out non-product words
        stop_words = {"ad", "advertisement", "image", "photo", "picture", "product"}
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]

        return keywords

    def detect_psychology_driver(self, prompt: str, intent: Any) -> str:
        """
        Detect appropriate psychology driver based on prompt and intent.

        Args:
            prompt: The generic input prompt
            intent: Detected intent

        Returns:
            Psychology driver string (e.g., "trust", "aspiration")
        """

        prompt_lower = prompt.lower()

        # Trust & Reliability indicators
        if any(word in prompt_lower for word in ["professional", "quality", "reliable", "effective"]):
            return "trust"

        # Aspiration indicators
        if any(word in prompt_lower for word in ["dream", "perfect", "ideal", "beautiful", "premium"]):
            return "aspiration"

        # Urgency/Scarcity indicators
        if any(word in prompt_lower for word in ["limited", "now", "today", "fast", "quick"]):
            return "urgency"

        # Social Proof indicators
        if any(word in prompt_lower for word in ["popular", "everyone", "people", "customers"]):
            return "social_proof"

        # Default to trust for product photography
        return "trust"

    def get_roas_optimized_features(
        self,
        product_context: Optional[ProductContext],
    ) -> Dict[str, Any]:
        """
        Get ROAS-optimized features for the product.

        In production, this would query the ad miner's ROAS data.
        For now, returns sensible defaults.

        Args:
            product_context: Product context

        Returns:
            Dictionary with ROAS-optimized feature recommendations
        """

        if not product_context:
            return {}

        # Mock ROAS-optimized features
        return {
            "product_position": "foreground-center",
            "color_balance": "warm" if product_context.category == "cleaning_tools" else "neutral",
            "lighting": "natural_daylight",
            "composition_style": "authentic",
            "human_element": True if product_context.category == "cleaning_tools" else False,
        }


class PsychologyMapper:
    """
    Map psychology drivers to visual language recommendations.

    Based on the 14 psychology types in the config system.
    """

    PSYCHOLOGY_VISUAL_MAP = {
        "trust": {
            "colors": ["#003366", "#FFFFFF", "#CCCCCC"],
            "lighting": "natural, even, professional",
            "mood": "calm, confident, reliable",
            "composition": "balanced, stable",
        },
        "aspiration": {
            "colors": ["#FFD700", "#FFFFFF", "#87CEEB"],
            "lighting": "golden hour, warm, dreamy",
            "mood": "optimistic, inspiring, elevated",
            "composition": "dynamic, upward movement",
        },
        "urgency": {
            "colors": ["#FF0000", "#FFA500", "#FFFF00"],
            "lighting": "high contrast, dramatic",
            "mood": "energetic, pressing, immediate",
            "composition": "off-center, dynamic angles",
        },
        "social_proof": {
            "colors": ["#4169E1", "#32CD32", "#FFFFFF"],
            "lighting": "friendly, approachable, natural",
            "mood": "inclusive, popular, trusted",
            "composition": "crowded but balanced",
        },
    }

    def get_visual_recommendations(self, psychology_driver: str) -> Dict[str, str]:
        """
        Get visual language recommendations for a psychology driver.

        Args:
            psychology_driver: The detected psychology driver

        Returns:
            Dictionary with visual recommendations
        """

        return self.PSYCHOLOGY_VISUAL_MAP.get(
            psychology_driver,
            self.PSYCHOLOGY_VISUAL_MAP["trust"]  # Default to trust
        )
