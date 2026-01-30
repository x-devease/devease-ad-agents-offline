"""Psychology classification from creative features.

This module analyzes creative features to detect psychological drivers
used in advertisements. It maps visual and textual patterns to 14 distinct
psychology types based on marketing principles (Cialdini, behavioral economics).

Author: V1.9 Ad System
Date: 2026-01-30
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PsychologyDriver(str, Enum):
    """Complete psychology driver catalog (14 types).

    Organized into 5 categories:
    - Authority & Trust (2 types)
    - Urgency & Scarcity (3 types)
    - Desire & Aspiration (3 types)
    - Social Proof (2 types)
    - Emotional States (4 types)
    """

    # Authority & Trust
    TRUST = "trust"
    ACHIEVEMENT = "achievement"

    # Urgency & Scarcity
    FOMO = "fomo"
    SCARCITY = "scarcity"
    LOSS_AVERSION = "loss_aversion"

    # Desire & Aspiration
    ASPIRATION = "aspiration"
    EXCLUSIVITY = "exclusivity"
    CURIOSITY = "curiosity"

    # Social Proof
    SOCIAL_PROOF = "social_proof"
    RECIPROCITY = "reciprocity"

    # Emotional States
    NOSTALGIA = "nostalgia"
    COMFORT = "comfort"
    EXCITEMENT = "excitement"
    CALM = "calm"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "PsychologyDriver":
        """Convert string to PsychologyDriver enum."""
        try:
            return cls(value.lower())
        except ValueError:
            logger.warning(f"Unknown psychology driver: {value}, defaulting to TRUST")
            return cls.TRUST

    def get_category(self) -> str:
        """Get the category this driver belongs to."""
        category_map = {
            # Authority & Trust
            PsychologyDriver.TRUST: "authority_trust",
            PsychologyDriver.ACHIEVEMENT: "authority_trust",

            # Urgency & Scarcity
            PsychologyDriver.FOMO: "urgency_scarcity",
            PsychologyDriver.SCARCITY: "urgency_scarcity",
            PsychologyDriver.LOSS_AVERSION: "urgency_scarcity",

            # Desire & Aspiration
            PsychologyDriver.ASPIRATION: "desire_aspiration",
            PsychologyDriver.EXCLUSIVITY: "desire_aspiration",
            PsychologyDriver.CURIOSITY: "desire_aspiration",

            # Social Proof
            PsychologyDriver.SOCIAL_PROOF: "social_proof",
            PsychologyDriver.RECIPROCITY: "social_proof",

            # Emotional States
            PsychologyDriver.NOSTALGIA: "emotional_states",
            PsychologyDriver.COMFORT: "emotional_states",
            PsychologyDriver.EXCITEMENT: "emotional_states",
            PsychologyDriver.CALM: "emotional_states",
        }
        return category_map.get(self, "unknown")


@dataclass
class PsychologyIndicators:
    """Psychology detection indicators from creative analysis.

    Attributes:
        primary_driver: Main psychological motivation detected
        confidence: Statistical confidence (0.0 - 1.0)
        secondary_driver: Optional secondary psychological motivation
        intensity_override: Optional manual intensity level override
        complexity_override: Optional manual complexity level override
        feature_scores: Dictionary of all psychology driver scores
        evidence: List of evidence supporting the classification
    """

    primary_driver: PsychologyDriver
    confidence: float
    secondary_driver: Optional[PsychologyDriver] = None
    intensity_override: Optional[str] = None
    complexity_override: Optional[str] = None
    feature_scores: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "primary_driver": self.primary_driver.value,
            "confidence": self.confidence,
            "secondary_driver": self.secondary_driver.value if self.secondary_driver else None,
            "intensity_override": self.intensity_override,
            "complexity_override": self.complexity_override,
            "feature_scores": {
                k.value if isinstance(k, PsychologyDriver) else k: v
                for k, v in self.feature_scores.items()
            },
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PsychologyIndicators":
        """Create from dictionary (YAML deserialization)."""
        return cls(
            primary_driver=PsychologyDriver.from_string(data["primary_driver"]),
            confidence=data["confidence"],
            secondary_driver=PsychologyDriver.from_string(data["secondary_driver"])
            if data.get("secondary_driver") else None,
            intensity_override=data.get("intensity_override"),
            complexity_override=data.get("complexity_override"),
            feature_scores=data.get("feature_scores", {}),
            evidence=data.get("evidence", []),
        )


class PsychologyClassifier:
    """
    Classify psychology drivers from creative features.

    Uses a scoring system based on:
    1. Visual features (materials, lighting, colors, camera angle, etc.)
    2. Copy patterns (keywords, phrases, tone)
    3. Text overlay style (font, size, position)
    4. Contextual factors (product category, campaign goal)

    Scoring:
    - Each matching feature adds 0.1-0.3 points
    - Each keyword match adds 0.05 points
    - Maximum score per driver: 1.0
    - Driver with highest score wins (if > 0.5 threshold)
    """

    # Psychology feature mapping
    # Each psychology type has positive indicators that increase its score
    PSYCHOLOGY_FEATURES = {
        PsychologyDriver.TRUST: {
            # Visual features (0.2 points each)
            "positive_materials": ["Marble", "Wood", "Leather", "Stone"],
            "positive_lighting": ["Window Light", "Soft Box", "Natural"],
            "positive_camera_angle": ["eye_level"],
            "positive_visual_prominence": ["balanced"],
            "positive_colors": ["Blue", "Navy", "Green", "Black", "White"],

            # Copy keywords (0.05 points each)
            "keywords": [
                "expert", "proven", "certified", "guaranteed", "trusted",
                "professional", "quality", "reliable", "authentic",
                "doctor", "recommended", "award", "winner", "years"
            ],

            # Text overlay style (0.1 points)
            "text_style": ["serif", "elegant", "subtle", "minimal"],
        },

        PsychologyDriver.ACHIEVEMENT: {
            "positive_colors": ["Green", "Gold", "Trophy"],
            "positive_lighting": ["Bright", "Celebratory"],
            "visual_elements": ["progress_bar", "trophy", "medal", "badge"],
            "keywords": [
                "level", "progress", "unlock", "achievement", "milestone",
                "earned", "reached", "completed", "percent", "there",
                "you're", "keep going", "congratulations"
            ],
            "text_style": ["bold", "celebratory", "dynamic"],
        },

        PsychologyDriver.FOMO: {
            "positive_colors": ["Red", "Orange", "Yellow", "Hot Pink"],
            "positive_lighting": ["Spotlight", "High Contrast", "Dramatic"],
            "visual_elements": ["countdown", "timer", "urgency_badge", "flame"],
            "keywords": [
                "limited", "now", "today", "hurry", "fast", "ending",
                "almost", "gone", "miss", "last", "chance", "people",
                "viewing", "hot", "selling", "don't wait"
            ],
            "text_style": ["bold", "uppercase", "large", "impact"],
        },

        PsychologyDriver.SCARCITY: {
            "positive_colors": ["Orange", "Amber", "Warning"],
            "visual_elements": ["stock_bar", "low_stock", "x_left", "decreasing"],
            "keywords": [
                "only", "left", "last", "few", "running", "reserve",
                "claim", "before", "ends", "hours", "minutes", "spots"
            ],
            "text_style": ["centered", "urgent", "focused"],
        },

        PsychologyDriver.LOSS_AVERSION: {
            "positive_colors": ["Red", "Black", "Dark Gray"],
            "visual_elements": ["strikethrough", "comparison", "warning", "x_mark"],
            "keywords": [
                "don't lose", "save", "missing", "wasting", "protect",
                "stop", "last chance", "money", "investment", "regret"
            ],
            "text_style": ["bold", "heavy", "dramatic"],
        },

        PsychologyDriver.ASPIRATION: {
            "positive_colors": ["Purple", "Sky Blue", "Gradient"],
            "positive_camera_angle": ["low_angle", "upward"],
            "visual_elements": ["upward_arrow", "horizon", "mountain", "light_ray"],
            "keywords": [
                "become", "best", "self", "potential", "future", "dreams",
                "reach", "stars", "transform", "elevate", "unlock", "new you"
            ],
            "text_style": ["light", "expanded", "elegant", "minimal"],
        },

        PsychologyDriver.EXCLUSIVITY: {
            "positive_colors": ["Gold", "Black", "Silver", "Platinum"],
            "positive_materials": ["Velvet", "Silk", "Premium"],
            "visual_elements": ["gold_border", "key", "lock", "invitation", "vip"],
            "keywords": [
                "invite", "vip", "member", "exclusive", "only", "access",
                "elite", "private", "selected", "chosen", "list", "special"
            ],
            "text_style": ["serif", "elegant", "premium", "refined"],
        },

        PsychologyDriver.CURIOSITY: {
            "positive_colors": ["Purple", "Mysterious", "Dark Blue"],
            "visual_elements": ["question_mark", "magnify", "veil", "hidden"],
            "keywords": [
                "discover", "secret", "mystery", "reveal", "surprise",
                "what", "how", "why", "inside", "unlock", "won't believe"
            ],
            "text_style": ["light", "minimal", "intriguing"],
        },

        PsychologyDriver.SOCIAL_PROOF: {
            "positive_colors": ["Green", "Gold", "Blue"],
            "visual_elements": ["stars", "avatars", "rating", "checkmark", "verified"],
            "keywords": [
                "join", "customers", "rating", "stars", "reviews", "best",
                "seller", "trending", "popular", "people", "bought", "verified"
            ],
            "text_style": ["medium", "friendly", "community"],
        },

        PsychologyDriver.RECIPROCITY: {
            "positive_colors": ["Orange", "Red", "Gift Gold"],
            "visual_elements": ["gift", "ribbon", "bonus", "free"],
            "keywords": [
                "free", "gift", "bonus", "get", "plus", "extra", "worth",
                "complimentary", "bundle", "save", "together"
            ],
            "text_style": ["rounded", "friendly", "warm"],
        },

        PsychologyDriver.NOSTALGIA: {
            "positive_colors": ["Sepia", "Brown", "Vintage", "Cream"],
            "positive_materials": ["Wood", "Vintage", "Retro"],
            "visual_elements": ["polaroid", "film", "grain", "retro", "vintage"],
            "keywords": [
                "remember", "good old", "classic", "timeless", "bring back",
                "era", "since", "tradition", "heritage", "original"
            ],
            "text_style": ["typewriter", "serif_old", "vintage"],
        },

        PsychologyDriver.COMFORT: {
            "positive_colors": ["Soft Blue", "Pastel", "Warm", "Gentle"],
            "visual_elements": ["cloud", "pillow", "hug", "soft", "rounded"],
            "keywords": [
                "relax", "safe", "comfort", "peace", "easy", "gentle",
                "calm", "breathe", "rest", "cozy", "warm", "secure"
            ],
            "text_style": ["rounded", "light", "soft", "friendly"],
        },

        PsychologyDriver.EXCITEMENT: {
            "positive_colors": ["Vibrant", "Rainbow", "Bright", "Neon"],
            "visual_elements": ["confetti", "balloon", "burst", "star", "firework"],
            "keywords": [
                "celebrate", "party", "woohoo", "exciting", "big", "announcement",
                "news", "amazing", "incredible", "wow", "finally", "yes"
            ],
            "text_style": ["bold", "playful", "dynamic", "exclamatory"],
        },

        PsychologyDriver.CALM: {
            "positive_colors": ["Sky Blue", "Mint", "Pale", "Soft"],
            "visual_elements": ["wave", "ripple", "minimal", "space", "breath"],
            "keywords": [
                "breathe", "calm", "clear", "simple", "peace", "mindful",
                "zen", "minimal", "clarity", "balance", "present", "still"
            ],
            "text_style": ["thin", "expanded", "minimal", "light"],
        },
    }

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize psychology classifier.

        Args:
            confidence_threshold: Minimum confidence to classify (default: 0.5)
        """
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        creative_features: Dict[str, Any],
        copy_analysis: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PsychologyIndicators:
        """
        Classify psychology driver from creative features.

        Args:
            creative_features: Detected features from Ad Miner
                Expected keys: primary_material, paired_lighting, camera_angle,
                                visual_prominence, color_palette, text_overlay_style
            copy_analysis: Optional NLP analysis of ad copy
                Expected keys: text, keywords, tone
            context: Optional context (product_category, campaign_goal)

        Returns:
            PsychologyIndicators with primary driver and confidence
        """
        scores = {}
        evidence_by_driver = {driver: [] for driver in PsychologyDriver}

        # Score each psychology driver
        for driver in PsychologyDriver:
            score, evidence = self._score_driver(
                driver,
                creative_features,
                copy_analysis,
                context
            )
            scores[driver] = score
            evidence_by_driver[driver] = evidence

        # Find highest score
        primary_driver = max(scores, key=scores.get)
        confidence = scores[primary_driver]

        # Find secondary driver (second highest)
        sorted_drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary_driver = sorted_drivers[1][0] if len(sorted_drivers) > 1 and sorted_drivers[1][1] > 0.3 else None

        # Gather evidence for primary driver
        primary_evidence = evidence_by_driver[primary_driver]

        logger.info(
            f"Psychology classification: primary={primary_driver.value} "
            f"(confidence={confidence:.2f}), "
            f"secondary={secondary_driver.value if secondary_driver else None}"
        )

        return PsychologyIndicators(
            primary_driver=primary_driver,
            confidence=confidence,
            secondary_driver=secondary_driver,
            feature_scores={k.value: v for k, v in scores.items()},
            evidence=primary_evidence,
        )

    def _score_driver(
        self,
        driver: PsychologyDriver,
        features: Dict[str, Any],
        copy_analysis: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> tuple[float, List[str]]:
        """
        Score a single psychology driver.

        Returns:
            Tuple of (score, evidence_list)
        """
        score = 0.0
        evidence = []

        feature_map = self.PSYCHOLOGY_FEATURES.get(driver, {})

        # Score visual features (0.2 points per match)
        visual_scoring_keys = [
            "positive_materials", "positive_lighting", "positive_camera_angle",
            "positive_visual_prominence", "positive_colors", "visual_elements"
        ]

        for key in visual_scoring_keys:
            if key not in feature_map:
                continue

            positive_values = feature_map[key]

            # Map key to feature name
            feature_key = key.replace("positive_", "")  # e.g., "positive_materials" -> "materials"

            if feature_key in features:
                actual_value = features[feature_key]
                if actual_value in positive_values:
                    score += 0.2
                    evidence.append(f"Visual: {feature_key}={actual_value}")

        # Score text overlay style (0.1 points)
        if "text_style" in feature_map and "text_overlay_style" in features:
            text_style = features["text_overlay_style"]
            positive_styles = feature_map["text_style"]

            if any(style in text_style.lower() for style in positive_styles):
                score += 0.1
                evidence.append(f"Text style: {text_style}")

        # Score copy keywords (0.05 points per keyword match)
        if copy_analysis and "keywords" in feature_map:
            keywords = feature_map["keywords"]
            copy_text = copy_analysis.get("text", "")
            copy_text_lower = copy_text.lower()

            matches = [
                kw for kw in keywords
                if kw.lower() in copy_text_lower
            ]

            if matches:
                keyword_score = min(len(matches) * 0.05, 0.3)  # Cap at 0.3
                score += keyword_score
                evidence.append(f"Keywords: {', '.join(matches[:3])}")  # Top 3 matches

        # Contextual boosting (0.1 points for matching use cases)
        if context and self._matches_context(driver, context):
            score += 0.1
            evidence.append(f"Context: {context.get('product_category', '')} + {context.get('campaign_goal', '')}")

        return min(score, 1.0), evidence

    def _matches_context(
        self,
        driver: PsychologyDriver,
        context: Dict[str, Any]
    ) -> bool:
        """Check if psychology driver matches the use case context."""
        product_category = context.get("product_category", "").lower()
        campaign_goal = context.get("campaign_goal", "").lower()

        # Context mappings
        context_mappings = {
            PsychologyDriver.TRUST: {
                "products": ["luxury", "professional", "healthcare", "financial"],
                "goals": ["brand_awareness", "consideration"]
            },
            PsychologyDriver.FOMO: {
                "products": ["fashion", "tech", "travel"],
                "goals": ["conversion", "sales"]
            },
            PsychologyDriver.ACHIEVEMENT: {
                "products": ["fitness", "education", "gaming"],
                "goals": ["engagement", "retention"]
            },
            # Add mappings for other drivers as needed
        }

        if driver not in context_mappings:
            return False

        driver_context = context_mappings[driver]

        # Check product category match
        product_match = any(
            cat in product_category
            for cat in driver_context.get("products", [])
        )

        # Check campaign goal match
        goal_match = any(
            goal in campaign_goal
            for goal in driver_context.get("goals", [])
        )

        return product_match or goal_match


# Convenience function for quick classification
def classify_psychology(
    creative_features: Dict[str, Any],
    copy_analysis: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.5
) -> PsychologyIndicators:
    """
    Quick psychology classification convenience function.

    Args:
        creative_features: Creative features dict
        copy_analysis: Optional copy analysis
        confidence_threshold: Minimum confidence threshold

    Returns:
        PsychologyIndicators
    """
    classifier = PsychologyClassifier(confidence_threshold=confidence_threshold)
    return classifier.classify(creative_features, copy_analysis)
