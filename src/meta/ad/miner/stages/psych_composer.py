"""
Stage 2.5: The Psych-Composer

Applies psychological filtering on top of visual co-occurrence.
Establishes [Visual Features] <-> [User Psychology] mapping relationships.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PsychologyType(Enum):
    """Core psychological drivers."""
    TRUST_AUTHORITY = "Trust_Authority"
    LUXURY_ASPIRATION = "Luxury_Aspiration"
    FOMO = "FOMO"
    SOCIAL_PROOF = "Social_Proof"


class StrictnessLevel(Enum):
    """Psychology filtering strictness."""
    STRICT = "strict"        # Only psych-aligned combinations
    MODERATE = "moderate"    # Allow mild deviations with warnings
    LOOSE = "loose"          # Visual-only (V1.8 behavior)


@dataclass
class PsychologicalAlignment:
    """Result of psychological alignment check."""
    aligned: bool
    strength: str  # "strong", "weak", "neutral"
    reasoning: str
    psychology_match: Optional[str] = None  # Which psychology does this feature match?


class PsychComposer:
    """
    Psychological Composer - V2.0

    Filters visual combinations based on psychological alignment.
    Implements dual-layer filtering:
    1. Visual Co-occurrence (from V1.8 Synthesizer)
    2. Psychological Alignment (NEW)
    """

    # Default psychology mappings (fallback if not in config)
    DEFAULT_PSYCHOLOGY_MAPPINGS = {
        "trust_authority": {
            "positive_features": [
                "Marble", "White Ceramic", "Metal", "Chrome", "Glass",
                "Studio Light", "Window Light", "Shadowless",
                "Overhead", "Front",
                "Cool", "White", "Blue",
                "Minimalist", "Centered", "Symmetrical"
            ],
            "negative_features": [
                "Neon", "Warm", "Cozy", "Cluttered", "Moody",
                "Dark", "Vibrant", "Colorful", "Vintage",
                "Plastic", "Cartoonish"
            ],
            "emotional_keywords": ["safe", "professional", "credible", "expert"]
        },
        "luxury_aspiration": {
            "positive_features": [
                "Marble", "Granite", "Velvet", "Gold", "Brass",
                "Dramatic", "Warm", "Moody",
                "Low Angle", "45-degree",
                "Rich", "Gold", "Black", "Jewel",
                "Minimalist Luxury", "Rule of Thirds", "Generous"
            ],
            "negative_features": [
                "Plastic", "Cluttered", "Cartoonish",
                "Flat", "Bright White", "Basic"
            ],
            "emotional_keywords": ["exclusive", "premium", "status", "achievement"]
        },
        "fomo": {
            "positive_features": [
                "High Contrast", "Bright", "Direct", "Red", "Orange",
                "Close-up", "Dutch Angle",
                "Red", "Orange", "Warm",
                "Tightly", "Center", "Dynamic", "Diagonal"
            ],
            "negative_features": [
                "Calm", "Serene", "Minimalist", "Clean",
                "Cool", "Spacious", "Slow"
            ],
            "emotional_keywords": ["urgent", "scarce", "limited", "now"]
        },
        "social_proof": {
            "positive_features": [
                "Natural Wood", "Fabric",
                "Natural Light", "Warm", "Golden Hour",
                "Eye Level", "45-degree",
                "Warm", "Natural", "Skin",
                "People", "Lifestyle", "Candid"
            ],
            "negative_features": [
                "Sterile", "Clinical", "Minimalist Isolated",
                "Dramatic", "Cool", "Unemotional"
            ],
            "emotional_keywords": ["popular", "belonging", "validated", "community"]
        }
    }

    def __init__(
        self,
        psychology_mappings: Optional[dict] = None,
        target_psychology: str = "Trust_Authority",
        strictness: str = "strict"
    ):
        """
        Initialize Psych-Composer.

        Args:
            psychology_mappings: Psychology feature mappings from config
            target_psychology: Target psychology (Trust_Authority, Luxury_Aspiration, FOMO, Social_Proof)
            strictness: Filtering strictness (strict, moderate, loose)
        """
        self.psychology_mappings = psychology_mappings or self.DEFAULT_PSYCHOLOGY_MAPPINGS
        self.target_psychology = target_psychology
        self.strictness = StrictnessLevel(strictness)

        # Normalize target psychology key
        self.psych_key = self._normalize_psychology_key(target_psychology)

        logger.info(f"Psych-Composer initialized: target={target_psychology}, strictness={strictness}")

    def _normalize_psychology_key(self, psychology: str) -> str:
        """Normalize psychology key to lowercase with underscores."""
        return psychology.lower().replace("-", "_").replace(" ", "_")

    def check_alignment(
        self,
        feature_value: str,
        feature_name: Optional[str] = None
    ) -> PsychologicalAlignment:
        """
        Check if a visual feature aligns with target psychology.

        Args:
            feature_value: Feature value (e.g., "Neon")
            feature_name: Optional feature category (e.g., "lighting_style")

        Returns:
            PsychologicalAlignment result
        """
        # Get mapping for target psychology
        mapping = self.psychology_mappings.get(self.psych_key, {})

        positive_features = mapping.get("positive_features", [])
        negative_features = mapping.get("negative_features", [])

        # Check positive alignment
        for positive in positive_features:
            if positive.lower() in feature_value.lower():
                return PsychologicalAlignment(
                    aligned=True,
                    strength="strong",
                    reasoning=f"{feature_value} is a positive trigger for {self.target_psychology}",
                    psychology_match=self.target_psychology
                )

        # Check negative alignment (mismatch)
        for negative in negative_features:
            if negative.lower() in feature_value.lower():
                return PsychologicalAlignment(
                    aligned=False,
                    strength="strong",
                    reasoning=f"{feature_value} contradicts {self.target_psychology}"
                )

        # Neutral features - behavior depends on strictness
        if self.strictness == StrictnessLevel.STRICT:
            return PsychologicalAlignment(
                aligned=False,
                strength="neutral",
                reasoning=f"{feature_value} is not a known positive trigger (strict mode)"
            )
        else:
            return PsychologicalAlignment(
                aligned=True,
                strength="weak",
                reasoning=f"{feature_value} is neutral (loose mode)"
            )

    def compose(
        self,
        visual_combinations: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Apply psychological filtering on visual combinations.

        Dual filtering logic:
        1. Visual co-occurrence (already validated by V1.8 Synthesizer)
        2. Psychological alignment (NEW in V2.0)

        Args:
            visual_combinations: Visual combinations from synthesizer

        Returns:
            Psychologically validated combinations
        """
        logger.info("="*60)
        logger.info(f"PSYCH-COMPOSER: Target Psychology = {self.target_psychology}")
        logger.info(f"Strictness = {self.strictness.value}")
        logger.info("="*60)

        psych_validated = {}
        stats = {
            "total": len(visual_combinations),
            "kept": 0,
            "rejected": 0,
            "strong_alignment": 0,
            "weak_alignment": 0
        }

        for combo_key, combo_data in visual_combinations.items():
            # Extract features
            primary_value = combo_data.get("primary_value", "")
            secondary_value = combo_data.get("secondary_value", "")

            # Check psychological alignment
            primary_alignment = self.check_alignment(primary_value)
            secondary_alignment = self.check_alignment(secondary_value)

            # Determine overall alignment
            if primary_alignment.aligned and secondary_alignment.aligned:
                # Both aligned
                overall = "strong" if primary_alignment.strength == "strong" and secondary_alignment.strength == "strong" else "weak"
                keep = True
                stats["strong_alignment" if overall == "strong" else "weak_alignment"] += 1
            elif not primary_alignment.aligned or not secondary_alignment.aligned:
                # At least one misaligned
                overall = "weak" if (primary_alignment.aligned or secondary_alignment.aligned) else "none"
                keep = self.strictness != StrictnessLevel.STRICT
            else:
                # Both neutral
                overall = "neutral"
                keep = self.strictness == StrictnessLevel.LOOSE

            if keep:
                psych_validated[combo_key] = {
                    **combo_data,
                    "psychological_alignment": {
                        "target_psychology": self.target_psychology,
                        "primary_alignment": {
                            "value": primary_value,
                            "aligned": primary_alignment.aligned,
                            "strength": primary_alignment.strength,
                            "reasoning": primary_alignment.reasoning
                        },
                        "secondary_alignment": {
                            "value": secondary_value,
                            "aligned": secondary_alignment.aligned,
                            "strength": secondary_alignment.strength,
                            "reasoning": secondary_alignment.reasoning
                        },
                        "overall_alignment": overall
                    }
                }

                stats["kept"] += 1
                logger.info(
                    f"✓ KEPT {combo_key}: "
                    f"Psych alignment = {overall} "
                    f"(primary: {primary_alignment.strength}, secondary: {secondary_alignment.strength})"
                )
            else:
                stats["rejected"] += 1
                logger.info(
                    f"✗ REJECTED {combo_key}: "
                    f"Psych mismatch with {self.target_psychology} "
                    f"(primary: {primary_value}, secondary: {secondary_value})"
                )

        logger.info("="*60)
        logger.info(f"Psych-Composer Results:")
        logger.info(f"  Total combinations: {stats['total']}")
        logger.info(f"  Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
        logger.info(f"  Rejected: {stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)")
        logger.info(f"  Strong alignment: {stats['strong_alignment']}")
        logger.info(f"  Weak alignment: {stats['weak_alignment']}")
        logger.info("="*60)

        return psych_validated

    def get_psychology_summary(self) -> dict:
        """
        Get summary of current psychology configuration.

        Returns:
            Psychology configuration summary
        """
        mapping = self.psychology_mappings.get(self.psych_key, {})

        return {
            "target_psychology": self.target_psychology,
            "strictness": self.strictness.value,
            "positive_features": mapping.get("positive_features", []),
            "negative_features": mapping.get("negative_features", []),
            "emotional_keywords": mapping.get("emotional_keywords", [])
        }


def create_psych_composer(
    psychology_mappings: Optional[dict] = None,
    target_psychology: str = "Trust_Authority",
    strictness: str = "strict"
) -> PsychComposer:
    """
    Factory function to create Psych-Composer.

    Args:
        psychology_mappings: Optional psychology mappings from config
        target_psychology: Target psychology
        strictness: Filtering strictness

    Returns:
        Initialized PsychComposer instance
    """
    return PsychComposer(
        psychology_mappings=psychology_mappings,
        target_psychology=target_psychology,
        strictness=strictness
    )
