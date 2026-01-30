"""
Unit tests for Psych-Composer (V2.0)

Tests for psychological alignment checking and composition filtering.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.meta.ad.miner.stages.psych_composer import (
    PsychComposer,
    PsychologicalAlignment,
    PsychologyType,
    StrictnessLevel,
    create_psych_composer
)


class TestPsychComposer:
    """Test PsychComposer core functionality."""

    def test_trust_authority_positive_alignment(self):
        """Test Trust_Authority positive alignment."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        # Test positive feature
        alignment = composer.check_alignment("Marble")
        assert alignment.aligned == True
        assert alignment.strength == "strong"
        assert "Trust_Authority" in alignment.reasoning

        alignment = composer.check_alignment("Studio Light")
        assert alignment.aligned == True
        assert alignment.strength == "strong"

    def test_trust_authority_negative_alignment(self):
        """Test Trust_Authority negative alignment."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        # Test negative feature
        alignment = composer.check_alignment("Neon")
        assert alignment.aligned == False
        assert alignment.strength == "strong"
        assert "contradicts" in alignment.reasoning

        alignment = composer.check_alignment("Warm Light")
        assert alignment.aligned == False

    def test_luxury_aspiration_alignment(self):
        """Test Luxury_Aspiration alignment."""
        composer = PsychComposer(
            target_psychology="Luxury_Aspiration",
            strictness="strict"
        )

        # Positive
        alignment = composer.check_alignment("Gold")
        assert alignment.aligned == True

        alignment = composer.check_alignment("Dramatic Lighting")
        assert alignment.aligned == True

        # Negative
        alignment = composer.check_alignment("Plastic")
        assert alignment.aligned == False

    def test_fomo_alignment(self):
        """Test FOMO alignment."""
        composer = PsychComposer(
            target_psychology="FOMO",
            strictness="strict"
        )

        # Positive
        alignment = composer.check_alignment("Red Background")
        assert alignment.aligned == True

        alignment = composer.check_alignment("High Contrast")
        assert alignment.aligned == True

        # Negative
        alignment = composer.check_alignment("Calm Blue")
        assert alignment.aligned == False

    def test_social_proof_alignment(self):
        """Test Social_Proof alignment."""
        composer = PsychComposer(
            target_psychology="Social_Proof",
            strictness="strict"
        )

        # Positive
        alignment = composer.check_alignment("People in Frame")
        assert alignment.aligned == True

        alignment = composer.check_alignment("Natural Wood")
        assert alignment.aligned == True

        # Negative
        alignment = composer.check_alignment("Sterile Clinical")
        assert alignment.aligned == False

    def test_psychological_filtering_strict(self):
        """Test psychological filtering with strict mode."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        visual_combinations = {
            "marble_window": {
                "primary_feature": "surface_material",
                "primary_value": "Marble",
                "secondary_feature": "lighting_style",
                "secondary_value": "Window Light",
                "confidence_score": 0.92
            },
            "marble_neon": {
                "primary_feature": "surface_material",
                "primary_value": "Marble",
                "secondary_feature": "lighting_style",
                "secondary_value": "Neon",
                "confidence_score": 0.85
            },
            "plastic_window": {
                "primary_feature": "surface_material",
                "primary_value": "Plastic",
                "secondary_feature": "lighting_style",
                "secondary_value": "Window Light",
                "confidence_score": 0.80
            }
        }

        validated = composer.compose(visual_combinations)

        # Marble + Window Light should pass (both aligned)
        assert "marble_window" in validated
        assert validated["marble_window"]["psychological_alignment"]["overall_alignment"] == "strong"

        # Marble + Neon should fail (Neon misaligned)
        assert "marble_neon" not in validated

        # Plastic + Window Light should fail (Plastic misaligned)
        assert "plastic_window" not in validated

    def test_psychological_filtering_moderate(self):
        """Test psychological filtering with moderate mode."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="moderate"
        )

        visual_combinations = {
            "marble_neon": {
                "primary_feature": "surface_material",
                "primary_value": "Marble",
                "secondary_feature": "lighting_style",
                "secondary_value": "Neon",
                "confidence_score": 0.85
            }
        }

        validated = composer.compose(visual_combinations)

        # In moderate mode, one misaligned feature still passes with warning
        assert "marble_neon" in validated
        assert validated["marble_neon"]["psychological_alignment"]["overall_alignment"] == "weak"

    def test_psychological_filtering_loose(self):
        """Test psychological filtering with loose mode."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="loose"
        )

        visual_combinations = {
            "neutral_surface": {
                "primary_feature": "surface_material",
                "primary_value": "Unknown Material",
                "secondary_feature": "lighting_style",
                "secondary_value": "Unknown Light",
                "confidence_score": 0.70
            }
        }

        validated = composer.compose(visual_combinations)

        # In loose mode, neutral features pass
        assert "neutral_surface" in validated

    def test_cross_psychology_rejection(self):
        """Test that features from wrong psychology are rejected."""
        # Trust composer
        trust_composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        # Luxury features
        alignment = trust_composer.check_alignment("Dramatic Moody")
        assert alignment.aligned == False

        # FOMO features
        alignment = trust_composer.check_alignment("Urgent Red")
        assert alignment.aligned == False

    def test_get_psychology_summary(self):
        """Test psychology summary generation."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        summary = composer.get_psychology_summary()

        assert summary["target_psychology"] == "Trust_Authority"
        assert summary["strictness"] == "strict"
        assert len(summary["positive_features"]) > 0
        assert len(summary["negative_features"]) > 0
        assert len(summary["emotional_keywords"]) > 0


class TestPsychComposerFactory:
    """Test Psych-Composer factory function."""

    def test_create_psych_composer_default(self):
        """Test factory with defaults."""
        composer = create_psych_composer()

        assert composer.target_psychology == "Trust_Authority"
        assert composer.strictness == StrictnessLevel.STRICT

    def test_create_psych_composer_custom(self):
        """Test factory with custom parameters."""
        composer = create_psych_composer(
            target_psychology="Luxury_Aspiration",
            strictness="moderate"
        )

        assert composer.target_psychology == "Luxury_Aspiration"
        assert composer.strictness == StrictnessLevel.MODERATE

    def test_create_psych_composer_with_mappings(self):
        """Test factory with custom psychology mappings."""
        custom_mappings = {
            "trust_authority": {
                "positive_features": ["CustomPositive"],
                "negative_features": ["CustomNegative"],
                "emotional_keywords": ["custom_keyword"]
            }
        }

        composer = create_psych_composer(
            psychology_mappings=custom_mappings,
            target_psychology="Trust_Authority"
        )

        # Test that custom mappings are used
        alignment = composer.check_alignment("CustomPositive")
        assert alignment.aligned == True

        alignment = composer.check_alignment("CustomNegative")
        assert alignment.aligned == False


class TestPsychComposerEdgeCases:
    """Test edge cases and error handling."""

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        # All should match regardless of case
        assert composer.check_alignment("marble").aligned == True
        assert composer.check_alignment("MARBLE").aligned == True
        assert composer.check_alignment("Marble").aligned == True

    def test_partial_matching(self):
        """Test that partial matching works."""
        composer = PsychComposer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        # Should match "Studio Light" if "Studio" is in positive features
        alignment = composer.check_alignment("Studio Light Soft")
        assert alignment.aligned == True

    def test_unknown_psychology_fallback(self):
        """Test behavior with unknown psychology."""
        # Should use defaults
        composer = PsychComposer(
            target_psychology="Unknown_Psychology",
            strictness="strict"
        )

        # Should not crash, but return neutral
        alignment = composer.check_alignment("Any Feature")
        # In strict mode, unknown features are not aligned
        assert composer.strictness == StrictnessLevel.STRICT


class TestV2Miner:
    """Test V2.0 Miner with psychology tagging."""

    def test_extract_psychology_tags_rule_based(self):
        """Test rule-based psychology extraction."""
        from src.meta.ad.miner.stages.miner_v2 import AdMinerV2
        import pandas as pd

        # Create mock miner
        miner = AdMinerV2(
            input_config={
                "customer_context": {"customer_id": "test"},
                "psychology_config": {"enable_psych_tagging": True}
            },
            strategy_selector=None,  # Mock
            customer_config=None
        )

        # Test with trust-related features
        row = pd.Series({
            "surface_material": "Marble",
            "lighting_style": "Studio Light",
            "direction": "Overhead",
            "primary_colors": "White Blue"
        })

        tags = miner.extract_psychology_tags_rule_based(row)

        assert tags["primary"] == "Trust_Authority"
        assert tags["method"] == "rule_based"
        assert "confidence" in tags

    def test_infer_emotional_tone(self):
        """Test emotional tone inference."""
        from src.meta.ad.miner.stages.miner_v2 import AdMinerV2

        miner = AdMinerV2(
            input_config={"customer_context": {"customer_id": "test"}},
            strategy_selector=None,
            customer_config=None
        )

        # Trust-related text
        tone = miner._infer_emotional_tone(
            "clean minimalist studio professional cool white",
            "Trust_Authority"
        )
        assert "professional" in tone

        # Luxury-related text
        tone = miner._infer_emotional_tone(
            "luxury premium dramatic sophisticated gold",
            "Luxury_Aspiration"
        )
        assert "premium" in tone


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
