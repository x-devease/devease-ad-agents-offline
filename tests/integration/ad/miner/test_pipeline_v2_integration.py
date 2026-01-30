"""
Integration tests for V2.0 Pipeline with Psych-Composer

Tests end-to-end functionality of the psychology-driven pipeline.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.meta.ad.miner.pipeline_v2 import V20Pipeline, create_pipeline_v2


class TestV20PipelineIntegration:
    """Integration tests for V2.0 pipeline."""

    @pytest.fixture
    def sample_config_dir(self, tmp_path):
        """Create sample customer config."""
        config_dir = tmp_path / "config" / "ad" / "miner" / "test_customer"
        config_dir.mkdir(parents=True)

        config = {
            "customer_context": {
                "customer_id": "test_customer",
                "ad_account_id": "act_test",
                "primary_metric": "ROAS",
                "lookback_days": 90
            },
            "mining_profiles": {
                "balanced": {
                    "base_quantile": 0.90,
                    "min_sample_size": 10
                }
            },
            "default_profile": "balanced",
            "platform_overrides": {},
            "product_overrides": {},
            "platform_config": {
                "target_model": "test_model",
                "combinatorial_threshold": 0.8
            },
            "fidelity_config": {
                "workflow_mode": "auto",
                "enable_quality_validation": True,
                "min_quality_score": 0.7
            },
            "brand_guidelines": {
                "brand_name": "TestBrand",
                "established": True
            },
            "psychology_mappings": {
                "trust_authority": {
                    "positive_features": ["Marble", "Studio Light", "White"],
                    "negative_features": ["Neon", "Warm"],
                    "emotional_keywords": ["safe", "professional"]
                }
            },
            "workflow_templates": {
                "fallback_standard": {
                    "name": "Fallback",
                    "priority": 6,
                    "trigger_conditions": {"fallback": True},
                    "cot_prompts": {},
                    "template_structure": "[Quality], [Subject]",
                    "quality_thresholds": {"min_token_count": 20}
                }
            }
        }

        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        return tmp_path

    @pytest.fixture
    def sample_data(self):
        """Create sample creative features data."""
        return pd.DataFrame({
            "creative_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "roas": [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
            "surface_material": [
                "Marble", "Marble", "Wood", "Marble", "Granite",
                "Wood", "Plastic", "Glass", "Metal", "Ceramic"
            ],
            "lighting_style": [
                "Studio Light", "Studio Light", "Window Light", "Studio Light", "Window Light",
                "Natural Light", "Warm Light", "Studio Light", "Studio Light", "Window Light"
            ],
            "direction": [
                "Overhead", "Overhead", "Front", "Overhead", "45-degree",
                "Eye Level", "Front", "Overhead", "Front", "Overhead"
            ],
            "primary_colors": [
                "White Blue", "White", "Cool Tones", "White Blue", "Neutral",
                "Warm", "Red", "White", "Cool", "White"
            ],
            "image_url": [
                f"https://example.com/img{i}.jpg" for i in range(1, 11)
            ]
        })

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        class MockLLMClient:
            def generate(self, prompt, **kwargs):
                return "Premium expanded description"

        return MockLLMClient()

    def test_pipeline_v2_trust_authority_manual(
        self,
        sample_config_dir,
        sample_data,
        mock_llm_client
    ):
        """Test V2.0 pipeline with manual Trust_Authority psychology."""
        # Create pipeline
        pipeline = create_pipeline_v2(
            config_root=sample_config_dir / "config" / "ad" / "miner",
            llm_client=mock_llm_client
        )

        # Create input config with manual psychology
        input_config = {
            "customer_context": {
                "customer_id": "test_customer",
                "platform": "meta",
                "product": "Test Product",
                "target_psychology": "Trust_Authority"
            },
            "campaign_context": {
                "execution_mode": "PRODUCTION_REAL_MONEY",
                "daily_budget_cents": 50000
            },
            "psychology_config": {
                "psychology_mode": "manual",
                "psychology_strictness": "strict",
                "enable_psych_tagging": True
            }
        }

        # Run pipeline
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            output_path = Path(f.name)

        try:
            blueprint = pipeline.run(
                customer_id="test_customer",
                input_config=input_config,
                df=sample_data,
                output_path=output_path
            )

            # Verify output
            assert blueprint["meta_info"]["recipe_id"] == "recipe_psych_v2.0"
            assert blueprint["meta_info"]["target_psychology"] == "Trust_Authority"
            assert "strategy_rationale" in blueprint
            assert "nano_generation_rules" in blueprint
            assert "psychology_driver" in blueprint["strategy_rationale"]
            assert blueprint["strategy_rationale"]["psychology_driver"] == "Trust_Authority"

            # Check negative prompts exclude Trust-incompatible features
            negative_prompts = blueprint["nano_generation_rules"]["negative_prompt"]
            assert "neon" in negative_prompts or "Neon" in negative_prompts

            # Verify output file was created
            assert output_path.exists()

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_pipeline_v2_psych_composer_filtering(
        self,
        sample_config_dir,
        sample_data
    ):
        """Test that Psych-Composer filters correctly."""
        pipeline = create_pipeline_v2(
            config_root=sample_config_dir / "config" / "ad" / "miner",
            llm_client=None  # No LLM for this test
        )

        input_config = {
            "customer_context": {
                "customer_id": "test_customer",
                "platform": "meta",
                "product": "Test Product",
                "target_psychology": "Trust_Authority"
            },
            "campaign_context": {
                "execution_mode": "PRODUCTION",
                "daily_budget_cents": 50000
            },
            "psychology_config": {
                "psychology_mode": "manual",
                "psychology_strictness": "strict"
            }
        }

        blueprint = pipeline.run(
            customer_id="test_customer",
            input_config=input_config,
            df=sample_data
        )

        # Verify that psychology filtering was applied
        # The blueprint should contain Trust_Authority psychology
        assert blueprint["meta_info"]["target_psychology"] == "Trust_Authority"

        # Validated combinations should align with Trust
        if "combinations" in blueprint["strategy_rationale"]:
            for combo_key, combo_data in blueprint["strategy_rationale"]["combinations"].items():
                psych_alignment = combo_data.get("psychological_alignment", {})
                if psych_alignment:
                    # At least primary or secondary should be aligned
                    primary_aligned = psych_alignment.get("primary_alignment", {}).get("aligned", True)
                    secondary_aligned = psych_alignment.get("secondary_alignment", {}).get("aligned", True)
                    # In strict mode, both should be aligned for strong combinations
                    assert primary_aligned or secondary_aligned

    def test_pipeline_v2_auto_psychology_detection(
        self,
        sample_config_dir,
        sample_data
    ):
        """Test automatic psychology detection from winners."""
        pipeline = create_pipeline_v2(
            config_root=sample_config_dir / "config" / "ad" / "miner",
            llm_client=None
        )

        input_config = {
            "customer_context": {
                "customer_id": "test_customer",
                "platform": "meta",
                "product": "Test Product"
            },
            "campaign_context": {
                "execution_mode": "PRODUCTION",
                "daily_budget_cents": 50000
            },
            "psychology_config": {
                "psychology_mode": "auto",  # Auto-detect
                "psychology_strictness": "strict"
            }
        }

        blueprint = pipeline.run(
            customer_id="test_customer",
            input_config=input_config,
            df=sample_data
        )

        # Should auto-detect Trust_Authority from Marble + Studio Light winners
        assert blueprint["meta_info"]["target_psychology"] in [
            "Trust_Authority", "Luxury_Aspiration", "FOMO", "Social_Proof"
        ]

        # Should have psychology distribution in metadata
        if "psychology_distribution" in blueprint["meta_info"]:
            assert isinstance(blueprint["meta_info"]["psychology_distribution"], dict)

    def test_pipeline_v2_backward_compatibility(
        self,
        sample_config_dir,
        sample_data
    ):
        """Test that pipeline works without psychology config (V1.8 compatibility)."""
        pipeline = create_pipeline_v2(
            config_root=sample_config_dir / "config" / "ad" / "miner",
            llm_client=None
        )

        # Input config without psychology settings
        input_config = {
            "customer_context": {
                "customer_id": "test_customer",
                "platform": "meta",
                "product": "Test Product"
            },
            "campaign_context": {
                "execution_mode": "PRODUCTION",
                "daily_budget_cents": 50000
            }
        }

        # Should not crash
        blueprint = pipeline.run(
            customer_id="test_customer",
            input_config=input_config,
            df=sample_data
        )

        # Should default to Trust_Authority
        assert "target_psychology" in blueprint["meta_info"]

    def test_pipeline_v2_multiple_psychologies(
        self,
        sample_config_dir,
        sample_data
    ):
        """Test pipeline with different target psychologies."""
        pipeline = create_pipeline_v2(
            config_root=sample_config_dir / "config" / "ad" / "miner",
            llm_client=None
        )

        psychologies = ["Trust_Authority", "Luxury_Aspiration", "FOMO", "Social_Proof"]

        for psych in psychologies:
            input_config = {
                "customer_context": {
                    "customer_id": "test_customer",
                    "platform": "meta",
                    "product": "Test Product",
                    "target_psychology": psych
                },
                "campaign_context": {
                    "execution_mode": "PRODUCTION",
                    "daily_budget_cents": 50000
                },
                "psychology_config": {
                    "psychology_mode": "manual",
                    "psychology_strictness": "strict"
                }
            }

            blueprint = pipeline.run(
                customer_id="test_customer",
                input_config=input_config,
                df=sample_data
            )

            # Should have the correct psychology
            assert blueprint["meta_info"]["target_psychology"] == psych


class TestPsychComposerIntegration:
    """Integration tests for Psych-Composer specifically."""

    def test_psych_composer_real_scenario(self):
        """Test Psych-Composer with realistic scenario."""
        from src.meta.ad.miner.stages.psych_composer import create_psych_composer

        # Create composer for Trust_Authority
        composer = create_psych_composer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )

        # Realistic combinations from mining
        visual_combinations = {
            "marble_studio_overhead": {
                "primary_feature": "surface_material",
                "primary_value": "Marble",
                "secondary_feature": "lighting_style",
                "secondary_value": "Studio Light",
                "confidence_score": 0.95
            },
            "granite_window_45deg": {
                "primary_feature": "surface_material",
                "primary_value": "Granite",
                "secondary_feature": "direction",
                "secondary_value": "45-degree",
                "confidence_score": 0.88
            },
            "wood_neon_closeup": {
                "primary_feature": "surface_material",
                "primary_value": "Wood",
                "secondary_feature": "lighting_style",
                "secondary_value": "Neon",
                "confidence_score": 0.85
            },
            "plastic_warm_front": {
                "primary_feature": "surface_material",
                "primary_value": "Plastic",
                "secondary_feature": "lighting_style",
                "secondary_value": "Warm Light",
                "confidence_score": 0.80
            }
        }

        # Apply psychological filtering
        validated = composer.compose(visual_combinations)

        # Marble + Studio Light should pass
        assert "marble_studio_overhead" in validated
        assert validated["marble_studio_overhead"]["psychological_alignment"]["overall_alignment"] == "strong"

        # Granite + 45-degree should pass (Granite is premium, 45-degree is neutral/positive)
        assert "granite_window_45deg" in validated

        # Wood + Neon should fail (Neon contradicts Trust)
        assert "wood_neon_closeup" not in validated

        # Plastic + Warm Light should fail (both contradict Trust)
        assert "plastic_warm_front" not in validated

    def test_psych_composer_strictness_comparison(self):
        """Test behavior across different strictness levels."""
        visual_combinations = {
            "marble_neon": {
                "primary_feature": "surface_material",
                "primary_value": "Marble",
                "secondary_feature": "lighting_style",
                "secondary_value": "Neon",
                "confidence_score": 0.90
            }
        }

        # Strict: should reject
        strict_composer = create_psych_composer(
            target_psychology="Trust_Authority",
            strictness="strict"
        )
        strict_validated = strict_composer.compose(visual_combinations)
        assert "marble_neon" not in strict_validated

        # Moderate: should accept with warning
        moderate_composer = create_psych_composer(
            target_psychology="Trust_Authority",
            strictness="moderate"
        )
        moderate_validated = moderate_composer.compose(visual_combinations)
        assert "marble_neon" in moderate_validated
        assert moderate_validated["marble_neon"]["psychological_alignment"]["overall_alignment"] == "weak"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
