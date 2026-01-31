"""
Unit tests for Ad Miner V1.8 Pipeline

Tests for MiningStrategySelector, AdMiner, CombinatorialSynthesizer, and CoTUpscaler.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.meta.ad.miner.config import MiningStrategySelector
from src.meta.ad.miner.stages.miner import AdMiner
from src.meta.ad.miner.stages.synthesizer import CombinatorialSynthesizer


class TestMiningStrategySelector:
    """Test MiningStrategySelector (Stage 0)."""

    def test_determine_winner_quantile_manual_override(self, tmp_path):
        """Test manual override (Priority 1)."""
        # Create test customer config
        config_dir = tmp_path / "config" / "ad" / "miner" / "test_customer"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text("""
default_profile: "balanced"
mining_profiles:
  balanced:
    base_quantile: 0.90
""")

        selector = MiningStrategySelector(config_dir.parent.parent.parent.parent)

        # Test manual override
        quantile = selector.determine_winner_quantile(
            customer_id="test_customer",
            platform="meta",
            product="Test Product",
            daily_budget_cents=50000,
            manual_quantile=0.92
        )

        assert quantile == 0.92

    def test_determine_winner_quantile_product_override(self, tmp_path):
        """Test product-level override (Priority 3)."""
        # Create test customer config with product override
        # MiningStrategySelector looks for config_root / customer_id / "config.yaml"
        config_dir = tmp_path / "test_customer"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text("""
default_profile: "balanced"
mining_profiles:
  balanced:
    base_quantile: 0.90
product_overrides:
  "Power Station":
    platform_overrides:
      meta:
        winner_quantile: 0.97
""")

        selector = MiningStrategySelector(tmp_path)

        # Test product override
        quantile = selector.determine_winner_quantile(
            customer_id="test_customer",
            platform="meta",
            product="Power Station",
            daily_budget_cents=50000
        )

        assert quantile == 0.97


class TestAdMiner:
    """Test AdMiner (Stage 1)."""

    def test_extract_winners_and_losers(self):
        """Test winner/loser extraction."""
        df = pd.DataFrame({
            "roas": [1.0, 2.0, 3.0, 4.0, 5.0],
            "direction": ["overhead", "45-degree", "front", "side", "overhead"]
        })

        # Mock strategy selector
        class MockSelector:
            def determine_winner_quantile(self, *args, **kwargs):
                return 0.80  # Top 20%

            def get_min_sample_size(self, *args, **kwargs):
                return 1

        input_config = {
            "customer_context": {
                "customer_id": "test",
                "platform": "meta"
            },
            "campaign_context": {
                "daily_budget_cents": 10000
            }
        }

        miner = AdMiner(
            input_config=input_config,
            strategy_selector=MockSelector()
        )

        winners, losers = miner.extract_winners_and_losers(df, winner_quantile=0.80)

        # Top 20% of 5 samples = 1 sample
        assert len(winners) == 1
        assert winners.iloc[0]["roas"] == 5.0

    def test_extract_raw_tags(self):
        """Test raw tag extraction."""
        df = pd.DataFrame({
            "roas": [3.0, 4.0, 5.0],
            "direction": ["overhead", "45-degree", "front"],
            "lighting_style": ["studio", "window", "studio"],
            "product_position": ["center", "bottom-right", "center"]
        })

        input_config = {
            "customer_context": {"customer_id": "test"},
            "campaign_context": {"daily_budget_cents": 10000}
        }

        miner = AdMiner(
            input_config=input_config,
            strategy_selector=None
        )

        raw_tags = miner.extract_raw_tags(df)

        assert "direction" in raw_tags
        assert set(raw_tags["direction"]) == {"overhead", "45-degree", "front"}
        assert set(raw_tags["lighting_style"]) == {"studio", "window"}


class TestCombinatorialSynthesizer:
    """Test CombinatorialSynthesizer (Stage 2)."""

    def test_build_co_occurrence_matrix(self):
        """Test co-occurrence matrix calculation."""
        df = pd.DataFrame({
            "surface_material": ["Marble", "Marble", "Wood", "Marble"],
            "lighting_style": ["Window", "Window", "Window", "Studio"]
        })

        synthesizer = CombinatorialSynthesizer(min_confidence=0.8)
        matrix = synthesizer.build_co_occurrence_matrix(
            df, [("surface_material", "lighting_style")]
        )

        # P(Window Light | Marble) = 2/3 = 0.67
        key = ("surface_material", "lighting_style", "Marble", "Window")
        assert key in matrix
        assert abs(matrix[key]["confidence"] - 0.67) < 0.01

    def test_find_locked_combinations(self):
        """Test locked combination discovery."""
        # Create co-occurrence matrix
        co_occurrence = {
            ("f1", "f2", "A", "X"): {"confidence": 0.92, "count": 92, "total": 100},
            ("f1", "f2", "A", "Y"): {"confidence": 0.08, "count": 8, "total": 100},
        }

        synthesizer = CombinatorialSynthesizer(min_confidence=0.8)
        locked = synthesizer.find_locked_combinations(
            co_occurrence, primary_feature="f1"
        )

        assert locked["confidence_score"] == 0.92
        assert locked["locked_combination"]["primary_value"] == "A"
        assert locked["locked_combination"]["secondary_value"] == "X"
        assert len(locked["excluded_features"]) == 1
        assert locked["excluded_features"][0]["value"] == "Y"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
