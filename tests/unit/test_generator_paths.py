"""
Unit tests for GeneratorPaths (template_system paths module).

Tests path management functionality including:
- Single config per customer architecture
- Platform-specific outputs
- Consolidated generator config
"""

import pytest
from pathlib import Path
from src.meta.ad.generator.template_system.paths import (
    GeneratorPaths,
    create_customer_structure,
)


class TestGeneratorPaths:
    """Test GeneratorPaths functionality."""

    def test_init_basic(self):
        """Test basic initialization."""
        paths = GeneratorPaths(
            customer="moprobo",
            platform="facebook"
        )

        assert paths.customer == "moprobo"
        assert paths.platform == "facebook"
        assert paths.config_dir == Path("config")
        assert paths.output_dir == Path("results")

    def test_init_normalizes_names(self):
        """Test that customer and platform names are normalized."""
        paths = GeneratorPaths(
            customer="MoProbo-Test",
            platform="FaceBook"
        )

        assert paths.customer == "moprobo_test"
        assert paths.platform == "facebook"

    def test_get_config_path(self):
        """Test config path returns customer/platform directory."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        config_path = paths.get_config_path()

        assert config_path == Path("config/moprobo/facebook")

    def test_get_blueprint_path(self):
        """Test blueprint path points to config.yaml."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        blueprint_path = paths.get_blueprint_path()

        assert blueprint_path == Path("config/moprobo/facebook/config.yaml")

    def test_get_campaign_content_path_base(self):
        """Test campaign content path returns base file."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        campaign_path = paths.get_campaign_content_path()

        assert campaign_path == Path("config/moprobo/facebook/campaign_content.yaml")

    def test_get_generator_config_path(self):
        """Test consolidated generator config path (from customer config)."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        generator_config_path = paths.get_generator_config_path()

        # Returns customer-specific config.yaml
        assert generator_config_path == Path("config/moprobo/facebook/config.yaml")

    def test_get_ad_miner_output_path(self):
        """Test Ad Miner output path."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        miner_path = paths.get_ad_miner_output_path(product="Power Station")

        path_str = str(miner_path)
        assert "results/moprobo/facebook/ad_miner" in path_str
        assert "power_station" in path_str

    def test_get_ad_generator_base_path(self):
        """Test Ad Generator base path."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        base_path = paths.get_ad_generator_base_path()

        assert base_path == Path("results/moprobo/facebook/ad_generator")

    def test_get_generated_output_path(self):
        """Test generated output path."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        output_path = paths.get_generated_output_path("Power Station")

        path_str = str(output_path)
        assert "results/moprobo/facebook/ad_generator/generated" in path_str
        assert "power_station" in path_str

    def test_get_backgrounds_output_path(self):
        """Test backgrounds output path."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        bg_path = paths.get_backgrounds_output_path("Power Station")

        path_str = str(bg_path)
        assert "results/moprobo/facebook/ad_generator/backgrounds" in path_str
        assert "power_station" in path_str

    def test_get_composited_output_path(self):
        """Test composited output path."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        comp_path = paths.get_composited_output_path("Power Station")

        path_str = str(comp_path)
        assert "results/moprobo/facebook/ad_generator/composited" in path_str
        assert "power_station" in path_str

    def test_get_product_input_path(self):
        """Test product input path."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        product_path = paths.get_product_input_path("Power Station")

        # Products are now in the platform-specific config directory
        path_str = str(product_path)
        assert "config/moprobo/facebook/products" in path_str
        assert "power_station" in path_str

    def test_get_product_input_path_custom_filename(self):
        """Test product input path with custom filename."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        product_path = paths.get_product_input_path(
            "Power Station",
            filename="custom_product.png"
        )

        path_str = str(product_path)
        assert "config/moprobo/facebook/products" in path_str
        assert "custom_product.png" in path_str

    def test_ensure_directories(self, tmp_path):
        """Test directory creation."""
        paths = GeneratorPaths(
            customer="test_customer",
            platform="facebook",
            config_dir=tmp_path / "config",
            output_dir=tmp_path / "results"
        )

        created = paths.ensure_directories(product="Test Product")

        assert len(created) > 0
        for directory in created:
            assert directory.exists()

    def test_repr(self):
        """Test string representation."""
        paths = GeneratorPaths(customer="moprobo", platform="facebook")
        repr_str = repr(paths)

        assert "moprobo" in repr_str
        assert "facebook" in repr_str


class TestCreateCustomerStructure:
    """Test create_customer_structure function."""

    def test_create_customer_structure(self, tmp_path):
        """Test customer structure creation."""
        directories = create_customer_structure(
            customer="test_customer",
            platforms=["facebook", "tiktok"],
            config_dir=tmp_path / "config"
        )

        assert "customer_config" in directories
        assert "facebook_output" in directories
        assert "tiktok_output" in directories

        # Verify directories were created
        assert directories["customer_config"].exists()
        assert directories["facebook_output"].exists()
        assert directories["tiktok_output"].exists()
